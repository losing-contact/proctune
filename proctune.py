import random
import threading
import time

import numpy as np
import psutil
import sounddevice as sd
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
MAX_VOICES = 32

GAIN = 4.0
FREQ_MIN = 80.0
FREQ_MAX = 2000.0

AMP_SMOOTH = 0.002
NAME_LEN_FULL = 40.0

ARP_SEMITONES = [0, 4, 7, 12]
ARP_MAX_RATE_HZ = 12.0
ARP_MIN_RATE_HZ = 0.0

_state_lock = threading.Lock()
_voices = []
_phase = 0.0  # общая фаза в сэмплах


# PID -> частота
def pid_to_freq(pid: int) -> float:
    f = float(max(pid, 1))
    while f < FREQ_MIN:
        f *= 2.0
    while f > FREQ_MAX:
        f /= 2.0
    return f


def _pick_waveform(mem_percent: float) -> str:
    if mem_percent < 1:
        return "sine"
    if mem_percent < 5:
        return "triangle"
    if mem_percent < 10:
        return "square"
    return "saw"


# арпеджиатор
def _arp_multiplier(voice, t_abs: np.ndarray) -> np.ndarray:
    rate = voice["arp_rate"]
    if rate <= 0.0:
        return np.full_like(t_abs, 2 ** (ARP_SEMITONES[0] / 12), dtype=np.float32)
    step_idx = (t_abs * rate).astype(np.int64) % len(ARP_SEMITONES)
    semis = np.array(ARP_SEMITONES, dtype=np.float32)
    multipliers = (2 ** (semis / 12)).astype(np.float32)
    return multipliers[step_idx]


# колбэк
def audio_callback(outdata, frames, time_info, status):
    global _phase

    with _state_lock:
        voices = _voices

    t_abs = (np.arange(frames) + _phase) / SAMPLE_RATE
    mix = np.zeros(frames, dtype=np.float32)

    for v in voices:
        diff = v["target_amp"] - v["amp"]
        v["amp"] += diff * AMP_SMOOTH
        if v["amp"] < 1e-4 and v["releasing"]:
            continue

        mult = _arp_multiplier(v, t_abs)
        eff_freq = v["freq"] * mult

        if "angle" not in v:
            v["angle"] = 0.0
        angles = v["angle"] + 2 * np.pi * np.cumsum(eff_freq) / SAMPLE_RATE

        if v["wave"] == "sine":
            base = np.sin(angles)
        elif v["wave"] == "triangle":
            base = 2.0 * np.abs(2.0 * ((angles / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        elif v["wave"] == "square":
            base = np.sign(np.sin(angles))
        else:  # saw
            base = 2.0 * ((angles / (2 * np.pi)) % 1.0) - 1.0

        v["angle"] = float(angles[-1] % (2 * np.pi))
        mix += base.astype(np.float32) * v["amp"]

    mix *= GAIN
    outdata[:, 0] = np.tanh(mix)

    _phase += frames
    if _phase > SAMPLE_RATE * 10:
        _phase %= SAMPLE_RATE


# обновление голосов
def _update_voices(shown):
    n = max(len(shown), 1)
    norm_n = np.sqrt(n)

    wanted = {}
    for info in shown:
        pid = info["pid"]
        name = info.get("name") or ""
        cpu = info.get("cpu_percent") or 0.0
        mem = info.get("memory_percent") or 0.0

        target_amp = min((len(name) / NAME_LEN_FULL) / norm_n, 1.0)
        arp_rate = min(cpu * 10.0, 100.0) / 100.0 * ARP_MAX_RATE_HZ

        wanted[pid] = {
            "freq": pid_to_freq(pid),
            "target_amp": float(target_amp),
            "wave": _pick_waveform(mem),
            "arp_rate": float(arp_rate),
        }

    with _state_lock:
        for v in _voices:
            if v["pid"] in wanted:
                w = wanted.pop(v["pid"])
                v["freq"] = w["freq"]
                v["target_amp"] = w["target_amp"]
                v["wave"] = w["wave"]
                v["arp_rate"] = w["arp_rate"]
                v["releasing"] = False
            else:
                v["target_amp"] = 0.0
                v["releasing"] = True

        for pid, w in wanted.items():
            _voices.append({
                "pid": pid,
                "freq": w["freq"],
                "target_amp": w["target_amp"],
                "amp": 0.0,
                "wave": w["wave"],
                "arp_rate": w["arp_rate"],
                "releasing": False,
                "angle": 0.0,
            })

        if len(_voices) > MAX_VOICES * 3:
            _voices.sort(key=lambda v: v["amp"], reverse=True)
            del _voices[MAX_VOICES * 3:]


# таблица
def _build_table(shown):
    table = Table(box=box.ASCII, header_style="", style="", show_edge=True)
    for col, right in [
        ("#", True), ("name", False), ("len", True), ("pid", True),
        ("cpu %", True), ("mem %", True), ("freq", True),
        ("arp Hz", True), ("wave", False),
    ]:
        table.add_column(col, justify="right" if right else "left")

    active = {v["pid"]: v for v in _voices if not v["releasing"]}
    for i, info in enumerate(shown, 1):
        name = info.get("name") or ""
        cpu = info.get("cpu_percent") or 0.0
        mem = info.get("memory_percent") or 0.0
        v = active.get(info["pid"])
        table.add_row(
            str(i),
            name[:22],
            str(len(name)),
            str(info["pid"]),
            f"{cpu:.1f}",
            f"{mem:.1f}",
            f"{v['freq']:.0f}" if v else "-",
            f"{v['arp_rate']:.1f}" if v else "-",
            v["wave"] if v else "",
        )
    return table


# выбор процессов
def _collect_procs():
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            if info["cpu_percent"] is not None:
                procs.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


def _pick(procs, n, mode):
    if not procs:
        return []
    if mode == "top":
        procs.sort(key=lambda x: x["cpu_percent"] or 0, reverse=True)
        return procs[:n]
    return random.sample(procs, min(n, len(procs)))


# мониторинг
def monitor_loop(top_n, interval, mode, live):
    for p in psutil.process_iter(["pid"]):
        try:
            p.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(interval)

    while True:
        procs = _collect_procs()
        if not procs:
            time.sleep(interval)
            continue

        shown = _pick(procs, top_n, mode)
        _update_voices(shown)
        live.update(_build_table(shown))
        time.sleep(interval)


# ввод
def ask_int(prompt, default, lo, hi):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass


def ask_float(prompt, default, choices):
    while True:
        s = input(f"{prompt} {choices} [{default}]: ").strip().replace(",", ".")
        if not s:
            return default
        try:
            v = float(s)
            if v > 0:
                return v
        except ValueError:
            pass


def ask_mode():
    while True:
        s = input("mode: 1 - random, 2 - top: ").strip()
        if s in ("", "1"):
            return "random"
        if s == "2":
            return "top"


def main():
    top_n = ask_int(f"processes (1-{MAX_VOICES})", 5, 1, MAX_VOICES)
    interval = ask_float("interval", 0.5, [0.25, 0.5, 1, 2])
    mode = ask_mode()

    console = Console(color_system=None)
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        mon = threading.Thread(
            target=monitor_loop,
            args=(top_n, interval, mode, live),
            daemon=True,
        )
        mon.start()
        try:
            with sd.OutputStream(
                callback=audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="float32",
            ):
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()