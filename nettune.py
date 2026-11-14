import math
import os
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
from scapy.all import AsyncSniffer, ICMP, IP, IPv6, TCP, UDP

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
MAX_VOICES = 32

GAIN = 4.0
FREQ_MIN = 80.0
FREQ_MAX = 2000.0

AMP_SMOOTH = 1 #0.002
BYTES_FULL = 100_000

ARP_SEMITONES = [0, 4, 7, 12]
ARP_MAX_RATE_HZ = 12.0
ARP_PPS_FULL = 50.0

_state_lock = threading.Lock()
_voices = []
_phase = 0.0

_count_lock = threading.Lock()
_counts = {}  # (proto, src, dst, dport)


# порт частота
def num_to_freq(n: int) -> float:
    f = float(max(n, 1))
    while f < FREQ_MIN:
        f *= 2.0
    while f > FREQ_MAX:
        f /= 2.0
    return f


def _pick_waveform(proto: str) -> str:
    if proto == "ICMP":
        return "sine"
    if proto == "UDP":
        return "triangle"
    if proto == "TCP":
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
        voices = list(_voices)

    t_abs = (np.arange(frames) + _phase) / SAMPLE_RATE
    mix = np.zeros(frames, dtype=np.float32)

    for v in voices:
        diff = v["target_amp"] - v["amp"]
        v["amp"] += diff * AMP_SMOOTH
        if v["amp"] < 1e-4 and v["releasing"]:
            continue

        mult = _arp_multiplier(v, t_abs)
        eff_freq = v["freq"] * mult
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
    norm_n = np.sqrt(max(len(shown), 1))

    wanted = {}
    for f in shown:
        amp = min(math.log10(f["bytes"] + 1) / math.log10(BYTES_FULL + 1), 1.0)
        arp_rate = min(f["pps"] / ARP_PPS_FULL, 1.0) * ARP_MAX_RATE_HZ
        wanted[f["key"]] = {
            "freq": num_to_freq(f["tone"]),
            "target_amp": float(amp / norm_n),
            "wave": _pick_waveform(f["proto"]),
            "arp_rate": float(arp_rate),
        }

    with _state_lock:
        for v in _voices:
            if v["key"] in wanted:
                w = wanted.pop(v["key"])
                v["freq"] = w["freq"]
                v["target_amp"] = w["target_amp"]
                v["wave"] = w["wave"]
                v["arp_rate"] = w["arp_rate"]
                v["releasing"] = False
            else:
                v["target_amp"] = 0.0
                v["releasing"] = True

        for key, w in wanted.items():
            _voices.append({
                "key": key,
                "freq": w["freq"],
                "target_amp": w["target_amp"],
                "amp": 0.0,
                "wave": w["wave"],
                "arp_rate": w["arp_rate"],
                "releasing": False,
                "angle": 0.0,
            })

        _voices[:] = [v for v in _voices if not (v["releasing"] and v["amp"] < 1e-3)]
        if len(_voices) > MAX_VOICES * 3:
            _voices.sort(key=lambda v: v["amp"], reverse=True)
            del _voices[MAX_VOICES * 3:]


# таблица
def _build_table(shown):
    table = Table(box=box.ASCII, header_style="", style="")
    for col, right in [
        ("#", True), ("proto", False), ("src", False), ("dst", False),
        ("port", True), ("pkts", True), ("bytes", True),
        ("freq", True), ("arp Hz", True), ("wave", False),
    ]:
        table.add_column(col, justify="right" if right else "left")

    active = {v["key"]: v for v in _voices if not v["releasing"]}
    for i, f in enumerate(shown, 1):
        v = active.get(f["key"])
        table.add_row(
            str(i),
            f["proto"],
            f["src"][:39],
            f["dst"][:39],
            str(f["port"]) if f["port"] else "-",
            str(f["pkts"]),
            str(f["bytes"]),
            f"{v['freq']:.0f}" if v else "-",
            f"{v['arp_rate']:.1f}" if v else "-",
            v["wave"] if v else "",
        )
    return table


# перехват пакетов
def on_packet(pkt):
    try:
        _handle(pkt)
    except Exception:
        pass


def _handle(pkt):
    if IP in pkt:
        src, dst = pkt[IP].src, pkt[IP].dst
        tone_fallback = int(dst.split(".")[-1]) + 1
    elif IPv6 in pkt:
        src, dst = pkt[IPv6].src, pkt[IPv6].dst
        tone_fallback = int(dst.split(":")[-1] or "0", 16) + 1
    else:
        return

    if TCP in pkt:
        proto, dport = "TCP", int(pkt[TCP].dport)
    elif UDP in pkt:
        proto, dport = "UDP", int(pkt[UDP].dport)
    elif ICMP in pkt:
        proto, dport = "ICMP", 0
    else:
        proto, dport = "OTHER", 0

    with _count_lock:
        c = _counts.setdefault((proto, src, dst, dport, tone_fallback), [0, 0])
        c[0] += 1
        c[1] += len(pkt)


def _snapshot(interval):
    with _count_lock:
        snap = dict(_counts)
        _counts.clear()
    flows = []
    for (proto, src, dst, dport, fb), (pkts, size) in snap.items():
        flows.append({
            "key": f"{proto}|{src}|{dst}|{dport}",
            "proto": proto,
            "src": src,
            "dst": dst,
            "port": dport,
            "tone": dport or fb,
            "pkts": pkts,
            "bytes": size,
            "pps": pkts / interval,
        })
    return flows


def _pick(flows, n, mode):
    if not flows:
        return []
    if mode == "top":
        flows.sort(key=lambda x: x["bytes"], reverse=True)
        return flows[:n]
    return random.sample(flows, min(n, len(flows)))


# мониторинг
def monitor_loop(top_n, interval, mode, live):
    while True:
        time.sleep(interval)
        shown = _pick(_snapshot(interval), top_n, mode)
        _update_voices(shown)
        live.update(_build_table(shown))


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
        s = input("mode: 1 - top , 2 - random [1]: ").strip()
        if s in ("", "1"):
            return "top"
        if s == "2":
            return "random"


def ask_iface():
    names = sorted(psutil.net_if_addrs().keys())
    for i, name in enumerate(names, 1):
        print(f"{i} - {name}")
    while True:
        s = input("interface (Enter - all): ").strip()
        if not s:
            return None
        if s.isdigit() and 1 <= int(s) <= len(names):
            return names[int(s) - 1]


def main():
    iface = ask_iface()
    top_n = ask_int(f"threads (1-{MAX_VOICES})", 10, 1, MAX_VOICES)
    interval = ask_float("интервал, сек", 1.0, [0.5, 1, 2, 3])
    mode = ask_mode()

    sniffer = AsyncSniffer(iface=iface, prn=on_packet, store=False)
    sniffer.start()
    time.sleep(0.5)
    exc = getattr(sniffer, "exception", None)
    if exc or not sniffer.running:
        print(f"err: {exc}")
        return

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

    try:
        sniffer.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()



   # sudo setcap cap_net_raw,cap_net_admin=eip (readlink -f .venv/bin/python)
#python nettune.py