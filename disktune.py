import math
import os
import threading
import time
import zlib

import numpy as np
import psutil
import sounddevice as sd
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

GAIN = 0.8
AMP_SMOOTH = 0.3

CHUNK = 4096
ROWS_SHOWN = 15

_voice = {
    "chunk": np.zeros(CHUNK, dtype=np.float32),
    "rate": 1.0,
    "ptr": 0.0,
    "amp": 0.0,
    "target_amp": 0.0,
}


# байты куска файла проигрываются по кругу как сэмплы
def audio_callback(outdata, frames, time_info, status):
    v = _voice
    chunk = v["chunk"]
    n = len(chunk)

    v["amp"] += (v["target_amp"] - v["amp"]) * AMP_SMOOTH

    pos = (v["ptr"] + np.arange(frames) * v["rate"]) % n
    out = chunk[pos.astype(np.int64)]
    v["ptr"] = float((v["ptr"] + frames * v["rate"]) % n)

    outdata[:, 0] = np.tanh(out * v["amp"] * GAIN)


# у каждого файла своя базовая скорость проигрывания
def file_rate(name: str) -> float:
    h = zlib.crc32(name.encode())
    return 2 ** ((h % 1000) / 1000 * 3 - 1.5)


def play(data: bytes, base_rate: float):
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    p = counts[counts > 0] / arr.size
    entropy = float(-(p * np.log2(p)).sum())

    chunk = (arr.astype(np.float32) - float(arr.mean())) / 128.0
    rate = base_rate * 2 ** ((zlib.crc32(data) % 100) / 100 - 0.5)

    _voice["chunk"] = chunk
    _voice["rate"] = rate
    return entropy, rate


# обход диска
def collect_files(root):
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            if os.path.isfile(path) and not os.path.islink(path):
                yield path


def chunk_offsets(size, max_notes):
    total = max(1, math.ceil(size / CHUNK))
    if max_notes and total > max_notes:
        idxs = np.linspace(0, total - 1, max_notes).astype(int)
        return [int(i) * CHUNK for i in idxs]
    return [i * CHUNK for i in range(total)]


def _short(path, width=36):
    return path if len(path) <= width else "..." + path[-(width - 3):]


# таблица
def build_table(rows):
    table = Table(box=box.ASCII, header_style="", style="")
    for col, right in [
        ("#", True), ("file", False), ("size", True), ("read %", True),
        ("entropy", True), ("rate", True),
    ]:
        table.add_column(col, justify="right" if right else "left")
    for r in rows[-ROWS_SHOWN:]:
        if r["error"]:
            entropy, rate = "error", ""
        elif r["rate"]:
            entropy, rate = f"{r['entropy']:.1f}", f"{r['rate']:.2f}"
        else:
            entropy, rate = "-", "-"
        table.add_row(
            str(r["n"]),
            _short(r["name"]),
            str(r["size"]),
            f"{r['pos']:.0f}",
            entropy,
            rate,
        )
    return table


# чтение///воспроизведение
def reader_loop(root, rate, max_notes, live, done):
    step = 1.0 / rate
    rows = []
    n = 0
    _voice["target_amp"] = 1.0

    for path in collect_files(root):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size == 0:
            continue

        n += 1
        row = {
            "n": n, "name": os.path.relpath(path, root), "size": size,
            "pos": 0.0, "entropy": 0.0, "rate": 0.0, "error": False,
        }
        rows.append(row)

        offsets = chunk_offsets(size, max_notes)
        base_rate = file_rate(row["name"])
        try:
            with open(path, "rb") as f:
                for i, off in enumerate(offsets, 1):
                    t0 = time.time()
                    f.seek(off)
                    data = f.read(CHUNK)
                    if not data:
                        break
                    row["entropy"], row["rate"] = play(data, base_rate)
                    row["pos"] = 100.0 * i / len(offsets)
                    live.update(build_table(rows))
                    time.sleep(max(0.0, step - (time.time() - t0)))
        except OSError:
            row["error"] = True
            live.update(build_table(rows))

    _voice["target_amp"] = 0.0
    time.sleep(1.0)
    done.set()


# ввод
def ask_disc():
    parts = psutil.disk_partitions(all=False)
    optical = [p for p in parts if p.fstype in ("iso9660", "udf")]
    ordered = optical + [p for p in parts if p not in optical]
    for i, p in enumerate(ordered, 1):
        print(f"{i} - {p.mountpoint} ({p.fstype})")
    default = "1" if ordered else ""
    while True:
        hint = f" [{default}]" if default else ""
        s = input(f"диск (номер или путь){hint}: ").strip() or default
        if s.isdigit() and 1 <= int(s) <= len(ordered):
            return ordered[int(s) - 1].mountpoint
        path = os.path.expanduser(s)
        if s and os.path.isdir(path):
            return path


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


def main():
    root = ask_disc()
    rate = ask_int("(1-30)", 8, 1, 30)
    cap = ask_int("max_file time", 10, 0, 3600)
    max_notes = cap * rate

    done = threading.Event()
    console = Console(color_system=None)
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        reader = threading.Thread(
            target=reader_loop,
            args=(root, rate, max_notes, live, done),
            daemon=True,
        )
        reader.start()
        try:
            with sd.OutputStream(
                callback=audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="float32",
            ):
                while not done.is_set():
                    time.sleep(0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
