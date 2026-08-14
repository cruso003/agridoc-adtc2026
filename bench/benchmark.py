"""Benchmark the AgriDoc bare model the way ADTC scores it.

Measures the two performance metrics on CPU (no GPU, as the Standard Laptop):
  - generation throughput  -> S_perf = 100 * (TPS_act / TPS_max)   [tg tok/s]
  - peak RAM               -> S_eff  = 100 * ((7 GB - PeakRAM) / 7 GB)

Uses llama.cpp's own llama-bench for throughput and a peak-RSS sampler for RAM.
Run this ON the ADTC Standard Laptop (Ubuntu 22.04, i5/Ryzen5, 8 GB, integrated GPU)
for the official figures — numbers on other machines are indicative only.

    python benchmark.py --gguf ../model/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf \
                        --llama-bin /path/to/llama.cpp/build/bin

Notes:
  - Generation is memory-bound, so use threads = PHYSICAL cores. Hyper-threading
    lowers tg tok/s (measured: 19 t/s at 4 threads vs 13 t/s at 8 on a 4-core i7).
  - RAM sampler needs `pip install psutil`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time


def _bin(name: str, llama_bin: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    exe = name + (".exe" if os.name == "nt" else "")
    return os.path.join(llama_bin, exe) if llama_bin else exe


def throughput(bench: str, gguf: str, threads: int) -> dict:
    out = subprocess.run(
        [bench, "-m", gguf, "-t", str(threads), "-p", "512", "-n", "128"],
        capture_output=True, text=True,
    ).stdout
    res = {}
    for line in out.splitlines():
        m = re.search(r"\|\s*(pp512|tg128)\s*\|\s*([\d.]+)", line)
        if m:
            res[m.group(1)] = float(m.group(2))
    return res


def peak_ram_mb(cli: str, gguf: str, threads: int, ctx: int) -> float | None:
    try:
        import psutil
    except ImportError:
        print("  (peak-RAM needs: pip install psutil)")
        return None
    p = subprocess.Popen(
        [cli, "-m", gguf, "-t", str(threads), "-n", "96", "-c", str(ctx), "-no-cnv",
         "-p", "Explain briefly why crop rotation improves soil health."],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    proc = psutil.Process(p.pid)
    peak, start = 0, time.time()
    while p.poll() is None and time.time() - start < 90:
        try:
            peak = max(peak, proc.memory_info().rss)
        except Exception:
            pass
        time.sleep(0.1)
    if p.poll() is None:
        p.kill()
    return peak / 1024 / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="../model/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf")
    ap.add_argument("--llama-bin", default=os.environ.get("LLAMA_BIN", ""))
    ap.add_argument("--threads", type=int, default=0, help="0 = physical cores")
    ap.add_argument("--ctx", type=int, default=4096)
    a = ap.parse_args()

    threads = a.threads
    if threads == 0:
        try:
            import psutil
            threads = psutil.cpu_count(logical=False) or 4
        except Exception:
            threads = max(1, (os.cpu_count() or 8) // 2)

    bench, cli = _bin("llama-bench", a.llama_bin), _bin("llama-cli", a.llama_bin)
    print(f"gguf={a.gguf}\nthreads={threads} (physical)  ctx={a.ctx}\n")

    tps = throughput(bench, a.gguf, threads)
    ram = peak_ram_mb(cli, a.gguf, threads, a.ctx)

    tg, pp = tps.get("tg128"), tps.get("pp512")
    TPS_REF, RAM_BUDGET_GB = 15.0, 7.0
    print("=== RESULTS ===")
    if pp:
        print(f"  prompt processing (pp512) : {pp:6.1f} tok/s")
    if tg:
        print(f"  generation      (tg128)   : {tg:6.1f} tok/s   (ADTC provisional ref {TPS_REF})")
    if ram:
        s_eff = 100 * ((RAM_BUDGET_GB - ram / 1024) / RAM_BUDGET_GB)
        print(f"  peak RAM                  : {ram:6.0f} MB  ({ram/1024:.2f} GB)")
        print(f"  S_eff estimate            : {s_eff:6.1f}   (100 x (7GB - {ram/1024:.2f}GB) / 7GB)")
    if tg:
        print(f"  S_perf                    : 100 x (tg / TPS_max); tg here is {tg/TPS_REF:.1f}x the provisional ref")


if __name__ == "__main__":
    main()
