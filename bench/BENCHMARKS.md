# Performance benchmarks — AgriDoc (run-9cp, Q4_0)

Measured with llama.cpp's own `llama-bench` (throughput) and a peak-RSS sampler (RAM),
CPU-only (no GPU, as the Standard Laptop). Reproduce with `benchmark.py`.

> **These figures are indicative.** They were taken on a developer machine
> (Intel i7-1165G7, 4c/8t, Windows, CPU build), *not* the ADTC Standard Laptop
> (Ubuntu 22.04, i5 10–12th / Ryzen 5 3000–5000, 8 GB, integrated GPU). Rerun
> `benchmark.py` on the reference laptop for the official S_perf / S_eff.

## Results

| Metric | Value | ADTC scoring |
|---|---|---|
| Model | qwen2 **1.54 B**, Q4_0, **886 MiB** on disk | — |
| Generation throughput (tg128) | **19.1 tok/s** @ 4 threads | S_perf = 100·(TPS/TPS_max); **1.27×** the provisional ref (15) |
| Prompt processing (pp512) | **79 tok/s** @ 4 threads | (prefill; not the scored metric) |
| Peak RAM (ctx 4096) | **~1.78 GB** | S_eff = 100·((7−1.78)/7) ≈ **75** |

## Key findings (design implications for the report)

1. **Use threads = physical cores, not logical.** Generation is memory-bandwidth-bound,
   so hyper-threading *hurts* it: **19.1 tok/s at 4 threads vs 13.3 tok/s at 8** on this
   4-core chip. Prompt processing (compute-bound) does scale up (79 → 89 tok/s), but the
   scored metric is generation. Pin threads to physical cores.

2. **RAM is nearly flat across context length.** Peak RSS was 1.69 GB @ ctx 512,
   1.73 GB @ 2048, 1.78 GB @ 4096 — dominated by weights + compute buffer + runtime, not
   the KV cache. So we keep a useful 4096 context **without** paying an S_eff penalty.

3. **Huge headroom under the 8 GB machine / 7 GB budget.** At ~1.8 GB peak the model uses
   about a quarter of the budget — no OOM risk (OOM = disqualification), and S_eff stays high.

4. **Thermal.** A chat turn completes in a few seconds of CPU load, not a sustained burn, so
   package-temperature throttling (>85 °C = −10) is unlikely in normal use. Confirm on the
   reference laptop under the audit's actual run.

## Why the 1.5 B / Q4_0 choice is rubric-aligned

S_perf (30%) + S_eff (20%) = **half the score rewards small-and-fast**, and OOM is an instant
disqualification. A 1.5 B Q4_0 is fast (≈19 tok/s), light (~1.8 GB), and safe on 8 GB — it is
matched to the rubric, not a compromise. Accuracy is lifted by the offline RAG corpus, which
costs nothing on throughput or memory (the corpus is retrieved, not loaded into the model).

## Reproduce

```sh
pip install psutil
# point --llama-bin at your llama.cpp build (containing llama-bench + llama-cli)
python benchmark.py --gguf ../model/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf \
                    --llama-bin /path/to/llama.cpp/build/bin
```
