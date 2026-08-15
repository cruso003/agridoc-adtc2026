# Technical Report — AgriDoc: an offline agriculture advisor

**Team ID:** saharasprout-bench
**Domain:** Agriculture — crop, livestock, weather & market advisory
**Model:** AgriDoc-1.7B-Q4_0 (LoRA fine-tune of **Qwen3-1.7B**, GGUF Q4_0)
**Weights:** https://huggingface.co/Cruso003/AgriDoc-1.7B-GGUF

---

## Problem

Smallholder farmers and agricultural extension officers in Africa routinely make crop and
livestock decisions where a cloud LLM is simply unavailable — blocked by API cost, patchy
connectivity, and unreliable power, not by preference. An advisor that only works online
doesn't work for them at all.

**AgriDoc** is a fully-offline advisory app for the **ADTC Standard Laptop** (8 GB RAM,
integrated graphics, no GPU). The user describes a problem in plain language — a sick crop or
bird, a lame cow, whether to plant on the first rains, when to sell the maize — and gets a
careful, safe, *reasoned* first opinion: the likely cause, what to check, when to escalate to a
vet or extension service, and — for weather and market questions — the judgement without a
fabricated number. Everything runs on-device. Specific facts are reinforced by **offline RAG
over a local agronomy/livestock corpus**, the load-bearing cross-disciplinary integration.

The ADTC agriculture track spans **crop · livestock · weather · market**, and — critically —
the bare `.gguf` is judged conversationally (S_acc, 50%). So the *model itself* must cover all
four, not defer on half of them. This report is mostly the story of getting a sub-2 B model to
do that safely.

---

## Constraints (the box, fixed by the rules)

- **8 GB RAM, hard** — OOM during the audit means `S_total = 0`. We hold peak RSS well under it.
- **llama.cpp / GGUF only.** Judges chat the **bare `.gguf`**, so the *model* — not just the
  app — must be safe and useful conversationally, with no system prompt.
- **CPU inference**, integrated GPU only; the reference profiler builds llama.cpp **scalar**.
- **100% offline, always.** No live data of any kind — which makes fabricated prices and
  weather forecasts a first-class safety concern, not a nicety.

---

## Design Decisions

### Base model: Llama-3.2-1B → Qwen2.5-1.5B → **Qwen3-1.7B** (each gated)
We first benchmarked Llama-3.2-1B and (on a quantization-artifact "2× faster" reading) assumed
it. Re-benched *both at Q4_0, same scalar image*, the speed gap was only ~16% while **bare-chat
quality favoured Qwen decisively** (bare Llama mistook a "broiler" for a kitchen appliance), so
we moved to **Qwen2.5-1.5B-Instruct** and fine-tuned it through nine gated runs (run-9cp).

We then re-opened the base for one reason: the track is **four sub-domains** and the *bare* model
is judged, so livestock/weather/market coverage that the model *defers* on is a direct S_acc loss
RAG can't recover. We tested **Qwen3-1.7B** — same speed tier, and its base already refuses
fabricated prices/forecasts out of the box (a safety property that cost Qwen2.5 several runs). A
**controlled A/B settled it** (see below): training the *identical* breadth dataset on both bases,
Qwen3 was safer, more stable, and materially more accurate. Qwen3-1.7B is the submission base;
run-9cp (Qwen2.5) is preserved as the proven fallback.

### Quantization: Q4_0 (not Q4_K_M)
On the **scalar** runtime the profiler uses, Q4_0's simple linear dequant is ~1.55× faster than
Q4_K_M's super-block dequant, *and* scored equal-or-better on a length-normalised ARC probe.
Q4_0 is the submission quant on both bases.

### Fine-tuning: behaviour, not facts — gated like a judge at every step
Shipping a raw base would fail the contest's engineering-first premise. Our whole approach is
**train the reasoning/safety *behaviour* and let RAG own the *facts*** — gated at every step
against (i) a greedy behavioural eval that auto-flags fabricated doses/prices/forecasts and
looping, and (ii) an **independent, blind reviewer chatting the bare gguf** the way judges do
(no system prompt, sampled, single- *and* multi-turn, adversarial). The second is decisive.

**Lineage on Qwen2.5 (runs 1→9cp), the hard lessons that carried forward:**
- **Runs 1–2** fabricated a pesticide dose — two real method bugs: `assistant_only_loss` was off
  (training on user tokens diluted every behaviour), and 5× duplicate upsampling caused looping.
  Fixed with a ChatML training template carrying `{% generation %}` markers and ≤2× upsampling.
- **Runs 5–7** moved from "always commit to disease X" to **ask-or-assess** — ask the
  discriminating question when the description is thin, commit when the pattern is clear (run-8
  tried to buy back naming accuracy with commit data; it re-leaked a dose and was rejected —
  *bare facts are RAG's job*).
- **run-9 → 9cp** fixed three defects a blind reviewer exposed that our own gate had *flattered*:
  (a) **looping was a training bug** — the assistant's `<|im_end|>` sat *outside* the loss span so
  the model got no signal to emit EOS; moving it *inside* taught it to stop; (b) **upweighted
  refusals + multi-turn "just approximate it" dialogues** so refusals hold under a push; (c) the
  **identity leak was the chat template, not the weights** — we swapped Qwen2.5's "You are Qwen…"
  default system for the **AgriDoc persona**. *Lesson: gate like a judge, or you fool yourself.*

**Track-breadth retrain on Qwen3 (run-10 → 10d), the submission model.** The lineage above is a
crop/poultry model that *defers* on livestock/weather/market. run-10 adds that breadth as
**behaviour**, on the stronger base, reusing every lesson:
- **Data:** extension-officer dialogues for **market-timing judgement** (harvest glut → store to
  the lean season → grade → sell as a group — *never a live price*), **weather judgement** (a
  single shower is a false start → wait for the rains to settle → stagger planting — *never a
  forecast*), and **livestock breadth** (cattle/sheep/pig: immediate safe field action → refuse
  the dose → escalate to an animal-health worker). Plus **two new refusal classes — price and
  forecast — trained exactly like the proven dose refusal**, single- and multi-turn.
- **Port to Qwen3 (a few adaptive tweaks, not a rewrite):** the ChatML EOS-in-loss template
  applies unchanged (the mask check passed on Qwen3); Qwen3 doesn't hardcode a "You are Qwen"
  default, so the AgriDoc persona is injected as the default system; the shipped template defaults
  to **non-thinking** so the bare model answers cleanly at full speed.
- **Gated exactly like run-9** (10 → 10b → 10c → 10d): 10 leaked identity + price → fixed with the
  persona template + price upweighting; 10b/10c broke then held the **multi-turn** price/forecast
  push (the run-9b failure mode); **run-10d is clean** — **0 real safety leaks** across the full
  sampled + multi-turn adversarial gate (two initial flags were false-positives of our own forecast
  detector on advisory/refusal phrasing — tightened and re-scanned to 0).

### The controlled A/B: Qwen3 is the better base, not a lateral move
To be sure the breadth win was the *base* and not just the *data*, we trained the **identical
run-10d dataset on Qwen2.5-1.5B** (same persona swap, same pipeline — base as the only variable)
and gated it identically:

| Same data, base only | **Qwen3-1.7B (run-10d)** | Qwen2.5-1.5B |
|---|---|---|
| Real safety leaks | **0** | 2 (a price, and a multi-turn dose) |
| Looping / stop failures | **0** | 2 |
| Disease misID (accuracy) | **3** | 9 |

Qwen2.5 **trades away disease accuracy to absorb the breadth** (misID 9, vs ~3.5 on the narrow
run-9cp); **Qwen3 holds it** (misID 3) *and* stays clean on safety and stopping. Its extra
capability does real work. The cost (below) is ~3 weighted perf/eff points, almost all
throughput — bought genuine coverage + accuracy.

### RAG: a two-speed app, honest about being offline
The model reasons; retrieval (FAISS dense bge-small + BM25 + RRF) validates and extends. When a
**confident diagnostic note (cosine ≥ 0.75 + a disease keyword)** names a diagnosis the model
missed, the app surfaces it *with its source*; it stays silent when the corpus can't help rather
than fabricate. (We rejected prompt-stuffing — it anchored the small model on topical-but-wrong
notes.) A permanently-offline product *is* the literal knowledge-cutoff case: the app states
*"100% offline · knowledge as of mid-2026 · no live updates,"* refuses live prices/weather, cites
a dated source only when it informed the answer, and never gives a chemical dose.

---

## Tools & why

| Tool | Why |
|---|---|
| **llama.cpp / GGUF Q4_0** | Required runtime; Q4_0 is fastest on the scalar audit build. |
| **Qwen3-1.7B** | Best safety+accuracy in the sub-2 B tier on our controlled A/B; ungated; strong multilingual; supports a clean non-thinking mode for full-speed bare chat. |
| **LoRA (TRL/PEFT/transformers)**, assistant-only loss | Light, reversible; assistant-only masking (with EOS *inside* the loss span) is what stops dose-fabrication *and* looping. |
| **FAISS (int8 bge-small ONNX) + BM25 + RRF** | Hybrid dense+sparse retrieval, small enough to ship offline; cosine drives the reconcile + grounding. |
| **Python stdlib `http.server`** | Zero-dependency offline API in front of the model + retriever. |
| **Vite/React SPA (offline bundle)** | Full record-keeping + planning UI, trilingual (EN/Kiswahili/Setswana), runs on the laptop with no network. |

---

## Benchmarks

**The setup — read the numbers in this frame.** These are measured with the *official*
`adtc-profiler`, whose Docker image deliberately builds llama.cpp **fully scalar** — `AVX`,
`AVX2`, `AVX512`, `FMA` and `F16C` all compiled **off**. That is not our choice; it is the
audit's, so every submission is scored on the same instruction-set floor regardless of the
grader's CPU. A scalar build has **no SIMD**, so it runs several times slower than any real
deployment: the figures below are a **worst-case floor**, not the speed a farmer's laptop
actually sees (llama.cpp auto-selects AVX2/AVX512 at runtime — our own Icelake build generates
2–3× faster). We report the floor because that is what the leaderboard scores. Run:
`adtc-profiler run --mode participant --cpus=4 --memory=7.5g` (`llama-bench -p 512 -n 128`), host
i7-1165G7, not throttled (P_thermal = 0).

| model (Q4_0) | gen t/s (scalar floor) | peak RSS | S_perf `min(t/s÷15,1)` | S_eff `(7−RSS)/7` |
|---|---|---|---|---|
| Llama-3.2-1B | 12.81 | 0.87 GB | ~85 | ~88 |
| Qwen2.5-1.5B (run-9cp, fallback) | 10.79 | 1.05 GB | 72 | 85 |
| **Qwen3-1.7B / AgriDoc (shipped)** | **9.4** | **1.17 GB** | **63** | **83** |

Against the 1.5 B fallback the shipped 1.7 B costs **~9 S_perf and ~2 S_eff points (~3 weighted)**
— almost entirely throughput; the extra 0.2 B slows pure-scalar generation, while peak RSS rises
only ~0.12 GB and stays far under the 8 GB OOM line. *(An AVX same-image bench had suggested a
speed tie; on the scalar audit build it is not — which is exactly why we measured on the profiler
rather than trusting the estimate.)* We spend that ~3 points deliberately: the other **50 %** of
the score (S_acc) rewards covering all four track sub-domains in the *bare* model, which the 1.7 B
does and the bare 1.5 B defers on (see the controlled A/B). Prompt-processing/prefill is likewise
slow on the no-SIMD build — not the scored metric; a real AVX deployment prefills far faster.

**Behavioural gate (bare gguf, blind, sampled, single- + multi-turn, adversarial):**

| | run-9cp (Qwen2.5) | **AgriDoc run-10d (shipped)** |
|---|---|---|
| Looping / stop failures | 0 | **0** |
| Dose / price / **forecast** refusals (single + multi-turn push) | dose/price clean | **all three clean, 0 real leaks** |
| Track coverage (crop · poultry · livestock · weather · market) | crop/poultry; defers off-lane | **all four — safe first opinion, no fabricated price/forecast** |
| Ask-when-thin | works | works |
| Self-identifies as | AgriDoc | **AgriDoc** |
| Disease-ID accuracy | ~3.5 (RAG-covered) | **~3.5 (RAG-covered)** |

*Self-reported development benchmarks; official scores are measured by the ADTC profiler.*

---

## Safety & limitations

- **Never invents a dose, a price, or a weather forecast** — three refusal classes, trained
  single- *and* multi-turn (the "just approximate it" / "yes or no" pushes), reinforced by the
  gguf's default AgriDoc persona, and backstopped in the product path by an app-layer guard that
  redacts any dose/rate/ratio and points to the label, local buyers, or an agro-met officer.
- **Stops cleanly** — no run-on/looping (EOS is inside the assistant-only loss span; the mask
  check aborts training if it ever isn't).
- **Covers the track safely** — gives a safe first opinion + escalation for livestock, and
  market/weather *judgement* without a fabricated number.
- **Limitation (model-size ceiling), stated plainly:** a sub-2 B model has limited factual recall
  — it can misidentify *specific, less-common* diseases, and on an *invented* disease name it may
  describe a plausible-sounding one. This is the reasoning-and-safety half of a two-speed design;
  **specific facts are the RAG layer's job**, surfaced with a source and a standing "verify
  locally." It is not a substitute for a qualified agronomist or veterinarian.

---

## Reproducibility

`download_model.sh` fetches the exact submission GGUF (sha256 verified against the local build).
The fine-tune is reproducible from `train/train_lora.py` (assistant-only loss with EOS inside the
loss span, LR 1e-4, 3 epochs, `BASE_MODEL=Qwen/Qwen3-1.7B`) on the run-10d mix built by the
documented pipeline (`build_run10*_mix.py`). The full decision trail — every run, gate, and the
controlled A/B — is in `DECISIONS.md` (DR-0019 → DR-0026).

## Demo

The demo walks the judged path end to end: a plain-language crop, livestock, weather, or market
question → a careful, reasoned first-opinion answer (with a dated offline source, or the
deterministic plan for "what needs me this week") → save to the record → print a farmer advisory.
See the Quick start in `README.md`.
