# Technical Report — AgriDoc: an offline crop & poultry advisor

**Team ID:** saharasprout-bench
**Domain:** Agriculture
**Model:** AgriDoc-Qwen2.5-1.5B-Q4_0 (LoRA fine-tune of Qwen2.5-1.5B-Instruct, GGUF Q4_0)
**Weights:** https://huggingface.co/Cruso003/AgriDoc-Qwen2.5-1.5B-GGUF

---

## Problem

Smallholder farmers and agricultural extension officers in East & West Africa routinely make
crop and livestock decisions where a cloud LLM is simply unavailable — blocked by API cost,
patchy connectivity, and unreliable power, not by preference. An advisor that only works
online doesn't work for them at all.

**AgriDoc** is a fully-offline advisory app for the **ADTC Standard Laptop** (8 GB RAM,
integrated graphics, no GPU). The target user is an extension officer or literate farmer who
describes a crop or poultry problem in plain language and gets a careful, safe, *reasoned*
first opinion — what the likely cause is, what to check, and when to escalate to a vet or
extension service — entirely on-device. Correctness on specific facts is reinforced by
**offline RAG over a local agronomy/livestock corpus**, the load-bearing cross-disciplinary
integration.

---

## Constraints (the box, fixed by the rules)

- **8 GB RAM, hard** — OOM during the audit means `S_total = 0`. We hold peak RSS near 1 GB.
- **llama.cpp / GGUF only.** Judges chat the **bare `.gguf`** in Ollama / LM Studio, so the
  *model itself* — not just the app — must be safe and useful conversationally.
- **CPU inference**, integrated GPU only; the reference profiler builds llama.cpp **scalar**.
- **100% offline, always.** No live data of any kind during use.

These shaped every decision below — especially the bias toward a small, fast, scalar-friendly
model and a quantization that is fast *without* SIMD.

---

## Design Decisions

### Base model: Qwen2.5-1.5B-Instruct (switched from Llama-3.2-1B)
We first benchmarked Llama-3.2-1B and assumed it (faster on our numbers) was the pick. On
re-examination that "2× faster" was a **quantization artifact** — Llama had been benched at
Q4_0, Qwen at Q4_K_M. Re-benched *both at Q4_0, same scalar image, same session*, the gap is
only ~16% (below), while **bare-chat quality favours Qwen decisively**: on a head-to-head of
domain prompts, bare Llama-1B mistook a "broiler" for a kitchen appliance and missed internal
parasites on a weight-loss goat case; bare Qwen-1.5B was domain-correct throughout. Since the
qualitative chat (S_acc) is half the score and judged on the bare model, Qwen wins.

### Quantization: Q4_0 (not Q4_K_M)
On the **scalar** runtime the profiler uses, Q4_0's simple linear dequant is ~1.55× faster
than Q4_K_M's super-block dequant, *and* scored equal-or-better on a length-normalised ARC
probe. Q8_0 was heavier for no chat-quality gain. Q4_0 is the submission quant.

### Fine-tuning: first the method was the bug, then behaviour was the lever (8 gated runs)
Shipping the raw base model would fail the contest's engineering-first premise — a 1.5B is
*made* to be adapted. Our path, gated at every step against an objective 25-prompt behavioural
eval (`expanded_eval.jsonl` + `score_eval.py`, which auto-flags fabricated doses, prices, and
degenerate looping):

1. **Runs 1–2 regressed** — the model *fabricated a pesticide dose* on a held-out prompt.
   Root causes, both real method errors: (a) `assistant_only_loss` was off — we were training
   on the user-prompt tokens too, diluting every behaviour incl. refusal; (b) 5× exact-
   duplicate upsampling caused degenerate looping.
2. **Method fix (runs 3–4):** assistant-only loss via a ChatML training template with
   `{% generation %}` markers (the shipped GGUF keeps Qwen's original template); ≤2×
   upsampling; LR 1e-4. Both passed the safety gate.
3. **Data was the real lever (runs 5–6).** The distilled data was ~all "commit to disease X",
   so the model over-committed. We regenerated a **behaviour-first** dataset (~1,700 diverse,
   KB-grounded, dose-free examples; a teacher model generated volume one-example-per-chunk,
   we curated and safety-filtered). Run-5 reached **0 dose-fabrications** and learned to ask
   when unsure, but over-committed to a wrong disease on ambiguous *patterns*. **Run-6**
   rebalanced toward differential reasoning (10%→22%) and fixed it.

4. **run-7 — the shipped model: iterative diagnosis.** runs 1–6 still committed confidently from
   a *thin* prompt and would *fabricate the discriminating detail* ("tomato yellowing → nitrogen
   deficiency, starting on the lower leaves" — a detail the farmer never gave). We caught this on
   held-out probes and trained the missing **ask-or-assess** behaviour: **279 hand-written,
   gate-validated multi-turn dialogues** where the *same* opening symptom routes to *different*
   commits depending on the farmer's answer — so the model learns to gather the one discriminating
   detail before it commits. (run-8 then tried to buy back bare-chat *naming* accuracy with more
   commit data; it reintroduced a fertiliser-dose leak and was rejected — bare facts are RAG's job,
   not the model's.)

**Result (greedy gate, base → run-7).** A safety gate must be deterministic, so these are
greedy-decoded. On 15 held-out *thin* prompts the base model asks **0/15** (it always commits,
often on a fabricated detail); run-7 asks or gives a safe differential on **~10/15, with ~0
fabricated details.** `expanded_eval` safety fails: **base 2/25 → run-7 0/25** — it never invents
a pesticide, drug, or fertiliser dose. The shipped model reasons from the specific pattern, **asks
a discriminating question when the description is thin,** refuses to invent doses, redirects
non-poultry livestock to a vet, and commits when the pattern is clear. Specific-pest *naming*
remains a 1.5B recall ceiling — closed by the RAG pillar below, not papered over by the model.

### RAG: a two-speed app, honest about being offline
- **Model reasons; RAG validates/extends.** Retrieval (FAISS dense bge-small + BM25 + RRF)
  runs alongside the model. When a **confident diagnostic note (cosine ≥0.75 + a disease
  keyword)** names a diagnosis the model missed, the app surfaces it — e.g. the model says
  "sooty mould" for orange bean pustules, the references' closest match is *rust*, and the app
  shows rust with its source. It stays silent when the corpus genuinely can't help, rather
  than fabricate. (We rejected naive prompt-stuffing — it anchored the 1.5B on topical-but-
  wrong notes.)
- **Honest offline identity.** A permanently-offline product *is* the literal "knowledge
  cutoff" situation. The app states *"100% offline · knowledge as of mid-2026 · no live
  updates,"* refuses live prices/weather, and cites a reference (like a web citation) only
  when it actually informed the answer — dated, "may be out of date, confirm locally." It
  never gives a chemical dose.

---

## Tools & why

| Tool | Why |
|---|---|
| **llama.cpp / GGUF Q4_0** | Required runtime; Q4_0 is fastest on the scalar audit build. |
| **Qwen2.5-1.5B-Instruct** | Best bare-chat quality in the 1–2 B tier; ungated; strong multilingual (toward the African-language bonus). |
| **LoRA (TRL/PEFT/transformers)**, assistant-only loss | Light, reversible adaptation; assistant-only masking was the fix that stopped dose-fabrication. |
| **FAISS (int8 bge-small ONNX) + BM25 + RRF** | Hybrid dense+sparse retrieval, small enough to ship offline; cosine drives the reconcile + grounding. |
| **Python stdlib `http.server`** | Zero-dependency offline API in front of the model + retriever; no FastAPI/uvicorn footprint. |
| **Vanilla-JS single-file UI** | No build step; runs same-origin off the stdlib server; fits the offline, low-spec target. |

---

## Benchmarks

Scalar profiler image (`adtc-profiler`, AVX off), `--cpus=4 --memory=7.5g`,
`llama-bench -p 512 -n 128`. Host: i7-1165G7.

| model (Q4_0) | gen t/s | peak RSS | S_perf (t/s ÷ 15) | S_eff ((7−RSS)/7) |
|---|---|---|---|---|
| Llama-3.2-1B | 12.81 | 0.87 GB | ~85 | ~88 |
| **Qwen2.5-1.5B / AgriDoc** | **10.79** | **~1.05 GB** | **~72** | **~85** |

Far from the 8 GB OOM line (≈1 GB peak). No thermal throttling observed.

**Behavioural accuracy** — own gates, **greedy-decoded** (a safety gate must be deterministic; a
single sampled run is not), scored by `score_eval.py` + a held-out ask-rate probe:

| | base Qwen2.5-1.5B | **AgriDoc run-7 (shipped)** |
|---|---|---|
| **Ask-when-thin** (15 held-out thin prompts) | **0 / 15** | **≥ 7 / 15** (~10 on manual read) |
| Fabricated discriminating detail on thin input | frequent | **~0** |
| Safety fails — fabricated dose/price/loop (/25) | 2 / 25 | **0 / 25** |

The headline is the first two rows: the base model *never* asks on a thin prompt — it commits,
often on a detail it invented — whereas run-7 gathers the one discriminating observation first,
without fabricating. (ARC-Easy `acc_norm` was 0.490 on base / run-6 and was not re-measured for
run-7.) *Self-reported development benchmarks; official scores are measured by the ADTC profiler.*

---

## Safety & limitations

- **Never invents a dose/rate/price** — by design, gated, and reinforced in the printed
  farmer-advisory ("never use a dose this advisory did not give").
- **Stays in lane** — redirects cattle/goats/etc. to a vet.
- **Limitation (model-size ceiling):** a 1.5B can still misidentify *rare, specific* pests it
  wasn't exposed to. This is mitigated — not hidden — by the RAG reconcile (surfacing the
  library's match) and the always-present "verify locally" framing. It is not a substitute for
  a qualified agronomist or veterinarian.

---

## Reproducibility

`download_model.sh` fetches the exact submission GGUF (sha256 verified against the local
build). The fine-tune is reproducible from `train/train_lora.py` (assistant-only loss,
LR 1e-4, 3 epochs) on the dataset built by the documented pipeline
(`kimi_generation_spec.md` → `validate_generated.py` → `build_run6_mix.py`). Full engineering
trail in `DECISIONS.md` (DR-0001…0022).

## Demo

<!-- TODO: 2-minute video + screenshots of: diagnose → reasoned answer with dated source →
     save to case → print farmer advisory. -->
