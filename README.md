# AgriDoc — offline crop & poultry advisor (ADTC 2026, Agriculture)

An offline agricultural & poultry advisory assistant for smallholder farmers and extension
officers in East & West Africa, built for the **Africa Deep Tech Challenge 2026** Laptop-LLM
track. It runs fully offline on a commodity 8 GB laptop via `llama.cpp`.

- **Model:** `AgriDoc-Qwen2.5-1.5B-Q4_0` — our LoRA fine-tune of Qwen2.5-1.5B-Instruct (GGUF Q4_0)
- **Weights:** https://huggingface.co/Cruso003/AgriDoc-Qwen2.5-1.5B-GGUF
- **Full writeup:** [`REPORT.md`](REPORT.md)

## Quick start (the judged path)
```bash
bash download_model.sh          # fetches model/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf (≈935 MB, public)
# then chat it offline in Ollama / LM Studio / llama.cpp — standard ChatML, no system prompt needed
llama-server -m model/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf --ctx-size 4096
```
The model reasons like a careful extension officer: it commits when symptoms are clear, gives
a differential and asks a question when they're ambiguous, **refuses to invent pesticide/drug
doses**, and redirects non-poultry livestock to a vet.

## The full app (cross-disciplinary RAG)
`app/` is the two-speed product: the model reasons, and **offline RAG over a local
crop/livestock corpus** validates/extends it (and surfaces a source when a reference informed
the answer). The model is the safe reasoner; RAG supplies specific factual grounding — the
load-bearing cross-disciplinary integration.
```bash
# starts the model server + the offline API/UI together (see app/run_agridoc.sh)
bash app/run_agridoc.sh         # → http://127.0.0.1:8000
```
> Note: the full app needs the local retrieval index built from the corpus; the **bare model**
> above runs with zero setup. The app code here documents the architecture described in REPORT.

## Structure
```
metadata.json        ADTC submission metadata (model, domain, 2 test prompts)
download_model.sh    fetches the GGUF weight into model/ (idempotent, no credentials)
REPORT.md            technical writeup — problem, design journey, benchmarks, safety
model/               weight lives here after download (git-ignored)
app/                 the offline app: rag/ (backend) + web/ (UI) + run_agridoc.sh
training/            reproducibility — fine-tune + data-pipeline + eval scripts (no bulky data)
```

## Reproducibility
The fine-tune is reproducible from `training/train_lora.py` (assistant-only loss, LR 1e-4,
3 epochs) on a behaviour-first dataset built by the documented pipeline
(`kimi_generation_spec.md` → `validate_generated.py` → `build_run6_mix.py`) and gated with
`score_eval.py` + `expanded_eval.jsonl`. See REPORT.md for the full 6-run journey.

## License & acknowledgements
Code: see [`LICENSE`](LICENSE). Model weights: Apache-2.0 (inherits Qwen2.5-1.5B-Instruct).
Built with open-source tools — llama.cpp, Hugging Face TRL/PEFT/transformers, FAISS, BM25,
bge-small — all cited in REPORT.md. Reference corpus includes public Wikipedia/FAO material
and AgriDoc synthesized advisory notes (flagged as not independently verified).
