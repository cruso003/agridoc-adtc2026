# SaharaSprout RAG

A thin, offline, hybrid-retrieval RAG pipeline for smallholder-farmer
agricultural advice. No LangChain, no torch — just `fastembed` + `faiss-cpu` +
`rank_bm25`, light enough to co-exist with the LLM on an 8 GB laptop.

This one harness is the rig that (1) decides the base model, (2) is Gate 0, and
(3) is the product inference path.

## Install

```bash
pip install -r rag/requirements.txt
```

The embedding model (`bge-small-en-v1.5`, int8 ONNX, CPU) downloads once on the
first index build and is cached. **Inference is then 100% offline** — no cloud
embeddings, no cloud LLM.

## Use

```bash
# 1. (re)build the index from chunks.jsonl — a one-liner; rerun whenever the
#    corpus grows. No code change needed: the pipeline depends on the schema
#    (chunk_id/source/license/topic/text), not the content.
python -m rag index

# 2. start a llama-server with any candidate GGUF (OpenAI-compatible), e.g.
#    via the existing adtc-profiler image, then ask a grounded question:
python -m rag query "My chickens are coughing and dying. What should I do?" \
    --server http://127.0.0.1:8080

# 3. base-model tiebreaker: run all eval prompts through the SAME retrieval
#    across candidates; emits base_tiebreaker.md (+ .json).
python -m rag bench \
    --prompts eval_prompts.jsonl \
    --models smollm2-360m-q4,llama-3.2-1b-q4,qwen2.5-1.5b-q4
```

`rag bench` starts/stops a llama-server per model. By default it uses the
existing `adtc-profiler:latest` Docker image (entrypoint `llama-server`) and
resolves weights from `adtc-bench-workspace/submissions/<key>/model/*.gguf`.
Pass `--launcher exec --server-bin llama-server` to drive a native binary
instead.

## How it works

```
query ─► hybrid retrieve ─► grounded prompt ─► llama-server ─► answer + citations
          │  dense: FAISS IndexFlatIP over normalized bge embeddings (cosine)
          │  sparse: BM25 (rank_bm25) over the same chunks
          └─ fuse: Reciprocal Rank Fusion (k=60); ~20 each → top-5 to the prompt
```

- **System prompt** (`config.SYSTEM_PROMPT`) is the exact string the fine-tune
  data uses, so retrieval-time and training-time contracts match. It forbids
  inventing any dose/rate/threshold/date not in the context and instructs
  abstention + "consult a local extension officer" when context is insufficient.
- **Citations** (chunk_id / source / license) are returned with every answer —
  both a quality signal and the CC-BY-SA attribution mechanism.

## Layout

| file | role |
| --- | --- |
| `config.py` | paths, embedding model, RRF/k & top-k, the exact system prompt |
| `corpus.py` | load + schema-validate chunks.jsonl; BM25 tokenizer |
| `embedder.py` | fastembed bge-small wrapper (normalized, query/passage asymmetric) |
| `index.py` | build/persist/load FAISS + BM25 + metadata |
| `retriever.py` | dense + sparse + RRF fusion |
| `prompt.py` | grounded prompt construction |
| `llm.py` | llama-server OpenAI-compatible client (urllib, no deps) |
| `pipeline.py` | query → answer + citations |
| `server.py` | llama-server lifecycle (docker/exec) for bench |
| `bench.py` | base-model tiebreaker → base_tiebreaker.md |
| `cli.py` / `__main__.py` | `index` / `query` / `bench` |

Indexes and `*.gguf` are git-ignored; the `rag/` code and system prompt are not.
