"""Central configuration for the SaharaSprout RAG pipeline.

Everything tunable lives here so re-indexing on a grown corpus is a one-liner
(`python -m rag index`) and the same code runs any candidate GGUF via a config /
CLI parameter. Paths are resolved relative to the repo root, not the CWD.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# repo root = parent of this rag/ package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / "adtc-bench-workspace"

# Default corpus + index locations. Override on the CLI for a grown corpus.
DEFAULT_CHUNKS = WORKSPACE / "chunks.jsonl"
DEFAULT_INDEX_DIR = WORKSPACE / "rag_index"

# Where candidate GGUFs live (submissions/<model-key>/model/*.gguf), matching the
# existing arc_eval / sweep layout so the bench runner reuses downloaded weights.
DEFAULT_SUBMISSIONS = WORKSPACE / "submissions"

# ── Embedding ─────────────────────────────────────────────────────────────────
# Small offline model. fastembed ships BAAI/bge-small-en-v1.5 as an int8 ONNX
# (NOT Q4) — embedding quality is retrieval quality, and this runs on CPU.
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384

# ── Hybrid retrieval ──────────────────────────────────────────────────────────
DENSE_CANDIDATES = 20   # top-k pulled from FAISS before fusion
SPARSE_CANDIDATES = 20  # top-k pulled from BM25 before fusion
RRF_K = 60              # Reciprocal Rank Fusion constant
TOP_K = 5              # chunks passed to the prompt (task says 4–6)

# ── Grounded prompt ───────────────────────────────────────────────────────────
# EXACT system prompt — shared with the fine-tune data so the two stay consistent.
SYSTEM_PROMPT = (
    "You are an agricultural advisor for smallholder farmers. Answer ONLY using the "
    "provided context. If the context does not contain the answer, say so plainly "
    "and recommend consulting a local agricultural extension officer. NEVER state a "
    "dose, application rate, threshold, or date that is not present in the context. "
    "When applicable, structure your answer as likely cause → what to do → "
    "prevention."
)
# NOTE: the model is NOT instructed to cite. Attribution is attached
# programmatically from the retrieved chunk_ids (see prompt.format_context),
# and 0/2221 training answers echo a handle — so a "cite the source(s)"
# instruction would train the model to ignore it. Train == serve.

# ── LLM call (llama-server, OpenAI-compatible) ────────────────────────────────
DEFAULT_SERVER = "http://127.0.0.1:8080"
GEN_TEMPERATURE = 0.0   # deterministic + grounded; we do not want creative dosing
# 384 tokens comfortably fits a cause → action → prevention answer and keeps
# worst-case latency sane on a ~5–9 tok/s CPU build (the slowest candidate,
# qwen-1.5b, otherwise runs the full budget and times out).
GEN_MAX_TOKENS = 384
LLM_TIMEOUT = 300.0     # CPU prefill of ~2–3k-token prompts + slow gen needs headroom
# Server context window. Top-k chunks + system + question can reach ~2.2k tokens,
# so 2048 overflows (HTTP 400). 4096 fits prompt + GEN_MAX_TOKENS with margin and
# stays light enough to co-exist with the LLM on the 8 GB target.
SERVER_CTX_SIZE = 4096
