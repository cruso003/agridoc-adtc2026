"""Build, persist, and load the hybrid index.

Artifacts (all under index_dir, none committed):
    faiss.index    — FAISS IndexFlatIP over L2-normalized embeddings (cosine)
    store.pkl      — chunk metadata+text and BM25 token lists (BM25 rebuilt on
                     load; reconstruction over ~1k docs is instant and avoids
                     pickling library internals across versions)
    manifest.json  — human-readable provenance (model, counts, config)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from . import config, corpus
from .embedder import Embedder

FAISS_FILE = "faiss.index"
STORE_FILE = "store.pkl"
MANIFEST_FILE = "manifest.json"


def build(
    chunks_path: str | Path = config.DEFAULT_CHUNKS,
    index_dir: str | Path = config.DEFAULT_INDEX_DIR,
    embed_model: str = config.EMBED_MODEL,
) -> dict:
    """(Re)build the index from chunks.jsonl. Idempotent — overwrites in place."""
    import faiss

    chunks_path = Path(chunks_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"[index] loading corpus: {chunks_path}")
    chunks = corpus.load_chunks(chunks_path)
    print(f"[index] {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]

    # ── Dense: embed + FAISS inner-product (cosine on normalized vectors) ──
    print(f"[index] embedding with {embed_model} (CPU, int8 ONNX)…")
    embedder = Embedder(embed_model)
    vectors = embedder.embed_passages(texts)  # (N, dim), normalized
    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(index_dir / FAISS_FILE))
    print(f"[index] FAISS IndexFlatIP: {faiss_index.ntotal} vectors, dim={dim}")

    # ── Sparse: BM25 token lists (BM25Okapi rebuilt cheaply at load) ──
    tokens = [corpus.tokenize(t) for t in texts]

    # Metadata table — keep text too, so the prompt builder needs only the store.
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "license": c["license"],
            "topic": c["topic"],
            "text": c["text"],
        }
        for c in chunks
    ]

    with (index_dir / STORE_FILE).open("wb") as fh:
        pickle.dump({"meta": meta, "tokens": tokens}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "embed_model": embed_model,
        "embed_dim": int(dim),
        "n_chunks": len(chunks),
        "chunks_path": str(chunks_path),
        "rrf_k": config.RRF_K,
        "dense_candidates": config.DENSE_CANDIDATES,
        "sparse_candidates": config.SPARSE_CANDIDATES,
        "top_k": config.TOP_K,
    }
    with (index_dir / MANIFEST_FILE).open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[index] wrote {index_dir / FAISS_FILE}, {index_dir / STORE_FILE}, "
          f"{index_dir / MANIFEST_FILE}")
    return manifest


class LoadedIndex:
    """In-memory hybrid index: FAISS handle + BM25 + metadata."""

    def __init__(self, index_dir: str | Path = config.DEFAULT_INDEX_DIR):
        import faiss
        from rank_bm25 import BM25Okapi

        index_dir = Path(index_dir)
        manifest_path = index_dir / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no index at {index_dir} — run `python -m rag index` first"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.faiss = faiss.read_index(str(index_dir / FAISS_FILE))

        with (index_dir / STORE_FILE).open("rb") as fh:
            store = pickle.load(fh)
        self.meta: list[dict] = store["meta"]
        self.bm25 = BM25Okapi(store["tokens"])
        self.embed_model: str = self.manifest["embed_model"]

    def __len__(self) -> int:
        return len(self.meta)
