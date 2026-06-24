"""Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with RRF.

Reciprocal Rank Fusion is rank-based, so it sidesteps the apples-to-oranges
problem of combining cosine similarities with BM25 scores. For a domain full of
exact-match disease/crop names, the sparse channel catches literal terms the
dense channel smooths over, and vice-versa.
"""
from __future__ import annotations

import numpy as np

from . import config, corpus
from .embedder import Embedder
from .index import LoadedIndex


def _rrf(rank_lists: list[list[int]], k: int) -> dict[int, float]:
    """Fuse ranked id-lists. score(d) = Σ 1/(k + rank), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    """Holds the loaded index + embedder; retrieves fused top-k chunks."""

    def __init__(self, index: LoadedIndex, embedder: Embedder | None = None):
        self.index = index
        # Reuse the SAME embedding model the index was built with.
        self.embedder = embedder or Embedder(index.embed_model)

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K,
        dense_n: int = config.DENSE_CANDIDATES,
        sparse_n: int = config.SPARSE_CANDIDATES,
        rrf_k: int = config.RRF_K,
    ) -> list[dict]:
        """Return up to top_k chunk dicts (meta+text) with fusion diagnostics."""
        # ── Dense ──
        qvec = self.embedder.embed_query(query).astype(np.float32)[None, :]
        dn = min(dense_n, len(self.index))
        dscores, dense_ids = self.index.faiss.search(qvec, dn)
        dense_ranked = [int(i) for i in dense_ids[0] if i != -1]
        # cosine score per id (IndexFlatIP over normalized vectors == cosine), for the
        # app-side grounding decision (system decides "grounded", not the model).
        dense_score = {int(i): float(s) for i, s in zip(dense_ids[0], dscores[0]) if i != -1}

        # ── Sparse ──
        q_tokens = corpus.tokenize(query)
        bm25_scores = self.index.bm25.get_scores(q_tokens)
        sn = min(sparse_n, len(self.index))
        sparse_ranked = [int(i) for i in np.argsort(bm25_scores)[::-1][:sn]]

        # ── Fuse ──
        fused = _rrf([dense_ranked, sparse_ranked], k=rrf_k)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        dense_rank = {idx: r for r, idx in enumerate(dense_ranked, 1)}
        sparse_rank = {idx: r for r, idx in enumerate(sparse_ranked, 1)}

        results: list[dict] = []
        for idx, score in ordered:
            rec = dict(self.index.meta[idx])  # copy: chunk_id/source/license/topic/text
            rec["_rrf_score"] = round(score, 6)
            rec["_dense_rank"] = dense_rank.get(idx)
            rec["_sparse_rank"] = sparse_rank.get(idx)
            rec["_dense_score"] = round(dense_score.get(idx, 0.0), 4)  # cosine, for grounding
            results.append(rec)
        return results
