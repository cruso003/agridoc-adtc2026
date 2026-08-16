"""Offline embedding via fastembed (bge-small-en-v1.5, int8 ONNX, CPU).

bge models use an asymmetric retrieval scheme: queries get an instruction
prefix, passages do not. fastembed's query_embed/passage_embed handle that for
us. We L2-normalize on our side so a FAISS inner-product index == cosine.
"""
from __future__ import annotations

import numpy as np

from . import config


class Embedder:
    """Thin wrapper around fastembed.TextEmbedding.

    The ONNX weights download once on first use (build-time, one-off) and are
    cached locally; every call after that is fully offline.
    """

    def __init__(self, model_name: str = config.EMBED_MODEL):
        import os

        from fastembed import TextEmbedding  # imported lazily — heavy-ish

        self.model_name = model_name
        # In the packaged app the embedder ONNX is bundled under this cache so a clean machine
        # never downloads; unset (dev) → fastembed's default cache, unchanged behaviour.
        cache_dir = os.environ.get("AGRIDOC_FASTEMBED_CACHE") or None
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed corpus chunks (no query instruction). Returns (N, dim) float32."""
        vecs = list(self._model.passage_embed(texts))
        return self._normalize(np.vstack(vecs))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query (with bge query instruction). Returns (dim,)."""
        vec = next(iter(self._model.query_embed([text])))
        return self._normalize(np.asarray(vec))[0]
