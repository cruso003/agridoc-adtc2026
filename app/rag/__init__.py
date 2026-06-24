"""SaharaSprout offline RAG pipeline.

A thin, debuggable hybrid retriever (FAISS dense + BM25 sparse, fused with
Reciprocal Rank Fusion) over the local agricultural corpus, plus a grounded
prompt + llama-server inference path and a base-model tiebreaker runner.

100% offline at inference time: no cloud embeddings, no cloud LLM.
"""

__all__ = ["config", "corpus", "embedder", "index", "retriever", "prompt", "llm", "pipeline"]
