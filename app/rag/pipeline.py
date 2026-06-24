"""End-to-end grounded answer path: query → retrieve → prompt → LLM → answer.

This is the single harness used by (1) the base-model tiebreaker, (2) Gate 0,
and (3) product inference. Output always carries the chunk_ids/sources used —
both a quality signal and the CC-BY-SA attribution mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, llm, prompt
from .retriever import Retriever


@dataclass
class RagResult:
    query: str
    answer: str
    citations: list[dict]          # [{chunk_id, source, license, topic}]
    retrieved: list[dict]          # full retrieved chunks + fusion diagnostics
    messages: list[dict] = field(default_factory=list)

    def format_cli(self) -> str:
        lines = [self.answer.strip(), "", "Sources:"]
        if not self.citations:
            lines.append("  (no chunks retrieved)")
        for c in self.citations:
            lines.append(f"  - {c['chunk_id']}  {c['source']}  [{c['license']}]")
        return "\n".join(lines)


class Pipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def answer(
        self,
        query: str,
        server: str = config.DEFAULT_SERVER,
        top_k: int = config.TOP_K,
        temperature: float = config.GEN_TEMPERATURE,
        max_tokens: int = config.GEN_MAX_TOKENS,
    ) -> RagResult:
        chunks = self.retriever.retrieve(query, top_k=top_k)
        messages = prompt.build_messages(query, chunks)
        answer = llm.chat(
            messages,
            server=server,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        citations = [
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "license": c["license"],
                "topic": c["topic"],
            }
            for c in chunks
        ]
        return RagResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieved=chunks,
            messages=messages,
        )
