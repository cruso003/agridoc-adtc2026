"""Grounded prompt construction.

The system prompt is the EXACT string the fine-tune data will use (see
config.SYSTEM_PROMPT) so the retrieval-time and training-time contracts match.
The user message format is fixed:

    CONTEXT:
    {each retrieved chunk, prefixed with its source}

    QUESTION:
    {query}
"""
from __future__ import annotations

from . import config


def format_context(chunks: list[dict]) -> str:
    """One block per chunk, each prefixed with its chunk_id and source.

    The chunk_id prefix doubles as a citation handle the model can echo and the
    CC-BY-SA attribution anchor we report back regardless of what the model says.
    """
    blocks = []
    for c in chunks:
        header = f"[{c['chunk_id']} | {c['source']}]"
        blocks.append(f"{header}\n{c['text'].strip()}")
    return "\n\n".join(blocks)


def build_user_message(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """OpenAI-style chat messages: fixed system prompt + grounded user turn."""
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(query, chunks)},
    ]
