"""Load and tokenize the corpus.

The pipeline depends on the chunk SCHEMA, not the content:
    {chunk_id, source, license, topic, text}
so re-indexing a grown chunks.jsonl needs no code change.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

# Required schema fields. We fail loudly if a line is missing one rather than
# silently indexing malformed records.
REQUIRED_FIELDS = ("chunk_id", "source", "license", "topic", "text")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_chunks(path: str | Path) -> list[dict]:
    """Read chunks.jsonl into a list of dicts, validating the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = [f for f in REQUIRED_FIELDS if f not in rec]
            if missing:
                raise ValueError(
                    f"{path}:{lineno} missing required field(s) {missing}; "
                    f"got keys {sorted(rec)}"
                )
            chunks.append(rec)
    if not chunks:
        raise ValueError(f"no chunks loaded from {path}")
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25.

    Deliberately simple and deterministic — exact-match disease/crop names
    ('chlorpyrifos', 'armyworm', 'newcastle') survive intact, which is the
    whole point of keeping a sparse channel alongside the dense one.
    """
    return _TOKEN_RE.findall(text.lower())


def iter_chunks(path: str | Path) -> Iterator[dict]:
    yield from load_chunks(path)
