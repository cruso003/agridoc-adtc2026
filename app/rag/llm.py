"""llama-server client (OpenAI-compatible /v1/chat/completions).

The model is whatever GGUF the server has loaded — `model` in the payload is
ignored by llama-server, so the SAME code drives any candidate. Mirrors the
plain-urllib style of arc_eval.py (no extra deps).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


def health(server: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def chat(
    messages: list[dict],
    server: str = config.DEFAULT_SERVER,
    temperature: float = config.GEN_TEMPERATURE,
    max_tokens: int = config.GEN_MAX_TOKENS,
    model: str = "local-gguf",
    timeout: float = config.LLM_TIMEOUT,
) -> str:
    """POST a chat completion and return the assistant message content."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        server.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"llama-server {e.code}: {body}") from e

    return (data["choices"][0]["message"]["content"] or "").strip()


def chat_stream(
    messages: list[dict],
    server: str = config.DEFAULT_SERVER,
    temperature: float = config.GEN_TEMPERATURE,
    max_tokens: int = config.GEN_MAX_TOKENS,
    model: str = "local-gguf",
    timeout: float = config.LLM_TIMEOUT,
):
    """Yield assistant text deltas as the model generates (SSE from llama-server).

    Lets the UI show the model reasoning live instead of staring at a spinner.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        server.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:  # SSE: one "data: {...}" per line, blank line between events
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                    delta = obj["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"llama-server {e.code}: {detail}") from e
