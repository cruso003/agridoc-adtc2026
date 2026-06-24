"""AgriDoc app backend — a zero-dependency HTTP API in front of the RAG pipeline.

Stdlib http.server only (no FastAPI/uvicorn) — keeps the offline, laptop-friendly
footprint and runs anywhere Python does. The UI calls a single endpoint.

API CONTRACT (locked — see DECISIONS.md DR-0011)
  GET  /health
      -> 200 {"status":"ok","llm":true|false,"chunks":<int>}
  POST /ask   {"question": "..."}
      -> 200 {
           "answer":  "<grounded answer text>",
           "sources": [ {"id","title","origin","snippet","topic"} , ... ],  # top-k, ordered
           "grounded": true|false
         }
  CORS: Access-Control-Allow-Origin: *  (UI is served from a different origin in dev)

Run:
  # 1) start a llama-server with the chosen GGUF on :8080 (Docker or native)
  # 2) python -m rag.app   (or: python -m rag serve)   -> API on :8000
Env: AGRIDOC_PORT (default 8000), RAG_LLM_SERVER (default http://127.0.0.1:8080),
     RAG_INDEX_DIR (default config.DEFAULT_INDEX_DIR).
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, llm
from . import diagnose as diagnose_mod
from .embedder import Embedder
from .index import LoadedIndex
from .pipeline import Pipeline
from .retriever import Retriever

LLM_SERVER = os.environ.get("RAG_LLM_SERVER", config.DEFAULT_SERVER)
INDEX_DIR = os.environ.get("RAG_INDEX_DIR", str(config.DEFAULT_INDEX_DIR))
APP_PORT = int(os.environ.get("AGRIDOC_PORT", "8000"))
WEB_DIR = (config.REPO_ROOT / "web").resolve()  # the AgriDoc UI, served same-origin

# Load the index + embedder ONCE at startup (slow part); reuse per request.
print(f"[app] loading index from {INDEX_DIR} …")
_INDEX = LoadedIndex(INDEX_DIR)
_PIPELINE = Pipeline(Retriever(_INDEX, Embedder(_INDEX.embed_model)))
print(f"[app] index ready: {len(_INDEX)} chunks | LLM @ {LLM_SERVER}")


def _source_view(chunk: dict) -> dict:
    """Map a retrieved chunk to the UI-facing source card."""
    src = chunk.get("source", "") or ""
    if "synthesized advisory" in src.lower():
        origin, title = "AgriDoc knowledge base", "Synthesized advisory"
    elif " — " in src:
        origin, title = (p.strip() for p in src.split(" — ", 1))
    else:
        origin, title = src, chunk.get("topic", "")
    text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
    snippet = text[:200].rstrip()
    if len(text) > 200:
        snippet = snippet.rsplit(" ", 1)[0] + "…"
    return {
        "id": chunk.get("chunk_id", ""),
        "title": title or origin,
        "origin": origin,
        "snippet": snippet,
        "topic": chunk.get("topic", ""),
    }


def build_query(payload: dict) -> str:
    """Structured intake (crop / growth_stage / affected_part / symptom) -> a retrieval
    query. The app builds a good query so a non-expert doesn't have to (PRODUCT_SPEC §4)."""
    parts = [payload.get(k, "") for k in ("crop", "growth_stage", "affected_part", "symptom")]
    return " ".join(p.strip() for p in parts if p and p.strip())


def handle_ask(question: str) -> dict:
    result = _PIPELINE.answer(question, server=LLM_SERVER)
    sources = [_source_view(c) for c in result.retrieved]
    return {"answer": result.answer, "sources": sources, "grounded": bool(sources)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # CORS preflight
        self._send(204, {})

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.rstrip("/") == "/health":
            self._send(200, {"status": "ok", "llm": llm.health(LLM_SERVER), "chunks": len(_INDEX)})
            return
        # static UI from web/ (same-origin with the API → no CORS issues)
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (WEB_DIR / rel).resolve()
        if (f == WEB_DIR or WEB_DIR in f.parents) and f.is_file():
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(f))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        route = self.path.rstrip("/")
        if route not in ("/ask", "/diagnose"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            # /diagnose accepts structured intake (crop/stage/part/symptom) or a question
            question = (payload.get("question") or build_query(payload)).strip()
            if not question:
                self._send(400, {"error": "missing 'question' or intake fields"})
                return
            if not llm.health(LLM_SERVER):
                self._send(503, {"error": f"no LLM server at {LLM_SERVER}"})
                return
            if route == "/diagnose":
                self._stream_diagnose(question)
            else:
                self._send(200, handle_ask(question))
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def _stream_diagnose(self, question: str) -> None:
        """Stream the diagnosis as SSE so the UI can show reasoning as it's written."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        try:
            for kind, payload in diagnose_mod.diagnose_stream(
                _PIPELINE.retriever, question, server=LLM_SERVER):
                if kind == "token":
                    emit({"type": "token", "text": payload})
                elif kind == "meta":
                    emit({"type": "meta", **payload})
                else:
                    emit({"type": "result", "result": payload})
        except Exception as e:  # noqa: BLE001
            try:
                emit({"type": "error", "error": str(e)})
            except Exception:  # noqa: BLE001
                pass

    def log_message(self, fmt, *args):  # quieter logs
        print(f"[app] {self.address_string()} {fmt % args}")


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", APP_PORT), Handler)
    print(f"[app] AgriDoc API on http://127.0.0.1:{APP_PORT}  (POST /ask, GET /health)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
