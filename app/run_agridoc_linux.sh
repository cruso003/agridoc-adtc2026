#!/usr/bin/env bash
# AgriDoc launcher — Linux, using Ollama as the local model server (no Docker needed).
# The original run_agridoc.sh targets Windows + the adtc-profiler Docker image; this is
# the native-Linux equivalent used to bring the full offline app up on this machine.
#
# Prereqs (one-time):
#   bash ../download_model.sh                 # fetch the GGUF into ../model/
#   python3 -m venv .venv && .venv/bin/pip install -r rag/requirements.txt
#   .venv/bin/python -m rag index             # build the FAISS+BM25 index from the corpus
#
# Usage:  bash run_agridoc_linux.sh           # Ctrl+C stops the app (Ollama keeps running)
set -uo pipefail
cd "$(dirname "$0")"

VENV="./.venv/bin/python"
GGUF="../model/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf"
OLLAMA_URL="http://127.0.0.1:11434"
MODEL_NAME="local-gguf"   # MUST match the model name the app sends in its payload (llm.py)
APP_PORT="${AGRIDOC_PORT:-8000}"

[[ -f "$GGUF" ]] || { echo "[agridoc] missing model: $GGUF — run: bash ../download_model.sh"; exit 1; }
[[ -x "$VENV" ]] || { echo "[agridoc] missing venv — run: python3 -m venv .venv && .venv/bin/pip install -r rag/requirements.txt"; exit 1; }
[[ -d "adtc-bench-workspace/rag_index" ]] || { echo "[agridoc] no index — run: .venv/bin/python -m rag index"; exit 1; }

# 1) Ensure the Ollama server is up.
if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
  echo "[agridoc] starting ollama serve…"
  ollama serve >/tmp/agridoc-ollama.log 2>&1 &
  for i in $(seq 1 30); do curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1 && break; sleep 1; done
fi
curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1 || { echo "[agridoc] ollama did not start (see /tmp/agridoc-ollama.log)"; exit 1; }

# 2) Import the GGUF into Ollama once (idempotent).
if ! ollama list 2>/dev/null | grep -q "^${MODEL_NAME}\b"; then
  echo "[agridoc] importing model into ollama as '${MODEL_NAME}'…"
  ollama create "$MODEL_NAME" -f Modelfile.agridoc
fi

# 3) Run the AgriDoc API + UI, pointed at Ollama's OpenAI-compatible endpoint.
echo "[agridoc] model ready. Open  http://127.0.0.1:${APP_PORT}  in your browser."
echo "[agridoc] (Ctrl+C stops the app; Ollama keeps running in the background.)"
exec "$VENV" -m rag serve --port "$APP_PORT" --server "$OLLAMA_URL"
