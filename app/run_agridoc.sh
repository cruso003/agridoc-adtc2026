#!/usr/bin/env bash
# AgriDoc launcher — starts the local model server + the app, then you open the UI.
# Usage:  bash run_agridoc.sh        (Ctrl+C stops everything)
# Optional: GGUF=/path/to/model.gguf bash run_agridoc.sh
set -uo pipefail
export MSYS_NO_PATHCONV=1   # keep container paths intact under git-bash/MINGW64

GGUF="${GGUF:-C:/dev/agri-doc/adtc-bench-workspace/artifacts/finetuned/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf}"
MODELDIR="$(dirname "$GGUF")"; MODELFILE="$(basename "$GGUF")"
IMG="${IMG:-adtc-profiler:latest}"
NAME="agridoc-model"

cleanup(){ echo; echo "[agridoc] stopping model server…"; docker stop "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[agridoc] starting model server (llama.cpp on :8080)…"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" -p 8080:8080 --entrypoint llama-server \
  -v "${MODELDIR}:/m:ro" "$IMG" \
  -m "/m/${MODELFILE}" --host 0.0.0.0 --port 8080 --log-disable \
  --parallel 1 --ctx-size 8192 --threads 4 >/dev/null

echo "[agridoc] waiting for the model to load…"
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
if ! curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
  echo "[agridoc] model server didn't come up. Logs:"; docker logs "$NAME" 2>&1 | tail -15; exit 1
fi

echo "[agridoc] model ready. Open  http://127.0.0.1:8000  in your browser."
echo "[agridoc] (Ctrl+C here stops both the app and the model server.)"
python -m rag serve --port 8000 --server http://127.0.0.1:8080
