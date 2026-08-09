#!/usr/bin/env bash
# Fetch the AgriDoc weight (LoRA-fine-tuned Qwen2.5-1.5B, Q4_0 GGUF) into model/.
# Idempotent, no credentials. Output path must match `_runtime.model_path` in metadata.json.
#
# This is our OWN fine-tune of Qwen2.5-1.5B-Instruct (run-9cp): assistant-only-loss SFT that adds
# the iterative ask-or-assess behaviour — it ASKS a discriminating question when a symptom is thin
# (instead of committing to a fabricated detail), commits when the pattern is clear, refuses
# pesticide/drug/fertiliser doses/rates/prices, and redirects non-poultry livestock to a vet. It
# ships an AgriDoc extension-officer persona as the gguf's default system prompt, so it behaves and
# identifies correctly even when chatted with no system prompt. Two independent raw-chat reviews
# confirm: clean stopping (0/114 hit the token cap), single-turn dose refusals hold, AgriDoc
# identity. As a 1.5B it has a factual-recall ceiling on specific, less-common disease IDs — that
# is the RAG layer's job in the app, and the app also runs a dose-guard on every reply. See
# REPORT.md / DECISIONS.md DR-0019..0025.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf"

# Public, credential-free. Source: https://huggingface.co/Cruso003/AgriDoc-Qwen2.5-1.5B-GGUF
MODEL_URL="${AGRIDOC_MODEL_URL:-https://huggingface.co/Cruso003/AgriDoc-Qwen2.5-1.5B-GGUF/resolve/main/AgriDoc-Qwen2.5-1.5B-Q4_0.gguf}"

mkdir -p "$MODEL_DIR"
if [[ -f "$MODEL_FILE" ]]; then
  echo "model already present at $MODEL_FILE — skipping download"
  exit 0
fi

echo "downloading $MODEL_URL → $MODEL_FILE …"
if command -v curl > /dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget > /dev/null 2>&1; then
  wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "error: neither curl nor wget found" >&2; exit 1
fi
mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE"
