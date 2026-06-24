#!/usr/bin/env bash
# Fetch the AgriDoc weight (LoRA-fine-tuned Qwen2.5-1.5B, Q4_0 GGUF) into model/.
# Idempotent, no credentials. Output path must match `_runtime.model_path` in metadata.json.
#
# This is our OWN fine-tune of Qwen2.5-1.5B-Instruct (run-6): assistant-only-loss SFT on a
# behaviour-first dataset (extension-officer reasoning, differential-when-ambiguous, refuse
# pesticide/drug doses, redirect non-poultry to a vet). It measurably beats the base on our
# 25-prompt gate (0 safety fails vs base 2; +behaviour passes). The earlier all-commit /
# full-sequence-loss attempts fabricated doses — fixed by assistant-only loss + diverse data.
# See REPORT.md / DECISIONS.md DR-0019..0021.
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
