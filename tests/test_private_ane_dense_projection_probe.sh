#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${FLASHCHAT_PRIVATE_ANE:-}" != "1" ]]; then
    echo "SKIP: set FLASHCHAT_PRIVATE_ANE=1 to run the private ANE dense projection probe"
    exit 0
fi

if [[ ! -d /System/Library/PrivateFrameworks/AppleNeuralEngine.framework ]]; then
    echo "SKIP: AppleNeuralEngine private framework not found"
    exit 0
fi

PYTHON_BIN="${FLASHCHAT_ANE_PYTHON:-python3}"
if ! "$PYTHON_BIN" -c 'import coremltools, numpy' >/dev/null 2>&1; then
    echo "SKIP: coremltools is not available for $PYTHON_BIN"
    exit 0
fi

MODEL_ID="${FLASHCHAT_ANE_DENSE_MODEL_ID:-Qwen-Qwen36-27B}"
MODEL="${FLASHCHAT_ANE_DENSE_MODEL:-}"
if [[ -z "$MODEL" ]]; then
    source lib/config.sh
    MODEL="$(flashchat_model_path_for_id "$MODEL_ID" || true)"
fi

if [[ ! -f "$MODEL/model.safetensors.index.json" ]]; then
    echo "SKIP: dense BF16 model snapshot not found: $MODEL"
    exit 0
fi

TENSOR="${FLASHCHAT_ANE_DENSE_TENSOR:-model.language_model.layers.3.self_attn.q_proj.weight}"
TOKENS="${FLASHCHAT_ANE_DENSE_TOKENS:-8}"
INPUT_LIMIT="${FLASHCHAT_ANE_DENSE_INPUT_LIMIT:-0}"
OUTPUT_LIMIT="${FLASHCHAT_ANE_DENSE_OUTPUT_LIMIT:-0}"
OUT="${FLASHCHAT_ANE_DENSE_OUT:-debug/private-ane-dense-probe}"

rm -rf "$OUT"

args=(
    scripts/build_ane_dense_projection.py
    --model "$MODEL"
    --model-id "$MODEL_ID"
    --tensor "$TENSOR"
    --tokens "$TOKENS"
    --output "$OUT"
)
if [[ "$INPUT_LIMIT" != "0" ]]; then
    args+=(--input-limit "$INPUT_LIMIT")
fi
if [[ "$OUTPUT_LIMIT" != "0" ]]; then
    args+=(--output-limit "$OUTPUT_LIMIT")
fi

"$PYTHON_BIN" "${args[@]}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/flashchat-private-ane-dense.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

BIN="$TMP_DIR/private_ane_dense_probe"
clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
    tests/private_ane_dense_probe.m \
    -o "$BIN"
"$BIN" "$OUT"
