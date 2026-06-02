#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${FLASHCHAT_NATIVE_QWEN_SMOKE_MODEL:-/Volumes/usr/Users/speedster/dev/models/hf/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0}"
MODEL_ID="${FLASHCHAT_NATIVE_QWEN_SMOKE_MODEL_ID:-mlx-community-Qwen36-35B-A3B-4bit}"
OUT="debug/native-qwen-compile-smoke"

if [[ ! -f "$MODEL/model.safetensors.index.json" ]]; then
    echo "SKIP: native Qwen BF16 smoke model not found: $MODEL"
    exit 0
fi

rm -rf "$OUT"

python3 scripts/compile_native_qwen.py \
    --model-id "$MODEL_ID" \
    --model "$MODEL" \
    --output "$OUT/non-experts" \
    --non-experts \
    --include-mtp \
    --name-regex '(^model\.layers\.0\.mlp\.gate\.weight$|^mtp\.layers\.0\.mlp\.gate\.weight$)'

python3 - <<'PY'
import json
from pathlib import Path

manifest = json.load(open("debug/native-qwen-compile-smoke/non-experts/model_weights.json"))
expected = {
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.gate.scales",
    "model.layers.0.mlp.gate.biases",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.gate.scales",
    "mtp.layers.0.mlp.gate.biases",
}
actual = set(manifest["tensors"])
missing = expected - actual
if missing:
    raise SystemExit(f"missing expected tensors: {sorted(missing)}")
if not Path("debug/native-qwen-compile-smoke/non-experts/model_weights.bin").exists():
    raise SystemExit("model_weights.bin was not written")
PY

python3 scripts/compile_native_qwen.py \
    --model-id "$MODEL_ID" \
    --model "$MODEL" \
    --output "$OUT/experts" \
    --experts \
    --layers 0 \
    --max-experts 1

test -f "$OUT/experts/packed_experts/layout.json"
test -f "$OUT/experts/packed_experts/layer_00.bin"

python3 scripts/compile_native_qwen.py \
    --model-id "$MODEL_ID" \
    --model "$MODEL" \
    --output "$OUT/mtp-experts" \
    --mtp-experts \
    --layers 0 \
    --max-experts 1

test -f "$OUT/mtp-experts/packed_mtp_experts/layout.json"
test -f "$OUT/mtp-experts/packed_mtp_experts/layer_00.bin"

python3 scripts/compile_native_qwen.py \
    --model-id "$MODEL_ID" \
    --model "$MODEL" \
    --output "$OUT/mtp-preflight/flashchat/q4" \
    --non-experts \
    --include-mtp \
    --name-regex '^mtp\.'

python3 scripts/compile_native_qwen.py \
    --model-id "$MODEL_ID" \
    --model "$MODEL" \
    --output "$OUT/mtp-preflight/flashchat/q4" \
    --mtp-experts \
    --layers 0 \
    --max-experts 1

./metal_infer/infer \
    --model-id "$MODEL_ID" \
    --model "$OUT/mtp-preflight" \
    --weights "$OUT/mtp-preflight/flashchat/q4/model_weights.bin" \
    --manifest "$OUT/mtp-preflight/flashchat/q4/model_weights.json" \
    --mtp-preflight \
    >/tmp/flashchat-mtp-preflight.out \
    2>/tmp/flashchat-mtp-preflight.err

grep -q '\[mtp\] preflight fc_out_rms=' /tmp/flashchat-mtp-preflight.err
grep -q '\[mtp\] decoder preflight fc_rms=' /tmp/flashchat-mtp-preflight.err
grep -q '\[mtp\] reusable forward preflight out_rms=' /tmp/flashchat-mtp-preflight.err

echo "native Qwen compile smoke passed"
