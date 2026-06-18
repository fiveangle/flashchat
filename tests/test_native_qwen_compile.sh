#!/usr/bin/env bash
set -euo pipefail

# Native BF16 compile smoke: drives modelmgr.steps.compile_native (the same
# code the launcher's build flow runs) against a real snapshot, then boots
# the engine's MTP preflight on the result.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${FLASHCHAT_NATIVE_QWEN_SMOKE_MODEL:-/Volumes/usr/Users/speedster/dev/models/hf/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0}"
MODEL_ID="${FLASHCHAT_NATIVE_QWEN_SMOKE_MODEL_ID:-Qwen-Qwen36-35B-A3B}"
OUT="debug/native-qwen-compile-smoke"

if [[ ! -f "$MODEL/model.safetensors.index.json" ]]; then
    echo "SKIP: native Qwen BF16 smoke model not found: $MODEL"
    exit 0
fi

rm -rf "$OUT"

VENV_PY="$ROOT/metal_infer/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=python3

# compile_step OUTPUT MODE [name_regex | layer max_experts]
compile_step() {
    "$VENV_PY" - "$ROOT" "$MODEL_ID" "$MODEL" "$@" <<'PYEOF'
import json, os, sys

root, model_id, model, output, mode = sys.argv[1:6]
extra = sys.argv[6:]
sys.path.insert(0, root)
from modelmgr.steps.compile_native import (
    compile_non_experts, compile_routed_experts,
    load_headers, load_weight_map, native_text_config)

entry = json.load(open(os.path.join(root, "assets/model_configs.json")))["models"][model_id]
wm = load_weight_map(model)
hd = load_headers(model, wm)
nc = native_text_config(model)
os.makedirs(output, exist_ok=True)
if mode == "non-experts":
    with open(os.path.join(output, "model_weights.bin"), "wb") as sink:
        manifest = compile_non_experts(model, wm, hd, entry, nc, True, sink,
                                       name_regex=extra[0])
    with open(os.path.join(output, "model_weights.json"), "w") as f:
        json.dump(manifest, f, indent=2)
elif mode == "experts":
    compile_routed_experts(model, os.path.join(output, "packed_experts"), wm, hd, entry,
                           [int(extra[0])], "model.layers.", "packed_experts",
                           entry["num_hidden_layers"], max_experts=int(extra[1]))
elif mode == "mtp-experts":
    mtp_layers = nc.get("mtp_num_hidden_layers", entry.get("mtp_num_hidden_layers", 0))
    assert mtp_layers, "model declares no MTP layers"
    compile_routed_experts(model, os.path.join(output, "packed_mtp_experts"), wm, hd, entry,
                           [int(extra[0])], "mtp.layers.", "packed_mtp_experts",
                           mtp_layers, max_experts=int(extra[1]))
else:
    raise SystemExit(f"unknown mode {mode}")
PYEOF
}

compile_step "$OUT/non-experts" non-experts '(^model\.layers\.0\.mlp\.gate\.weight$|^mtp\.layers\.0\.mlp\.gate\.weight$)'

python3 - <<'PYEOF'
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
PYEOF

compile_step "$OUT/experts" experts 0 1

test -f "$OUT/experts/packed_experts/layout.json"
test -f "$OUT/experts/packed_experts/layer_00.bin"

compile_step "$OUT/mtp-experts" mtp-experts 0 1

test -f "$OUT/mtp-experts/packed_mtp_experts/layout.json"
test -f "$OUT/mtp-experts/packed_mtp_experts/layer_00.bin"

# The MTP weight cache also needs embed_tokens (token embedding for drafts)
# and lm_head (draft logits) from the same manifest.
compile_step "$OUT/mtp-preflight/flashchat/q4" non-experts '(^mtp\.|^model\.embed_tokens\.|^lm_head\.)'
compile_step "$OUT/mtp-preflight/flashchat/q4" mtp-experts 0 1

# The engine's startup validation requires the full artifact set — vocab.bin
# and every packed_experts layer — not just the MTP pieces this smoke
# compiles. Borrow them from the source snapshot's own runtime artifacts
# when present; otherwise skip the engine half.
SRC_RUNTIME="$MODEL/flashchat/q4"
if [[ -f "$SRC_RUNTIME/vocab.bin" && -f "$SRC_RUNTIME/packed_experts/layer_00.bin" ]]; then
    ln -sf "$SRC_RUNTIME/vocab.bin" "$OUT/mtp-preflight/flashchat/q4/vocab.bin"
    ln -sfn "$SRC_RUNTIME/packed_experts" "$OUT/mtp-preflight/flashchat/q4/packed_experts"

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
else
    echo "SKIP: engine preflight (no full runtime artifacts at $SRC_RUNTIME)"
fi

echo "native Qwen compile smoke passed"
