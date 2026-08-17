#!/bin/bash
# quick_decode_bench.sh — inner-loop decode benchmark for exploration A/Bs.
#
# Runs metal_infer/infer N times with a fixed prompt/token budget and captures
# decode tok/s, TTFT, and the per-layer timing breakdown per run. Prints a
# compact summary table. Intended for fast iteration; final sign-off is
# `bash tests/bench_api.sh --model-id <id>`.
#
# Usage:
#   tools/quick_decode_bench.sh [RESULTS_DIR] [TOKENS] [RUNS]
#
# Environment passthrough: all FLASHCHAT_* env vars reach the engine. Explicit
# values set by the caller win. Use RESULTS_DIR to keep per-experiment output
# under docs/explorations/<path>/.
set -u

RESULTS_DIR="${1:-/tmp/flashchat-quick-bench}"
TOKENS="${2:-150}"
RUNS="${3:-3}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INFER="$ROOT/metal_infer/infer"
CONFIG="${FLASHCHAT_CONFIG:-$HOME/.config/flashchat/config}"

MODEL_GLOB="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/"
MODEL_PATH="${FLASHCHAT_MODEL:-$(ls -d $MODEL_GLOB 2>/dev/null | head -1)}"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model snapshot not found ($MODEL_GLOB)" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
PROMPT="Explain how a gated delta-net linear attention layer works, covering the recurrence, the decay factor, and how it differs from full softmax attention."

echo "quick_decode_bench: model=$(basename "$MODEL_PATH") tokens=$TOKENS runs=$RUNS"
echo "  env: PIN_GB=${FLASHCHAT_EXPERT_PIN_MAX_GB:-<unset>} MTP=${FLASHCHAT_MTP:-<unset>} KV=${FLASHCHAT_KV_QUANT:-<unset>}"
echo "  results -> $RESULTS_DIR"

sum_tokps=""
for r in $(seq 1 "$RUNS"); do
    out="$RESULTS_DIR/run$r.txt"
    "$INFER" --config "$CONFIG" --model "$MODEL_PATH" \
        --prompt "$PROMPT" --tokens "$TOKENS" --timing >"$out" 2>&1
    tokps=$(grep -oE '\([0-9]+\.[0-9]+ tok/s\)' "$out" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
    ttft=$(grep -oE 'TTFT: *[0-9]+ ms' "$out" | grep -oE '[0-9]+')
    echo "  run$r: decode=${tokps:-FAIL} tok/s ttft=${ttft:-?}ms"
    sum_tokps="$sum_tokps $tokps"
done

python3 - "$RESULTS_DIR" $sum_tokps <<'EOF'
import sys, re, statistics, glob
d = sys.argv[1]
vals = [float(x) for x in sys.argv[2:] if x]
if not vals:
    print("  MEDIAN: FAIL (no successful runs)"); sys.exit(1)
med = statistics.median(vals)
print(f"  MEDIAN decode: {med:.2f} tok/s (runs: {', '.join(f'{v:.2f}' for v in vals)})")
best = max(vals)
print(f"  BEST   decode: {best:.2f} tok/s")
phases = ["cmd1_submit","cmd1_wait","cpu_attn","cmd2_encode","cmd2_wait",
          "routing_cpu","expert_io","cmd3_encode","total_layer"]
agg = {}
for f in sorted(glob.glob(f"{d}/run*.txt")):
    for line in open(f):
        m = re.match(r"\s+(\w+):\s+([0-9.]+)", line)
        if m and m.group(1) in phases:
            agg.setdefault(m.group(1), []).append(float(m.group(2)))
if agg:
    print("  layer-phase medians (ms): " + "  ".join(
        f"{k}={statistics.median(v):.3f}" for k, v in sorted(agg.items())))
EOF
