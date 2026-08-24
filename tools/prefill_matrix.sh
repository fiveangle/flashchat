#!/bin/bash
# prefill_matrix.sh — measure prefill_ms for one config at one prompt length.
#
# Fresh server per invocation; 3 identical requests; reports the LAST one
# (fully warm: Metal pipelines compiled, sys-prompt snapshot hot, expert page
# cache populated) — the steady-state prefill a user experiences mid-session.
# Also captures the ANE engagement phase breakdown for that request.
#
# Usage: prefill_matrix.sh <tag> <prompt_repeats> [KEY=VAL ...]
set -u
TAG="$1"; REPS="$2"; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/explorations/11-prefill-ane-crossover"
mkdir -p "$OUT/raw"
PORT=$((9800 + RANDOM % 150))
M=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/ | head -1)
LOG="$OUT/raw/$TAG.log"

ENVARGS=()
for kv in "$@"; do ENVARGS+=("env" "$kv"); done
[ ${#ENVARGS[@]} -eq 0 ] && ENVARGS=("env" "FC_NOOP=1")

cd "$ROOT/metal_infer"
"${ENVARGS[@]}" ./infer --serve "$PORT" --model "$M" > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 40); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

PROMPT=$(python3 -c "print('Summarize the following passage in detail. ' + 'The history of computing spans several decades of innovation across hardware, software, and theory, from mechanical calculators through vacuum tubes, transistors, integrated circuits, microprocessors, and modern system-on-chip designs. ' * $REPS)")
for req in 1 2 3; do
  curl -s -m 400 -N "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"bench\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"stream\":true,\"max_tokens\":4,\"temperature\":0}" \
    -o /dev/null -w "" 2>/dev/null
done

sleep 1
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2

P3=$(grep -oE "prefill=[0-9]+ tokens in [0-9]+ms" "$LOG" | tail -1)
E3=$(grep -E "ane-prefill.*engagement" "$LOG" | tail -1)
printf "%-24s %s\n" "$TAG" "${P3:-FAIL}"
printf "%-24s %s\n" "" "${E3:-no-engagement-line}"
