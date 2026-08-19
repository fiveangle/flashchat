#!/bin/bash
# ram_discipline_matrix.sh — A/B one config of the exp/02 knobs under memory
# pressure. Starts a FRESH server (cold pin cache) with the given env, sends
# one long-prompt streaming request, extracts decode tok/s / TTFT / prefill /
# pin-hit stats from the server log, kills the server.
#
# Usage: ram_discipline_matrix.sh <tag> [extra FLASHCHAT_* env as KEY=VAL...]
set -u
TAG="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/explorations/02-ram-discipline"
mkdir -p "$OUT/raw"
PORT=$((9700 + RANDOM % 200))
M=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/ | head -1)
LOG="$OUT/raw/$TAG.log"

# env assembled from remaining KEY=VAL args
ENVARGS=()
for kv in "$@"; do ENVARGS+=("env" "$kv"); done

cd "$ROOT/metal_infer"
"${ENVARGS[@]}" ./infer --serve "$PORT" --model "$M" > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 40); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

PROMPT=$(python3 -c "print('Summarize the following passage in detail. ' + 'The history of computing spans several decades of innovation across hardware, software, and theory, from mechanical calculators through vacuum tubes, transistors, integrated circuits, microprocessors, and modern system-on-chip designs. ' * 12)")
T0=$(python3 -c 'import time;print(int(time.time()*1000))')
curl -s -m 400 -N "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"bench\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"stream\":true,\"max_tokens\":128,\"temperature\":0}" \
  -o /dev/null -w "http=%{http_code}\n"
T1=$(python3 -c 'import time;print(int(time.time()*1000))')

sleep 2
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
sleep 2

GEN=$(grep -oE "generated=[0-9]+ tokens in [0-9.]+ms \([0-9.]+ tok/s" "$LOG" | tail -1)
TTFT=$(grep -oE "TTFT [0-9.]+s" "$LOG" | tail -1)
PIN=$(grep -oE "hits [0-9]+ / [0-9]+ \([0-9.]+%\)" "$LOG" | tail -1)
PREFILL=$(grep -oE "prefill=[0-9]+ tokens in [0-9]+ms" "$LOG" | tail -1)
printf "%-28s %-42s %-12s %-10s %s | wall=%dms\n" "$TAG" "${GEN:-NO-GEN-LINE}" "${TTFT:-?}" "${PIN:-no-pin}" "${PREFILL:-?}" $((T1-T0))
