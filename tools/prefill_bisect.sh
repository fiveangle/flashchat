#!/bin/bash
# prefill_bisect.sh <commit> — build one commit in an isolated worktree and
# measure its prefill speed with the standard bench spec.
#
# Purpose: localize the ~2.2x prefill regression between 4f2356a (Jul-8 fast
# rows' HEAD) and 9fd7c83 (last pre-exploration main commit).
#
# Protocol per commit:
#   - isolated git worktree at <commit> (clean tree, no WIP)
#   - commits containing the second-request serve segfault (86c8846, 9fd7c83)
#     get d507059 (the regrow fix, not prefill-path code) cherry-picked so the
#     bench can complete; noted in the results as "+fix"
#   - make infer (same machine, same flags)
#   - tests/bench_api.sh --model-id Qwen-Qwen36-35B-A3B --no-perf-log
#     (warmup=1 + repeats=2 per scenario, identical spec to the Jul-8 rows)
#   - server stderr captured via FLASHCHAT_SERVER_LOG (engine reads it
#     directly) so ANE ready/failed state is recorded per commit
#
# Output: results.tsv row + raw/ logs under the exploration dir.
set -u

COMMIT="${1:?usage: prefill_bisect.sh <commit> [tag]}"
TAG="${2:-$(git rev-parse --short "$COMMIT")}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/explorations/10-prefill-regression"
WT="/tmp/fc-bisect/$TAG"
mkdir -p "$OUT/raw"

# Commits with the released-buffer segfault need the regrow fix to bench at all.
NEEDS_FIX=0
case "$(git rev-parse "$COMMIT")" in
    "$(git rev-parse 86c8846)"|"$(git rev-parse 9fd7c83)") NEEDS_FIX=1 ;;
esac

if [ ! -d "$WT" ]; then
    if [ "$NEEDS_FIX" -eq 1 ]; then
        git worktree add -f -b "bisect-tmp/$TAG" "$WT" "$COMMIT" >/dev/null 2>&1
        (cd "$WT" && git cherry-pick --no-edit d507059 >/dev/null 2>&1) || { echo "ERROR: cherry-pick failed for $TAG"; exit 1; }
    else
        git worktree add -f --detach "$WT" "$COMMIT" >/dev/null 2>&1
    fi
fi

echo "=== $TAG ($(git -C "$WT" log -1 --format='%h %s')) ==="
make -C "$WT" infer >/dev/null 2>&1 || { echo "BUILD FAILED"; exit 1; }

LOG="$OUT/raw/$TAG-bench.stdout"
SLOG="$OUT/raw/$TAG-server.log"
rm -f "$SLOG"
# Unique port per tag: a stale server squatting on a shared port would answer
# /health with the WRONG binary and silently taint the measurement (this
# actually happened with a leaked bench server on 9999 — ctrl-A redo).
PORT=$((9500 + ($(printf '%s' "$TAG" | cksum | cut -d' ' -f1) % 400)))
( cd "$WT" && FLASHCHAT_SERVER_LOG="$SLOG" bash tests/bench_api.sh \
      --port "$PORT" --model-id Qwen-Qwen36-35B-A3B --no-perf-log ) > "$LOG" 2>&1
RC=$?

if grep -q "bind failed" "$SLOG" 2>/dev/null; then
    echo "  !! SERVER BIND FAILED — results tainted by a squatter on port $PORT; abort"
    exit 1
fi

grep -E "prefill=" "$LOG" | sed 's/^/  /'
if grep -q "Segmentation fault" "$LOG"; then echo "  !! SEGFAULT during bench"; fi

# TSV: tag, scenario, prefill_ms, decode_tps
grep -E "prefill=" "$LOG" | while read -r _ scenario _ decode rest; do
    pms=$(echo "$rest" | grep -oE 'prefill=[0-9]+' | cut -d= -f2)
    echo -e "$TAG\t$scenario\t$pms\t$decode" >> "$OUT/results.tsv"
done

echo "--- server ANE lines:"
grep -E "ane-prefill" "$SLOG" 2>/dev/null | head -3 | sed 's/^/  /'
echo "--- done $TAG (rc=$RC)"
