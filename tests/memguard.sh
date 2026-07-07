#!/usr/bin/env bash
# memguard.sh — run a command under a swap-growth guard.
#
#   tests/memguard.sh [-g max_growth_gb] [-f min_free_pct] <command...>
#
# Polls vm.swapusage and memory_pressure every few seconds while the command
# runs. If swap grows more than max_growth_gb over its starting value (default
# 2 GiB) or free memory drops below min_free_pct (default 15%), the command's
# process group is killed and memguard exits 99. Exists because unmonitored
# engine test runs accumulated 40GB+ of swap on 2026-07-06.
set -u

MAX_GROWTH_GB=2
MIN_FREE_PCT=15
INTERVAL=5
while getopts "g:f:i:" opt; do
    case "$opt" in
        g) MAX_GROWTH_GB="$OPTARG" ;;
        f) MIN_FREE_PCT="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        *) echo "usage: $0 [-g max_growth_gb] [-f min_free_pct] [-i poll_sec] cmd..." >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { echo "memguard: no command given" >&2; exit 2; }

swap_used_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
free_pct()     { memory_pressure -Q 2>/dev/null | sed -nE 's/.*free percentage: ([0-9]+)%.*/\1/p'; }

BASE_SWAP="$(swap_used_mb)"
LIMIT_MB=$((BASE_SWAP + MAX_GROWTH_GB * 1024))
echo "[memguard] baseline swap ${BASE_SWAP}MB, kill at ${LIMIT_MB}MB or free<${MIN_FREE_PCT}%" >&2

# Run the command in its own process group so the whole tree can be killed.
set -m
"$@" &
CHILD=$!
set +m

RC=""
while kill -0 "$CHILD" 2>/dev/null; do
    sleep "$INTERVAL"
    SWAP="$(swap_used_mb)"; FREE="$(free_pct)"
    if [ -n "$SWAP" ] && [ "$SWAP" -gt "$LIMIT_MB" ]; then
        echo "[memguard] SWAP GUARD TRIPPED: ${SWAP}MB used (baseline ${BASE_SWAP}MB) — killing pgid $CHILD" >&2
        kill -TERM -- -"$CHILD" 2>/dev/null; sleep 3; kill -KILL -- -"$CHILD" 2>/dev/null
        RC=99; break
    fi
    if [ -n "$FREE" ] && [ "$FREE" -lt "$MIN_FREE_PCT" ]; then
        echo "[memguard] FREE-MEMORY GUARD TRIPPED: ${FREE}% free — killing pgid $CHILD" >&2
        kill -TERM -- -"$CHILD" 2>/dev/null; sleep 3; kill -KILL -- -"$CHILD" 2>/dev/null
        RC=99; break
    fi
done
if [ -z "$RC" ]; then
    wait "$CHILD"; RC=$?
fi
# Post-run: verify nothing from the tree is still alive.
if pgrep -g "$CHILD" >/dev/null 2>&1; then
    echo "[memguard] WARNING: survivors in pgid $CHILD after exit" >&2
    pgrep -gl "$CHILD" >&2
fi
echo "[memguard] done rc=$RC swap now $(swap_used_mb)MB (baseline ${BASE_SWAP}MB)" >&2
exit "$RC"
