#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER="${REPO_ROOT}/metal_infer/infer"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

if [ ! -x "$INFER" ]; then
    echo -e "${RED}FAIL${NC}  infer binary missing at $INFER (run 'make infer')"
    exit 1
fi

echo ""
echo "=== Flashchat Disk Cache Roundtrip ==="
echo ""

# The infer binary's --cache-roundtrip-test flag synthesizes deterministic
# snapshot data sized like real KV/conv/SSM tensors, runs it through the
# real save_system_prompt_disk_cache + load_system_prompt_disk_cache paths
# into a temp dir, and memcmp's loaded vs original byte-for-byte. Any
# divergence is reported with (kind, layer, byte offset). Exits non-zero on
# any mismatch.
#
# Catches: serializer regressions (header rewrite at line 8135, LZFSE
# round-trip, FNV1a checksum, chunk ordering, struct layout drift). Does
# NOT catch GPU MTLBuffer coherency issues at capture time — those need
# the runtime validator (FLASHCHAT_CACHE_VALIDATE=1 during a live prefill).

LOG=/tmp/flashchat_cache_roundtrip.log
"$INFER" --cache-roundtrip-test >"$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    echo -e "${RED}FAIL${NC}  infer --cache-roundtrip-test exit=$rc"
    echo "--- output ---"
    cat "$LOG"
    exit 1
fi
if grep -qF '[cache-roundtrip] PASS' "$LOG"; then
    echo -e "${GREEN}PASS${NC}  disk cache save/load is byte-identical for synthetic data"
    rm -f "$LOG"
    exit 0
fi
echo -e "${RED}FAIL${NC}  no PASS line found"
echo "--- output ---"
cat "$LOG"
exit 1
