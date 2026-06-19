#!/bin/bash
set -euo pipefail

# Integration test for the modelmgr management core through the
# launcher: status, migrate, verify (quick/deep + corruption), manage TUI
# basics, ensure gating, resolved-view generation. Replaces the legacy
# bash-TUI tests (test_flashchat_manage.sh, test_model_*_config.sh,
# test_profile_edit_config.sh) whose flows moved into Python — deep logic
# coverage lives in tests/python/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLASHCHAT="${REPO_ROOT}/flashchat"
VENV_PY="${REPO_ROOT}/metal_infer/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=python3

FAILED=0
PASSED=0
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { PASSED=$((PASSED + 1)); echo -e "${GREEN}PASS${NC}  $1"; }
fail() { FAILED=$((FAILED + 1)); echo -e "${RED}FAIL${NC}  $1${2:+: $2}"; }

assert_contains() {
    if printf "%s" "$3" | grep -q "$2"; then pass "$1"; else fail "$1" "expected '$2'"; fi
}
assert_not_contains() {
    if printf "%s" "$3" | grep -q "$2"; then fail "$1" "unexpected '$2'"; else pass "$1"; fi
}
assert_exists() { if [ -e "$2" ]; then pass "$1"; else fail "$1" "missing $2"; fi; }
assert_is_link() { if [ -L "$2" ]; then pass "$1"; else fail "$1" "not a symlink: $2"; fi; }

TMPDIR="$(mktemp -d /tmp/flashchat-models-cli-test.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

export HOME="${TMPDIR}/home"
mkdir -p "$HOME/.config/flashchat"
HF_CACHE="${TMPDIR}/hf-cache"
OFFLOAD_DIR="${TMPDIR}/offload"
mkdir -p "$HF_CACHE" "$OFFLOAD_DIR"

NATIVE_SNAPSHOT="${HF_CACHE}/models--Qwen--Qwen3.6-35B-A3B/snapshots/fixturehash0000"

cat > "$HOME/.config/flashchat/config" <<EOF
MODEL="Qwen-Qwen36-35B-A3B"
HUGGINGFACE_CACHE_DIR="$HF_CACHE"
OFFLOAD_DIR="$OFFLOAD_DIR"
SERVER_PORT="19876"
COLOR_OUTPUT="0"
EOF

run_fc() { "$FLASHCHAT" "$@"; }

# Build a legacy-layout fixture (duplicated vocab per variant, no manifests)
# with the same builder the Python unit tests use.
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/tests/python" "$VENV_PY" - "$HF_CACHE" <<'PY'
import sys
from modelmgr.registry import Registry
from treebuilder import make_legacy_snapshot

registry = Registry.load()
make_legacy_snapshot(sys.argv[1], registry.get("qwen3.6-35b-a3b"), with_bf16=True)
PY

echo ""
echo "=== Status / models ==="
output=$(run_fc models 2>&1)
assert_contains "models shows fixture cache" "$HF_CACHE" "$output"
assert_contains "models shows native model" "Qwen/Qwen3.6-35B-A3B" "$output"
assert_contains "models marks selection" "<- selected" "$output"
assert_contains "models shows variant lines" "q8:" "$output"

echo ""
echo "=== Migration ==="
output=$(run_fc migrate --dry-run 2>&1)
assert_contains "dry-run plans vocab dedup" "dedup vocab.bin" "$output"
assert_contains "dry-run plans bf16 dedup" "dedup bf16/" "$output"
assert_contains "dry-run reports reclaim" "total reclaimed" "$output"
assert_exists "dry-run touched nothing" "${NATIVE_SNAPSHOT}/flashchat/q8/vocab.bin"
if [ -L "${NATIVE_SNAPSHOT}/flashchat/q8/vocab.bin" ]; then
    fail "dry-run left q8 vocab as real file"
else
    pass "dry-run left q8 vocab as real file"
fi

# --hash records sha256 baselines so deep verify can detect corruption
output=$(run_fc migrate -y --hash 2>&1)
assert_contains "migrate completes" "migration complete" "$output"
assert_exists "shared vocab created" "${NATIVE_SNAPSHOT}/flashchat/shared/vocab.bin"
assert_exists "shared bf16 created" "${NATIVE_SNAPSHOT}/flashchat/shared/bf16/mtp_weights.bin"
assert_is_link "q4 vocab is a symlink" "${NATIVE_SNAPSHOT}/flashchat/q4/vocab.bin"
assert_is_link "q8 vocab is a symlink" "${NATIVE_SNAPSHOT}/flashchat/q8/vocab.bin"
assert_exists "q4 integrity manifest written" "${NATIVE_SNAPSHOT}/flashchat/q4/.flashchat_artifacts.json"
grep -q '^MODEL_BASE="qwen3.6-35b-a3b"$' "$HOME/.config/flashchat/config" \
    && pass "config gained MODEL_BASE" || fail "config gained MODEL_BASE"
grep -q '^MODEL_VARIANT="q4"$' "$HOME/.config/flashchat/config" \
    && pass "config gained MODEL_VARIANT" || fail "config gained MODEL_VARIANT"

output=$(run_fc migrate --dry-run 2>&1)
assert_contains "second migrate is a no-op" "nothing to migrate" "$output"

echo ""
echo "=== Verify ==="
output=$(run_fc verify --model qwen3.6-35b-a3b 2>&1 || true)
assert_contains "quick verify passes" "PASS" "$output"

# corrupt one expert layer (same size) -> quick passes, deep catches it
LAYER="${NATIVE_SNAPSHOT}/flashchat/q4/packed_experts/layer_03.bin"
printf 'X' | dd of="$LAYER" bs=1 count=1 conv=notrunc 2>/dev/null
output=$(run_fc verify --model qwen3.6-35b-a3b --variant q4 2>&1 || true)
assert_contains "quick verify misses same-size flip" "PASS" "$output"
output=$(run_fc verify --model qwen3.6-35b-a3b --variant q4 --deep 2>&1 || true)
assert_contains "deep verify catches corruption" "hash-mismatch" "$output"
assert_contains "deep verify fails overall" "FAIL" "$output"

echo ""
echo "=== Resolved view ==="
RESOLVED="$HOME/.config/flashchat/resolved_models.json"
assert_exists "resolved view generated" "$RESOLVED"
python3 -m json.tool "$RESOLVED" >/dev/null && pass "resolved view is valid json" \
    || fail "resolved view is valid json"
grep -q '"Qwen-Qwen36-35B-A3B"' "$RESOLVED" \
    && pass "resolved view keeps legacy id" || fail "resolved view keeps legacy id"

echo ""
echo "=== Manage TUI (non-destructive) ==="
output=$(printf 'q\n' | run_fc manage 2>&1)
assert_contains "manage lists models" "Manage models" "$output"
# index 4 = the native qwen3.6-35b-a3b in the sorted manage list
output=$(printf '4\nq\nq\n' | run_fc manage 2>&1)
assert_contains "manage model view shows snapshot" "snapshot:" "$output"
assert_contains "manage model view shows artifacts" "packed_experts/" "$output"
assert_contains "manage offers verify actions" "deep verify" "$output"
assert_exists "manage left weights intact" "${NATIVE_SNAPSHOT}/flashchat/q4/model_weights.bin"

echo ""
echo "=== Ensure gating ==="
# Exhausted stdin must never consent to downloads/builds/migrations.
output=$(printf '' | run_fc config </dev/null 2>&1 || true)
assert_not_contains "eof config does not download" "downloading" "$output"
[ ! -d "${HF_CACHE}/models--mlx-community--Qwen3.6-35B-A3B-4bit" ] \
    && pass "eof config fetched nothing" || fail "eof config fetched nothing"

summary=$(printf 'q\n' | run_fc 2>&1 | head -20)
assert_contains "interactive menu renders" "Flashchat Management" "$summary"

echo ""
echo "================================"
echo "Passed: $PASSED, Failed: $FAILED"
[ "$FAILED" -eq 0 ]