#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLASHCHAT="${REPO_ROOT}/flashchat"

FAILED=0
PASSED=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

MODEL_ID="qwen3.6-35B-A3B"
REPO_DIR_NAME="models--mlx-community--Qwen3.6-35B-A3B-4bit"
SNAPSHOT_ID="test-snapshot"
LAYER_COUNT=40

assert_pass() {
    PASSED=$((PASSED + 1))
    echo -e "${GREEN}PASS${NC}  $1"
}

assert_fail() {
    FAILED=$((FAILED + 1))
    echo -e "${RED}FAIL${NC}  $1${2:+: $2}"
}

assert_exists() {
    if [ -e "$2" ]; then
        assert_pass "$1"
    else
        assert_fail "$1" "missing $2"
    fi
}

assert_not_exists() {
    if [ ! -e "$2" ]; then
        assert_pass "$1"
    else
        assert_fail "$1" "still exists $2"
    fi
}

assert_contains() {
    local name="$1"
    local expected="$2"
    local output="$3"
    if printf "%s" "$output" | grep -q "$expected"; then
        assert_pass "$name"
    else
        assert_fail "$name" "expected '$expected'"
    fi
}

TMPDIR="$(mktemp -d /tmp/flashchat-manage-test.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

export HOME="${TMPDIR}/home"
CONFIG_DIR="${HOME}/.config/flashchat"
HF_CACHE="${HOME}/.cache/huggingface/hub"
LOCAL_REPO="${HF_CACHE}/${REPO_DIR_NAME}"
OFFLOAD_DIR="${TMPDIR}/offload"
OFFLOADED_REPO="${OFFLOAD_DIR}/${REPO_DIR_NAME}"
LOCAL_SNAPSHOT="${LOCAL_REPO}/snapshots/${SNAPSHOT_ID}"
OFFLOADED_SNAPSHOT="${OFFLOADED_REPO}/snapshots/${SNAPSHOT_ID}"

mkdir -p "$CONFIG_DIR"
cat > "${CONFIG_DIR}/config" <<EOF
MODEL="${MODEL_ID}"
MAX_TOKENS="1"
SAMPLING_PROFILE="custom"
REASONING="0"
TEMPERATURE="0.1"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
SERVER_PORT="19998"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="${TMPDIR}/logs"
OFFLOAD_DIR="${OFFLOAD_DIR}"
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

reset_storage() {
    rm -rf "$LOCAL_REPO" "$OFFLOADED_REPO"
    mkdir -p "${LOCAL_SNAPSHOT}/flashchat/packed_experts" "${LOCAL_REPO}/blobs"
    printf "weights" > "${LOCAL_SNAPSHOT}/flashchat/model_weights.bin"
    printf "{}" > "${LOCAL_SNAPSHOT}/flashchat/model_weights.json"
    printf "vocab" > "${LOCAL_SNAPSHOT}/flashchat/vocab.bin"
    printf "{}" > "${LOCAL_SNAPSHOT}/flashchat/expert_index.json"
    local i layer
    i=0
    while [ "$i" -lt "$LAYER_COUNT" ]; do
        printf -v layer "%02d" "$i"
        printf "layer" > "${LOCAL_SNAPSHOT}/flashchat/packed_experts/layer_${layer}.bin"
        i=$((i + 1))
    done
    printf "blob" > "${LOCAL_REPO}/blobs/blob1"
    ln -s "../../blobs/blob1" "${LOCAL_SNAPSHOT}/model-00001-of-00001.safetensors"
}

run_manage() {
    local input="$1"
    printf "%b" "$input" | "$FLASHCHAT" manage --interactive 2>&1
}

echo ""
echo "=== Flashchat Manage Storage Smoke ==="
echo ""

reset_storage
output=$(run_manage "${MODEL_ID}\n1\nwrong-id\nq\n")
assert_contains "wrong model ID refuses blob removal" "Confirmation did not match" "$output"
assert_exists "wrong model ID keeps blob link" "${LOCAL_SNAPSHOT}/model-00001-of-00001.safetensors"
assert_exists "wrong model ID keeps blob target" "${LOCAL_REPO}/blobs/blob1"

output=$(run_manage "${MODEL_ID}\n1\n${MODEL_ID}\nq\n")
assert_contains "runtime-ready blob removal succeeds" "Original blobs removed" "$output"
assert_not_exists "blob link removed" "${LOCAL_SNAPSHOT}/model-00001-of-00001.safetensors"
assert_not_exists "blob target removed" "${LOCAL_REPO}/blobs/blob1"
assert_exists "runtime remains after blob removal" "${LOCAL_SNAPSHOT}/flashchat/model_weights.bin"

reset_storage
rm -f "${LOCAL_SNAPSHOT}/flashchat/packed_experts/layer_39.bin"
output=$(run_manage "${MODEL_ID}\n1\nq\n")
assert_contains "incomplete runtime blocks blob removal" "runtime artifacts are complete" "$output"
assert_exists "incomplete runtime keeps blob" "${LOCAL_REPO}/blobs/blob1"

reset_storage
output=$(run_manage "${MODEL_ID}\n3\n${MODEL_ID}\nq\n")
assert_contains "offload moves repo" "Model offloaded" "$output"
assert_not_exists "offload removes local repo" "$LOCAL_REPO"
assert_exists "offload creates offloaded repo" "$OFFLOADED_REPO"

output=$(run_manage "${MODEL_ID}\n5\nq\n")
assert_contains "runtime-only restore succeeds" "Runtime artifacts restored" "$output"
assert_exists "runtime-only restore copies runtime" "${LOCAL_SNAPSHOT}/flashchat/model_weights.bin"
assert_not_exists "runtime-only restore does not copy safetensors" "${LOCAL_SNAPSHOT}/model-00001-of-00001.safetensors"
assert_exists "runtime-only restore leaves offload intact" "$OFFLOADED_REPO"

printf "local" > "${LOCAL_SNAPSHOT}/flashchat/local-marker"
printf "offload" > "${OFFLOADED_SNAPSHOT}/flashchat/offload-marker"
output=$(run_manage "${MODEL_ID}\n5\nwrong-id\nq\n")
assert_contains "runtime overwrite requires exact model ID" "Confirmation did not match" "$output"
assert_exists "wrong runtime overwrite keeps local runtime" "${LOCAL_SNAPSHOT}/flashchat/local-marker"

output=$(run_manage "${MODEL_ID}\n4\nq\n")
assert_contains "full reload refuses local collision" "Refusing to overwrite existing local model" "$output"
assert_exists "collision leaves offload repo" "$OFFLOADED_REPO"

rm -rf "$LOCAL_REPO"
output=$(run_manage "${MODEL_ID}\n4\n${MODEL_ID}\nq\n")
assert_contains "full reload moves repo back" "Model fully reloaded" "$output"
assert_exists "full reload restores local repo" "$LOCAL_REPO"
assert_not_exists "full reload removes offload copy" "$OFFLOADED_REPO"

reset_storage
mkdir -p "$OFFLOADED_REPO"
output=$(run_manage "${MODEL_ID}\n3\nq\n")
assert_contains "offload refuses destination collision" "Refusing to overwrite existing offloaded model" "$output"
assert_exists "collision keeps local repo" "$LOCAL_REPO"
assert_exists "collision keeps offload repo" "$OFFLOADED_REPO"

rm -rf "$OFFLOADED_REPO"
output=$(run_manage "${MODEL_ID}\n2\nwrong-id\nq\n")
assert_contains "wrong model ID refuses local delete" "Confirmation did not match" "$output"
assert_exists "wrong model ID keeps local repo" "$LOCAL_REPO"

output=$(run_manage "${MODEL_ID}\n2\n${MODEL_ID}\nq\n")
assert_contains "local delete succeeds" "Local model deleted" "$output"
assert_not_exists "local delete removes repo" "$LOCAL_REPO"

echo ""
echo "========================================"
echo "Flashchat Manage Storage Smoke Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC}  $PASSED"
echo -e "${RED}Failed:${NC}  $FAILED"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi

echo -e "${GREEN}All tests passed!${NC}"
