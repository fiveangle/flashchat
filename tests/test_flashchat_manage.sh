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

MODEL_ID="mlx-community-Qwen36-35B-A3B-4bit"
REPO_DIR_NAME="models--mlx-community--Qwen3.6-35B-A3B-4bit"
NATIVE_Q4_ID="Qwen-Qwen36-35B-A3B"
NATIVE_Q8_ID="Qwen-Qwen36-35B-A3B-8bit"
NATIVE_REPO_DIR_NAME="models--Qwen--Qwen3.6-35B-A3B"
DENSE_ID="Qwen-Qwen36-27B"
DENSE_REPO_DIR_NAME="models--Qwen--Qwen3.6-27B"
SNAPSHOT_ID="test-snapshot"

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

assert_not_contains() {
    local name="$1"
    local unexpected="$2"
    local output="$3"
    if printf "%s" "$output" | grep -q "$unexpected"; then
        assert_fail "$name" "unexpected '$unexpected'"
    else
        assert_pass "$name"
    fi
}

TMPDIR="$(mktemp -d /tmp/flashchat-manage-test.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

export HOME="${TMPDIR}/home"
CONFIG_DIR="${HOME}/.config/flashchat"
HF_CACHE="${TMPDIR}/custom-hf-cache"
LOCAL_REPO="${HF_CACHE}/${REPO_DIR_NAME}"
OFFLOAD_DIR="${TMPDIR}/offload"
OFFLOADED_REPO="${OFFLOAD_DIR}/${REPO_DIR_NAME}"
LOCAL_SNAPSHOT="${LOCAL_REPO}/snapshots/${SNAPSHOT_ID}"
OFFLOADED_SNAPSHOT="${OFFLOADED_REPO}/snapshots/${SNAPSHOT_ID}"
NATIVE_LOCAL_REPO="${HF_CACHE}/${NATIVE_REPO_DIR_NAME}"
NATIVE_OFFLOADED_REPO="${OFFLOAD_DIR}/${NATIVE_REPO_DIR_NAME}"
NATIVE_LOCAL_SNAPSHOT="${NATIVE_LOCAL_REPO}/snapshots/${SNAPSHOT_ID}"
NATIVE_OFFLOADED_SNAPSHOT="${NATIVE_OFFLOADED_REPO}/snapshots/${SNAPSHOT_ID}"
DENSE_LOCAL_REPO="${HF_CACHE}/${DENSE_REPO_DIR_NAME}"
DENSE_OFFLOADED_REPO="${OFFLOAD_DIR}/${DENSE_REPO_DIR_NAME}"
DENSE_LOCAL_SNAPSHOT="${DENSE_LOCAL_REPO}/snapshots/${SNAPSHOT_ID}"

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
HUGGINGFACE_CACHE_DIR="${HF_CACHE}"
OFFLOAD_DIR="${OFFLOAD_DIR}"
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

source "${REPO_ROOT}/lib/config.sh"

runtime_dir_for() {
    flashchat_model_runtime_dir "$1" "$2"
}

write_runtime_fixture() {
    local model_id="$1"
    local snapshot="$2"
    local runtime_dir
    runtime_dir=$(runtime_dir_for "$model_id" "$snapshot")
    mkdir -p "$runtime_dir"
    python3 - "$REPO_ROOT/assets/model_configs.json" "$model_id" "$snapshot" "$runtime_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

registry_path, model_id, snapshot, runtime_dir = sys.argv[1:]
with open(registry_path) as f:
    registry = json.load(f)
model = registry["models"][model_id]
runtime = Path(runtime_dir)
runtime.mkdir(parents=True, exist_ok=True)

quant = model.get("quantization", {})
config = {
    "hidden_size": model["hidden_size"],
    "num_hidden_layers": model["num_hidden_layers"],
    "num_attention_heads": model["num_attention_heads"],
    "num_key_value_heads": model["num_key_value_heads"],
    "head_dim": model["head_dim"],
    "vocab_size": model["vocab_size"],
    "rms_norm_eps": model["rms_norm_eps"],
    "num_experts": model["num_experts"],
    "num_experts_per_tok": model["num_experts_per_tok"],
    "moe_intermediate_size": model["moe_intermediate_size"],
    "shared_expert_intermediate_size": model["shared_expert_intermediate_size"],
    "intermediate_size": model.get("intermediate_size", 0),
    "full_attention_interval": model["full_attention_interval"],
    "linear_num_value_heads": model["linear_num_value_heads"],
    "linear_num_key_heads": model["linear_num_key_heads"],
    "linear_key_head_dim": model["linear_key_head_dim"],
    "linear_value_head_dim": model["linear_value_head_dim"],
    "linear_conv_kernel_dim": model["linear_conv_kernel_dim"],
    "partial_rotary_factor": model["partial_rotary_factor"],
    "rope_theta": model["rope_theta"],
    "quantization": {
        "bits": quant.get("bits", 4),
        "group_size": quant.get("group_size", 64),
    },
}
config["layer_types"] = [
    "full_attention" if (i + 1) % config["full_attention_interval"] == 0 else "linear_attention"
    for i in range(config["num_hidden_layers"])
]
config["mtp_num_hidden_layers"] = model.get("mtp_num_hidden_layers", 0)

tensors = {"model.embed_tokens.weight": {"offset": 0, "size": 1, "shape": [1], "dtype": "BF16"}}
mtp_required = model.get("source_format") == "native_bf16" and int(model.get("mtp_max_predictions") or 0) > 0
if mtp_required:
    for name in [
        "mtp.fc.weight",
        "mtp.fc.scales",
        "mtp.fc.biases",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.q_proj.scales",
        "mtp.layers.0.self_attn.q_proj.biases",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.gate.scales",
        "mtp.layers.0.mlp.gate.biases",
        "mtp.norm.weight",
    ]:
        tensors[name] = {"offset": 0, "size": 1, "shape": [1], "dtype": "BF16"}

(runtime / "model_weights.bin").write_text("weights")
with open(runtime / "model_weights.json", "w") as f:
    json.dump({"model": snapshot, "num_tensors": len(tensors), "tensors": tensors, "config": config}, f, indent=2)
(runtime / "vocab.bin").write_text("vocab")

num_experts = int(model.get("num_experts") or 0)
if num_experts > 0:
    hidden = int(model["hidden_size"])
    moe = int(model["moe_intermediate_size"])
    bits = int(quant.get("bits", 4) or 4)
    group_size = int(quant.get("group_size", 64) or 64)
    values_per_word = 32 // bits
    gate_w = moe * (hidden // values_per_word) * 4
    gate_s = moe * (hidden // group_size) * 2
    down_w = hidden * (moe // values_per_word) * 4
    down_s = hidden * (moe // group_size) * 2
    expert_size = gate_w + gate_s + gate_s + gate_w + gate_s + gate_s + down_w + down_s + down_s

    with open(runtime / "expert_index.json", "w") as f:
        json.dump({"model_path": os.path.realpath(snapshot), "expert_reads": {}}, f)

    packed = runtime / "packed_experts"
    packed.mkdir(parents=True, exist_ok=True)
    with open(packed / "layout.json", "w") as f:
        json.dump({"expert_size": expert_size, "num_layers": model["num_hidden_layers"], "num_experts": num_experts, "artifact": "packed_experts"}, f)
    for i in range(model["num_hidden_layers"]):
        (packed / f"layer_{i:02d}.bin").write_text("layer")

    if mtp_required:
        mtp_layers = int(model.get("mtp_num_hidden_layers") or 1)
        packed_mtp = runtime / "packed_mtp_experts"
        packed_mtp.mkdir(parents=True, exist_ok=True)
        with open(packed_mtp / "layout.json", "w") as f:
            json.dump({"expert_size": expert_size, "num_layers": mtp_layers, "num_experts": num_experts, "artifact": "packed_mtp_experts"}, f)
        for i in range(mtp_layers):
            (packed_mtp / f"layer_{i:02d}.bin").write_text("mtp-layer")
PY
}

write_original_blob() {
    local repo="$1"
    local snapshot="$2"
    mkdir -p "${repo}/blobs" "$snapshot"
    printf "blob" > "${repo}/blobs/blob1"
    ln -s "../../blobs/blob1" "${snapshot}/model-00001-of-00001.safetensors"
}

reset_storage() {
    rm -rf "$LOCAL_REPO" "$OFFLOADED_REPO"
    write_runtime_fixture "$MODEL_ID" "$LOCAL_SNAPSHOT"
    write_original_blob "$LOCAL_REPO" "$LOCAL_SNAPSHOT"
}

reset_native_storage() {
    rm -rf "$NATIVE_LOCAL_REPO" "$NATIVE_OFFLOADED_REPO"
    write_runtime_fixture "$NATIVE_Q4_ID" "$NATIVE_LOCAL_SNAPSHOT"
    write_runtime_fixture "$NATIVE_Q8_ID" "$NATIVE_LOCAL_SNAPSHOT"
    write_original_blob "$NATIVE_LOCAL_REPO" "$NATIVE_LOCAL_SNAPSHOT"
}

reset_dense_storage() {
    rm -rf "$DENSE_LOCAL_REPO" "$DENSE_OFFLOADED_REPO"
    write_runtime_fixture "$DENSE_ID" "$DENSE_LOCAL_SNAPSHOT"
    write_original_blob "$DENSE_LOCAL_REPO" "$DENSE_LOCAL_SNAPSHOT"
}

run_manage() {
    local input="$1"
    printf "%b" "$input" | "$FLASHCHAT" manage --interactive 2>&1
}

run_flashchat() {
    local input="$1"
    printf "%b" "$input" | "$FLASHCHAT" 2>&1
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
output=$(run_manage "${MODEL_ID}\n3\n${MODEL_ID}\nq\n")
assert_contains "startup restore setup offloads model" "Model offloaded" "$output"
output=$(run_flashchat "\nq\n")
assert_contains "startup restore offers offload copy" "Offloaded Model Found" "$output"
assert_contains "startup restore offers offload run" "Run from offload directory" "$output"
assert_contains "startup restore reloads model" "Model fully reloaded" "$output"
assert_not_contains "startup restore skips HF download prompt" "Model Download Required" "$output"
assert_exists "startup restore creates local repo" "$LOCAL_REPO"
assert_not_exists "startup restore consumes offload repo" "$OFFLOADED_REPO"

reset_storage
output=$(run_manage "${MODEL_ID}\n3\n${MODEL_ID}\nq\n")
assert_contains "startup offload run setup offloads model" "Model offloaded" "$output"
output=$(run_flashchat "u\nq\n")
assert_contains "startup offload run offers offload copy" "Offloaded Model Found" "$output"
assert_contains "startup offload run uses offloaded model" "Using offloaded model for this run" "$output"
assert_not_contains "startup offload run skips HF download prompt" "Model Download Required" "$output"
assert_not_exists "startup offload run leaves local repo absent" "$LOCAL_REPO"
assert_exists "startup offload run keeps offload repo" "$OFFLOADED_REPO"

reset_storage
output=$(run_manage "${MODEL_ID}\n3\n${MODEL_ID}\nq\n")
mkdir -p "$LOCAL_REPO"
output=$(run_flashchat "y\nq\n")
assert_contains "startup offload collision reports reload unavailable" "Reloading to the HuggingFace cache is unavailable" "$output"
assert_contains "startup offload collision uses offloaded model" "Using offloaded model for this run" "$output"
assert_exists "startup offload collision keeps local repo" "$LOCAL_REPO"
assert_exists "startup offload collision keeps offload repo" "$OFFLOADED_REPO"

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

reset_native_storage
output=$(run_manage "${NATIVE_Q8_ID}\n3\n${NATIVE_Q8_ID}\nq\n")
assert_contains "native shared repo offloads by q8 ID" "Model offloaded" "$output"
assert_not_exists "native offload removes shared local repo" "$NATIVE_LOCAL_REPO"
assert_exists "native offload creates shared offload repo" "$NATIVE_OFFLOADED_REPO"

write_runtime_fixture "$NATIVE_Q4_ID" "$NATIVE_LOCAL_SNAPSHOT"
printf "q4-local" > "$(runtime_dir_for "$NATIVE_Q4_ID" "$NATIVE_LOCAL_SNAPSHOT")/q4-marker"
output=$(run_manage "${NATIVE_Q8_ID}\n5\nq\n")
assert_contains "native q8 runtime-only restore succeeds" "Runtime artifacts restored" "$output"
assert_exists "native q8 restore creates q8 runtime" "$(runtime_dir_for "$NATIVE_Q8_ID" "$NATIVE_LOCAL_SNAPSHOT")/model_weights.bin"
assert_exists "native q8 restore includes MTP experts" "$(runtime_dir_for "$NATIVE_Q8_ID" "$NATIVE_LOCAL_SNAPSHOT")/packed_mtp_experts/layer_00.bin"
assert_exists "native q8 restore preserves q4 runtime files" "$(runtime_dir_for "$NATIVE_Q4_ID" "$NATIVE_LOCAL_SNAPSHOT")/q4-marker"
assert_not_exists "native q8 runtime-only restore does not copy original blobs" "${NATIVE_LOCAL_SNAPSHOT}/model-00001-of-00001.safetensors"

rm -rf "$NATIVE_LOCAL_REPO"
write_runtime_fixture "$NATIVE_Q8_ID" "$NATIVE_LOCAL_SNAPSHOT"
printf "q8-local" > "$(runtime_dir_for "$NATIVE_Q8_ID" "$NATIVE_LOCAL_SNAPSHOT")/q8-marker"
output=$(run_manage "${NATIVE_Q4_ID}\n5\nq\n")
assert_contains "native q4 runtime-only restore succeeds" "Runtime artifacts restored" "$output"
assert_exists "native q4 restore creates q4 runtime" "$(runtime_dir_for "$NATIVE_Q4_ID" "$NATIVE_LOCAL_SNAPSHOT")/model_weights.bin"
assert_exists "native q4 restore preserves q8 runtime files" "$(runtime_dir_for "$NATIVE_Q8_ID" "$NATIVE_LOCAL_SNAPSHOT")/q8-marker"

reset_dense_storage
output=$(run_manage "${DENSE_ID}\n1\n${DENSE_ID}\nq\n")
assert_contains "dense runtime-ready blob removal succeeds" "Original blobs removed" "$output"
assert_not_exists "dense blob link removed" "${DENSE_LOCAL_SNAPSHOT}/model-00001-of-00001.safetensors"
assert_not_exists "dense blob target removed" "${DENSE_LOCAL_REPO}/blobs/blob1"
assert_exists "dense runtime remains without packed experts" "$(runtime_dir_for "$DENSE_ID" "$DENSE_LOCAL_SNAPSHOT")/model_weights.bin"

reset_dense_storage
output=$(run_manage "${DENSE_ID}\n3\n${DENSE_ID}\nq\n")
assert_contains "dense offload moves repo" "Model offloaded" "$output"
output=$(run_manage "${DENSE_ID}\n5\nq\n")
assert_contains "dense runtime-only restore succeeds" "Runtime artifacts restored" "$output"
assert_exists "dense runtime-only restore copies MTP weights" "$(runtime_dir_for "$DENSE_ID" "$DENSE_LOCAL_SNAPSHOT")/model_weights.json"
assert_not_exists "dense runtime-only restore does not copy safetensors" "${DENSE_LOCAL_SNAPSHOT}/model-00001-of-00001.safetensors"

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
