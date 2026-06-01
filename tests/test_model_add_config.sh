#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEST_ROOT="$ROOT/debug/fresh-envs/$(date +%Y%m%d-%H%M%S)-model-add-config"
HOME_DIR="$TEST_ROOT/home"
HF_CACHE="$TEST_ROOT/hf"
BIN_DIR="$TEST_ROOT/bin"
CONFIG_JSON="$TEST_ROOT/model_configs.json"
mkdir -p "$HOME_DIR/.config/flashchat" "$BIN_DIR"

python3 - "$ROOT/assets/model_configs.json" "$CONFIG_JSON" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    data = json.load(f)
data.get("models", {}).pop("Qwen-Qwen36-35B-A3B", None)
with open(dst, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

cat > "$BIN_DIR/hf" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$BIN_DIR/hf"

SNAP="$HF_CACHE/models--Qwen--Qwen3.6-35B-A3B/snapshots/test-snapshot"
mkdir -p "$SNAP"
cat > "$SNAP/config.json" <<'EOF'
{
  "model_type": "qwen3_5_moe",
  "hidden_size": 2048,
  "num_hidden_layers": 40,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "vocab_size": 248320,
  "rms_norm_eps": 1e-6,
  "num_experts": 256,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 512,
  "full_attention_interval": 4,
  "linear_num_value_heads": 32,
  "linear_num_key_heads": 16,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "partial_rotary_factor": 0.25,
  "rope_theta": 10000000.0
}
EOF

cat > "$HOME_DIR/.config/flashchat/config" <<EOF
MODEL="mlx-community-Qwen36-35B-A3B-4bit"
MAX_TOKENS="1"
SAMPLING_PROFILE="custom"
REASONING="0"
TEMPERATURE="0.7"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
SERVER_PORT="19998"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="$TEST_ROOT/logs"
HUGGINGFACE_CACHE_DIR="$HF_CACHE"
OFFLOAD_DIR=""
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
MTP="0"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

output=$(
    {
        printf 'y'
        printf '0\n'
        printf 'Qwen/Qwen3.6-35B-A3B\n'
        printf 'y'
        for _ in $(seq 1 40); do
            printf '\n'
        done
    } | HOME="$HOME_DIR" PATH="$BIN_DIR:$PATH" FLASHCHAT_MODEL_CONFIG="$CONFIG_JSON" ./flashchat config
)

if echo "$output" | grep -q "WARNING: model 'Registry updated"; then
    echo "FAIL: registry status leaked into selected model ID" >&2
    exit 1
fi

model_value=$(HOME="$HOME_DIR" FLASHCHAT_MODEL_CONFIG="$CONFIG_JSON" bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MODEL')
if [ "$model_value" != "Qwen-Qwen36-35B-A3B" ]; then
    echo "FAIL: expected selected model Qwen-Qwen36-35B-A3B, got '$model_value'" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
model = data["models"]["Qwen-Qwen36-35B-A3B"]
assert model["source_format"] == "native_bf16"
assert model["scripts"]["extract_weights"] == "scripts/compile_native_qwen.py"
assert model["scripts"]["repack_experts"] == "scripts/compile_native_qwen.py"
assert model["mtp_default_predictions"] == 0
assert model["mtp_max_predictions"] == 0
PY

echo "model add config smoke passed"
