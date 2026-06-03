#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEST_ROOT="$ROOT/debug/fresh-envs/$(date +%Y%m%d-%H%M%S)-model-quant-config"
HOME_DIR="$TEST_ROOT/home"
mkdir -p "$HOME_DIR/.config/flashchat"

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
SERVER_PORT="19995"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="$TEST_ROOT/logs"
HUGGINGFACE_CACHE_DIR="$TEST_ROOT/hf"
OFFLOAD_DIR=""
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
MTP="0"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

helper_output=$(
    HOME="$HOME_DIR" bash -c '
        source lib/config.sh
        echo "bits=$(flashchat_model_quant_bits mlx-community-Qwen36-35B-A3B-8bit)"
        echo "bytes=$(flashchat_model_expert_pack_bytes mlx-community-Qwen36-35B-A3B-8bit)"
        echo "native_bits=$(flashchat_model_quant_bits Qwen-Qwen36-35B-A3B-8bit)"
        echo "native_bytes=$(flashchat_model_expert_pack_bytes Qwen-Qwen36-35B-A3B-8bit)"
        echo "native_q4_runtime=$(flashchat_model_runtime_dir Qwen-Qwen36-35B-A3B /tmp/native)"
        echo "native_q8_runtime=$(flashchat_model_runtime_dir Qwen-Qwen36-35B-A3B-8bit /tmp/native)"
        echo "mlx_q4_runtime=$(flashchat_model_runtime_dir mlx-community-Qwen36-35B-A3B-4bit /tmp/mlx)"
        echo "mlx_q8_runtime=$(flashchat_model_runtime_dir mlx-community-Qwen36-35B-A3B-8bit /tmp/mlx)"
    '
)

if ! echo "$helper_output" | grep -q '^bits=8$'; then
    echo "FAIL: 8-bit model helper did not report bits=8" >&2
    exit 1
fi
if ! echo "$helper_output" | grep -q '^bytes=34225520640$'; then
    echo "FAIL: 8-bit expert pack size changed unexpectedly" >&2
    echo "$helper_output" >&2
    exit 1
fi
if ! echo "$helper_output" | grep -q '^native_bits=8$'; then
    echo "FAIL: native Qwen 8-bit model helper did not report bits=8" >&2
    exit 1
fi
if ! echo "$helper_output" | grep -q '^native_bytes=34225520640$'; then
    echo "FAIL: native Qwen 8-bit expert pack size changed unexpectedly" >&2
    echo "$helper_output" >&2
    exit 1
fi
if ! echo "$helper_output" | grep -q '^native_q4_runtime=/tmp/native/flashchat/q4$'; then
    echo "FAIL: native 4-bit runtime path should use flashchat/q4" >&2
    echo "$helper_output" >&2
    exit 1
fi
if ! echo "$helper_output" | grep -q '^native_q8_runtime=/tmp/native/flashchat/q8$'; then
    echo "FAIL: native 8-bit runtime path should use flashchat/q8" >&2
    echo "$helper_output" >&2
    exit 1
fi
if ! echo "$helper_output" | grep -q '^mlx_q4_runtime=/tmp/mlx/flashchat/q4$'; then
    echo "FAIL: MLX 4-bit runtime path should use flashchat/q4" >&2
    echo "$helper_output" >&2
    exit 1
fi
if ! echo "$helper_output" | grep -q '^mlx_q8_runtime=/tmp/mlx/flashchat/q8$'; then
    echo "FAIL: MLX 8-bit runtime path should use flashchat/q8" >&2
    echo "$helper_output" >&2
    exit 1
fi

output=$(
    {
        printf 'y'
        printf 'Qwen-Qwen36-35B-A3B-8bit\n'
        for _ in $(seq 1 80); do printf '\n'; done
    } | HOME="$HOME_DIR" ./flashchat config 2>&1
)

if ! echo "$output" | grep -q "Qwen/Qwen3.6-35B-A3B"; then
    echo "FAIL: wizard did not list the native Qwen repo" >&2
    exit 1
fi
if ! echo "$output" | grep -q "Quantization: 8-bit"; then
    echo "FAIL: wizard did not show 8-bit quantization" >&2
    exit 1
fi
if ! echo "$output" | grep -q "expert pack: 31.9 GiB"; then
    echo "FAIL: wizard did not show derived 8-bit expert pack size" >&2
    exit 1
fi

model_value=$(
    HOME="$HOME_DIR" bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MODEL'
)
if [ "$model_value" != "Qwen-Qwen36-35B-A3B-8bit" ]; then
    echo "FAIL: expected selected 8-bit model, got '$model_value'" >&2
    exit 1
fi

weights_dir=$(
    HOME="$HOME_DIR" bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get WEIGHTS_DIR'
)
if [[ "$weights_dir" != */models--Qwen--Qwen3.6-35B-A3B/snapshots/\<snapshot\>/flashchat/q8 ]]; then
    echo "FAIL: selected native 8-bit model should store runtime artifacts under flashchat/q8, got '$weights_dir'" >&2
    exit 1
fi

cat > "$HOME_DIR/.config/flashchat/config" <<EOF
MODEL="Qwen-Qwen36-35B-A3B"
MAX_TOKENS="32768"
SAMPLING_PROFILE="instruct"
REASONING="0"
TEMPERATURE="0.7"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
SERVER_PORT="19995"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="$TEST_ROOT/logs"
HUGGINGFACE_CACHE_DIR="$TEST_ROOT/hf"
OFFLOAD_DIR=""
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
MTP=""
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

output=$(
    {
        printf 'y'
        printf '6\n'
        printf '\n'
        printf '2\n'
        for _ in $(seq 1 12); do printf '\n'; done
    } | HOME="$HOME_DIR" ./flashchat config 2>&1
)

if echo "$output" | grep -q "No changes made"; then
    echo "FAIL: selecting native 8-bit by visible number reported no changes" >&2
    echo "$output" >&2
    exit 1
fi
if ! echo "$output" | grep -q "Selected model: Qwen-Qwen36-35B-A3B-8bit (8-bit)"; then
    echo "FAIL: wizard did not confirm selected native 8-bit model" >&2
    echo "$output" >&2
    exit 1
fi
model_value=$(
    HOME="$HOME_DIR" bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MODEL'
)
if [ "$model_value" != "Qwen-Qwen36-35B-A3B-8bit" ]; then
    echo "FAIL: visible model number 6 should select native 8-bit model, got '$model_value'" >&2
    exit 1
fi
profile_value=$(
    HOME="$HOME_DIR" bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get SAMPLING_PROFILE'
)
if [ "$profile_value" != "thinking-coding" ]; then
    echo "FAIL: canned profile selection should persist thinking-coding, got '$profile_value'" >&2
    exit 1
fi

echo "model quant config smoke passed"
