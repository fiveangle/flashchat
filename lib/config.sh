#!/bin/bash
# config.sh — Flashchat Configuration Loader
#
# Loads configuration with the following priority (highest to lowest):
#   1. --config FILE (explicit override)
#   2. ~/.config/flashchat/config (user)
#   3. Environment variables (FLASHCHAT_*)
#   4. Registry/default values
#
# Usage:
#   source lib/config.sh          # Load config
#   flashchat_load_config          # Initialize (creates default if missing)
#   flashchat_get "KEY"           # Get a config value

set -e

FLASHCHAT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASHCHAT_REPO_ROOT="$(cd "${FLASHCHAT_LIB_DIR}/.." && pwd)"
FLASHCHAT_CONFIG_DIR="${HOME}/.config/flashchat"
FLASHCHAT_CONFIG_FILE=""
FLASHCHAT_MODEL_CONFIG="${FLASHCHAT_MODEL_CONFIG:-${FLASHCHAT_REPO_ROOT}/assets/model_configs.json}"

# Default configuration values
FLASHCHAT_DEFAULT_MODEL="mlx-community-Qwen36-35B-A3B-4bit"
FLASHCHAT_DEFAULT_MAX_TOKENS="8192"
FLASHCHAT_DEFAULT_SERVER_PORT="8000"
FLASHCHAT_DEFAULT_SERVER_HOST="127.0.0.1"
FLASHCHAT_DEFAULT_SERVER_LOG_PATH="${FLASHCHAT_CONFIG_DIR}/logs/server.log"
FLASHCHAT_DEFAULT_SHOW_THINKING="0"
FLASHCHAT_DEFAULT_SERVER_DEBUG="0"
FLASHCHAT_DEFAULT_SERVER_HTTP_LOG="0"
FLASHCHAT_DEFAULT_SYSTEM_PROMPT_CACHE="1"
FLASHCHAT_DEFAULT_SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
FLASHCHAT_DEFAULT_MTP=""
FLASHCHAT_DEFAULT_MTP_BF16="0"
FLASHCHAT_DEFAULT_COLOR_OUTPUT="1"
FLASHCHAT_DEFAULT_SAMPLING_PROFILE=""
FLASHCHAT_DEFAULT_REASONING="0"
FLASHCHAT_DEFAULT_TEMPERATURE="0.7"
FLASHCHAT_DEFAULT_TOP_P="0.8"
FLASHCHAT_DEFAULT_TOP_K="20"
FLASHCHAT_DEFAULT_MIN_P="0.0"
FLASHCHAT_DEFAULT_PRESENCE_PENALTY="1.5"
FLASHCHAT_DEFAULT_REPETITION_PENALTY="1.0"
FLASHCHAT_DEFAULT_HUGGINGFACE_CACHE_DIR="${HOME}/.cache/huggingface/hub"
FLASHCHAT_DEFAULT_OFFLOAD_DIR=""

# Config values (set after loading)
MODEL=""
MODEL_REPO=""
MAX_TOKENS=""
SERVER_PORT=""
SERVER_HOST=""
SERVER_LOG_PATH=""
SHOW_THINKING=""
SERVER_DEBUG=""
SERVER_HTTP_LOG=""
SYSTEM_PROMPT_CACHE=""
SYSTEM_PROMPT_CACHE_MAX_ENTRIES=""
MTP=""
MTP_BF16=""
COLOR_OUTPUT=""
SAMPLING_PROFILE=""
REASONING=""
TEMPERATURE=""
TOP_P=""
TOP_K=""
MIN_P=""
PRESENCE_PENALTY=""
REPETITION_PENALTY=""
HUGGINGFACE_CACHE_DIR=""
OFFLOAD_DIR=""

# Derived paths (computed after config load)
MODEL_PATH=""
WEIGHTS_DIR=""
EXPERTS_DIR=""

# -----------------------------------------------------------------------------
# Look up model defaults from the bundled model registry.
# -----------------------------------------------------------------------------
flashchat_default_model() {
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    if [ -f "$config_file" ] && command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    print(data.get('default_model') or sys.argv[2])
except Exception:
    print(sys.argv[2])
" "$config_file" "$FLASHCHAT_DEFAULT_MODEL" 2>/dev/null
    else
        echo "$FLASHCHAT_DEFAULT_MODEL"
    fi
}

_flashchat_lookup_model_repo() {
    local model_id="${1:-$(flashchat_default_model)}"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    if [ -f "$config_file" ] && command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    model = data.get('models', {}).get(sys.argv[2], {})
    print(model.get('hf_repo', ''))
except Exception:
    pass
" "$config_file" "$model_id" 2>/dev/null
    fi
}

# -----------------------------------------------------------------------------
# Model registry helpers
# -----------------------------------------------------------------------------
flashchat_model_exists() {
    local model_id="$1"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -n "$model_id" ] || return 1
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
sys.exit(0 if sys.argv[2] in data.get('models', {}) else 1)
" "$config_file" "$model_id" 2>/dev/null
}

flashchat_model_field() {
    local model_id="$1"
    local field="$2"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -n "$model_id" ] || return 1
    [ -n "$field" ] || return 1
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
model = data.get('models', {}).get(sys.argv[2], {})
value = model.get(sys.argv[3], '')
print(value if value is not None else '')
" "$config_file" "$model_id" "$field" 2>/dev/null
}

flashchat_model_name() {
    flashchat_model_field "$1" "name"
}

flashchat_model_repo() {
    flashchat_model_field "$1" "hf_repo"
}

flashchat_model_layers() {
    local layers
    layers=$(flashchat_model_field "$1" "num_hidden_layers")
    echo "${layers:-60}"
}

flashchat_model_default_sampling_profile() {
    flashchat_model_field "$1" "default_sampling_profile"
}

flashchat_model_mtp_default() {
    flashchat_model_field "$1" "mtp_default_predictions"
}

flashchat_server_default_field() {
    local field="$1"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -n "$field" ] || return 1
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
value = data.get('server_defaults', {}).get(sys.argv[2], '')
print(value if value is not None else '')
" "$config_file" "$field" 2>/dev/null
}

flashchat_server_mtp_default() {
    flashchat_server_default_field "mtp_default_predictions"
}

flashchat_model_quant_bits() {
    local model_id="$1"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -n "$model_id" ] || return 1
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
model = data.get('models', {}).get(sys.argv[2], {})
quant = model.get('quantization', {})
print(quant.get('bits', 4))
" "$config_file" "$model_id" 2>/dev/null
}

flashchat_model_expert_pack_bytes() {
    local model_id="$1"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -n "$model_id" ] || return 1
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
model = data.get('models', {}).get(sys.argv[2], {})
num_experts = int(model.get('num_experts') or 0)
layers = int(model.get('num_hidden_layers') or 0)
if num_experts <= 0 or layers <= 0:
    print(0)
    sys.exit(0)
hidden = int(model.get('hidden_size') or 0)
moe = int(model.get('moe_intermediate_size') or 0)
quant = model.get('quantization', {})
bits = int(quant.get('bits', 4) or 4)
group_size = int(quant.get('group_size', 64) or 64)
values_per_word = 32 // bits
gate_w = moe * (hidden // values_per_word) * 4
gate_s = moe * (hidden // group_size) * 2
down_w = hidden * (moe // values_per_word) * 4
down_s = hidden * (moe // group_size) * 2
expert_size = gate_w + gate_s + gate_s + gate_w + gate_s + gate_s + down_w + down_s + down_s
print(expert_size * num_experts * layers)
" "$config_file" "$model_id" 2>/dev/null
}

# Resolve the model's trained active-experts (K) from the registry. Returns
# empty string if the field is missing — caller decides what default to use.
flashchat_model_active_experts() {
    flashchat_model_field "$1" "num_experts_per_tok"
}

# Total routed experts. 0 (or empty) => dense model: no packed_experts/ or
# expert_index.json runtime artifacts (the MLP is quantized into model_weights.bin).
flashchat_model_num_experts() {
    local n
    n=$(flashchat_model_field "$1" "num_experts")
    echo "${n:-0}"
}

# Resolve the *effective* active-experts value: $FLASHCHAT_ACTIVE_EXPERTS env
# override if set and >0, else the model's trained num_experts_per_tok from
# the registry. The env var is the user's escape hatch (e.g. for sweeping K
# during quality-vs-streaming-cost benchmarks); otherwise the registry is the
# authoritative source for K.
flashchat_effective_active_experts() {
    local override="${FLASHCHAT_ACTIVE_EXPERTS:-}"
    if [ -n "$override" ] && [ "$override" -gt 0 ] 2>/dev/null; then
        echo "$override"
        return
    fi
    flashchat_model_active_experts "${MODEL:-$(flashchat_default_model)}"
}

flashchat_model_sampling_profile_field() {
    local model_id="$1"
    local profile_id="$2"
    local field="$3"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -n "$model_id" ] || return 1
    [ -n "$profile_id" ] || return 1
    [ -n "$field" ] || return 1
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
profile = data.get('models', {}).get(sys.argv[2], {}).get('sampling_profiles', {}).get(sys.argv[3], {})
field = sys.argv[4]
value = profile.get(field, '')
if field == 'mtp_default_predictions' and value == '':
    value = profile.get('mtp', '')
print(value if value is not None else '')
" "$config_file" "$model_id" "$profile_id" "$field" 2>/dev/null
}

flashchat_model_sampling_profiles() {
    local model_id="$1"
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -n "$model_id" ] || return 1
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
profiles = data.get('models', {}).get(sys.argv[2], {}).get('sampling_profiles', {})
for profile_id, profile in profiles.items():
    print('\\t'.join([
        profile_id,
        str(profile.get('label', profile_id)),
        str(profile.get('description', '')),
        str(profile.get('temperature', '')),
        str(profile.get('top_p', '')),
        str(profile.get('top_k', '')),
        str(profile.get('min_p', '')),
        str(profile.get('presence_penalty', '')),
        str(profile.get('repetition_penalty', '')),
        str(profile.get('reasoning', '')),
        str(profile.get('mtp_default_predictions', profile.get('mtp', ''))),
    ]))
" "$config_file" "$model_id" 2>/dev/null
}

flashchat_model_path_for_id() {
    local repo
    repo=$(flashchat_model_repo "$1")
    [ -n "$repo" ] || return 1
    _flashchat_detect_model_path "$repo"
}

flashchat_model_runtime_dir() {
    local model_id="$1"
    local model_path="$2"
    local bits
    [ -n "$model_path" ] || return 1
    bits=$(flashchat_model_quant_bits "$model_id")
    bits="${bits:-4}"
    echo "${model_path}/flashchat/q${bits}"
}

flashchat_list_models() {
    local config_file="$FLASHCHAT_MODEL_CONFIG"
    [ -f "$config_file" ] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for model_id, model in data.get('models', {}).items():
    print('\\t'.join([
        model_id,
        str(model.get('name', model_id)),
        str(model.get('hf_repo', '')),
    ]))
" "$config_file" 2>/dev/null
}

# -----------------------------------------------------------------------------
# Detect the HuggingFace snapshot path from model repo
# -----------------------------------------------------------------------------
_flashchat_detect_model_path() {
    local repo="$1"
    local escaped_repo="${repo//\//--}"
    local hf_cache="${HUGGINGFACE_CACHE_DIR:-$FLASHCHAT_DEFAULT_HUGGINGFACE_CACHE_DIR}"
    
    # Try to find the latest snapshot
    local snapshot_dir="${hf_cache}/models--${escaped_repo}/snapshots"
    
    if [ -d "$snapshot_dir" ]; then
        # Find the latest snapshot (directory with longest name)
        local latest=$(ls -1 "$snapshot_dir" 2>/dev/null | sort | tail -1)
        if [ -n "$latest" ]; then
            echo "${snapshot_dir}/${latest}"
            return 0
        fi
    fi
    
    # Return expected path even if it doesn't exist (for setup prompts)
    echo "${snapshot_dir}/<snapshot>"
}

# -----------------------------------------------------------------------------
# Compute derived paths based on config
# -----------------------------------------------------------------------------
_flashchat_compute_paths() {
    local runtime_dir
    if [ -n "$MODEL_PATH" ]; then
        runtime_dir=$(flashchat_model_runtime_dir "$MODEL" "$MODEL_PATH")
        WEIGHTS_DIR="${WEIGHTS_DIR:-$runtime_dir}"
        EXPERTS_DIR="${EXPERTS_DIR:-${WEIGHTS_DIR}/packed_experts}"
    else
        local detected_path
        detected_path=$(_flashchat_detect_model_path "$MODEL_REPO")
        MODEL_PATH="$detected_path"
        runtime_dir=$(flashchat_model_runtime_dir "$MODEL" "$detected_path")
        WEIGHTS_DIR="${WEIGHTS_DIR:-$runtime_dir}"
        EXPERTS_DIR="${EXPERTS_DIR:-${WEIGHTS_DIR}/packed_experts}"
    fi
}

# -----------------------------------------------------------------------------
# Load configuration from file
# -----------------------------------------------------------------------------
_flashchat_source_config() {
    local config_file="$1"
    if [ -f "$config_file" ]; then
        source "$config_file"
    fi
}

# -----------------------------------------------------------------------------
# Load configuration with priority
# -----------------------------------------------------------------------------
flashchat_load_config() {
    mkdir -p "$FLASHCHAT_CONFIG_DIR"

    MODEL=""
    MODEL_REPO=""
    MAX_TOKENS=""
    SERVER_PORT=""
    SERVER_HOST=""
    SERVER_LOG_PATH=""
    SHOW_THINKING=""
    SERVER_DEBUG=""
    SERVER_HTTP_LOG=""
    SYSTEM_PROMPT_CACHE=""
    SYSTEM_PROMPT_CACHE_MAX_ENTRIES=""
    MTP=""
    COLOR_OUTPUT=""
    SAMPLING_PROFILE=""
    REASONING=""
    TEMPERATURE=""
    TOP_P=""
    TOP_K=""
    MIN_P=""
    PRESENCE_PENALTY=""
    REPETITION_PENALTY=""
    HUGGINGFACE_CACHE_DIR=""
    OFFLOAD_DIR=""
    MODEL_PATH=""
    WEIGHTS_DIR=""
    EXPERTS_DIR=""

    local override_config="${FLASHCHAT_CONFIG_FILE_OVERRIDE:-${CONFIG_FILE:-}}"

    if [ -n "$override_config" ]; then
        FLASHCHAT_CONFIG_FILE="$override_config"
        _flashchat_source_config "$override_config"
    elif [ -f "${FLASHCHAT_CONFIG_DIR}/config" ]; then
        FLASHCHAT_CONFIG_FILE="${FLASHCHAT_CONFIG_DIR}/config"
        _flashchat_source_config "${FLASHCHAT_CONFIG_DIR}/config"
    else
        FLASHCHAT_CONFIG_FILE="${FLASHCHAT_CONFIG_DIR}/config"
    fi
    
    # 3. Environment variables override
    [ -n "$FLASHCHAT_MODEL" ] && MODEL="$FLASHCHAT_MODEL"
    [ -n "$FLASHCHAT_MODEL_PATH" ] && MODEL_PATH="$FLASHCHAT_MODEL_PATH"
    [ -n "$FLASHCHAT_MAX_TOKENS" ] && MAX_TOKENS="$FLASHCHAT_MAX_TOKENS"
    [ -n "$FLASHCHAT_SERVER_PORT" ] && SERVER_PORT="$FLASHCHAT_SERVER_PORT"
    [ -n "$FLASHCHAT_SERVER_HOST" ] && SERVER_HOST="$FLASHCHAT_SERVER_HOST"
    [ -n "$FLASHCHAT_SERVER_LOG" ] && SERVER_LOG_PATH="$FLASHCHAT_SERVER_LOG"
    [ -n "$FLASHCHAT_SHOW_THINKING" ] && SHOW_THINKING="$FLASHCHAT_SHOW_THINKING"
    [ -n "$FLASHCHAT_SERVER_DEBUG" ] && SERVER_DEBUG="$FLASHCHAT_SERVER_DEBUG"
    [ -n "$FLASHCHAT_SERVER_HTTP_LOG" ] && SERVER_HTTP_LOG="$FLASHCHAT_SERVER_HTTP_LOG"
    [ -n "$FLASHCHAT_SYSTEM_PROMPT_CACHE" ] && SYSTEM_PROMPT_CACHE="$FLASHCHAT_SYSTEM_PROMPT_CACHE"
    [ -n "$FLASHCHAT_SYSTEM_PROMPT_CACHE_MAX_ENTRIES" ] && SYSTEM_PROMPT_CACHE_MAX_ENTRIES="$FLASHCHAT_SYSTEM_PROMPT_CACHE_MAX_ENTRIES"
    [ -n "$FLASHCHAT_MTP" ] && MTP="$FLASHCHAT_MTP"
    [ -n "$FLASHCHAT_MTP_BF16" ] && MTP_BF16="$FLASHCHAT_MTP_BF16"
    [ -n "$FLASHCHAT_COLOR_OUTPUT" ] && COLOR_OUTPUT="$FLASHCHAT_COLOR_OUTPUT"
    [ -n "$FLASHCHAT_SAMPLING_PROFILE" ] && SAMPLING_PROFILE="$FLASHCHAT_SAMPLING_PROFILE"
    [ -n "$FLASHCHAT_HUGGINGFACE_CACHE_DIR" ] && HUGGINGFACE_CACHE_DIR="$FLASHCHAT_HUGGINGFACE_CACHE_DIR"
    [ -n "$FLASHCHAT_OFFLOAD_DIR" ] && OFFLOAD_DIR="$FLASHCHAT_OFFLOAD_DIR"
    [ -n "$FLASHCHAT_WEIGHTS_DIR" ] && WEIGHTS_DIR="$FLASHCHAT_WEIGHTS_DIR"
    [ -n "$FLASHCHAT_EXPERTS_DIR" ] && EXPERTS_DIR="$FLASHCHAT_EXPERTS_DIR"
    
    local default_model
    default_model=$(flashchat_default_model)
    if [ -z "$MODEL" ]; then
        MODEL="$default_model"
    fi
    if ! flashchat_model_exists "$MODEL"; then
        if [ -n "$MODEL_REPO" ]; then
            local derived_id="${MODEL_REPO//\//-}"
            derived_id="${derived_id//./}"
            if flashchat_model_exists "$derived_id"; then
                MODEL="$derived_id"
                if [ -f "$FLASHCHAT_CONFIG_FILE" ]; then
                    sed -i '' "s/^MODEL=.*/MODEL=\"$derived_id\"/" "$FLASHCHAT_CONFIG_FILE" 2>/dev/null || true
                fi
            else
                echo "WARNING: model '$MODEL' is not in $FLASHCHAT_MODEL_CONFIG." >&2
                echo "  Run 'flashchat config' to select your model, or set FLASHCHAT_MODEL_REPO." >&2
                echo "  Using default model '$default_model' for now." >&2
                MODEL="$default_model"
            fi
        else
            echo "WARNING: model '$MODEL' is not in $FLASHCHAT_MODEL_CONFIG (may be an old-format ID)." >&2
            echo "  Run 'flashchat config' to select your model again." >&2
            echo "  Using default model '$default_model' for now." >&2
            MODEL="$default_model"
        fi
    fi
    local looked_up_repo
    looked_up_repo=$(_flashchat_lookup_model_repo "$MODEL")
    if [ -n "$looked_up_repo" ]; then
        MODEL_REPO="$looked_up_repo"
    fi
    local default_profile
    default_profile=$(flashchat_model_default_sampling_profile "$MODEL")
    SAMPLING_PROFILE="${SAMPLING_PROFILE:-${default_profile:-$FLASHCHAT_DEFAULT_SAMPLING_PROFILE}}"
    if [ -n "$SAMPLING_PROFILE" ] && [ "$SAMPLING_PROFILE" != "custom" ]; then
        local profile_temperature profile_top_p profile_top_k profile_min_p profile_presence_penalty profile_repetition_penalty profile_reasoning profile_mtp
        profile_temperature=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "temperature")
        profile_top_p=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "top_p")
        profile_top_k=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "top_k")
        profile_min_p=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "min_p")
        profile_presence_penalty=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "presence_penalty")
        profile_repetition_penalty=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "repetition_penalty")
        profile_reasoning=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "reasoning")
        profile_mtp=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "mtp_default_predictions")
        if [ -z "$profile_mtp" ]; then
            profile_mtp=$(flashchat_model_sampling_profile_field "$MODEL" "$SAMPLING_PROFILE" "mtp")
        fi
        [ -n "$profile_temperature" ] && TEMPERATURE="$profile_temperature"
        [ -n "$profile_top_p" ] && TOP_P="$profile_top_p"
        [ -n "$profile_top_k" ] && TOP_K="$profile_top_k"
        [ -n "$profile_min_p" ] && MIN_P="$profile_min_p"
        [ -n "$profile_presence_penalty" ] && PRESENCE_PENALTY="$profile_presence_penalty"
        [ -n "$profile_repetition_penalty" ] && REPETITION_PENALTY="$profile_repetition_penalty"
        [ -n "$profile_reasoning" ] && REASONING="$profile_reasoning"
        [ -z "$MTP" ] && [ -n "$profile_mtp" ] && MTP="$profile_mtp"
    fi
    if [ -z "$MTP" ]; then
        local server_mtp_default
        server_mtp_default=$(flashchat_server_mtp_default)
        [ -n "$server_mtp_default" ] && MTP="$server_mtp_default"
    fi
    if [ -z "$MTP" ]; then
        local model_mtp_default
        model_mtp_default=$(flashchat_model_mtp_default "$MODEL")
        [ -n "$model_mtp_default" ] && MTP="$model_mtp_default"
    fi
    MAX_TOKENS="${MAX_TOKENS:-$FLASHCHAT_DEFAULT_MAX_TOKENS}"
    SERVER_PORT="${SERVER_PORT:-$FLASHCHAT_DEFAULT_SERVER_PORT}"
    SERVER_HOST="${SERVER_HOST:-$FLASHCHAT_DEFAULT_SERVER_HOST}"
    SERVER_LOG_PATH="${SERVER_LOG_PATH:-$FLASHCHAT_DEFAULT_SERVER_LOG_PATH}"
    SHOW_THINKING="${SHOW_THINKING:-$FLASHCHAT_DEFAULT_SHOW_THINKING}"
    SERVER_DEBUG="${SERVER_DEBUG:-$FLASHCHAT_DEFAULT_SERVER_DEBUG}"
    SERVER_HTTP_LOG="${SERVER_HTTP_LOG:-$FLASHCHAT_DEFAULT_SERVER_HTTP_LOG}"
    SYSTEM_PROMPT_CACHE="${SYSTEM_PROMPT_CACHE:-$FLASHCHAT_DEFAULT_SYSTEM_PROMPT_CACHE}"
    SYSTEM_PROMPT_CACHE_MAX_ENTRIES="${SYSTEM_PROMPT_CACHE_MAX_ENTRIES:-$FLASHCHAT_DEFAULT_SYSTEM_PROMPT_CACHE_MAX_ENTRIES}"
    MTP="${MTP:-$FLASHCHAT_DEFAULT_MTP}"
    MTP_BF16="${MTP_BF16:-$FLASHCHAT_DEFAULT_MTP_BF16}"
    COLOR_OUTPUT="${COLOR_OUTPUT:-$FLASHCHAT_DEFAULT_COLOR_OUTPUT}"
    REASONING="${REASONING:-$FLASHCHAT_DEFAULT_REASONING}"
    TEMPERATURE="${TEMPERATURE:-$FLASHCHAT_DEFAULT_TEMPERATURE}"
    TOP_P="${TOP_P:-$FLASHCHAT_DEFAULT_TOP_P}"
    TOP_K="${TOP_K:-$FLASHCHAT_DEFAULT_TOP_K}"
    MIN_P="${MIN_P:-$FLASHCHAT_DEFAULT_MIN_P}"
    PRESENCE_PENALTY="${PRESENCE_PENALTY:-$FLASHCHAT_DEFAULT_PRESENCE_PENALTY}"
    REPETITION_PENALTY="${REPETITION_PENALTY:-$FLASHCHAT_DEFAULT_REPETITION_PENALTY}"
    HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-$FLASHCHAT_DEFAULT_HUGGINGFACE_CACHE_DIR}"
    OFFLOAD_DIR="${OFFLOAD_DIR:-$FLASHCHAT_DEFAULT_OFFLOAD_DIR}"

    [ -n "$FLASHCHAT_REASONING" ] && REASONING="$FLASHCHAT_REASONING"
    [ -n "$FLASHCHAT_TEMPERATURE" ] && TEMPERATURE="$FLASHCHAT_TEMPERATURE"
    [ -n "$FLASHCHAT_TOP_P" ] && TOP_P="$FLASHCHAT_TOP_P"
    [ -n "$FLASHCHAT_TOP_K" ] && TOP_K="$FLASHCHAT_TOP_K"
    [ -n "$FLASHCHAT_MIN_P" ] && MIN_P="$FLASHCHAT_MIN_P"
    [ -n "$FLASHCHAT_PRESENCE_PENALTY" ] && PRESENCE_PENALTY="$FLASHCHAT_PRESENCE_PENALTY"
    [ -n "$FLASHCHAT_REPETITION_PENALTY" ] && REPETITION_PENALTY="$FLASHCHAT_REPETITION_PENALTY"
    
    # Compute derived paths
    _flashchat_compute_paths
}

# -----------------------------------------------------------------------------
# Get a config value
# -----------------------------------------------------------------------------
flashchat_get() {
    local key="$1"
    case "$key" in
        MODEL) echo "$MODEL" ;;
        MODEL_REPO) echo "$MODEL_REPO" ;;
        MAX_TOKENS) echo "$MAX_TOKENS" ;;
        SERVER_PORT) echo "$SERVER_PORT" ;;
        SERVER_HOST) echo "$SERVER_HOST" ;;
        SERVER_LOG_PATH) echo "$SERVER_LOG_PATH" ;;
        SHOW_THINKING) echo "$SHOW_THINKING" ;;
        SERVER_DEBUG) echo "$SERVER_DEBUG" ;;
        SERVER_HTTP_LOG) echo "$SERVER_HTTP_LOG" ;;
        SYSTEM_PROMPT_CACHE) echo "$SYSTEM_PROMPT_CACHE" ;;
        SYSTEM_PROMPT_CACHE_MAX_ENTRIES) echo "$SYSTEM_PROMPT_CACHE_MAX_ENTRIES" ;;
        MTP) echo "$MTP" ;;
        MTP_BF16) echo "$MTP_BF16" ;;
        COLOR_OUTPUT) echo "$COLOR_OUTPUT" ;;
        SAMPLING_PROFILE) echo "$SAMPLING_PROFILE" ;;
        REASONING) echo "$REASONING" ;;
        TEMPERATURE) echo "$TEMPERATURE" ;;
        TOP_P) echo "$TOP_P" ;;
        TOP_K) echo "$TOP_K" ;;
        MIN_P) echo "$MIN_P" ;;
        PRESENCE_PENALTY) echo "$PRESENCE_PENALTY" ;;
        REPETITION_PENALTY) echo "$REPETITION_PENALTY" ;;
        HUGGINGFACE_CACHE_DIR) echo "$HUGGINGFACE_CACHE_DIR" ;;
        OFFLOAD_DIR) echo "$OFFLOAD_DIR" ;;
        MODEL_PATH) echo "$MODEL_PATH" ;;
        WEIGHTS_DIR) echo "$WEIGHTS_DIR" ;;
        EXPERTS_DIR) echo "$EXPERTS_DIR" ;;
        CONFIG_FILE) echo "$FLASHCHAT_CONFIG_FILE" ;;
        CONFIG_DIR) echo "$FLASHCHAT_CONFIG_DIR" ;;
        MODEL_CONFIG) echo "$FLASHCHAT_MODEL_CONFIG" ;;
        *) echo "" ;;
    esac
}

# -----------------------------------------------------------------------------
# Create default config file
# -----------------------------------------------------------------------------
flashchat_create_default_config() {
    local config_file="${FLASHCHAT_CONFIG_FILE:-${FLASHCHAT_CONFIG_DIR}/config}"
    mkdir -p "$(dirname "$config_file")"
    cat > "$config_file" << EOF
# Flashchat Configuration
# Generated on $(date)

# Model Settings
MODEL="${MODEL:-$(flashchat_default_model)}"

# Storage Settings
HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-$FLASHCHAT_DEFAULT_HUGGINGFACE_CACHE_DIR}"
OFFLOAD_DIR="${OFFLOAD_DIR:-$FLASHCHAT_DEFAULT_OFFLOAD_DIR}"

# Generation Defaults
MAX_TOKENS="${MAX_TOKENS:-$FLASHCHAT_DEFAULT_MAX_TOKENS}"
SAMPLING_PROFILE="${SAMPLING_PROFILE:-${FLASHCHAT_DEFAULT_SAMPLING_PROFILE:-$(flashchat_model_default_sampling_profile "${MODEL:-$(flashchat_default_model)}")}}"
REASONING="${REASONING:-$FLASHCHAT_DEFAULT_REASONING}"
TEMPERATURE="${TEMPERATURE:-$FLASHCHAT_DEFAULT_TEMPERATURE}"
TOP_P="${TOP_P:-$FLASHCHAT_DEFAULT_TOP_P}"
TOP_K="${TOP_K:-$FLASHCHAT_DEFAULT_TOP_K}"
MIN_P="${MIN_P:-$FLASHCHAT_DEFAULT_MIN_P}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-$FLASHCHAT_DEFAULT_PRESENCE_PENALTY}"
REPETITION_PENALTY="${REPETITION_PENALTY:-$FLASHCHAT_DEFAULT_REPETITION_PENALTY}"

# Server Settings
SERVER_PORT="${SERVER_PORT:-$FLASHCHAT_DEFAULT_SERVER_PORT}"
SERVER_HOST="${SERVER_HOST:-$FLASHCHAT_DEFAULT_SERVER_HOST}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$FLASHCHAT_DEFAULT_SERVER_LOG_PATH}"
SERVER_DEBUG="${SERVER_DEBUG:-$FLASHCHAT_DEFAULT_SERVER_DEBUG}"
SERVER_HTTP_LOG="${SERVER_HTTP_LOG:-$FLASHCHAT_DEFAULT_SERVER_HTTP_LOG}"
SYSTEM_PROMPT_CACHE="${SYSTEM_PROMPT_CACHE:-$FLASHCHAT_DEFAULT_SYSTEM_PROMPT_CACHE}"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="${SYSTEM_PROMPT_CACHE_MAX_ENTRIES:-$FLASHCHAT_DEFAULT_SYSTEM_PROMPT_CACHE_MAX_ENTRIES}"
MTP="${MTP:-$FLASHCHAT_DEFAULT_MTP}"
MTP_BF16="${MTP_BF16:-$FLASHCHAT_DEFAULT_MTP_BF16}"

# UI Settings
SHOW_THINKING="${SHOW_THINKING:-$FLASHCHAT_DEFAULT_SHOW_THINKING}"
COLOR_OUTPUT="${COLOR_OUTPUT:-$FLASHCHAT_DEFAULT_COLOR_OUTPUT}"
EOF
    FLASHCHAT_CONFIG_FILE="$config_file"
}

# -----------------------------------------------------------------------------
# Check if config exists
# -----------------------------------------------------------------------------
flashchat_has_config() {
    local override_config="${FLASHCHAT_CONFIG_FILE_OVERRIDE:-${CONFIG_FILE:-}}"
    if [ -n "$override_config" ]; then
        [ -f "$override_config" ]
    else
        [ -f "${FLASHCHAT_CONFIG_DIR}/config" ]
    fi
}

# -----------------------------------------------------------------------------
# Get PID file path
# -----------------------------------------------------------------------------
flashchat_get_pid_file() {
    echo "${FLASHCHAT_CONFIG_DIR}/server.pid"
}

flashchat_get_server_signature_file() {
    echo "${FLASHCHAT_CONFIG_DIR}/server.signature"
}

# -----------------------------------------------------------------------------
# Get Flashchat user/app state paths
# -----------------------------------------------------------------------------
flashchat_get_data_dir() {
    echo "$FLASHCHAT_CONFIG_DIR"
}

flashchat_get_sessions_dir() {
    echo "${FLASHCHAT_CONFIG_DIR}/sessions"
}

flashchat_get_history_file() {
    echo "${FLASHCHAT_CONFIG_DIR}/history"
}

flashchat_get_system_prompt_file() {
    echo "${FLASHCHAT_CONFIG_DIR}/system.md"
}

# Export functions for use in subshells
export -f flashchat_load_config
export -f flashchat_get
export -f flashchat_create_default_config
export -f flashchat_has_config
export -f flashchat_get_data_dir
export -f flashchat_get_pid_file
export -f flashchat_get_server_signature_file
export -f flashchat_get_sessions_dir
export -f flashchat_get_history_file
export -f flashchat_get_system_prompt_file
export -f flashchat_default_model
export -f flashchat_model_exists
export -f flashchat_model_field
export -f flashchat_model_name
export -f flashchat_model_repo
export -f flashchat_model_layers
export -f flashchat_model_default_sampling_profile
export -f flashchat_model_mtp_default
export -f flashchat_server_default_field
export -f flashchat_server_mtp_default
export -f flashchat_model_quant_bits
export -f flashchat_model_expert_pack_bytes
export -f flashchat_model_sampling_profile_field
export -f flashchat_model_sampling_profiles
export -f flashchat_model_path_for_id
export -f flashchat_list_models
