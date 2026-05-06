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
FLASHCHAT_DEFAULT_MODEL="qwen3.6-35B-A3B"
FLASHCHAT_DEFAULT_QUANTIZATION="4bit"
FLASHCHAT_DEFAULT_MAX_TOKENS="8192"
FLASHCHAT_DEFAULT_SERVER_PORT="8000"
FLASHCHAT_DEFAULT_SERVER_HOST="127.0.0.1"
FLASHCHAT_DEFAULT_SERVER_LOG_PATH="${FLASHCHAT_CONFIG_DIR}/logs/server.log"
FLASHCHAT_DEFAULT_SHOW_THINKING="0"
FLASHCHAT_DEFAULT_SERVER_DEBUG="0"
FLASHCHAT_DEFAULT_SERVER_HTTP_LOG="0"
FLASHCHAT_DEFAULT_COLOR_OUTPUT="1"
FLASHCHAT_DEFAULT_TEMPERATURE="0.7"
FLASHCHAT_DEFAULT_TOP_P="0.9"

# Config values (set after loading)
MODEL=""
MODEL_REPO=""
QUANTIZATION=""
MAX_TOKENS=""
SERVER_PORT=""
SERVER_HOST=""
SERVER_LOG_PATH=""
SHOW_THINKING=""
SERVER_DEBUG=""
SERVER_HTTP_LOG=""
COLOR_OUTPUT=""
TEMPERATURE=""
TOP_P=""

# Derived paths (computed after config load)
MODEL_PATH=""
WEIGHTS_DIR=""
EXPERTS_DIR=""
EXPERTS_2BIT_DIR=""

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

flashchat_model_path_for_id() {
    local repo
    repo=$(flashchat_model_repo "$1")
    [ -n "$repo" ] || return 1
    _flashchat_detect_model_path "$repo"
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
    local hf_cache="${HOME}/.cache/huggingface/hub"
    
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
    if [ -n "$MODEL_PATH" ]; then
        WEIGHTS_DIR="${WEIGHTS_DIR:-${MODEL_PATH}/flashchat}"
        EXPERTS_DIR="${MODEL_PATH}/flashchat/packed_experts"
        EXPERTS_2BIT_DIR="${MODEL_PATH}/flashchat/packed_experts_2bit"
    else
        local detected_path
        detected_path=$(_flashchat_detect_model_path "$MODEL_REPO")
        MODEL_PATH="$detected_path"
        WEIGHTS_DIR="${WEIGHTS_DIR:-${detected_path}/flashchat}"
        EXPERTS_DIR="${detected_path}/flashchat/packed_experts"
        EXPERTS_2BIT_DIR="${detected_path}/flashchat/packed_experts_2bit"
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
    QUANTIZATION=""
    MAX_TOKENS=""
    SERVER_PORT=""
    SERVER_HOST=""
    SERVER_LOG_PATH=""
    SHOW_THINKING=""
    SERVER_DEBUG=""
    SERVER_HTTP_LOG=""
    COLOR_OUTPUT=""
    TEMPERATURE=""
    TOP_P=""
    MODEL_PATH=""
    WEIGHTS_DIR=""
    EXPERTS_DIR=""
    EXPERTS_2BIT_DIR=""

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
    [ -n "$FLASHCHAT_QUANTIZATION" ] && QUANTIZATION="$FLASHCHAT_QUANTIZATION"
    [ -n "$FLASHCHAT_MAX_TOKENS" ] && MAX_TOKENS="$FLASHCHAT_MAX_TOKENS"
    [ -n "$FLASHCHAT_SERVER_PORT" ] && SERVER_PORT="$FLASHCHAT_SERVER_PORT"
    [ -n "$FLASHCHAT_SERVER_HOST" ] && SERVER_HOST="$FLASHCHAT_SERVER_HOST"
    [ -n "$FLASHCHAT_SERVER_LOG" ] && SERVER_LOG_PATH="$FLASHCHAT_SERVER_LOG"
    [ -n "$FLASHCHAT_SHOW_THINKING" ] && SHOW_THINKING="$FLASHCHAT_SHOW_THINKING"
    [ -n "$FLASHCHAT_SERVER_DEBUG" ] && SERVER_DEBUG="$FLASHCHAT_SERVER_DEBUG"
    [ -n "$FLASHCHAT_SERVER_HTTP_LOG" ] && SERVER_HTTP_LOG="$FLASHCHAT_SERVER_HTTP_LOG"
    [ -n "$FLASHCHAT_COLOR_OUTPUT" ] && COLOR_OUTPUT="$FLASHCHAT_COLOR_OUTPUT"
    [ -n "$FLASHCHAT_TEMPERATURE" ] && TEMPERATURE="$FLASHCHAT_TEMPERATURE"
    [ -n "$FLASHCHAT_TOP_P" ] && TOP_P="$FLASHCHAT_TOP_P"
    [ -n "$FLASHCHAT_WEIGHTS_DIR" ] && WEIGHTS_DIR="$FLASHCHAT_WEIGHTS_DIR"
    [ -n "$FLASHCHAT_EXPERTS_DIR" ] && EXPERTS_DIR="$FLASHCHAT_EXPERTS_DIR"
    
    local default_model
    default_model=$(flashchat_default_model)
    if [ -z "$MODEL" ]; then
        MODEL="$default_model"
    fi
    if ! flashchat_model_exists "$MODEL"; then
        echo "WARNING: configured model '$MODEL' is not in $FLASHCHAT_MODEL_CONFIG; using $default_model" >&2
        MODEL="$default_model"
    fi
    local looked_up_repo
    looked_up_repo=$(_flashchat_lookup_model_repo "$MODEL")
    if [ -n "$looked_up_repo" ]; then
        MODEL_REPO="$looked_up_repo"
    fi
    QUANTIZATION="${QUANTIZATION:-$FLASHCHAT_DEFAULT_QUANTIZATION}"
    if [ "$QUANTIZATION" = "2bit" ]; then
        QUANTIZATION="4bit"
    fi
    MAX_TOKENS="${MAX_TOKENS:-$FLASHCHAT_DEFAULT_MAX_TOKENS}"
    SERVER_PORT="${SERVER_PORT:-$FLASHCHAT_DEFAULT_SERVER_PORT}"
    SERVER_HOST="${SERVER_HOST:-$FLASHCHAT_DEFAULT_SERVER_HOST}"
    SERVER_LOG_PATH="${SERVER_LOG_PATH:-$FLASHCHAT_DEFAULT_SERVER_LOG_PATH}"
    SHOW_THINKING="${SHOW_THINKING:-$FLASHCHAT_DEFAULT_SHOW_THINKING}"
    SERVER_DEBUG="${SERVER_DEBUG:-$FLASHCHAT_DEFAULT_SERVER_DEBUG}"
    SERVER_HTTP_LOG="${SERVER_HTTP_LOG:-$FLASHCHAT_DEFAULT_SERVER_HTTP_LOG}"
    COLOR_OUTPUT="${COLOR_OUTPUT:-$FLASHCHAT_DEFAULT_COLOR_OUTPUT}"
    TEMPERATURE="${TEMPERATURE:-$FLASHCHAT_DEFAULT_TEMPERATURE}"
    TOP_P="${TOP_P:-$FLASHCHAT_DEFAULT_TOP_P}"
    
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
        QUANTIZATION) echo "$QUANTIZATION" ;;
        MAX_TOKENS) echo "$MAX_TOKENS" ;;
        SERVER_PORT) echo "$SERVER_PORT" ;;
        SERVER_HOST) echo "$SERVER_HOST" ;;
        SERVER_LOG_PATH) echo "$SERVER_LOG_PATH" ;;
        SHOW_THINKING) echo "$SHOW_THINKING" ;;
        SERVER_DEBUG) echo "$SERVER_DEBUG" ;;
        SERVER_HTTP_LOG) echo "$SERVER_HTTP_LOG" ;;
        COLOR_OUTPUT) echo "$COLOR_OUTPUT" ;;
        TEMPERATURE) echo "$TEMPERATURE" ;;
        TOP_P) echo "$TOP_P" ;;
        MODEL_PATH) echo "$MODEL_PATH" ;;
        WEIGHTS_DIR) echo "$WEIGHTS_DIR" ;;
        EXPERTS_DIR) echo "$EXPERTS_DIR" ;;
        EXPERTS_2BIT_DIR) echo "$EXPERTS_2BIT_DIR" ;;
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

# Quantization: 4bit. 2bit is deprecated.
QUANTIZATION="${QUANTIZATION:-$FLASHCHAT_DEFAULT_QUANTIZATION}"

# Generation Defaults
MAX_TOKENS="${MAX_TOKENS:-$FLASHCHAT_DEFAULT_MAX_TOKENS}"
TEMPERATURE="${TEMPERATURE:-$FLASHCHAT_DEFAULT_TEMPERATURE}"
TOP_P="${TOP_P:-$FLASHCHAT_DEFAULT_TOP_P}"

# Server Settings
SERVER_PORT="${SERVER_PORT:-$FLASHCHAT_DEFAULT_SERVER_PORT}"
SERVER_HOST="${SERVER_HOST:-$FLASHCHAT_DEFAULT_SERVER_HOST}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$FLASHCHAT_DEFAULT_SERVER_LOG_PATH}"
SERVER_DEBUG="${SERVER_DEBUG:-$FLASHCHAT_DEFAULT_SERVER_DEBUG}"
SERVER_HTTP_LOG="${SERVER_HTTP_LOG:-$FLASHCHAT_DEFAULT_SERVER_HTTP_LOG}"

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

# -----------------------------------------------------------------------------
# Get sessions directory
# Note: Uses ~/.flashchat/sessions for compatibility with existing chat.m
# -----------------------------------------------------------------------------
flashchat_get_sessions_dir() {
    echo "${HOME}/.flashchat/sessions"
}

# Export functions for use in subshells
export -f flashchat_load_config
export -f flashchat_get
export -f flashchat_create_default_config
export -f flashchat_has_config
export -f flashchat_get_pid_file
export -f flashchat_get_sessions_dir
export -f flashchat_default_model
export -f flashchat_model_exists
export -f flashchat_model_field
export -f flashchat_model_name
export -f flashchat_model_repo
export -f flashchat_model_layers
export -f flashchat_model_path_for_id
export -f flashchat_list_models
