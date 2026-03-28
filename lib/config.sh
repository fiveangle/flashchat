#!/bin/bash
# config.sh — Flash-MoE Configuration Loader
#
# Loads configuration with the following priority (highest to lowest):
#   1. ./flashmoe.config (project-local)
#   2. ~/.config/flash-moe/config (user)
#   3. Environment variables (FLASHMOE_*)
#   4. Hardcoded defaults
#
# Usage:
#   source lib/config.sh          # Load config
#   flashmoe_load_config          # Initialize (creates default if missing)
#   flashmoe_get "KEY"           # Get a config value

set -e

FLASHMOE_CONFIG_DIR="${HOME}/.config/flash-moe"
FLASHMOE_CONFIG_FILE=""
FLASHMOE_PROJECT_CONFIG="./flashmoe.config"

# Default configuration values
FLASHMOE_DEFAULT_MODEL_REPO="mlx-community/Qwen3.5-397B-A17B-4bit"
FLASHMOE_DEFAULT_QUANTIZATION="4bit"
FLASHMOE_DEFAULT_MAX_TOKENS="8192"
FLASHMOE_DEFAULT_SERVER_PORT="8000"
FLASHMOE_DEFAULT_SERVER_HOST="127.0.0.1"
FLASHMOE_DEFAULT_SHOW_THINKING="0"
FLASHMOE_DEFAULT_COLOR_OUTPUT="1"
FLASHMOE_DEFAULT_TEMPERATURE="0.7"
FLASHMOE_DEFAULT_TOP_P="0.9"

# Config values (set after loading)
MODEL_REPO=""
QUANTIZATION=""
MAX_TOKENS=""
SERVER_PORT=""
SERVER_HOST=""
SHOW_THINKING=""
COLOR_OUTPUT=""
TEMPERATURE=""
TOP_P=""

# Derived paths (computed after config load)
MODEL_PATH=""
WEIGHTS_DIR=""
EXPERTS_DIR=""
EXPERTS_2BIT_DIR=""

# -----------------------------------------------------------------------------
# Detect the HuggingFace snapshot path from model repo
# -----------------------------------------------------------------------------
_flashmoe_detect_model_path() {
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
_flashmoe_compute_paths() {
    if [ -n "$MODEL_PATH" ]; then
        WEIGHTS_DIR="${WEIGHTS_DIR:-./metal_infer}"
        EXPERTS_DIR="${MODEL_PATH}/packed_experts"
        EXPERTS_2BIT_DIR="${MODEL_PATH}/packed_experts_2bit"
    else
        local detected_path
        detected_path=$(_flashmoe_detect_model_path "$MODEL_REPO")
        MODEL_PATH="$detected_path"
        WEIGHTS_DIR="${WEIGHTS_DIR:-./metal_infer}"
        EXPERTS_DIR="${detected_path}/packed_experts"
        EXPERTS_2BIT_DIR="${detected_path}/packed_experts_2bit"
    fi
}

# -----------------------------------------------------------------------------
# Load configuration from file
# -----------------------------------------------------------------------------
_flashmoe_source_config() {
    local config_file="$1"
    if [ -f "$config_file" ]; then
        source "$config_file"
    fi
}

# -----------------------------------------------------------------------------
# Load configuration with priority
# -----------------------------------------------------------------------------
flashmoe_load_config() {
    # Ensure config directory exists
    mkdir -p "$FLASHMOE_CONFIG_DIR"
    
    # 1. Project-local config (highest priority)
    if [ -f "$FLASHMOE_PROJECT_CONFIG" ]; then
        FLASHMOE_CONFIG_FILE="$FLASHMOE_PROJECT_CONFIG"
        _flashmoe_source_config "$FLASHMOE_PROJECT_CONFIG"
    # 2. User config
    elif [ -f "${FLASHMOE_CONFIG_DIR}/config" ]; then
        FLASHMOE_CONFIG_FILE="${FLASHMOE_CONFIG_DIR}/config"
        _flashmoe_source_config "${FLASHMOE_CONFIG_DIR}/config"
    # 3. No config file - use default path
    else
        FLASHMOE_CONFIG_FILE="${FLASHMOE_CONFIG_DIR}/config"
    fi
    
    # 3. Environment variables override
    [ -n "$FLASHMOE_MODEL_REPO" ] && MODEL_REPO="$FLASHMOE_MODEL_REPO"
    [ -n "$FLASHMOE_MODEL_PATH" ] && MODEL_PATH="$FLASHMOE_MODEL_PATH"
    [ -n "$FLASHMOE_QUANTIZATION" ] && QUANTIZATION="$FLASHMOE_QUANTIZATION"
    [ -n "$FLASHMOE_MAX_TOKENS" ] && MAX_TOKENS="$FLASHMOE_MAX_TOKENS"
    [ -n "$FLASHMOE_SERVER_PORT" ] && SERVER_PORT="$FLASHMOE_SERVER_PORT"
    [ -n "$FLASHMOE_SERVER_HOST" ] && SERVER_HOST="$FLASHMOE_SERVER_HOST"
    [ -n "$FLASHMOE_SHOW_THINKING" ] && SHOW_THINKING="$FLASHMOE_SHOW_THINKING"
    [ -n "$FLASHMOE_COLOR_OUTPUT" ] && COLOR_OUTPUT="$FLASHMOE_COLOR_OUTPUT"
    [ -n "$FLASHMOE_TEMPERATURE" ] && TEMPERATURE="$FLASHMOE_TEMPERATURE"
    [ -n "$FLASHMOE_TOP_P" ] && TOP_P="$FLASHMOE_TOP_P"
    [ -n "$FLASHMOE_WEIGHTS_DIR" ] && WEIGHTS_DIR="$FLASHMOE_WEIGHTS_DIR"
    [ -n "$FLASHMOE_EXPERTS_DIR" ] && EXPERTS_DIR="$FLASHMOE_EXPERTS_DIR"
    
    # 4. Apply defaults for any missing values
    MODEL_REPO="${MODEL_REPO:-$FLASHMOE_DEFAULT_MODEL_REPO}"
    QUANTIZATION="${QUANTIZATION:-$FLASHMOE_DEFAULT_QUANTIZATION}"
    MAX_TOKENS="${MAX_TOKENS:-$FLASHMOE_DEFAULT_MAX_TOKENS}"
    SERVER_PORT="${SERVER_PORT:-$FLASHMOE_DEFAULT_SERVER_PORT}"
    SERVER_HOST="${SERVER_HOST:-$FLASHMOE_DEFAULT_SERVER_HOST}"
    SHOW_THINKING="${SHOW_THINKING:-$FLASHMOE_DEFAULT_SHOW_THINKING}"
    COLOR_OUTPUT="${COLOR_OUTPUT:-$FLASHMOE_DEFAULT_COLOR_OUTPUT}"
    TEMPERATURE="${TEMPERATURE:-$FLASHMOE_DEFAULT_TEMPERATURE}"
    TOP_P="${TOP_P:-$FLASHMOE_DEFAULT_TOP_P}"
    
    # Compute derived paths
    _flashmoe_compute_paths
}

# -----------------------------------------------------------------------------
# Get a config value
# -----------------------------------------------------------------------------
flashmoe_get() {
    local key="$1"
    case "$key" in
        MODEL_REPO) echo "$MODEL_REPO" ;;
        QUANTIZATION) echo "$QUANTIZATION" ;;
        MAX_TOKENS) echo "$MAX_TOKENS" ;;
        SERVER_PORT) echo "$SERVER_PORT" ;;
        SERVER_HOST) echo "$SERVER_HOST" ;;
        SHOW_THINKING) echo "$SHOW_THINKING" ;;
        COLOR_OUTPUT) echo "$COLOR_OUTPUT" ;;
        TEMPERATURE) echo "$TEMPERATURE" ;;
        TOP_P) echo "$TOP_P" ;;
        MODEL_PATH) echo "$MODEL_PATH" ;;
        WEIGHTS_DIR) echo "$WEIGHTS_DIR" ;;
        EXPERTS_DIR) echo "$EXPERTS_DIR" ;;
        EXPERTS_2BIT_DIR) echo "$EXPERTS_2BIT_DIR" ;;
        CONFIG_FILE) echo "$FLASHMOE_CONFIG_FILE" ;;
        CONFIG_DIR) echo "$FLASHMOE_CONFIG_DIR" ;;
        *) echo "" ;;
    esac
}

# -----------------------------------------------------------------------------
# Create default config file
# -----------------------------------------------------------------------------
flashmoe_create_default_config() {
    mkdir -p "$FLASHMOE_CONFIG_DIR"
    cat > "${FLASHMOE_CONFIG_DIR}/config" << EOF
# Flash-MoE Configuration
# Generated on $(date)

# Model Settings
MODEL_REPO="${FLASHMOE_DEFAULT_MODEL_REPO}"

# Quantization: 4bit or 2bit
QUANTIZATION="${FLASHMOE_DEFAULT_QUANTIZATION}"

# Generation Defaults
MAX_TOKENS="${FLASHMOE_DEFAULT_MAX_TOKENS}"
TEMPERATURE="${FLASHMOE_DEFAULT_TEMPERATURE}"
TOP_P="${FLASHMOE_DEFAULT_TOP_P}"

# Server Settings
SERVER_PORT="${FLASHMOE_DEFAULT_SERVER_PORT}"
SERVER_HOST="${FLASHMOE_DEFAULT_SERVER_HOST}"

# UI Settings
SHOW_THINKING="${FLASHMOE_DEFAULT_SHOW_THINKING}"
COLOR_OUTPUT="${FLASHMOE_DEFAULT_COLOR_OUTPUT}"
EOF
    FLASHMOE_CONFIG_FILE="${FLASHMOE_CONFIG_DIR}/config"
}

# -----------------------------------------------------------------------------
# Check if config exists
# -----------------------------------------------------------------------------
flashmoe_has_config() {
    [ -f "$FLASHMOE_PROJECT_CONFIG" ] || [ -f "${FLASHMOE_CONFIG_DIR}/config" ]
}

# -----------------------------------------------------------------------------
# Get PID file path
# -----------------------------------------------------------------------------
flashmoe_get_pid_file() {
    echo "${FLASHMOE_CONFIG_DIR}/server.pid"
}

# -----------------------------------------------------------------------------
# Get sessions directory
# Note: Uses ~/.flash-moe/sessions for compatibility with existing chat.m
# -----------------------------------------------------------------------------
flashmoe_get_sessions_dir() {
    echo "${HOME}/.flash-moe/sessions"
}

# Export functions for use in subshells
export -f flashmoe_load_config
export -f flashmoe_get
export -f flashmoe_create_default_config
export -f flashmoe_has_config
export -f flashmoe_get_pid_file
export -f flashmoe_get_sessions_dir
