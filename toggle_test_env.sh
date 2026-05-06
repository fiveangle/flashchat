#!/bin/bash
# Toggle between live environment and fresh test environment
# Safety: verifies data integrity before overwriting

set -e

# Parse arguments
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--force]"
            echo "  --force  Skip integrity checks and force toggle"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

FLASHCHAT_CONFIG_DIR="${HOME}/.config/flashchat"
FLASHCHAT_SESSIONS_DIR="${HOME}/.config/flashchat/sessions"
HF_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit"
SUFFIX=".live"

# Verify model cache has required files
verify_model_cache() {
    local cache_dir="$1"
    if [ -d "$cache_dir/snapshots" ]; then
        local snapshot
        snapshot=$(ls -1 "$cache_dir/snapshots/" 2>/dev/null | head -1)
        if [ -n "$snapshot" ] && [ -f "$cache_dir/snapshots/$snapshot/model.safetensors.index.json" ]; then
            return 0
        fi
    fi
    return 1
}

# Get cache size in human readable format
get_cache_size() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1 || echo "?"
    else
        echo "none"
    fi
}

echo "=== Flashchat Environment Toggle ==="
echo ""

# Check current state
if [ -d "${HF_CACHE_DIR}${SUFFIX}" ]; then
    # Currently in test mode - restore live
    
    # Safety check: verify backup is valid before restoring
    echo "Checking backup integrity..."
    if [ $FORCE -eq 1 ]; then
        echo "  --force flag set, skipping integrity check"
    elif verify_model_cache "${HF_CACHE_DIR}${SUFFIX}"; then
        local backup_size
        backup_size=$(get_cache_size "${HF_CACHE_DIR}${SUFFIX}")
        echo "  Backup size: $backup_size (valid)"
    else
        if [ $FORCE -eq 1 ]; then
            echo "  --force flag set, proceeding anyway"
        else
            echo "ERROR: Backup is invalid or incomplete!"
            echo "  Backup location: ${HF_CACHE_DIR}${SUFFIX}"
            echo ""
            echo "Manual recovery needed. Your live data may be lost."
            echo "  Live data was moved to: ${HF_CACHE_DIR}${SUFFIX}"
            exit 1
        fi
    fi
    
    # Check if current is also valid (would be overwritten)
    if [ -d "$HF_CACHE_DIR" ] && verify_model_cache "$HF_CACHE_DIR"; then
        local current_size
        current_size=$(get_cache_size "$HF_CACHE_DIR")
        echo "  Current size: $current_size (will be replaced)"
        echo ""
        echo "WARNING: Replacing current model with backup."
        echo "  Current: $current_size"
        if [ -d "${HF_CACHE_DIR}${SUFFIX}" ]; then
            local backup_size
            backup_size=$(get_cache_size "${HF_CACHE_DIR}${SUFFIX}")
            echo "  Backup:  $backup_size"
        fi
        echo ""
        read -p "Continue? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        rm -rf "$HF_CACHE_DIR"
    fi
    
    # Restore HuggingFace cache
    echo "Restoring live environment..."
    mv "${HF_CACHE_DIR}${SUFFIX}" "$HF_CACHE_DIR"
    
    # Restore config
    if [ -f "${FLASHCHAT_CONFIG_DIR}/config${SUFFIX}" ]; then
        mv "${FLASHCHAT_CONFIG_DIR}/config${SUFFIX}" "${FLASHCHAT_CONFIG_DIR}/config"
    fi
    
    # Restore sessions
    if [ -d "${FLASHCHAT_SESSIONS_DIR}${SUFFIX}" ]; then
        mv "${FLASHCHAT_SESSIONS_DIR}${SUFFIX}" "$FLASHCHAT_SESSIONS_DIR"
    fi
    
    echo "Live environment restored."
    
else
    # Currently in live mode - switch to test
    
    # Safety check: verify current model is valid before backing up
    echo "Checking live model integrity..."
    if [ $FORCE -eq 1 ]; then
        echo "  --force flag set, skipping integrity check"
    elif [ -d "$HF_CACHE_DIR" ] && verify_model_cache "$HF_CACHE_DIR"; then
        local live_size
        live_size=$(get_cache_size "$HF_CACHE_DIR")
        echo "  Live model size: $live_size (valid)"
    else
        echo "ERROR: Live model is invalid or incomplete!"
        echo "  Location: $HF_CACHE_DIR"
        echo ""
        echo "Cannot toggle - live data is already corrupted."
        echo "Use --force to override (data may be lost)."
        exit 1
    fi
    
    echo "Switching to fresh test environment..."
    
    # Backup HuggingFace cache
    if [ -d "$HF_CACHE_DIR" ]; then
        mv "$HF_CACHE_DIR" "${HF_CACHE_DIR}${SUFFIX}"
    fi
    
    # Backup config
    if [ -f "${FLASHCHAT_CONFIG_DIR}/config" ]; then
        mkdir -p "$FLASHCHAT_CONFIG_DIR"
        mv "${FLASHCHAT_CONFIG_DIR}/config" "${FLASHCHAT_CONFIG_DIR}/config${SUFFIX}"
    fi
    
    # Backup sessions
    if [ -d "$FLASHCHAT_SESSIONS_DIR" ]; then
        mv "$FLASHCHAT_SESSIONS_DIR" "${FLASHCHAT_SESSIONS_DIR}${SUFFIX}"
    fi
    
    echo "Fresh test environment ready."
    echo ""
    echo "To run flashchat fresh: ./flashchat"
    echo "To restore live: ./toggle_test_env.sh"
fi
