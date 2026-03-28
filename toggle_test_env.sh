#!/bin/bash
# Toggle between live environment and fresh test environment
# Safety: verifies data integrity before overwriting

set -e

FLASHMOE_CONFIG_DIR="${HOME}/.config/flash-moe"
FLASHMOE_SESSIONS_DIR="${HOME}/.flash-moe/sessions"
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

echo "=== Flash-MoE Environment Toggle ==="
echo ""

# Check current state
if [ -d "${HF_CACHE_DIR}${SUFFIX}" ]; then
    # Currently in test mode - restore live
    
    # Safety check: verify backup is valid before restoring
    echo "Checking backup integrity..."
    if verify_model_cache "${HF_CACHE_DIR}${SUFFIX}"; then
        local backup_size
        backup_size=$(get_cache_size "${HF_CACHE_DIR}${SUFFIX}")
        echo "  Backup size: $backup_size (valid)"
        
        # Check if current is also valid (would be overwritten)
        if [ -d "$HF_CACHE_DIR" ] && verify_model_cache "$HF_CACHE_DIR"; then
            local current_size
            current_size=$(get_cache_size "$HF_CACHE_DIR")
            echo "  Current size: $current_size (will be replaced)"
            echo ""
            echo "WARNING: Replacing current model with backup."
            echo "  Current: $current_size"
            echo "  Backup:  $backup_size"
            echo ""
            read -p "Continue? [y/N] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Cancelled."
                exit 0
            fi
            # Remove current (invalid) model
            rm -rf "$HF_CACHE_DIR"
        fi
        
        # Restore HuggingFace cache
        echo "Restoring live environment..."
        mv "${HF_CACHE_DIR}${SUFFIX}" "$HF_CACHE_DIR"
        
        # Restore config
        if [ -f "${FLASHMOE_CONFIG_DIR}/config${SUFFIX}" ]; then
            mv "${FLASHMOE_CONFIG_DIR}/config${SUFFIX}" "${FLASHMOE_CONFIG_DIR}/config"
        fi
        
        # Restore sessions
        if [ -d "${FLASHMOE_SESSIONS_DIR}${SUFFIX}" ]; then
            mv "${FLASHMOE_SESSIONS_DIR}${SUFFIX}" "$FLASHMOE_SESSIONS_DIR"
        fi
        
        echo "Live environment restored."
    else
        echo "ERROR: Backup is invalid or incomplete!"
        echo "  Backup location: ${HF_CACHE_DIR}${SUFFIX}"
        echo ""
        echo "Manual recovery needed. Your live data may be lost."
        echo "  Live data was moved to: ${HF_CACHE_DIR}${SUFFIX}"
        exit 1
    fi
    
else
    # Currently in live mode - switch to test
    
    # Safety check: verify current model is valid before backing up
    echo "Checking live model integrity..."
    if [ -d "$HF_CACHE_DIR" ] && verify_model_cache "$HF_CACHE_DIR"; then
        local live_size
        live_size=$(get_cache_size "$HF_CACHE_DIR")
        echo "  Live model size: $live_size (valid)"
    else
        echo "ERROR: Live model is invalid or incomplete!"
        echo "  Location: $HF_CACHE_DIR"
        echo ""
        echo "Cannot toggle - live data is already corrupted."
        echo "Aborting to prevent further damage."
        exit 1
    fi
    
    echo "Switching to fresh test environment..."
    
    # Backup HuggingFace cache
    if [ -d "$HF_CACHE_DIR" ]; then
        mv "$HF_CACHE_DIR" "${HF_CACHE_DIR}${SUFFIX}"
    fi
    
    # Backup config
    if [ -f "${FLASHMOE_CONFIG_DIR}/config" ]; then
        mkdir -p "$FLASHMOE_CONFIG_DIR"
        mv "${FLASHMOE_CONFIG_DIR}/config" "${FLASHMOE_CONFIG_DIR}/config${SUFFIX}"
    fi
    
    # Backup sessions
    if [ -d "$FLASHMOE_SESSIONS_DIR" ]; then
        mv "$FLASHMOE_SESSIONS_DIR" "${FLASHMOE_SESSIONS_DIR}${SUFFIX}"
    fi
    
    echo "Fresh test environment ready."
    echo ""
    echo "To run flashchat fresh: ./flashchat"
    echo "To restore live: ./toggle_test_env.sh"
fi
