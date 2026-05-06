#!/bin/bash
# Test ZStandard transparent compression on expert data.
# Compresses a copy of one layer file with ZStandard compression,
# then benchmarks pread throughput on compressed vs uncompressed.

set -e

EXPERTS_DIR="/Users/speedster/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/packed_experts"
SRC="$EXPERTS_DIR/layer_00.bin"
WORKDIR="/tmp/zstd_compress_test"

echo "=== ZStandard Transparent Compression Test ==="

# Check source
if [ ! -f "$SRC" ]; then
    echo "ERROR: source file not found: $SRC"
    exit 1
fi

RAW_SIZE=$(stat -f%z "$SRC")
echo "Source: $SRC"
echo "Raw size: $((RAW_SIZE / 1024 / 1024)) MB ($RAW_SIZE bytes)"

# Setup
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# Copy uncompressed
echo ""
echo "--- Copying uncompressed ---"
cp "$SRC" "$WORKDIR/layer_uncompressed.bin"

# Benchmark with ZStandard compression (using zstd -b1 -e6 --nocheck)
echo ""
echo "--- Benchmark with ZStandard compression ---"
if command -v zstd &>/dev/null; then
    cp "$SRC" "$WORKDIR/layer_compressed.bin"
    zstd -e1 -b6 --no-check "$WORKDIR/layer_compressed.bin"
    #COMP_SIZE=$(stat -f%z "$WORKDIR/layer_compressed.bin")
    #COMP_DISK=$(du -k "$WORKDIR/layer_compressed.bin" | cut -f1)
    #echo "afsctool LZFSE: logical=$((COMP_SIZE / 1024 / 1024)) MB, disk=${COMP_DISK}K"
fi

# Cleanup
echo ""
echo "Workdir: $WORKDIR (not cleaned up — inspect manually)"
