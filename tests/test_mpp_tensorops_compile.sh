#!/usr/bin/env bash
set -euo pipefail

SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
HEADER="${SDK_PATH}/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h"

if [[ -z "$SDK_PATH" || ! -f "$HEADER" ]]; then
    echo "SKIP: MetalPerformancePrimitives TensorOps headers not found"
    exit 0
fi

SRC="$(mktemp /tmp/flashchat-mpp-tensorops.XXXXXX.metal)"
AIR="$(mktemp /tmp/flashchat-mpp-tensorops.XXXXXX.air)"
trap 'rm -f "$SRC" "$AIR"' EXIT

cat >"$SRC" <<'METAL'
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_matmul_uint4_probe(
    tensor<device half, dextents<int32_t, 2>> A [[buffer(0)]],
    tensor<device uint4b_format, dextents<int32_t, 2>> B [[buffer(1)]],
    tensor<device float, dextents<int32_t, 2>> C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    matmul2d<d, execution_simdgroups<4>> op;
    auto a = A.slice(0, tgid.y * 64);
    auto b = B.slice(tgid.x * 32, 0);
    auto c = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(a, b, c);
}
METAL

xcrun -sdk macosx metal -x metal -std=metal4.0 -c "$SRC" -o "$AIR"

echo "MPP TensorOps matmul compile smoke passed"
