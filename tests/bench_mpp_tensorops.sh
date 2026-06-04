#!/usr/bin/env bash
set -euo pipefail

SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
MPP_HEADER="${SDK_PATH}/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h"

if [[ -z "${SDK_PATH}" || ! -f "${MPP_HEADER}" ]]; then
    echo "MPP TensorOps headers unavailable; skipping"
    exit 0
fi

OS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo 0)"
OS_MAJOR="${OS_VERSION%%.*}"
if [[ ! "${OS_MAJOR}" =~ ^[0-9]+$ || "${OS_MAJOR}" -lt 26 ]]; then
    echo "MPP TensorOps runtime APIs require macOS 26; skipping"
    exit 0
fi

OUT_DIM="${FLASHCHAT_MPP_BENCH_OUT_DIM:-32}"
GROUP_COUNT="${FLASHCHAT_MPP_BENCH_GROUPS:-39}"
CHUNK_GROUPS="${FLASHCHAT_MPP_BENCH_CHUNK_GROUPS:-13}"
MAX_CHUNKS_PER_COMMAND="${FLASHCHAT_MPP_BENCH_MAX_CHUNKS_PER_COMMAND:-3}"
REPEATS="${FLASHCHAT_MPP_BENCH_REPEATS:-20}"

for value_name in OUT_DIM GROUP_COUNT CHUNK_GROUPS MAX_CHUNKS_PER_COMMAND REPEATS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -le 0 ]]; then
        echo "${value_name} must be a positive integer"
        exit 1
    fi
done

if (( OUT_DIM % 32 != 0 )); then
    echo "FLASHCHAT_MPP_BENCH_OUT_DIM must be divisible by 32"
    exit 1
fi

if (( GROUP_COUNT > 192 )); then
    echo "FLASHCHAT_MPP_BENCH_GROUPS is capped at 192 for this generated benchmark"
    exit 1
fi

if (( CHUNK_GROUPS > 13 )); then
    echo "FLASHCHAT_MPP_BENCH_CHUNK_GROUPS is capped at 13 by Metal's buffer index limit"
    exit 1
fi

if (( MAX_CHUNKS_PER_COMMAND > 3 )); then
    echo "FLASHCHAT_MPP_BENCH_MAX_CHUNKS_PER_COMMAND is capped at 3 by current MTL4 chunk-dispatch validation"
    exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/flashchat-mpp-bench.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

KERNEL="${TMP_DIR}/mpp_tensorops_bench.metal"
SRC="${TMP_DIR}/mpp_tensorops_bench.m"
BIN="${TMP_DIR}/mpp_tensorops_bench"

cat > "${KERNEL}" <<EOF
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define BENCH_OUT_DIM ${OUT_DIM}u
#define BENCH_GROUPS ${GROUP_COUNT}u
#define BENCH_GROUP_SIZE 64u
#define BENCH_N 8u
#define ROWS_PER_TG 8u
#define ROWS_PER_SIMD 4u

static inline float bench_bf16_to_f32(ushort bf16) {
    return as_type<float>(uint(bf16) << 16);
}

kernel void baseline_dequant_matmulN_v5(
    device const uint* W_packed   [[buffer(0)]],
    device const ushort* scales   [[buffer(1)]],
    device const ushort* biases   [[buffer(2)]],
    device const float* X         [[buffer(3)]],
    device float* OUT             [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = (tgid * ROWS_PER_TG + simd_group) * ROWS_PER_SIMD;
    uint packed_cols = (BENCH_GROUPS * BENCH_GROUP_SIZE) / 8u;
    device const uint* w_row[ROWS_PER_SIMD];
    device const ushort* s_row[ROWS_PER_SIMD];
    device const ushort* b_row[ROWS_PER_SIMD];
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        uint rr = row_base + r;
        uint srr = rr < BENCH_OUT_DIM ? rr : 0u;
        w_row[r] = W_packed + (size_t)srr * packed_cols;
        s_row[r] = scales + (size_t)srr * BENCH_GROUPS;
        b_row[r] = biases + (size_t)srr * BENCH_GROUPS;
    }
    float acc[ROWS_PER_SIMD][BENCH_N];
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        for (uint n = 0; n < BENCH_N; n++) acc[r][n] = 0.0f;
    }
    for (uint col = simd_lane; col < packed_cols; col += 32u) {
        uint g = col / (BENCH_GROUP_SIZE / 8u);
        uint packed[ROWS_PER_SIMD];
        float scale[ROWS_PER_SIMD], bias[ROWS_PER_SIMD];
        for (uint r = 0; r < ROWS_PER_SIMD; r++) {
            packed[r] = w_row[r][col];
            scale[r] = bench_bf16_to_f32(s_row[r][g]);
            bias[r] = bench_bf16_to_f32(b_row[r][g]);
        }
        uint x_base = col * 8u;
        for (uint n = 0; n < BENCH_N; n++) {
            device const float* xn = X + (size_t)n * BENCH_GROUPS * BENCH_GROUP_SIZE;
            for (uint k = 0; k < 8u; k++) {
                float xv = xn[x_base + k];
                for (uint r = 0; r < ROWS_PER_SIMD; r++) {
                    float nib = float((packed[r] >> (k * 4u)) & 0xFu);
                    acc[r][n] = fma(nib, scale[r] * xv, fma(bias[r], xv, acc[r][n]));
                }
            }
        }
    }
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        uint rr = row_base + r;
        if (rr >= BENCH_OUT_DIM) continue;
        for (uint n = 0; n < BENCH_N; n++) {
            float s = simd_sum(acc[r][n]);
            if (simd_lane == 0) OUT[n * BENCH_OUT_DIM + rr] = s;
        }
    }
}

struct BenchParams {
    uint group_base;
    uint group_count;
    uint zero_accum;
    uint pad;
};

kernel void mpp_affine_matmulN_chunk(
EOF

for ((g = 0; g < CHUNK_GROUPS; g++)); do
    comma=","
    printf '    tensor<device bfloat, dextents<int32_t, 2>> A%d [[buffer(%d)]]%s\n' "${g}" "${g}" "${comma}" >> "${KERNEL}"
done
for ((g = 0; g < CHUNK_GROUPS; g++)); do
    printf '    tensor<device uint4b_format, dextents<int32_t, 2>> B%d [[buffer(%d)]],\n' "${g}" "$((CHUNK_GROUPS + g))" >> "${KERNEL}"
done
cat >> "${KERNEL}" <<EOF
    tensor<device float, dextents<int32_t, 2>> C [[buffer($((CHUNK_GROUPS * 2)))]],
    device const float* scale [[buffer($((CHUNK_GROUPS * 2 + 1)))]],
    device const float* bias [[buffer($((CHUNK_GROUPS * 2 + 2)))]],
    device const float* row_sum [[buffer($((CHUNK_GROUPS * 2 + 3)))]],
    device const BenchParams* params [[buffer($((CHUNK_GROUPS * 2 + 4)))]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr auto desc = matmul2d_descriptor(8, 32, 64, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto c = C.slice<32, 8>(tgid.x * 32, 0);
EOF

if (( CHUNK_GROUPS > 0 )); then
    cat >> "${KERNEL}" <<'EOF'
    auto a0 = A0.slice<64, 8>(0, 0);
    auto b0 = B0.slice<64, 32>(0, tgid.x * 32);
    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    if (params->zero_accum != 0u) {
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) acc[i] = 0.0f;
        }
    } else {
        acc.load(c);
    }
EOF
fi

for ((g = 0; g < CHUNK_GROUPS; g++)); do
    cat >> "${KERNEL}" <<EOF
    if (${g}u < params->group_count) {
        uint group_index = params->group_base + ${g}u;
        auto a = A${g}.slice<64, 8>(0, 0);
        auto b = B${g}.slice<64, 32>(0, tgid.x * 32);
        auto raw = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
        for (ushort i = 0; i < raw.get_capacity(); ++i) {
            if (raw.is_valid_element(i)) raw[i] = 0.0f;
        }
        op.run(a, b, raw);
        for (ushort i = 0; i < raw.get_capacity(); ++i) {
            if (raw.is_valid_element(i)) {
                auto ids = raw.get_multidimensional_index(i);
                uint n = (uint)ids[0];
                uint m = (uint)ids[1];
                uint out_col = tgid.x * 32 + n;
                acc[i] += raw[i] * scale[out_col * BENCH_GROUPS + group_index] + row_sum[m * BENCH_GROUPS + group_index] * bias[out_col * BENCH_GROUPS + group_index];
            }
        }
    }
EOF
done

cat >> "${KERNEL}" <<'EOF'
    acc.store(c);
}
EOF

cat > "${SRC}" <<EOF
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    BENCH_M = 8,
    BENCH_OUT_DIM = ${OUT_DIM},
    BENCH_GROUP_SIZE = 64,
    BENCH_GROUPS = ${GROUP_COUNT},
    BENCH_CHUNK_GROUPS = ${CHUNK_GROUPS},
    BENCH_CHUNKS = (BENCH_GROUPS + BENCH_CHUNK_GROUPS - 1) / BENCH_CHUNK_GROUPS,
    BENCH_MAX_CHUNKS_PER_COMMAND = ${MAX_CHUNKS_PER_COMMAND},
    BENCH_IN_DIM = BENCH_GROUP_SIZE * BENCH_GROUPS,
    BENCH_PACKED_COLS = BENCH_IN_DIM / 8,
    BENCH_REPEATS = ${REPEATS}
};

typedef struct {
    uint32_t group_base;
    uint32_t group_count;
    uint32_t zero_accum;
    uint32_t pad;
} BenchParams;

static MTLTensorExtents *extents(NSUInteger rank, const NSInteger *values) {
    return [[MTLTensorExtents alloc] initWithRank:rank values:values];
}

static id<MTLTensor> make_tensor(id<MTLDevice> device, MTLTensorDataType type, NSInteger d0, NSInteger d1) {
    MTLTensorDescriptor *desc = [[MTLTensorDescriptor alloc] init];
    NSInteger dims[2] = { d0, d1 };
    desc.dimensions = extents(2, dims);
    desc.dataType = type;
    desc.usage = MTLTensorUsageCompute;
    desc.storageMode = MTLStorageModeShared;
    NSError *error = nil;
    id<MTLTensor> tensor = [device newTensorWithDescriptor:desc error:&error];
    if (!tensor) fprintf(stderr, "tensor create failed: %s\\n", error.localizedDescription.UTF8String);
    return tensor;
}

static uint16_t bf16_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static uint8_t q_value(NSUInteger out, NSUInteger group) {
    return (uint8_t)((out + group * 3) % 15 + 1);
}

static double now_ms(void) {
    return CFAbsoluteTimeGetCurrent() * 1000.0;
}

static int run_baseline(id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> w_buffer,
                        id<MTLBuffer> scale_buffer,
                        id<MTLBuffer> bias_buffer,
                        id<MTLBuffer> x_buffer,
                        id<MTLBuffer> out_buffer) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:w_buffer offset:0 atIndex:0];
    [encoder setBuffer:scale_buffer offset:0 atIndex:1];
    [encoder setBuffer:bias_buffer offset:0 atIndex:2];
    [encoder setBuffer:x_buffer offset:0 atIndex:3];
    [encoder setBuffer:out_buffer offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((BENCH_OUT_DIM + 31) / 32, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "baseline command failed: %s\\n", command_buffer.error.localizedDescription.UTF8String);
        return -1;
    }
    return 0;
}

static int run_mpp(id<MTLDevice> device,
                   id<MTL4CommandQueue> queue,
                   id<MTLComputePipelineState> pipeline,
                   NSArray *arg_tables,
                   id<MTLSharedEvent> event,
                   uint64_t *event_value) {
    for (NSUInteger base_chunk = 0; base_chunk < arg_tables.count; base_chunk += BENCH_MAX_CHUNKS_PER_COMMAND) {
        NSUInteger end_chunk = base_chunk + BENCH_MAX_CHUNKS_PER_COMMAND;
        if (end_chunk > arg_tables.count) end_chunk = arg_tables.count;
        id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
        id<MTL4CommandBuffer> command_buffer = [device newCommandBuffer];
        if (!allocator || !command_buffer) {
            fprintf(stderr, "MTL4 command objects unavailable\\n");
            return -1;
        }
        [command_buffer beginCommandBufferWithAllocator:allocator];
        id<MTL4ComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger chunk = base_chunk; chunk < end_chunk; chunk++) {
            id<MTL4ArgumentTable> args = [arg_tables objectAtIndex:chunk];
            [encoder setArgumentTable:args];
            [encoder dispatchThreadgroups:MTLSizeMake(BENCH_OUT_DIM / 32, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth * 4, 1, 1)];
            if (chunk + 1 < end_chunk) {
                [encoder barrierAfterEncoderStages:MTLStageDispatch
                               beforeEncoderStages:MTLStageDispatch
                                 visibilityOptions:MTL4VisibilityOptionDevice];
            }
        }
        [encoder endEncoding];
        [command_buffer endCommandBuffer];
        id<MTL4CommandBuffer> command_buffers[1] = { command_buffer };
        [queue commit:command_buffers count:1];
        (*event_value)++;
        [queue signalEvent:event value:*event_value];
        if (![event waitUntilSignaledValue:*event_value timeoutMS:30000]) {
            fprintf(stderr, "MPP command timed out\\n");
            return -1;
        }
    }
    return 0;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("Metal device unavailable; skipping\\n");
            return 0;
        }
        if (![device respondsToSelector:@selector(newMTL4CommandQueue)] ||
            ![device respondsToSelector:@selector(newCommandAllocator)] ||
            ![device respondsToSelector:@selector(newCommandBuffer)] ||
            ![device respondsToSelector:@selector(newArgumentTableWithDescriptor:error:)] ||
            ![device respondsToSelector:@selector(newTensorWithDescriptor:error:)]) {
            printf("MTL4 tensor APIs unavailable; skipping\\n");
            return 0;
        }

        NSString *kernel_path = @"${KERNEL}";
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:kernel_path encoding:NSUTF8StringEncoding error:&error];
        if (!source) {
            fprintf(stderr, "kernel read failed: %s\\n", error.localizedDescription.UTF8String);
            return 1;
        }
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion4_0;
        double compile_start = now_ms();
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        double compile_ms = now_ms() - compile_start;
        if (!library) {
            fprintf(stderr, "library compile failed: %s\\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> baseline_function = [library newFunctionWithName:@"baseline_dequant_matmulN_v5"];
        id<MTLFunction> mpp_function = [library newFunctionWithName:@"mpp_affine_matmulN_chunk"];
        id<MTLComputePipelineState> baseline_pipeline = [device newComputePipelineStateWithFunction:baseline_function error:&error];
        if (!baseline_pipeline) {
            fprintf(stderr, "baseline pipeline create failed: %s\\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> mpp_pipeline = [device newComputePipelineStateWithFunction:mpp_function error:&error];
        if (!mpp_pipeline) {
            fprintf(stderr, "MPP pipeline create failed: %s\\n", error.localizedDescription.UTF8String);
            return 1;
        }

        size_t w_count = (size_t)BENCH_OUT_DIM * BENCH_PACKED_COLS;
        size_t sb_count = (size_t)BENCH_OUT_DIM * BENCH_GROUPS;
        size_t x_count = (size_t)BENCH_M * BENCH_IN_DIM;
        size_t out_count = (size_t)BENCH_M * BENCH_OUT_DIM;
        uint32_t *w_data = calloc(w_count, sizeof(uint32_t));
        uint16_t *scale16_data = calloc(sb_count, sizeof(uint16_t));
        uint16_t *bias16_data = calloc(sb_count, sizeof(uint16_t));
        float *scale32_data = calloc(sb_count, sizeof(float));
        float *bias32_data = calloc(sb_count, sizeof(float));
        float *row_sum_data = calloc((size_t)BENCH_M * BENCH_GROUPS, sizeof(float));
        float *x_data = calloc(x_count, sizeof(float));
        float *baseline_out = calloc(out_count, sizeof(float));
        float *mpp_out = calloc(out_count, sizeof(float));
        uint16_t *a_data = calloc((size_t)BENCH_M * BENCH_GROUP_SIZE, sizeof(uint16_t));
        uint8_t *b_group_data = calloc((size_t)BENCH_OUT_DIM * BENCH_GROUP_SIZE / 2, sizeof(uint8_t));
        if (!w_data || !scale16_data || !bias16_data || !scale32_data || !bias32_data ||
            !row_sum_data || !x_data || !baseline_out || !mpp_out || !a_data || !b_group_data) {
            fprintf(stderr, "allocation failed\\n");
            return 1;
        }

        for (size_t i = 0; i < x_count; i++) x_data[i] = 1.0f;
        for (size_t i = 0; i < (size_t)BENCH_M * BENCH_GROUP_SIZE; i++) a_data[i] = 0x3f80;
        for (NSUInteger m = 0; m < BENCH_M; m++) {
            for (NSUInteger g = 0; g < BENCH_GROUPS; g++) row_sum_data[m * BENCH_GROUPS + g] = (float)BENCH_GROUP_SIZE;
        }
        for (NSUInteger out = 0; out < BENCH_OUT_DIM; out++) {
            for (NSUInteger g = 0; g < BENCH_GROUPS; g++) {
                float scale = (float)(1 + ((out + g) % 3));
                float bias = (float)(g % 2);
                scale32_data[out * BENCH_GROUPS + g] = scale;
                bias32_data[out * BENCH_GROUPS + g] = bias;
                scale16_data[out * BENCH_GROUPS + g] = bf16_from_f32(scale);
                bias16_data[out * BENCH_GROUPS + g] = bf16_from_f32(bias);
                uint8_t q = q_value(out, g);
                for (NSUInteger packed = 0; packed < BENCH_GROUP_SIZE / 8; packed++) {
                    uint32_t word = 0;
                    for (NSUInteger k = 0; k < 8; k++) word |= (uint32_t)q << (k * 4);
                    w_data[out * BENCH_PACKED_COLS + g * (BENCH_GROUP_SIZE / 8) + packed] = word;
                }
            }
        }

        id<MTLBuffer> w_buffer = [device newBufferWithBytes:w_data length:w_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale16_buffer = [device newBufferWithBytes:scale16_data length:sb_count * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bias16_buffer = [device newBufferWithBytes:bias16_data length:sb_count * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> x_buffer = [device newBufferWithBytes:x_data length:x_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> baseline_out_buffer = [device newBufferWithLength:out_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale32_buffer = [device newBufferWithBytes:scale32_data length:sb_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bias32_buffer = [device newBufferWithBytes:bias32_data length:sb_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> row_sum_buffer = [device newBufferWithBytes:row_sum_data length:(size_t)BENCH_M * BENCH_GROUPS * sizeof(float) options:MTLResourceStorageModeShared];
        if (!w_buffer || !scale16_buffer || !bias16_buffer || !x_buffer || !baseline_out_buffer ||
            !scale32_buffer || !bias32_buffer || !row_sum_buffer) {
            fprintf(stderr, "buffer create failed\\n");
            return 1;
        }

        NSMutableArray<id<MTLTensor>> *a_tensors = [NSMutableArray arrayWithCapacity:BENCH_GROUPS];
        NSMutableArray<id<MTLTensor>> *b_tensors = [NSMutableArray arrayWithCapacity:BENCH_GROUPS];
        NSInteger origin_values[2] = { 0, 0 };
        MTLTensorExtents *origin = extents(2, origin_values);
        NSInteger a_dims[2] = { BENCH_GROUP_SIZE, BENCH_M };
        NSInteger a_strides[2] = { 1, BENCH_GROUP_SIZE };
        NSInteger b_dims[2] = { BENCH_GROUP_SIZE, BENCH_OUT_DIM };
        NSInteger b_strides[2] = { 1, BENCH_GROUP_SIZE };
        for (NSUInteger g = 0; g < BENCH_GROUPS; g++) {
            id<MTLTensor> A = make_tensor(device, MTLTensorDataTypeBFloat16, BENCH_GROUP_SIZE, BENCH_M);
            id<MTLTensor> B = make_tensor(device, MTLTensorDataTypeUInt4, BENCH_GROUP_SIZE, BENCH_OUT_DIM);
            if (!A || !B) return 1;
            [A replaceSliceOrigin:origin sliceDimensions:extents(2, a_dims) withBytes:a_data strides:extents(2, a_strides)];
            for (NSUInteger out = 0; out < BENCH_OUT_DIM; out++) {
                uint8_t q = q_value(out, g);
                for (NSUInteger k = 0; k < BENCH_GROUP_SIZE; k += 2) {
                    b_group_data[out * (BENCH_GROUP_SIZE / 2) + k / 2] = (uint8_t)(q | (q << 4));
                }
            }
            [B replaceSliceOrigin:origin sliceDimensions:extents(2, b_dims) withBytes:b_group_data strides:extents(2, b_strides)];
            [a_tensors addObject:A];
            [b_tensors addObject:B];
        }
        id<MTLTensor> c_tensor = make_tensor(device, MTLTensorDataTypeFloat32, BENCH_OUT_DIM, BENCH_M);
        if (!c_tensor) return 1;

        NSMutableArray *arg_tables = [NSMutableArray arrayWithCapacity:BENCH_CHUNKS];
        NSMutableArray *param_buffers = [NSMutableArray arrayWithCapacity:BENCH_CHUNKS];
        for (NSUInteger chunk = 0; chunk < BENCH_CHUNKS; chunk++) {
            NSUInteger group_base = chunk * BENCH_CHUNK_GROUPS;
            NSUInteger group_count = BENCH_GROUPS - group_base;
            if (group_count > BENCH_CHUNK_GROUPS) group_count = BENCH_CHUNK_GROUPS;
            BenchParams params = {
                (uint32_t)group_base,
                (uint32_t)group_count,
                chunk == 0 ? 1u : 0u,
                0u
            };
            id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            if (!params_buffer) {
                fprintf(stderr, "params buffer create failed\\n");
                return 1;
            }
            MTL4ArgumentTableDescriptor *arg_desc = [[MTL4ArgumentTableDescriptor alloc] init];
            arg_desc.maxBufferBindCount = BENCH_CHUNK_GROUPS * 2 + 5;
            arg_desc.initializeBindings = YES;
            id<MTL4ArgumentTable> args = [device newArgumentTableWithDescriptor:arg_desc error:&error];
            if (!args) {
                fprintf(stderr, "argument table create failed: %s\\n", error.localizedDescription.UTF8String);
                return 1;
            }
            for (NSUInteger slot = 0; slot < BENCH_CHUNK_GROUPS; slot++) {
                NSUInteger real_group = group_base + slot;
                if (real_group >= BENCH_GROUPS) real_group = 0;
                [args setResource:[a_tensors objectAtIndex:real_group].gpuResourceID atBufferIndex:slot];
                [args setResource:[b_tensors objectAtIndex:real_group].gpuResourceID atBufferIndex:BENCH_CHUNK_GROUPS + slot];
            }
            [args setResource:c_tensor.gpuResourceID atBufferIndex:BENCH_CHUNK_GROUPS * 2];
            [args setAddress:scale32_buffer.gpuAddress atIndex:BENCH_CHUNK_GROUPS * 2 + 1];
            [args setAddress:bias32_buffer.gpuAddress atIndex:BENCH_CHUNK_GROUPS * 2 + 2];
            [args setAddress:row_sum_buffer.gpuAddress atIndex:BENCH_CHUNK_GROUPS * 2 + 3];
            [args setAddress:params_buffer.gpuAddress atIndex:BENCH_CHUNK_GROUPS * 2 + 4];
            [arg_tables addObject:args];
            [param_buffers addObject:params_buffer];
        }

        id<MTLCommandQueue> baseline_queue = [device newCommandQueue];
        id<MTL4CommandQueue> mpp_queue = [device newMTL4CommandQueue];
        id<MTLSharedEvent> event = [device newSharedEvent];
        if (!baseline_queue || !mpp_queue || !event) {
            fprintf(stderr, "queue/event create failed\\n");
            return 1;
        }
        uint64_t mpp_event_value = 0;

        if (run_baseline(baseline_queue, baseline_pipeline, w_buffer, scale16_buffer, bias16_buffer, x_buffer, baseline_out_buffer) != 0 ||
            run_mpp(device, mpp_queue, mpp_pipeline, arg_tables, event, &mpp_event_value) != 0) {
            return 1;
        }
        memcpy(baseline_out, baseline_out_buffer.contents, out_count * sizeof(float));
        NSInteger c_dims[2] = { BENCH_OUT_DIM, BENCH_M };
        NSInteger c_strides[2] = { 1, BENCH_OUT_DIM };
        [c_tensor getBytes:mpp_out strides:extents(2, c_strides) fromSliceOrigin:origin sliceDimensions:extents(2, c_dims)];
        double max_err = 0.0;
        for (size_t i = 0; i < out_count; i++) {
            double err = fabs((double)baseline_out[i] - (double)mpp_out[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 0.001) {
            fprintf(stderr, "MPP mismatch max_err=%.6f baseline0=%.6f mpp0=%.6f\\n", max_err, baseline_out[0], mpp_out[0]);
            return 1;
        }

        for (int i = 0; i < 3; i++) {
            if (run_baseline(baseline_queue, baseline_pipeline, w_buffer, scale16_buffer, bias16_buffer, x_buffer, baseline_out_buffer) != 0 ||
                run_mpp(device, mpp_queue, mpp_pipeline, arg_tables, event, &mpp_event_value) != 0) {
                return 1;
            }
        }

        double baseline_start = now_ms();
        for (int i = 0; i < BENCH_REPEATS; i++) {
            if (run_baseline(baseline_queue, baseline_pipeline, w_buffer, scale16_buffer, bias16_buffer, x_buffer, baseline_out_buffer) != 0) return 1;
        }
        double baseline_ms = now_ms() - baseline_start;

        double mpp_start = now_ms();
        for (int i = 0; i < BENCH_REPEATS; i++) {
            if (run_mpp(device, mpp_queue, mpp_pipeline, arg_tables, event, &mpp_event_value) != 0) return 1;
        }
        double mpp_ms = now_ms() - mpp_start;

        double baseline_avg = baseline_ms / BENCH_REPEATS;
        double mpp_avg = mpp_ms / BENCH_REPEATS;
        double dense_ops = 2.0 * (double)BENCH_M * (double)BENCH_OUT_DIM * (double)BENCH_IN_DIM;
        printf("MPP TensorOps dense-prefill bench\\n");
        printf("device=%s compile=%.1f ms M=%d OUT=%d IN=%d groups=%d chunk_groups=%d chunks=%d chunks_per_command=%d repeats=%d\\n",
               device.name.UTF8String, compile_ms, BENCH_M, BENCH_OUT_DIM, BENCH_IN_DIM,
               BENCH_GROUPS, BENCH_CHUNK_GROUPS, BENCH_CHUNKS, BENCH_MAX_CHUNKS_PER_COMMAND, BENCH_REPEATS);
        printf("baseline_v5_avg_ms=%.4f effective_tflops=%.3f\\n", baseline_avg, dense_ops / (baseline_avg * 1.0e9));
        printf("mpp_affine_avg_ms=%.4f effective_tflops=%.3f\\n", mpp_avg, dense_ops / (mpp_avg * 1.0e9));
        printf("mpp_vs_baseline=%.3fx max_err=%.6f\\n", baseline_avg / mpp_avg, max_err);
    }
    return 0;
}
EOF

xcrun clang -fobjc-arc -ObjC -mmacosx-version-min=26.0 \
    -Wno-unguarded-availability-new \
    -framework Foundation -framework Metal \
    "${SRC}" -o "${BIN}"

"${BIN}"
