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

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/flashchat-mpp-runtime.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

SRC="${TMP_DIR}/mpp_tensorops_runtime.m"
BIN="${TMP_DIR}/mpp_tensorops_runtime"

cat > "${SRC}" <<'EOF'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>

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
    if (!tensor) {
        fprintf(stderr, "tensor create failed: %s\n", error.localizedDescription.UTF8String);
    }
    return tensor;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("Metal device unavailable; skipping\n");
            return 0;
        }
        if (![device respondsToSelector:@selector(newMTL4CommandQueue)] ||
            ![device respondsToSelector:@selector(newCommandAllocator)] ||
            ![device respondsToSelector:@selector(newCommandBuffer)] ||
            ![device respondsToSelector:@selector(newArgumentTableWithDescriptor:error:)] ||
            ![device respondsToSelector:@selector(newTensorWithDescriptor:error:)]) {
            printf("MTL4 tensor APIs unavailable; skipping\n");
            return 0;
        }

        enum { M = 8, N = 32, GROUP_SIZE = 64, GROUPS = 2 };
        id<MTLTensor> A0 = make_tensor(device, MTLTensorDataTypeBFloat16, GROUP_SIZE, M);
        id<MTLTensor> A1 = make_tensor(device, MTLTensorDataTypeBFloat16, GROUP_SIZE, M);
        id<MTLTensor> B0 = make_tensor(device, MTLTensorDataTypeUInt4, GROUP_SIZE, N);
        id<MTLTensor> B1 = make_tensor(device, MTLTensorDataTypeUInt4, GROUP_SIZE, N);
        id<MTLTensor> C = make_tensor(device, MTLTensorDataTypeFloat32, N, M);
        if (!A0 || !A1 || !B0 || !B1 || !C) return 1;

        uint16_t a0_data[M * GROUP_SIZE];
        uint16_t a1_data[M * GROUP_SIZE];
        uint8_t b0_data[N * GROUP_SIZE / 2];
        uint8_t b1_data[N * GROUP_SIZE / 2];
        float c_data[M * N];
        float scale_data[N * GROUPS];
        float bias_data[N * GROUPS];
        float row_sum_data[M * GROUPS];
        for (NSUInteger i = 0; i < M * GROUP_SIZE; i++) a0_data[i] = 0x3f80;
        for (NSUInteger i = 0; i < M * GROUP_SIZE; i++) a1_data[i] = 0x3f80;
        for (NSUInteger o = 0; o < N; o++) {
            uint8_t q0 = (uint8_t)((o + 0 * 3) % 7 + 1);
            uint8_t q1 = (uint8_t)((o + 1 * 3) % 7 + 1);
            for (NSUInteger k = 0; k < GROUP_SIZE; k += 2) {
                b0_data[o * (GROUP_SIZE / 2) + k / 2] = (uint8_t)(q0 | (q0 << 4));
                b1_data[o * (GROUP_SIZE / 2) + k / 2] = (uint8_t)(q1 | (q1 << 4));
            }
        }
        for (NSUInteger i = 0; i < M * N; i++) c_data[i] = 0.0f;
        for (NSUInteger o = 0; o < N; o++) {
            for (NSUInteger g = 0; g < GROUPS; g++) {
                scale_data[o * GROUPS + g] = (float)(2 + ((o + g) % 3));
                bias_data[o * GROUPS + g] = (float)(1 + g);
            }
        }
        for (NSUInteger m = 0; m < M; m++) {
            for (NSUInteger g = 0; g < GROUPS; g++) row_sum_data[m * GROUPS + g] = (float)GROUP_SIZE;
        }

        NSInteger origin_values[2] = { 0, 0 };
        NSInteger a_dims[2] = { GROUP_SIZE, M };
        NSInteger a_strides[2] = { 1, GROUP_SIZE };
        NSInteger b_dims[2] = { GROUP_SIZE, N };
        NSInteger b_strides[2] = { 1, GROUP_SIZE };
        NSInteger c_dims[2] = { N, M };
        NSInteger c_strides[2] = { 1, N };
        MTLTensorExtents *origin = extents(2, origin_values);
        [A0 replaceSliceOrigin:origin sliceDimensions:extents(2, a_dims) withBytes:a0_data strides:extents(2, a_strides)];
        [A1 replaceSliceOrigin:origin sliceDimensions:extents(2, a_dims) withBytes:a1_data strides:extents(2, a_strides)];
        [B0 replaceSliceOrigin:origin sliceDimensions:extents(2, b_dims) withBytes:b0_data strides:extents(2, b_strides)];
        [B1 replaceSliceOrigin:origin sliceDimensions:extents(2, b_dims) withBytes:b1_data strides:extents(2, b_strides)];
        [C replaceSliceOrigin:origin sliceDimensions:extents(2, c_dims) withBytes:c_data strides:extents(2, c_strides)];

        id<MTLBuffer> scale_buffer = [device newBufferWithBytes:scale_data length:sizeof(scale_data) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bias_buffer = [device newBufferWithBytes:bias_data length:sizeof(bias_data) options:MTLResourceStorageModeShared];
        id<MTLBuffer> row_sum_buffer = [device newBufferWithBytes:row_sum_data length:sizeof(row_sum_data) options:MTLResourceStorageModeShared];
        if (!scale_buffer || !bias_buffer || !row_sum_buffer) {
            fprintf(stderr, "buffer create failed\n");
            return 1;
        }

        NSString *source =
            @"#include <metal_stdlib>\n"
             "#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>\n"
             "using namespace metal;\n"
             "using namespace mpp::tensor_ops;\n"
             "kernel void mpp_matmul_uint4_runtime(\n"
             "    tensor<device bfloat, dextents<int32_t, 2>> A0 [[buffer(0)]],\n"
             "    tensor<device uint4b_format, dextents<int32_t, 2>> B0 [[buffer(1)]],\n"
             "    tensor<device bfloat, dextents<int32_t, 2>> A1 [[buffer(2)]],\n"
             "    tensor<device uint4b_format, dextents<int32_t, 2>> B1 [[buffer(3)]],\n"
             "    tensor<device float, dextents<int32_t, 2>> C [[buffer(4)]],\n"
             "    device const float* scale [[buffer(5)]],\n"
             "    device const float* bias [[buffer(6)]],\n"
             "    device const float* row_sum [[buffer(7)]],\n"
             "    uint2 tgid [[threadgroup_position_in_grid]]) {\n"
             "    constexpr auto desc = matmul2d_descriptor(8, 32, 64, false, true, false);\n"
             "    matmul2d<desc, execution_simdgroups<4>> op;\n"
             "    auto a0 = A0.slice<64, 8>(0, tgid.y * 8);\n"
             "    auto b0 = B0.slice<64, 32>(0, tgid.x * 32);\n"
             "    auto c = C.slice<32, 8>(tgid.x * 32, tgid.y * 8);\n"
             "    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n"
             "    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {\n"
             "        if (acc.is_valid_element(i)) acc[i] = 0.0f;\n"
             "    }\n"
             "    auto raw0 = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n"
             "    for (uint16_t i = 0; i < raw0.get_capacity(); ++i) if (raw0.is_valid_element(i)) raw0[i] = 0.0f;\n"
             "    op.run(a0, b0, raw0);\n"
             "    for (uint16_t i = 0; i < raw0.get_capacity(); ++i) {\n"
             "        if (raw0.is_valid_element(i)) {\n"
             "            auto ids = raw0.get_multidimensional_index(i);\n"
             "            uint n = (uint)ids[0];\n"
             "            uint m = (uint)ids[1];\n"
             "            uint out_col = tgid.x * 32 + n;\n"
             "            uint token_row = tgid.y * 8 + m;\n"
             "            acc[i] += raw0[i] * scale[out_col * 2] + row_sum[token_row * 2] * bias[out_col * 2];\n"
             "        }\n"
             "    }\n"
             "    auto a1 = A1.slice<64, 8>(0, tgid.y * 8);\n"
             "    auto b1 = B1.slice<64, 32>(0, tgid.x * 32);\n"
             "    auto raw1 = op.get_destination_cooperative_tensor<decltype(a1), decltype(b1), float>();\n"
             "    for (uint16_t i = 0; i < raw1.get_capacity(); ++i) if (raw1.is_valid_element(i)) raw1[i] = 0.0f;\n"
             "    op.run(a1, b1, raw1);\n"
             "    for (uint16_t i = 0; i < raw1.get_capacity(); ++i) {\n"
             "        if (raw1.is_valid_element(i)) {\n"
             "            auto ids = raw1.get_multidimensional_index(i);\n"
             "            uint n = (uint)ids[0];\n"
             "            uint m = (uint)ids[1];\n"
             "            uint out_col = tgid.x * 32 + n;\n"
             "            uint token_row = tgid.y * 8 + m;\n"
             "            acc[i] += raw1[i] * scale[out_col * 2 + 1] + row_sum[token_row * 2 + 1] * bias[out_col * 2 + 1];\n"
             "        }\n"
             "    }\n"
             "    acc.store(c);\n"
             "}\n";
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion4_0;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "library compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"mpp_matmul_uint4_runtime"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "pipeline create failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        MTL4ArgumentTableDescriptor *arg_desc = [[MTL4ArgumentTableDescriptor alloc] init];
        arg_desc.maxBufferBindCount = 8;
        arg_desc.initializeBindings = YES;
        id<MTL4ArgumentTable> args = [device newArgumentTableWithDescriptor:arg_desc error:&error];
        if (!args) {
            fprintf(stderr, "argument table create failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        [args setResource:A0.gpuResourceID atBufferIndex:0];
        [args setResource:B0.gpuResourceID atBufferIndex:1];
        [args setResource:A1.gpuResourceID atBufferIndex:2];
        [args setResource:B1.gpuResourceID atBufferIndex:3];
        [args setResource:C.gpuResourceID atBufferIndex:4];
        [args setAddress:scale_buffer.gpuAddress atIndex:5];
        [args setAddress:bias_buffer.gpuAddress atIndex:6];
        [args setAddress:row_sum_buffer.gpuAddress atIndex:7];

        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
        id<MTL4CommandBuffer> command_buffer = [device newCommandBuffer];
        if (!queue || !allocator || !command_buffer) {
            fprintf(stderr, "MTL4 command objects unavailable\n");
            return 1;
        }
        [command_buffer beginCommandBufferWithAllocator:allocator];
        id<MTL4ComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:args];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth * 4, 1, 1)];
        [encoder endEncoding];
        [command_buffer endCommandBuffer];

        id<MTLSharedEvent> event = [device newSharedEvent];
        id<MTL4CommandBuffer> command_buffers[1] = { command_buffer };
        [queue commit:command_buffers count:1];
        [queue signalEvent:event value:1];
        if (![event waitUntilSignaledValue:1 timeoutMS:5000]) {
            fprintf(stderr, "MTL4 command queue timed out\n");
            return 1;
        }

        [C getBytes:c_data strides:extents(2, c_strides) fromSliceOrigin:origin sliceDimensions:extents(2, c_dims)];
        double sum = 0.0;
        double max_err = 0.0;
        for (NSUInteger m = 0; m < M; m++) {
            for (NSUInteger o = 0; o < N; o++) {
                double expected_value = 0.0;
                for (NSUInteger g = 0; g < GROUPS; g++) {
                    double q = (double)((o + g * 3) % 7 + 1);
                    expected_value += (double)GROUP_SIZE * q * scale_data[o * GROUPS + g];
                    expected_value += (double)GROUP_SIZE * bias_data[o * GROUPS + g];
                }
                double got = c_data[m * N + o];
                double err = fabs(got - expected_value);
                if (err > max_err) max_err = err;
                sum += got;
            }
        }
        double expected_sum = 0.0;
        for (NSUInteger m = 0; m < M; m++) {
            for (NSUInteger o = 0; o < N; o++) {
                double expected_value = 0.0;
                for (NSUInteger g = 0; g < GROUPS; g++) {
                    double q = (double)((o + g * 3) % 7 + 1);
                    expected_value += (double)GROUP_SIZE * q * scale_data[o * GROUPS + g];
                    expected_value += (double)GROUP_SIZE * bias_data[o * GROUPS + g];
                }
                expected_sum += expected_value;
            }
        }
        if (fabs(sum - expected_sum) > 0.001 || max_err > 0.001) {
            fprintf(stderr, "unexpected MPP result sum=%.6f expected=%.6f max_err=%.6f first=%.6f\n",
                    sum, expected_sum, max_err, c_data[0]);
            return 1;
        }
        printf("MPP TensorOps runtime smoke passed sum=%.0f first=%.0f\n", sum, c_data[0]);
    }
    return 0;
}
EOF

xcrun clang -fobjc-arc -ObjC -mmacosx-version-min=26.0 \
    -Wno-unguarded-availability-new \
    -framework Foundation -framework Metal \
    "${SRC}" -o "${BIN}"

"${BIN}"
