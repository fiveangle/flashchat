#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float bf16_value(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "Metal device unavailable\n"); return 1; }
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@"metal_infer/shaders.metal"
                                                   encoding:NSUTF8StringEncoding error:&error];
        id<MTLLibrary> library = source ? [device newLibraryWithSource:source options:nil error:&error] : nil;
        id<MTLFunction> function = [library newFunctionWithName:@"rms_norm_q_weighted"];
        id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const uint32_t dimensions[] = {16, 32, 64, 128, 256, 512};
        const uint32_t heads = 3;
        for (unsigned shape = 0; shape < sizeof(dimensions) / sizeof(dimensions[0]); shape++) {
            uint32_t dim = dimensions[shape];
            id<MTLBuffer> q = [device newBufferWithLength:heads * dim * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> weights = [device newBufferWithLength:dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            float reference[3 * 512];
            uint16_t *w = weights.contents;
            for (uint32_t i = 0; i < dim; i++) w[i] = (i % 3 == 0) ? 0x3f80 : 0x3f00;
            for (int zero = 0; zero < 2; zero++) {
                float *values = q.contents;
                for (uint32_t h = 0; h < heads; h++) {
                    float sum = 0;
                    for (uint32_t i = 0; i < dim; i++) {
                        // Different magnitudes across 32-element slices expose partial reductions.
                        float value = zero ? 0 : ((int)(i % 13) - 6) * (float)(1 + i / 32) * (h + 1) * 0.07f;
                        values[h * dim + i] = value;
                        sum += value * value;
                    }
                    float inv = 1.0f / sqrtf(sum / dim + 1e-6f);
                    for (uint32_t i = 0; i < dim; i++)
                        reference[h * dim + i] = values[h * dim + i] * inv * bf16_value(w[i]);
                }
                float eps = 1e-6f;
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:q offset:0 atIndex:0];
                [encoder setBuffer:weights offset:0 atIndex:1];
                [encoder setBytes:&dim length:sizeof(dim) atIndex:2];
                [encoder setBytes:&eps length:sizeof(eps) atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(heads, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status == MTLCommandBufferStatusError) {
                    fprintf(stderr, "%s\n", command.error.description.UTF8String); return 1;
                }
                for (uint32_t i = 0; i < heads * dim; i++) {
                    if (!isfinite(values[i]) || fabsf(values[i] - reference[i]) > 2e-5f * (1 + fabsf(reference[i]))) {
                        fprintf(stderr, "dim=%u zero=%d index=%u: GPU=%g CPU=%g\n",
                                dim, zero, i, values[i], reference[i]);
                        return 1;
                    }
                }
            }
        }
        puts("Q normalization: six head sizes, multiple heads, and zero inputs match CPU reference.");
    }
    return 0;
}
