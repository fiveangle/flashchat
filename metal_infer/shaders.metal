/*
 * shaders.metal — Optimized Metal compute shaders for 4-bit quantized MoE inference
 *
 * Core operations:
 *   1. dequant_matvec_4bit: Naive 4-bit affine dequant matvec (reference)
 *   2. dequant_matvec_4bit_fast: SIMD-optimized with simd_sum reduction
 *   3. dequant_matvec_4bit_v3: Fully optimized — tiled threadgroup, vector loads,
 *      coalesced access, shared input cache. Target: <0.1ms per matmul.
 *   4. swiglu_fused / swiglu_fused_vec4: SwiGLU activation
 *   5. weighted_sum: combine expert outputs with routing weights
 *   6. rms_norm: RMS normalization
 *
 * Quantization format (MLX affine 4-bit, group_size=64):
 *   - Weights stored as uint32, each holding 8 x 4-bit values
 *   - Per-group scale and bias in bfloat16
 *   - Dequantized value = uint4_val * scale + bias
 *   - Groups of 64 elements share one (scale, bias) pair
 *
 * Matrix layout for expert projections (dimensions are model-dependent):
 *   gate_proj/up_proj: [moe_intermediate, hidden_dim/8] uint32
 *   down_proj: [hidden_dim, moe_intermediate/8] uint32
 *
 *   Scales/biases: [out_dim, in_dim/group_size] in bf16
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

inline uint16_t f32_to_bf16(float f) {
    return uint16_t(as_type<uint>(f) >> 16);
}


// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply (NAIVE — reference)
// ============================================================================

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    float acc = 0.0f;

    device const uint32_t* w_row = W_packed + tid * packed_cols;
    device const uint16_t* s_row = scales + tid * num_groups;
    device const uint16_t* b_row = biases + tid * num_groups;

    for (uint g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            for (uint n = 0; n < 8; n++) {
                uint nibble = (packed >> (n * 4)) & 0xF;
                float w_val = float(nibble) * scale + bias;
                acc += w_val * x[x_base + n];
            }
        }
    }

    out[tid] = acc;
}


// ============================================================================
// Kernel 1b: 4-bit dequant matvec — SIMD-optimized (legacy, kept for compat)
// ============================================================================

kernel void dequant_matvec_4bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            acc += (float((packed >>  0) & 0xF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x[x_base + 7];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}

kernel void dequant_matvec_8bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 4;
    uint packed_cols = in_dim / 4;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 4;

            acc += (float((packed >>  0) & 0xFF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  8) & 0xFF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >> 16) & 0xFF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 24) & 0xFF) * scale + bias) * x[x_base + 3];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}

// ============================================================================
// Fused gate+up+SwiGLU: reads x ONCE, computes silu(gate(x)) * up(x)
// Saves one input read + one kernel dispatch per expert
// ============================================================================
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],
    device const uint16_t* gate_b    [[buffer(2)]],
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],
    device const uint16_t* up_b      [[buffer(5)]],
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint16_t* gs = gate_s + tgid * num_groups;
    device const uint16_t* gb = gate_b + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint16_t* us = up_s   + tgid * num_groups;
    device const uint16_t* ub = up_b   + tgid * num_groups;
    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsc = bf16_to_f32(gs[g]), gbi = bf16_to_f32(gb[g]);
        float usc = bf16_to_f32(us[g]), ubi = bf16_to_f32(ub[g]);
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p*8 + i];
                ga += (float((gp>>(i*4))&0xF)*gsc+gbi)*xv;
                ua += (float((up>>(i*4))&0xF)*usc+ubi)*xv;
            }
        }
    }
    threadgroup float sg[32], su[32];
    float rg = simd_sum(ga), ru = simd_sum(ua);
    uint sl = lid%32, si = lid/32, ns = (tg_size+31)/32;
    if (sl==0) { sg[si]=rg; su[si]=ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (si==0 && sl<ns) {
        float vg=simd_sum(sg[sl]), vu=simd_sum(su[sl]);
        if (sl==0) out[tgid] = (vg/(1.0f+exp(-vg))) * vu;
    }
}

// ============================================================================
// Kernel 1c: FULLY OPTIMIZED 4-bit dequant matvec
// ============================================================================
//
// Design for M3 Max (40-core GPU, SIMD width 32):
//
// Strategy: Each threadgroup handles ROWS_PER_TG output rows.
//   - Threadgroup size = 256 (8 SIMD groups of 32)
//   - Each SIMD group handles one output row
//   - Within a SIMD group, 32 threads split the input dimension
//   - Each thread processes in_dim/32 input elements using vector loads
//   - Reduction via simd_sum (single instruction)
//
// Memory optimizations:
//   - Input vector x cached in threadgroup shared memory (loaded once)
//   - uint4 vector loads for weights (128 bits = 32 nibbles per load)
//   - float4 vector loads for x (128 bits = 4 floats per load)
//   - Coalesced weight reads: adjacent threads read adjacent uint4 vectors
//
// For gate/up_proj [1024, 4096]: 1024/8 = 128 threadgroups, 256 threads each
//   - 128 * 256 = 32768 threads across 40 cores = good occupancy
//   - Each thread processes 4096/32 = 128 input elements = 16 uint32 packed words
//     = 4 uint4 loads per thread per row
//
// For down_proj [4096, 1024]: 4096/8 = 512 threadgroups
//   - Each thread processes 1024/32 = 32 input elements = 4 uint32 packed words
//     = 1 uint4 load per thread per row

// Number of output rows per threadgroup = number of SIMD groups (256/32 = 8)
#define ROWS_PER_TG 8

kernel void dequant_matvec_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],     // which tile of rows
    uint lid    [[thread_position_in_threadgroup]],    // 0..255
    uint simd_lane  [[thread_index_in_simdgroup]],    // 0..31
    uint simd_group [[simdgroup_index_in_threadgroup]] // 0..7
) {
    // Which output row this SIMD group handles
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;      // uint32 columns per row
    uint num_groups  = in_dim / group_size;

    // ---- Cache input vector in threadgroup shared memory ----
    // Max in_dim = 4096, so we need 4096 floats = 16KB shared memory
    // This is well within the 32KB threadgroup memory limit on M3
    threadgroup float x_shared[4096];

    // Cooperative load: 256 threads load 4096 floats (16 per thread)
    // ALL threads must participate in this load + barrier, even if their
    // row is out of bounds. Early return before the barrier causes only
    // partial loading of x_shared, corrupting results for valid rows.
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now safe to bail out for out-of-bounds rows
    if (row >= out_dim) return;

    // ---- Pointer setup for this row ----
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // ---- Each lane processes a strided slice of the packed columns ----
    // Lane k processes columns: k, k+32, k+64, ...
    // This gives coalesced reads: adjacent lanes read adjacent uint32 words.

    float acc = 0.0f;

    // Process packed columns in strides of 32 (one per SIMD lane)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Determine which group this column belongs to
        // packed_per_group = group_size / 8 = 64 / 8 = 8
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        float xv0 = x_shared[x_base + 0];
        float xv1 = x_shared[x_base + 1];
        float xv2 = x_shared[x_base + 2];
        float xv3 = x_shared[x_base + 3];
        float xv4 = x_shared[x_base + 4];
        float xv5 = x_shared[x_base + 5];
        float xv6 = x_shared[x_base + 6];
        float xv7 = x_shared[x_base + 7];

        float partial = float((packed >>  0) & 0xF) * xv0;
        partial = fma(float((packed >>  4) & 0xF), xv1, partial);
        partial = fma(float((packed >>  8) & 0xF), xv2, partial);
        partial = fma(float((packed >> 12) & 0xF), xv3, partial);
        partial = fma(float((packed >> 16) & 0xF), xv4, partial);
        partial = fma(float((packed >> 20) & 0xF), xv5, partial);
        partial = fma(float((packed >> 24) & 0xF), xv6, partial);
        partial = fma(float((packed >> 28) & 0xF), xv7, partial);

        float sum_x = (xv0 + xv1) + (xv2 + xv3);
        sum_x += (xv4 + xv5) + (xv6 + xv7);

        acc = fma(scale, partial, fma(bias, sum_x, acc));
    }

    // ---- SIMD reduction: sum across 32 lanes ----
    float sum = simd_sum(acc);

    // Lane 0 writes the result
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

kernel void dequant_matvec_8bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 4;
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];

    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 4);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 4;

        float xv0 = x_shared[x_base + 0];
        float xv1 = x_shared[x_base + 1];
        float xv2 = x_shared[x_base + 2];
        float xv3 = x_shared[x_base + 3];

        float partial = float((packed >>  0) & 0xFF) * xv0;
        partial = fma(float((packed >>  8) & 0xFF), xv1, partial);
        partial = fma(float((packed >> 16) & 0xFF), xv2, partial);
        partial = fma(float((packed >> 24) & 0xFF), xv3, partial);

        float sum_x = (xv0 + xv1) + (xv2 + xv3);

        acc = fma(scale, partial, fma(bias, sum_x, acc));
    }

    float sum = simd_sum(acc);

    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1e: BATCHED (N=2) 4-bit dequant matmul — MTP draft/verify foundation.
// Reads each packed weight word ONCE and applies it to two input vectors,
// amortizing the (bandwidth-bound) weight read across both tokens. This is the
// whole premise of batched speculative verify on this engine.
// x0 is cached in threadgroup memory; x1 is read from device (small + hot).
// ============================================================================
kernel void dequant_matmul2_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x0         [[buffer(3)]],  // [in_dim]  token 0
    device float*          out0       [[buffer(4)]],  // [out_dim] token 0
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    device const float*    x1         [[buffer(8)]],  // [in_dim]  token 1
    device float*          out1       [[buffer(9)]],  // [out_dim] token 1
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x0[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);
        uint32_t packed = w_row[col];   // ONE weight read, shared by both tokens
        uint x_base = col * 8;

        for (uint n = 0; n < 8; n++) {
            float nib = float((packed >> (n * 4)) & 0xF);
            float xv0 = x_shared[x_base + n];
            float xv1 = x1[x_base + n];
            acc0 += fma(nib, scale * xv0, bias * xv0);
            acc1 += fma(nib, scale * xv1, bias * xv1);
        }
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        out0[row] = sum0;
        out1[row] = sum1;
    }
}

kernel void dequant_matmul2_8bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x0         [[buffer(3)]],
    device float*          out0       [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    device const float*    x1         [[buffer(8)]],
    device float*          out1       [[buffer(9)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 4;
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x0[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 4);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);
        uint32_t packed = w_row[col];
        uint x_base = col * 4;

        for (uint n = 0; n < 4; n++) {
            float q = float((packed >> (n * 8)) & 0xFF);
            float xv0 = x_shared[x_base + n];
            float xv1 = x1[x_base + n];
            acc0 += fma(q, scale * xv0, bias * xv0);
            acc1 += fma(q, scale * xv1, bias * xv1);
        }
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        out0[row] = sum0;
        out1[row] = sum1;
    }
}


// ============================================================================
// Kernel 1e-N: BATCHED (N-wide) 4-bit dequant matmul — depth-N MTP verify.
// Generalizes matmul2 to N input vectors (N<=8). Reads each packed weight word
// ONCE and applies it to all N tokens. Inputs/outputs are packed contiguously:
// X is [N][in_dim], OUT is [N][out_dim]. No threadgroup x cache (N*in_dim may
// exceed shared limits); the small X stays hot in L1/L2. The amortized bandwidth
// is the weight read, shared across all N tokens — better per-token as N grows.
// ============================================================================
kernel void dequant_matmulN_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    X          [[buffer(3)]],  // [N][in_dim]
    device float*          OUT        [[buffer(4)]],  // [N][out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         N          [[buffer(8)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc[8];
    for (uint n = 0; n < N; n++) acc[n] = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);
        uint32_t packed = w_row[col];   // ONE weight read, shared across all N
        uint x_base = col * 8;
        for (uint n = 0; n < N; n++) {
            device const float* xn = X + n * in_dim;
            for (uint k = 0; k < 8; k++) {
                float nib = float((packed >> (k * 4)) & 0xF);
                float xv = xn[x_base + k];
                acc[n] = fma(nib, scale * xv, fma(bias, xv, acc[n]));
            }
        }
    }
    for (uint n = 0; n < N; n++) {
        float s = simd_sum(acc[n]);
        if (simd_lane == 0) OUT[n * out_dim + row] = s;
    }
}

kernel void dequant_matmulN_8bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    X          [[buffer(3)]],
    device float*          OUT        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         N          [[buffer(8)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;
    uint packed_cols = in_dim / 4;
    uint num_groups  = in_dim / group_size;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc[8];
    for (uint n = 0; n < N; n++) acc[n] = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 4);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);
        uint32_t packed = w_row[col];
        uint x_base = col * 4;
        for (uint n = 0; n < N; n++) {
            device const float* xn = X + n * in_dim;
            for (uint k = 0; k < 4; k++) {
                float q = float((packed >> (k * 8)) & 0xFF);
                float xv = xn[x_base + k];
                acc[n] = fma(q, scale * xv, fma(bias, xv, acc[n]));
            }
        }
    }
    for (uint n = 0; n < N; n++) {
        float s = simd_sum(acc[n]);
        if (simd_lane == 0) OUT[n * out_dim + row] = s;
    }
}


// ============================================================================
// Kernel 1e-tiled: matmulN with X staged in threadgroup memory (any in_dim).
// ============================================================================
// v3 re-reads all of X from device once per output row: the 8 row-simdgroups in
// a threadgroup each stream the full activation vector. For wide projections
// (gate/up out_dim 17408) that X traffic dwarfs the weights. v5 (matvec) solved
// this with x_shared[4096] but can't handle in_dim>4096 (down_proj=17408). Here we
// TILE the contraction dim: cooperatively stage a tile of X into threadgroup memory
// (shared by all 8 rows), accumulate, advance. X device reads drop ~8x (once per
// threadgroup instead of once per row). Tiles align to group_size so scale/bias
// lookups stay simple. All threads must reach the barriers, so out-of-range rows
// participate in the loads and skip only the math/store.
#define MATMULN_TILE_PACK 64        // packed cols per tile = 512 inputs = 8 groups of 64
#define MATMULN_MAXN 8

kernel void dequant_matmulN_4bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    X          [[buffer(3)]],  // [N][in_dim]
    device float*          OUT        [[buffer(4)]],  // [N][out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         N          [[buffer(8)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 8;
    uint packed_per_group = group_size / 8;
    bool active = row < out_dim;

    threadgroup float xtile[MATMULN_MAXN][MATMULN_TILE_PACK * 8];

    // Clamp row for pointer math so inactive simdgroups still form valid (unused) addrs.
    uint srow = active ? row : 0;
    device const uint32_t* w_row = W_packed + (size_t)srow * packed_cols;
    device const uint16_t* s_row = scales + (size_t)srow * (in_dim / group_size);
    device const uint16_t* b_row = biases + (size_t)srow * (in_dim / group_size);

    float acc[MATMULN_MAXN];
    for (uint n = 0; n < N; n++) acc[n] = 0.0f;

    for (uint tile = 0; tile < packed_cols; tile += MATMULN_TILE_PACK) {
        uint tile_pack = min((uint)MATMULN_TILE_PACK, packed_cols - tile);
        uint tile_inputs = tile_pack * 8;
        // All 256 threads cooperatively stage this tile of X (every n) into threadgroup mem.
        for (uint n = 0; n < N; n++) {
            device const float* xn = X + (size_t)n * in_dim + tile * 8;
            for (uint i = lid; i < tile_inputs; i += 256) xtile[n][i] = xn[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint c = simd_lane; c < tile_pack; c += 32) {
                uint col = tile + c;
                uint g = col / packed_per_group;
                float scale = bf16_to_f32(s_row[g]);
                float bias  = bf16_to_f32(b_row[g]);
                uint32_t packed = w_row[col];
                uint xb = c * 8;
                for (uint n = 0; n < N; n++) {
                    threadgroup const float* xn = xtile[n];
                    for (uint k = 0; k < 8; k++) {
                        float nib = float((packed >> (k * 4)) & 0xF);
                        float xv = xn[xb + k];
                        acc[n] = fma(nib, scale * xv, fma(bias, xv, acc[n]));
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        for (uint n = 0; n < N; n++) {
            float s = simd_sum(acc[n]);
            if (simd_lane == 0) OUT[n * out_dim + row] = s;
        }
    }
}

kernel void dequant_matmulN_8bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    X          [[buffer(3)]],
    device float*          OUT        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         N          [[buffer(8)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 4;
    uint packed_per_group = group_size / 4;
    bool active = row < out_dim;

    threadgroup float xtile[MATMULN_MAXN][MATMULN_TILE_PACK * 8];

    uint srow = active ? row : 0;
    device const uint32_t* w_row = W_packed + (size_t)srow * packed_cols;
    device const uint16_t* s_row = scales + (size_t)srow * (in_dim / group_size);
    device const uint16_t* b_row = biases + (size_t)srow * (in_dim / group_size);

    float acc[MATMULN_MAXN];
    for (uint n = 0; n < N; n++) acc[n] = 0.0f;

    for (uint tile = 0; tile < packed_cols; tile += MATMULN_TILE_PACK) {
        uint tile_pack = min((uint)MATMULN_TILE_PACK, packed_cols - tile);
        uint tile_inputs = tile_pack * 4;
        for (uint n = 0; n < N; n++) {
            device const float* xn = X + (size_t)n * in_dim + tile * 4;
            for (uint i = lid; i < tile_inputs; i += 256) xtile[n][i] = xn[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint c = simd_lane; c < tile_pack; c += 32) {
                uint col = tile + c;
                uint g = col / packed_per_group;
                float scale = bf16_to_f32(s_row[g]);
                float bias  = bf16_to_f32(b_row[g]);
                uint32_t packed = w_row[col];
                uint xb = c * 4;
                for (uint n = 0; n < N; n++) {
                    threadgroup const float* xn = xtile[n];
                    for (uint k = 0; k < 4; k++) {
                        float q = float((packed >> (k * 8)) & 0xFF);
                        float xv = xn[xb + k];
                        acc[n] = fma(q, scale * xv, fma(bias, xv, acc[n]));
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        for (uint n = 0; n < N; n++) {
            float s = simd_sum(acc[n]);
            if (simd_lane == 0) OUT[n * out_dim + row] = s;
        }
    }
}


// ============================================================================
// Kernel 1e-ILP: matmulN with multiple rows per simdgroup (load-latency hiding).
// ============================================================================
// matmulN measured ~60 GB/s (1/8 of bandwidth) and the tiled-X experiment showed it
// is NOT memory-traffic bound — it's load-latency bound: each simdgroup processes one
// row, issuing w_row[col] then immediately consuming it in dependent FMAs, so the
// load latency stalls the lane. Fix: give each simdgroup ROWS_PER_SIMD rows. The
// per-row weight loads (w_row0[col], w_row1[col], ...) are independent, so the GPU
// keeps several in flight and overlaps their latency — classic ILP latency hiding,
// without needing more threadgroups/occupancy. Rows also share the (L2-hot) X reads.
// No threadgroup memory / no barriers, so out-of-range rows just skip their store.
#define ROWS_PER_SIMD 4

kernel void dequant_matmulN_4bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    X          [[buffer(3)]],  // [N][in_dim]
    device float*          OUT        [[buffer(4)]],  // [N][out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         N          [[buffer(8)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = (tgid * ROWS_PER_TG + simd_group) * ROWS_PER_SIMD;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    // Per-row weight/scale/bias pointers; clamp OOB rows so loads stay in-bounds.
    device const uint32_t* w_row[ROWS_PER_SIMD];
    device const uint16_t* s_row[ROWS_PER_SIMD];
    device const uint16_t* b_row[ROWS_PER_SIMD];
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        uint rr = row_base + r;
        uint srr = (rr < out_dim) ? rr : 0;
        w_row[r] = W_packed + (size_t)srr * packed_cols;
        s_row[r] = scales   + (size_t)srr * num_groups;
        b_row[r] = biases   + (size_t)srr * num_groups;
    }

    float acc[ROWS_PER_SIMD][MATMULN_MAXN];
    for (uint r = 0; r < ROWS_PER_SIMD; r++)
        for (uint n = 0; n < N; n++) acc[r][n] = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;
        // Issue the independent per-row weight loads up front so they overlap.
        uint32_t packed[ROWS_PER_SIMD];
        float scale[ROWS_PER_SIMD], bias[ROWS_PER_SIMD];
        for (uint r = 0; r < ROWS_PER_SIMD; r++) {
            packed[r] = w_row[r][col];
            scale[r]  = bf16_to_f32(s_row[r][g]);
            bias[r]   = bf16_to_f32(b_row[r][g]);
        }
        uint x_base = col * 8;
        for (uint n = 0; n < N; n++) {
            device const float* xn = X + (size_t)n * in_dim;
            float partial[ROWS_PER_SIMD];
            for (uint r = 0; r < ROWS_PER_SIMD; r++) partial[r] = 0.0f;
            float sum_x = 0.0f;
            for (uint k = 0; k < 8; k++) {
                float xv = xn[x_base + k];
                sum_x += xv;
                for (uint r = 0; r < ROWS_PER_SIMD; r++) {
                    float nib = float((packed[r] >> (k * 4)) & 0xF);
                    partial[r] = fma(nib, xv, partial[r]);
                }
            }
            for (uint r = 0; r < ROWS_PER_SIMD; r++) {
                acc[r][n] = fma(scale[r], partial[r], fma(bias[r], sum_x, acc[r][n]));
            }
        }
    }
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        uint rr = row_base + r;
        if (rr >= out_dim) continue;       // uniform across the simdgroup
        for (uint n = 0; n < N; n++) {
            float s = simd_sum(acc[r][n]);
            if (simd_lane == 0) OUT[n * out_dim + rr] = s;
        }
    }
}

kernel void dequant_matmulN_8bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    X          [[buffer(3)]],
    device float*          OUT        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         N          [[buffer(8)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = (tgid * ROWS_PER_TG + simd_group) * ROWS_PER_SIMD;
    uint packed_cols = in_dim / 4;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 4;

    device const uint32_t* w_row[ROWS_PER_SIMD];
    device const uint16_t* s_row[ROWS_PER_SIMD];
    device const uint16_t* b_row[ROWS_PER_SIMD];
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        uint rr = row_base + r;
        uint srr = (rr < out_dim) ? rr : 0;
        w_row[r] = W_packed + (size_t)srr * packed_cols;
        s_row[r] = scales   + (size_t)srr * num_groups;
        b_row[r] = biases   + (size_t)srr * num_groups;
    }

    float acc[ROWS_PER_SIMD][MATMULN_MAXN];
    for (uint r = 0; r < ROWS_PER_SIMD; r++)
        for (uint n = 0; n < N; n++) acc[r][n] = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;
        uint32_t packed[ROWS_PER_SIMD];
        float scale[ROWS_PER_SIMD], bias[ROWS_PER_SIMD];
        for (uint r = 0; r < ROWS_PER_SIMD; r++) {
            packed[r] = w_row[r][col];
            scale[r]  = bf16_to_f32(s_row[r][g]);
            bias[r]   = bf16_to_f32(b_row[r][g]);
        }
        uint x_base = col * 4;
        for (uint n = 0; n < N; n++) {
            device const float* xn = X + (size_t)n * in_dim;
            float partial[ROWS_PER_SIMD];
            for (uint r = 0; r < ROWS_PER_SIMD; r++) partial[r] = 0.0f;
            float sum_x = 0.0f;
            for (uint k = 0; k < 4; k++) {
                float xv = xn[x_base + k];
                sum_x += xv;
                for (uint r = 0; r < ROWS_PER_SIMD; r++) {
                    float q = float((packed[r] >> (k * 8)) & 0xFF);
                    partial[r] = fma(q, xv, partial[r]);
                }
            }
            for (uint r = 0; r < ROWS_PER_SIMD; r++) {
                acc[r][n] = fma(scale[r], partial[r], fma(bias[r], sum_x, acc[r][n]));
            }
        }
    }
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        uint rr = row_base + r;
        if (rr >= out_dim) continue;
        for (uint n = 0; n < N; n++) {
            float s = simd_sum(acc[r][n]);
            if (simd_lane == 0) OUT[n * out_dim + rr] = s;
        }
    }
}


// ============================================================================
// Kernel 1f: 4-bit dequant matvec with LUT (eliminates uint→float conversions)
// ============================================================================
// Instead of converting each nibble to float (expensive conversion instruction),
// pre-compute a 16-entry LUT per group: lut[v] = float(v) * scale + bias.
// Then inner loop is just: acc += lut[nibble] * x_shared[i] — pure math, no conversions.
// The LUT is recomputed every group_size/8 iterations (amortized).

#define ROWS_PER_TG_V5 8

kernel void dequant_matvec_4bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_V5 + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;
    uint prev_g = 0xFFFFFFFF;
    float lut[16];

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;

        // Rebuild LUT when group changes
        if (g != prev_g) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            for (uint v = 0; v < 16; v++) {
                lut[v] = float(v) * scale + bias;
            }
            prev_g = g;
        }

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += lut[(packed >>  0) & 0xF] * x_shared[x_base + 0];
        acc += lut[(packed >>  4) & 0xF] * x_shared[x_base + 1];
        acc += lut[(packed >>  8) & 0xF] * x_shared[x_base + 2];
        acc += lut[(packed >> 12) & 0xF] * x_shared[x_base + 3];
        acc += lut[(packed >> 16) & 0xF] * x_shared[x_base + 4];
        acc += lut[(packed >> 20) & 0xF] * x_shared[x_base + 5];
        acc += lut[(packed >> 24) & 0xF] * x_shared[x_base + 6];
        acc += lut[(packed >> 28) & 0xF] * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1d: FULLY OPTIMIZED with uint4 vector loads
// ============================================================================
//
// Same structure as v3 but uses uint4 loads (128-bit / 16 bytes) to maximize
// memory bandwidth per thread. Each uint4 = 4 uint32 = 32 nibbles.
//
// For gate/up (packed_cols=512): each thread processes 512/32 = 16 uint32
//   = 4 uint4 loads per thread
// For down (packed_cols=128): each thread processes 128/32 = 4 uint32
//   = 1 uint4 load per thread

kernel void dequant_matvec_4bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache input vector — ALL threads must participate before the barrier
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    // Pointers — cast to uint4 for vector loads
    device const uint4* w_row_v = (device const uint4*)(W_packed + row * packed_cols);
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    uint vec4_cols = packed_cols / 4;  // number of uint4 vectors per row

    float acc = 0.0f;

    // Each lane processes vec4_cols / 32 vectors (coalesced: adjacent lanes read adjacent uint4)
    for (uint vi = simd_lane; vi < vec4_cols; vi += 32) {
        uint4 packed4 = w_row_v[vi];

        // Each uint4 covers 4 * 8 = 32 input elements
        // Starting packed column index = vi * 4
        uint base_col = vi * 4;
        uint x_base = base_col * 8;  // starting input element

        // Process each of the 4 uint32 words in the uint4
        // Unroll all 4 words x 8 nibbles = 32 multiply-adds
        #pragma unroll
        for (uint w = 0; w < 4; w++) {
            uint32_t packed = packed4[w];
            uint col = base_col + w;
            uint g = col / (group_size / 8);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint xb = x_base + w * 8;
            acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[xb + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[xb + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[xb + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[xb + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[xb + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[xb + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[xb + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[xb + 7];
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1e: Multi-expert batched matvec
// ============================================================================
//
// Dispatch multiple experts simultaneously. The grid's Y dimension indexes
// the expert, so K experts' matmuls run as parallel threadgroups.
//
// Buffer layout: W_packed, scales, biases are arrays of K experts concatenated.
// x_inputs:  K input vectors concatenated [K * in_dim]
// out:       K output vectors concatenated [K * out_dim]
// expert_offsets: byte offset into W_packed buffer for each expert's weights
//                 (allows non-contiguous expert data in a shared buffer)

kernel void dequant_matvec_4bit_batched(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x_inputs   [[buffer(3)]],  // [K, in_dim]
    device float*          out        [[buffer(4)]],  // [K, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    // Per-expert offsets into the weight/scale/bias buffers (in elements)
    device const uint*     w_offsets  [[buffer(8)]],  // [K] offset in uint32 elements
    device const uint*     s_offsets  [[buffer(9)]],  // [K] offset in uint16 elements
    device const uint*     b_offsets  [[buffer(10)]], // [K] offset in uint16 elements
    constant uint&         num_row_tiles [[buffer(11)]], // ceil(out_dim / ROWS_PER_TG)
    uint tgid_flat [[threadgroup_position_in_grid]],  // linearized (row_tile + expert * num_row_tiles)
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // De-linearize: tgid_flat = row_tile + expert_k * num_row_tiles
    uint expert_k = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row = row_tile * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache this expert's input vector
    threadgroup float x_shared[4096];
    device const float* x_k = x_inputs + expert_k * in_dim;
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_k[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Point to this expert's weights
    device const uint32_t* w_row = W_packed + w_offsets[expert_k] + row * packed_cols;
    device const uint16_t* s_row = scales   + s_offsets[expert_k] + row * num_groups;
    device const uint16_t* b_row = biases   + b_offsets[expert_k] + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[x_base + 0];
        acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[x_base + 1];
        acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[x_base + 2];
        acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[x_base + 3];
        acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[x_base + 4];
        acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[x_base + 5];
        acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[x_base + 6];
        acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Vectorized SwiGLU: process 4 elements per thread
kernel void swiglu_fused_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up   [[buffer(1)]],
    device float4*       out  [[buffer(2)]],
    constant uint&       dim  [[buffer(3)]],  // original dim (must be multiple of 4)
    uint tid [[thread_position_in_grid]]
) {
    uint vec_dim = dim / 4;
    if (tid >= vec_dim) return;

    float4 g = gate[tid];
    float4 silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 2b: Batched SwiGLU for K experts
// ============================================================================

kernel void swiglu_fused_batched(
    device const float* gate [[buffer(0)]],  // [K * dim]
    device const float* up   [[buffer(1)]],  // [K * dim]
    device float*       out  [[buffer(2)]],  // [K * dim]
    constant uint&      dim  [[buffer(3)]],
    constant uint&      K    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = K * dim;
    if (tid >= total) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 3: Weighted sum of expert outputs
// ============================================================================

kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}


// ============================================================================
// Kernel 4: RMS Normalization
// ============================================================================

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}


// ============================================================================
// Kernel 4b: RMS Normalization with bf16 weights
// ============================================================================
// Same as rms_norm_apply but reads weights as bfloat16 (uint16_t) and
// converts to float32 inline. Used in the fused o_proj+norm+routing path
// where norm weights come directly from the mmap'd weight file (bf16).

kernel void rms_norm_apply_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],  // bf16 weights
    device const float*    sum_sq  [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant uint&         dim     [[buffer(4)]],
    constant float&        eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}


// ============================================================================
// Kernel 5: Residual add
// ============================================================================
// out[i] = a[i] + b[i]
// Used to fuse the residual connection into a GPU command buffer,
// eliminating a CPU round-trip between o_proj and routing.

kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}


// ============================================================================
// Kernel 6: Batched GPU attention scores (Q @ K^T, scaled) — all heads at once
// ============================================================================
//
// Computes scores[h, p] = sum_d(Q[h, d] * K[p, kv_h*head_dim + d]) * scale
// for all heads h in [0, num_heads) and positions p in [0, seq_len).
//
// Grid: linearized (pos + h * num_seq_tgs) — one threadgroup per (position, head).
// Each threadgroup of 256 threads reduces over head_dim=256.
//
// GQA mapping: kv_head = h / heads_per_kv (e.g. 16 query heads share 1 KV head)
//
// Output layout: scores[h * seq_stride + p] where seq_stride = MAX_SEQ_LEN

kernel void attn_scores_batched(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       scores     [[buffer(2)]],  // [num_heads, seq_stride]
    constant uint&      head_dim   [[buffer(3)]],  // 256
    constant uint&      kv_dim     [[buffer(4)]],  // 512
    constant uint&      seq_len    [[buffer(5)]],  // current seq length
    constant uint&      seq_stride [[buffer(6)]],  // MAX_SEQ_LEN
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant uint&      num_seq_tgs  [[buffer(9)]],  // = seq_len
    uint tgid  [[threadgroup_position_in_grid]],    // linearized: pos + h * num_seq_tgs
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += qh[d] * kp[d];
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 7: Batched softmax — one threadgroup per head
// ============================================================================

kernel void attn_softmax_batched(
    device float*    scores     [[buffer(0)]],  // [num_heads, seq_stride]
    constant uint&   seq_len    [[buffer(1)]],
    constant uint&   seq_stride [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],     // head index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device float* s = scores + tgid * seq_stride;

    // Pass 1: find max
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, s[i]);
    }
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float broadcast_max = -1e30f;
    if (lid == 0) broadcast_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = broadcast_max;

    // Pass 2: exp and sum
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    float simd_s = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float broadcast_sum = 0.0f;
    if (lid == 0) broadcast_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = broadcast_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = lid; i < seq_len; i += tg_size) {
        s[i] *= inv_sum;
    }
}


// ============================================================================
// Kernel 8: Batched attention value aggregation (scores @ V) — all heads
// ============================================================================
//
// For each head h: output[h*head_dim + d] = sum_p(scores[h*seq_stride+p] * V[p*kv_dim + kv_h*head_dim + d])
//
// Grid: linearized over (head_dim * num_heads) — one thread per (dimension, head).

kernel void attn_values_batched(
    device const float* scores   [[buffer(0)]],  // [num_heads, seq_stride]
    device const float* V_cache  [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       out      [[buffer(2)]],  // [num_heads, head_dim]
    constant uint&      head_dim  [[buffer(3)]],  // 256
    constant uint&      kv_dim    [[buffer(4)]],  // 512
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    uint tid [[thread_position_in_grid]]          // linearized: d + h * head_dim
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        acc += s[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Quantized-KV variants (FLASHCHAT_KV_QUANT=q8|q4). Dequant-on-read: the K/V
// cache holds symmetric per-(token, kv_head) quantized values plus an fp16
// scale per head vector. bits=8 -> signed int8; bits=4 -> two nibbles/byte at
// a +8 offset (low nibble = even index). Math matches the C kv_quantize_token.
// ============================================================================

kernel void attn_scores_quant(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const uchar* K_cache    [[buffer(1)]],  // packed q8/q4 K cache
    device float*       scores     [[buffer(2)]],
    constant uint&      head_dim   [[buffer(3)]],
    constant uint&      kv_dim     [[buffer(4)]],  // logical kv_dim (num_kv_heads*head_dim)
    constant uint&      seq_len    [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant float&     scale      [[buffer(7)]],
    constant uint&      heads_per_kv [[buffer(8)]],
    constant uint&      num_seq_tgs  [[buffer(9)]],
    device const half*  k_scale    [[buffer(10)]], // [seq, num_kv_heads]
    constant uint&      bits       [[buffer(11)]], // 8 or 4
    uint tgid  [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint num_kv_heads = kv_dim / head_dim;
    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    float ks = float(k_scale[pos * num_kv_heads + kv_h]);

    float acc = 0.0f;
    if (bits == 8) {
        device const char* kp = (device const char*)K_cache + pos * kv_dim + kv_h * head_dim;
        for (uint d = lid; d < head_dim; d += tg_size) {
            acc += qh[d] * (float(kp[d]) * ks);
        }
    } else {
        uint half_hd = head_dim >> 1;
        device const uchar* kp = K_cache + pos * (kv_dim >> 1) + kv_h * half_hd;
        for (uint d = lid; d < head_dim; d += tg_size) {
            uchar byte = kp[d >> 1];
            int nib = (d & 1) ? (byte >> 4) : (byte & 0xF);
            acc += qh[d] * (float(nib - 8) * ks);
        }
    }

    // SIMD reduction (identical to attn_scores_batched)
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}

kernel void attn_values_quant(
    device const float* scores   [[buffer(0)]],
    device const uchar* V_cache  [[buffer(1)]],  // packed q8/q4 V cache
    device float*       out      [[buffer(2)]],
    constant uint&      head_dim  [[buffer(3)]],
    constant uint&      kv_dim    [[buffer(4)]],
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    device const half*  v_scale   [[buffer(8)]], // [seq, num_kv_heads]
    constant uint&      bits      [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint num_kv_heads = kv_dim / head_dim;
    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    if (bits == 8) {
        device const char* V = (device const char*)V_cache;
        for (uint p = 0; p < seq_len; p++) {
            float vs = float(v_scale[p * num_kv_heads + kv_h]);
            acc += s[p] * (float(V[p * kv_dim + kv_h * head_dim + d]) * vs);
        }
    } else {
        uint half_hd = head_dim >> 1;
        for (uint p = 0; p < seq_len; p++) {
            float vs = float(v_scale[p * num_kv_heads + kv_h]);
            uchar byte = V_cache[p * (kv_dim >> 1) + kv_h * half_hd + (d >> 1)];
            int nib = (d & 1) ? (byte >> 4) : (byte & 0xF);
            acc += s[p] * (float(nib - 8) * vs);
        }
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 9: Sigmoid element-wise gate
// ============================================================================
// out[i] = x[i] * sigmoid(gate[i])

kernel void sigmoid_gate(
    device float*       x_out  [[buffer(0)]],  // [dim] in/out
    device const float* gate   [[buffer(1)]],  // [dim] gate values
    constant uint&      dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = 1.0f / (1.0f + exp(-gate[tid]));
    x_out[tid] = x_out[tid] * g;
}


// ============================================================================
// Kernel 9b: Fused flash attention for decode (single query token)
// ============================================================================
//
// Replaces attn_scores_batched + attn_softmax_batched + attn_values_batched +
// sigmoid_gate with a single kernel. Scores stay in registers/threadgroup
// (never written to global memory). Uses online softmax (log-sum-exp).
//
// One threadgroup per query head (num_attn_heads TGs), 256 threads each.
// Each TG loads Q for its head once, then iterates over the KV cache in blocks
// of BK=8 positions. K/V for the kv_head are loaded into threadgroup memory
// (shared between K and V since they're never needed simultaneously).
// Each simdgroup computes one score (dot product via simd_sum), the online
// softmax rescales the running max/sum and output accumulator, then V is
// loaded and accumulated. The sigmoid gate is fused into the final write.
//
// This eliminates the scores buffer round-trip (1MB of global memory traffic
// for seq_len=8192) and 3 encoder dispatch overheads vs the 3-kernel approach.
//
// Dispatch: grid = (num_attn_heads, 1, 1), 256 threads per TG.
// Requires: head_dim=256, fp32 KV cache (no quantization), seq_len >= 1.

#define FA_HD 256
#define FA_BK 8

kernel void flash_attn_fused_decode(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim] fp32
    device const float* V_cache    [[buffer(2)]],  // [max_seq, kv_dim]
    device const float* gate       [[buffer(3)]],  // [num_heads, head_dim]
    device float*       out        [[buffer(4)]],  // [num_heads, head_dim]
    constant uint&      kv_dim     [[buffer(5)]],  // num_kv_heads * head_dim
    constant uint&      seq_len    [[buffer(6)]],
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(HD)
    constant uint&      heads_per_kv [[buffer(8)]],
    uint h     [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]]
) {
    if (tid >= FA_HD) return;

    constexpr int FA_STRIDE = FA_HD + 8;

    threadgroup float Qs[FA_STRIDE];
    threadgroup float KVs[FA_BK * FA_STRIDE];
    threadgroup float shared_scores[FA_BK];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * FA_HD;

    float o_acc = 0.0f;

    if (tid == 0) {
        shared_max = -1e30f;
        shared_sum = 0.0f;
    }

    Qs[tid] = qh[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kb = 0; kb < seq_len; kb += FA_BK) {
        uint block_size = min((uint)FA_BK, seq_len - kb);

        for (uint i = 0; i < FA_BK; i++) {
            uint pos = kb + i;
            if (pos < seq_len) {
                KVs[i * FA_STRIDE + tid] = K_cache[pos * kv_dim + kv_h * FA_HD + tid];
            } else {
                KVs[i * FA_STRIDE + tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint sg_id = tid / 32;
        uint sg_lane = tid % 32;

        if (sg_id < FA_BK) {
            float partial = 0.0f;
            for (uint d = 0; d < FA_HD / 32; d++) {
                uint elem = sg_lane + d * 32;
                partial += Qs[elem] * KVs[sg_id * FA_STRIDE + elem];
            }
            float score = simd_sum(partial) * scale;
            if (sg_lane == 0) {
                shared_scores[sg_id] = (sg_id < block_size) ? score : -1e30f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_max = -1e30f;
        for (uint i = 0; i < block_size; i++) {
            block_max = max(block_max, shared_scores[i]);
        }

        float running_max = shared_max;
        float new_max = max(running_max, block_max);
        float factor = exp(running_max - new_max);

        float running_sum = shared_sum * factor;
        o_acc *= factor;

        for (uint i = 0; i < FA_BK; i++) {
            uint pos = kb + i;
            if (pos < seq_len) {
                KVs[i * FA_STRIDE + tid] = V_cache[pos * kv_dim + kv_h * FA_HD + tid];
            } else {
                KVs[i * FA_STRIDE + tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < block_size; i++) {
            float es = exp(shared_scores[i] - new_max);
            running_sum += es;
            o_acc += es * KVs[i * FA_STRIDE + tid];
        }

        if (tid == 0) {
            shared_max = new_max;
            shared_sum = running_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = 1.0f / shared_sum;
    float g = 1.0f / (1.0f + exp(-gate[h * FA_HD + tid]));
    out[h * FA_HD + tid] = o_acc * inv_sum * g;
}


// ============================================================================
// Kernel 9c: GPU RoPE (rotary position embedding) for decode
// ============================================================================
//
// Applies RoPE in-place to Q (and optionally K) using a precomputed frequency
// table. Uses metal::fast::cos/sin (SFU instructions, ~1 cycle) instead of CPU
// cosf/sinf. The frequency table (inv_freq * pos) is precomputed once per
// position on CPU, so the GPU kernel only does cos/sin + 2 FMAs per pair.
//
// MLX-default pairing: (x[i], x[i + half_dim]) where half_dim = rotary_dim / 2.
// Only the first rotary_dim dimensions of each head are rotated; the rest
// pass through unchanged.
//
// Dispatch: grid = (ceil(half_dim/32), 1, 1), threadsPerThreadgroup = 32.
// Each thread handles one frequency pair across all Q heads (and K if apply_k=1).

kernel void rope_apply(
    device float*       q          [[buffer(0)]],  // [num_heads, head_dim] in/out
    device float*       k          [[buffer(1)]],  // [num_kv_heads, head_dim] in/out (or NULL)
    device const float* freq_table [[buffer(2)]],  // [half_dim] precomputed: inv_freq[i] * pos
    constant uint&      num_heads  [[buffer(3)]],
    constant uint&      num_kv_heads [[buffer(4)]],
    constant uint&      head_dim   [[buffer(5)]],
    constant uint&      rotary_dim [[buffer(6)]],
    constant uint&      apply_k    [[buffer(7)]],  // 1 = apply RoPE to K, 0 = Q only
    uint tid [[thread_position_in_grid]]
) {
    uint half_dim = rotary_dim / 2;
    if (tid >= half_dim) return;

    float angle = freq_table[tid];
    float cos_a = metal::fast::cos(angle);
    float sin_a = metal::fast::sin(angle);

    for (uint h = 0; h < num_heads; h++) {
        uint base = h * head_dim + tid;
        uint pair = base + half_dim;
        float q0 = q[base];
        float q1 = q[pair];
        q[base] = q0 * cos_a - q1 * sin_a;
        q[pair] = q0 * sin_a + q1 * cos_a;
    }

    if (apply_k) {
        for (uint h = 0; h < num_kv_heads; h++) {
            uint base = h * head_dim + tid;
            uint pair = base + half_dim;
            float k0 = k[base];
            float k1 = k[pair];
            k[base] = k0 * cos_a - k1 * sin_a;
            k[pair] = k0 * sin_a + k1 * cos_a;
        }
    }
}


// ============================================================================
// Kernel 10: GatedDeltaNet linear attention step (single token, all heads)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence for autoregressive generation:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// Dispatch: 64 threadgroups (one per v-head), 128 threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [64 * 128 * 128] float = 4MB total, persisted across tokens.
// k-head sharing: 4 v-heads share 1 k-head (64 v-heads / 16 k-heads).

kernel void gated_delta_net_step(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < 128; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        out_val += state[state_base + ki] * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 11: Conv1d depthwise step (single token, incremental inference)
// ============================================================================
//
// Depthwise 1D convolution for one new input token:
//   output[c] = sum_k(history[k][c] * weight[c][k]) + input[c] * weight[c][3]
//   then SiLU activation: output[c] = output[c] / (1 + exp(-output[c]))
//
// After computing, shifts the history buffer left and appends the new input.
//
// Weight layout: [channels * kernel_size] bf16, weight[c * kernel_size + k]
// Conv state layout: [(kernel_size-1) * channels] row-major, state[k * channels + c]
// kernel_size = 4 (hardcoded), so 3 history slots + 1 new input.
//
// Dispatch: conv_dim threads (12288), one per channel.

kernel void conv1d_step(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,        // [conv_dim] current input
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [conv_dim] convolution output
    constant uint &conv_dim,          // = 12288
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    // Convolution: dot product of history + new input with weights
    // weight layout: weight[c * 4 + k] for channel c, position k
    uint w_base = idx * 4;
    float acc = 0.0f;

    // 3 history slots (k=0,1,2)
    acc += conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[w_base + 2]);

    // New input (k=3)
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[w_base + 3]);

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history: move slots 1,2 -> 0,1, append input at slot 2
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors
// ============================================================================
// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
// Normalize each head independently, then scale by 1/sqrt(key_dim)^2 for q, 1/sqrt(key_dim) for k
// Dispatch: num_k_heads threadgroups, key_dim threads each

kernel void rms_norm_qk(
    device float *q,              // [num_k_heads * key_dim] in/out
    device float *k,              // [num_k_heads * key_dim] in/out
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * key_dim;

    // RMS norm for q
    threadgroup float q_sum_sq;
    if (tid == 0) q_sum_sq = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qval = (tid < key_dim) ? q[base + tid] : 0;
    // Use threadgroup atomic add for sum of squares
    float q_sq_local = qval * qval;
    // Simple reduction: thread 0 accumulates (key_dim=128, fits in one pass)
    threadgroup float q_partial[128];
    q_partial[tid] = q_sq_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += q_partial[i];
        q_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(q_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;  // q gets extra scale
    }

    // RMS norm for k
    threadgroup float k_sum_sq;
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    threadgroup float k_partial[128];
    k_partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += k_partial[i];
        k_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(k_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
// Dispatch: num_v_heads threads (64)

kernel void compute_decay_beta(
    device const float *alpha_out,   // [num_v_heads] from projection
    device const float *beta_out,    // [num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (persistent)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [num_v_heads] output
    device float *beta_gate,         // [num_v_heads] output
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per v-head: normalize values, gate with z, scale with weight
// Dispatch: num_v_heads threadgroups, value_dim threads each

kernel void gated_rms_norm(
    device const float *values,       // [num_v_heads * value_dim] delta-net output
    device const float *z,            // [num_v_heads * value_dim] gate values
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared across heads)
    device float *output,             // [num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;

    // RMS norm reduction
    threadgroup float partial[128];
    partial[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < value_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}


// ============================================================================
// Kernel 12: MoE combine + residual + shared expert gate (fused)
// ============================================================================
// Fused operation for CMD3 GPU-side combine:
//   hidden[i] = h_mid[i] + sum_k(expert_weight[k] * expert_out[k][i])
//               + sigmoid(shared_gate_score) * shared_out[i]
//
// All 8 expert output buffers are always bound (unused ones have weight=0).
// This avoids variable buffer bindings and keeps the dispatch simple.
//
// Dispatch: (dim + 255) / 256 threadgroups, 256 threads each.

// Combine up to MAX_K=16 expert outputs (must match model_config.h MAX_K). Buffers 3..18 are
// the 16 expert outputs; params (buffer 19) holds weights[0..15] then shared_gate_score at [16].
// Unused experts have weight 0 (host zeroes them), so K<16 is handled by the K>n guards.
kernel void moe_combine_residual(
    device const float* h_mid       [[buffer(0)]],   // [dim]
    device const float* shared_out  [[buffer(1)]],   // [dim]
    device float*       hidden_out  [[buffer(2)]],   // [dim] output
    device const float* e0  [[buffer(3)]],  device const float* e1  [[buffer(4)]],
    device const float* e2  [[buffer(5)]],  device const float* e3  [[buffer(6)]],
    device const float* e4  [[buffer(7)]],  device const float* e5  [[buffer(8)]],
    device const float* e6  [[buffer(9)]],  device const float* e7  [[buffer(10)]],
    device const float* e8  [[buffer(11)]], device const float* e9  [[buffer(12)]],
    device const float* e10 [[buffer(13)]], device const float* e11 [[buffer(14)]],
    device const float* e12 [[buffer(15)]], device const float* e13 [[buffer(16)]],
    device const float* e14 [[buffer(17)]], device const float* e15 [[buffer(18)]],
    device const float* params      [[buffer(19)]],  // [18]: weights[0..15], shared_gate_score at [16]
    constant uint&      dim         [[buffer(20)]],
    constant uint&      K           [[buffer(21)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float shared_gate = 1.0f / (1.0f + exp(-params[16]));  // sigmoid(shared_gate_score)

    float moe = 0.0f;
    if (K > 0)  moe += params[0]  * e0[tid];
    if (K > 1)  moe += params[1]  * e1[tid];
    if (K > 2)  moe += params[2]  * e2[tid];
    if (K > 3)  moe += params[3]  * e3[tid];
    if (K > 4)  moe += params[4]  * e4[tid];
    if (K > 5)  moe += params[5]  * e5[tid];
    if (K > 6)  moe += params[6]  * e6[tid];
    if (K > 7)  moe += params[7]  * e7[tid];
    if (K > 8)  moe += params[8]  * e8[tid];
    if (K > 9)  moe += params[9]  * e9[tid];
    if (K > 10) moe += params[10] * e10[tid];
    if (K > 11) moe += params[11] * e11[tid];
    if (K > 12) moe += params[12] * e12[tid];
    if (K > 13) moe += params[13] * e13[tid];
    if (K > 14) moe += params[14] * e14[tid];
    if (K > 15) moe += params[15] * e15[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}


// ============================================================================
// Kernel 13: BF16 matrix-vector multiply (for MTP head and other BF16 paths)
// ============================================================================
// Simple row-wise matvec over native BF16 weights.
// Weights are stored as uint16_t bfloat16 values in row-major order.
// Dispatch: (out_dim + 255) / 256 threadgroups, 256 threads each.

kernel void bf16_matvec(
    device const uint16_t* W [[buffer(0)]],
    device const float*    x [[buffer(1)]],
    device float*          out [[buffer(2)]],
    constant uint&         out_dim [[buffer(3)]],
    constant uint&         in_dim  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;
    float sum = 0.0f;
    device const uint16_t* row = W + tid * in_dim;
    for (uint j = 0; j < in_dim; j++) {
        sum += bf16_to_f32(row[j]) * x[j];
    }
    out[tid] = sum;
}
