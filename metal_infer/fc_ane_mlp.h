// ANE MoE-expert MLP (ported from ds4-ssd ds4_ane_mlp_int8w) — int8/fp16 SwiGLU
// MLP evaluated on the Apple Neural Engine via the private AppleNeuralEngine
// framework. Production path is mode 6 (i8w-i8x-tiled-fused).

#ifndef FC_ANE_MLP_H
#define FC_ANE_MLP_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fc_ane_mlp_int8w_ctx fc_ane_mlp_int8w_ctx;

fc_ane_mlp_int8w_ctx *fc_ane_mlp_int8w_create(int H, int I, int B, float w_scale, float x_scale);
fc_ane_mlp_int8w_ctx *fc_ane_mlp_fp16w_create(int H, int I, int B);
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_fp16x_create(int H, int I, int B, float w_scale);
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_i8x_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_i8x_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_i8x_gateup_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_i8x_tiled_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_i8x_tiled_fused_routed_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
/* Tiled-fused with int8 output (B*H bytes instead of B*H*2). */
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_i8x_tiled_fused_i8out_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
/* Single-call fused fp16-weight MLP lowered as conv2d-1x1 (ANE-native pattern). */
fc_ane_mlp_int8w_ctx *fc_ane_mlp_fp16w_fused_conv_create(int H, int I, int B);

/* Constexpr-weight version: weights are baked into the compiled MIL via a
 * side-loaded fp16 blob file.  Caller passes weights in conv-native [O, I]
 * layout (Wg/Wu: [I, H], Wd: [H, I] — both ggml's native ordering for these
 * tensors).  Only X is uploaded per eval, only Y is read. */
fc_ane_mlp_int8w_ctx *fc_ane_mlp_fp16w_constexpr_create(int H, int I, int B,
                                                          const uint16_t *Wgate_OI,
                                                          const uint16_t *Wup_OI,
                                                          const uint16_t *Wdown_OI);

bool fc_ane_mlp_int8w_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    const uint16_t *route_f16,
    uint16_t *output_f16);

bool fc_ane_mlp_fp16w_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wgate_f16,
    const uint16_t *Wup_f16,
    const uint16_t *Wdown_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Single-call fused fp16w eval using the conv2d-1x1 MIL.  Same I/O contract as
 * fc_ane_mlp_fp16w_eval (Wg/Wu [H,I], Wd [I,H], input [B,H], output [B,H]). */
bool fc_ane_mlp_fp16w_fused_conv_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wgate_f16,
    const uint16_t *Wup_f16,
    const uint16_t *Wdown_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Constexpr-weight conv eval.  No weight arguments — they're already in the
 * compiled model.  Input [B, H], output [B, H], both fp16. */
bool fc_ane_mlp_fp16w_constexpr_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Linear two-stage constexpr conv2d-1x1 (no activation) — for LoRA-style
 * projections like DSv4 attention output.  Wa [I, H], Wb [H, I] (ggml-native
 * [O, I] orientation matches conv weight layout).  Input [B, H], output [B, H].
 * Mode 10 ctx with weights baked into MIL via BLOBFILE constexpr. */
fc_ane_mlp_int8w_ctx *fc_ane_mlp_fp16w_linear_constexpr_create(
    int H, int I, int B,
    const uint16_t *Wa_OI,
    const uint16_t *Wb_OI);

bool fc_ane_mlp_fp16w_linear_constexpr_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Linear two-matmul fp16w eval: input · Wa → mid · Wb → output, no
 * activation between.  Reuses a mode-1 (fp16w split) ctx created with the
 * matching shape.  Wa is fed via the gate model, Wb via the down model.
 * Shapes: input [B, H], Wa [H, I], Wb [I, H], output [B, H]. */
bool fc_ane_mlp_fp16w_linear_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wa_f16,
    const uint16_t *Wb_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* int8-weight version of the above.  Uses a mode-2 (i8w-fp16x) ctx — the per-
 * tensor w_scale was baked in at ctx create time.  Weight upload bytes are
 * half the fp16 path (Wa+Wb together = H*I + I*H bytes int8 instead of fp16). */
bool fc_ane_mlp_i8w_fp16x_linear_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t   *Wa_i8,
    const int8_t   *Wb_i8,
    const uint16_t *input_f16,
    uint16_t       *output_f16);

/* Dedicated create for the linear-eval int8 path that lets gate and down have
 * DIFFERENT w_scales (mode-2 create_common uses one scale for both). */
fc_ane_mlp_int8w_ctx *fc_ane_mlp_i8w_fp16x_linear_create(int H, int I, int B,
                                                           float a_scale,
                                                           float b_scale);

/* int8 + per-output-channel scale variant of the linear constexpr conv path.
 * Weights baked into MIL via constexpr_blockwise_shift_scale (variant A in
 * INT4_MATMUL_ANE_WORKFLOW.md).  Wa_q/Wb_q [O, I] int8 row-major; Wa_off/Wb_off
 * [O] int8 (typically all zero for symmetric); Wa_scale/Wb_scale [O] fp16.
 * Mode 11.  X [B, H] in, Y [B, H] out. */
fc_ane_mlp_int8w_ctx *fc_ane_mlp_int8w_linear_constexpr_create(
    int H, int I, int B,
    const int8_t   *Wa_q_OI,
    const int8_t   *Wa_off_O,
    const uint16_t *Wa_scale_f16_O,
    const int8_t   *Wb_q_OI,
    const int8_t   *Wb_off_O,
    const uint16_t *Wb_scale_f16_O);

bool fc_ane_mlp_int8w_linear_constexpr_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const uint16_t *input_f16,
    uint16_t       *output_f16);

/* Same constexpr int8-weight conv path as mode 11, but X is supplied as int8
 * and dequantized inside the ANE graph with x_scale.  Output remains fp16.
 * Mode 12. */
fc_ane_mlp_int8w_ctx *fc_ane_mlp_int8w_i8x_linear_constexpr_create(
    int H, int I, int B,
    float x_scale,
    const int8_t   *Wa_q_OI,
    const int8_t   *Wa_off_O,
    const uint16_t *Wa_scale_f16_O,
    const int8_t   *Wb_q_OI,
    const int8_t   *Wb_off_O,
    const uint16_t *Wb_scale_f16_O);

bool fc_ane_mlp_int8w_i8x_linear_constexpr_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *input_i8,
    uint16_t     *output_f16);

/* Attach N external input IOSurfaces to an existing mode-11/mode-12 ctx —
 * creates one ANE request per IOSurface, all sharing the ctx's existing io_out.
 * After this, eval_at_chunk dispatches via the request for the requested
 * chunk index, and ANE reads the matching external IOSurface (no per-call
 * write_surface needed).  Caller owns the IOSurfaces' lifetime. */
#ifdef __OBJC__
#import <IOSurface/IOSurface.h>
bool fc_ane_mlp_int8w_linear_constexpr_attach_chunks(
    fc_ane_mlp_int8w_ctx *ctx,
    const IOSurfaceRef    *chunk_input_iosurfaces,
    int                    n_chunks);
#endif

bool fc_ane_mlp_int8w_linear_constexpr_eval_at_chunk(
    fc_ane_mlp_int8w_ctx *ctx,
    int                    chunk_idx,
    uint16_t              *output_f16);

bool fc_ane_mlp_i8w_fp16x_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const uint16_t *input_f16,
    uint16_t *output_f16);

bool fc_ane_mlp_i8w_i8x_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

bool fc_ane_mlp_i8w_i8x_fused_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

bool fc_ane_mlp_i8w_i8x_gateup_fused_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

bool fc_ane_mlp_i8w_i8x_tiled_fused_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

bool fc_ane_mlp_i8w_i8x_tiled_fused_routed_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    const uint16_t *route_f16,
    uint16_t *output_f16);

/* Mode 7: int8 output. Same i/o as tiled_fused_eval except output is int8. */
bool fc_ane_mlp_i8w_i8x_tiled_fused_i8out_eval(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    int8_t       *output_i8);

/* Zero-copy producer support (mode 6): direct pointers into the ctx's input
 * IOSurfaces (which: 0=gate 1=up 2=down 3=x) so a GPU kernel can write the
 * int8 operands in place, plus an eval that skips all operand writes. Output
 * is read via fc_ane_mlp_int8w_lock_output_f16. Caller synchronizes: never
 * produce into a ctx whose eval is in flight. */
void *fc_ane_mlp_operand_base(fc_ane_mlp_int8w_ctx *ctx, int which, size_t *bytes);
size_t fc_ane_mlp_operand_alloc_size(fc_ane_mlp_int8w_ctx *ctx, int which);
bool fc_ane_mlp_i8w_i8x_tiled_fused_eval_prewritten(fc_ane_mlp_int8w_ctx *ctx);

bool fc_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8);

/* Variant that skips re-writing the weight IOSurfaces: assumes a prior call
 * has already populated them with the desired expert's gate/up/down weights.
 * Only the input is written and the output is read. Used to measure how much
 * of per-call wall time is spent on weight upload vs. ANE evaluate itself. */
bool fc_ane_mlp_i8w_i8x_tiled_fused_eval_xonly(
    fc_ane_mlp_int8w_ctx *ctx,
    const int8_t *input_i8,
    uint16_t *output_f16);

const uint16_t *fc_ane_mlp_int8w_lock_output_f16(fc_ane_mlp_int8w_ctx *ctx, uint64_t *elems);
void fc_ane_mlp_int8w_unlock_output(fc_ane_mlp_int8w_ctx *ctx);

void fc_ane_mlp_int8w_quant_stats_reset(void);
void fc_ane_mlp_int8w_quant_stats(uint64_t *hidden_values,
                                   uint64_t *hidden_saturated,
                                   float *hidden_abs_max);

void fc_ane_mlp_int8w_destroy(fc_ane_mlp_int8w_ctx *ctx);

int fc_ane_mlp_int8w_H(const fc_ane_mlp_int8w_ctx *ctx);
int fc_ane_mlp_int8w_I(const fc_ane_mlp_int8w_ctx *ctx);
int fc_ane_mlp_int8w_B(const fc_ane_mlp_int8w_ctx *ctx);
float fc_ane_mlp_int8w_scale(const fc_ane_mlp_int8w_ctx *ctx);
float fc_ane_mlp_int8w_x_scale(const fc_ane_mlp_int8w_ctx *ctx);
float fc_ane_mlp_int8w_mid_scale(const fc_ane_mlp_int8w_ctx *ctx);
int fc_ane_mlp_int8w_mode(const fc_ane_mlp_int8w_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
