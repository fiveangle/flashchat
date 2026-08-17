# 07 — Fuse CMD1+CMD2 on GPU-linear layers; GPU-resident dataflow

Branch: `exp/07-cmd-fuse-linear` (from main@d507059)

## Hypothesis

Per linear layer, decode does two full GPU commit+waitUntilCompleted round
trips (CMD1, CMD2) plus CPU readback/upload ping-pong, even though with
gpu_linear_attn + prev_gpu_combined there is NO CPU dependency between CMD1
(attn projections + linear attention chain) and CMD2 (o_proj + norm +
routing): every CMD2 input is already GPU-resident (batch_out[6], weights).
Each round trip + finalize costs ~0.1ms/layer x 30 linear layers.

Stage A (scheduling only, byte-identical): when the layer takes the
prev_gpu_combined fast path AND gpu_linear_attn, append CMD2's encoders to
CMD1's command buffer; single commit+wait. Env: FLASHCHAT_FUSE_LINEAR
(default 0; 1 = enable).

Stage B (dataflow): bind GPU-resident sources instead of CPU round trips:
- residual_add reads buf_moe_hidden (== the uploaded residual bytes)
- expert input reads buf_input (== h_post upload)
- shared gate/up read batch_out[1]/[2] (== shared_gate/up uploads)
- skip per-layer finalize readback (hidden needed only at token end)
All same-bytes rebindings -> temp-0 outputs must remain byte-identical.

## Method

Interleaved A/B quick-bench; byte-identity via temp-0 diff OFF vs A vs B.

## Observations

(pending)

## Verdict

(pending)
