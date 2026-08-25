# 17 — Fuse CMD1+CMD2 on full-attention layers

Date: 2026-08-24  
Status: **NOT STARTED**

## Hypothesis

`fuse_cmd12` only fires when `gpu_linear_attn_will_run`. Full-attn layers
(~1/4 of layers at interval 4) still do:

CMD1 commit+wait → CPU RoPE/KV append → CMD2 commit+wait.

GPU attention already encodes into CMD2 when `kv->len >= 32`, but the
extra RTT remains. Extending fuse (or a single cmd buffer from QKV through
o_proj/router) could save ~1 wait per full-attn layer.

## Constraints

- KV append and RoPE must complete before scores — either GPU RoPE+append
  (partially present) or keep a thin CPU section and fuse only the pure-GPU
  spans.
- Must stay byte-identical at temp 0.

## Expected

~10 full-attn layers × ~0.3–0.6 ms saved → **+3–8%** decode if fully fused.

## Verdict

Pending. Medium priority after pin/lm_head/split-io.
