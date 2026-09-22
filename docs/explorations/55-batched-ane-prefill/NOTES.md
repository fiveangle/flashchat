# 55 — Batched and ANE prefill

Date: 2026-07-06 to 2026-07-08. Registered 2026-09-22.
Commits: `ae0544e` (Phase 1), `6c04132` (Phase 2), `2a34a7e` (Phase 3), `67bec6b` (Phase 4).
Status: **Closed.** Shipped before it had an exploration ID.
Verdict: **Keep, default on.**

## Hypothesis

Prompt processing ran one decode-path forward per token, reading every routed
expert once per token. Processing prompts in chunks reads each expert once per
chunk, and offloading the expert MLP to the Neural Engine adds throughput on top.

## What shipped

- **Phase 1, `ae0544e`:** one KV writer for decode, the CPU reference, and batched
  prefill; allocation sized to the context window instead of the 1M design maximum
  (root cause of a 2026-07-06 swap storm).
- **Phase 2, `6c04132`:** chunked batched prefill through `batched_layer_forward_N`.
  Chunk-scale MoE gathers each expert's routed tokens and runs them as matmulN
  GEMMs, so each expert's weights are read once per chunk. Fixed a latent matmulN
  overrun for N > 8 exposed by chunking.
- **Phase 3, `2a34a7e`:** ANE expert-MLP offload with zero-copy operands written by
  GPU producer kernels and two ping-ponging ANE contexts.
- **Phase 4, `67bec6b`:** batch prefill and ANE prefill default on, chunk 1024,
  schema v8, and a calm permanent fallback to the GPU path on any ANE failure, so
  correctness never depends on the ANE.

## Measurement

The implementing commits' own figures, on 35B with a q8 context cache, isolated:
prompt TTFT **~2.4x faster (1966–2932 ms → 802–1124 ms)** with decode flat; a
1948-token prompt went from **12 to 61 tok/s** on the GPU path and **80 tok/s** on
the ANE. Greedy byte-match was verified batched-versus-per-token and
ANE-versus-fallback, at q8 and off.

These are not `bench_api` A/Bs. The 2026-07-07/08 watchdog rows cannot re-derive
them: 10 established that the fast July rows came from an unrecoverable dirty tree.

Phase 4 also records two sub-mechanisms **dropped by measurement**: fp16 chunk
staging (≤5%, now 52) and pread double-buffering (prefill I/O is ~12% of prefill
time, recorded under 06).

## Prior attempts

Upstream `3253ef2` (Dan Woods, 2026-03-18) cut TTFT 5.6→2.6 s on a 16-token prompt
by skipping expert work for every prompt token but the last. That approximates
later layers' hidden states, so it is a different mechanism from this one. See 56.

## Conclusion

Keep. The 00 baseline configuration has this enabled, so every measurement from
00 onward already includes it; 00 is a decode baseline and does not quantify this
prefill win. 11 tunes this path's chunk-size crossover and 10 investigated an
apparent regression in it.
