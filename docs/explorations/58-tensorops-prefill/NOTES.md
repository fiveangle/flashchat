# 58 — Metal 4 TensorOps for prefill matmuls

Date: 2026-06-04. Registered 2026-09-22.
Commit: `f715591` (probe harness and engine probes).
Status: **Parked.** Probed, never wired into production.
Verdict: **Blocked** on resource-layout limits; no throughput result.

## Hypothesis

Metal 4 MPP/TensorOps `matmul2d` could replace the hand-written
`gpu_dequant_matmulN` / `gpu_dequant_matmulN_batch` kernels for multi-token
prefill, with the existing Metal 3 kernels as the fallback on older toolchains and
devices.

## What was built

`f715591` added engine-side probes and three test scripts:
`tests/bench_mpp_tensorops.sh` (571 lines), `tests/test_mpp_tensorops_compile.sh`,
and `tests/test_mpp_tensorops_runtime.sh`. The benchmark generates a group-wise
TensorOps kernel over configurable output width, group count, and chunking, and
skips cleanly when the SDK headers or macOS 26 runtime APIs are missing. All three
scripts were removed in `71fdcb7` (2026-06-12) and remain in history at `f715591`.

## Findings

The probe findings lived in AGENTS.md ("Dense Prefill TensorOps Direction") until
`8fa6ebf` removed that section on 2026-09-17. They are preserved here:

- MLX affine quantization semantics must be kept: TensorOps must process 64-wide
  quant groups and apply each group's scale and bias with a per-token group sum. A
  single full-K raw uint4 matmul is not equivalent.
- BF16 activations were the production-oriented input candidate. Zero-copy
  integration still needed an aligned or repacked 4-bit tensor layout, or another
  validated binding path.
- Metal buffer argument indices top out at 30, so group-wise TensorOps kernels must
  chunk groups.
- The simple one-`MTLTensor`-per-group shape was validated only for small output
  tiles; beyond that TensorOps returned zero results.
- `MPSNDArrayQuantizedMatrixMultiplication` was worth keeping only as a comparator
  or fallback probe, where the local SDK exposes it.

No throughput number was recorded anywhere in history.

## Conclusion

Parked. Do not wire this into production until a compact resource layout or
tensor-pool strategy is validated at full model dimensions (the original target
was 80 groups on the since-removed dense model). The motivating dense model is
gone, but the same matmulN kernels still carry 55's batched-prefill projections and
its GPU expert path (the fallback when the ANE is unavailable), so the direction
remains relevant to prefill.
