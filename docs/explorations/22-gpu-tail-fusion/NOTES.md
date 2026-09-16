# 22 — Fuse final normalization and output projection into the last GPU buffer

Status: **DESIGN ONLY — not implemented or measured in the recovered record.**
Reconstructed on 2026-09-16; no new tests run.

## Hypothesis

Append final normalization and `lm_head` (the vocabulary/output projection) to
the last layer's CMD3 GPU command buffer, then read back only logits for sampling.
The intended benefit is fewer command submissions or synchronization boundaries
at the end of each generated token, not reduced expert SSD traffic.

## Available evidence

- [Experiment 21, "Follow-ups not attempted"](../21-io-event-gating/NOTES.md)
  preserves the design and an estimated +1–3% upside, explicitly NOT implemented.
- [The status board](../STATUS.md) reserves index 22 and assigns P2 priority.
- Commit `b803619` preserves the originating experiment-21 notes. No dedicated
  implementation commit, correctness result, timing capture, or raw benchmark
  for 22 was found in the inspected history.

The +1–3% number is an old estimate, not a result. It is also within the usual
noise range for this engine, so it is not justification for repeated baselines.

## Requirements for a deliberate revisit

First inspect current command-buffer ownership and data dependencies: the final
hidden state must be GPU-resident and ready, and sampling must wait for completed
logits. Verify normalization/projection ordering and generated-output correctness.
Confirm that a removable boundary still exists before implementing anything.
Any bounded test should separate direct submission/wait evidence from noisy
end-to-end throughput. These are review criteria, not a recovered implementation.
