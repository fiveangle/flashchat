# 19 — On-disk hot-expert co-location

Date: 2026-08-24  
Status: **NOT STARTED**

## Hypothesis

Packed files store experts in index order. LFU hot set is scattered →
random 1.7 MiB reads. A secondary layout (or rewrite) that packs the
top-N experts per layer contiguously turns miss I/O into sequential DMA,
which Apple Fabric handles far better.

## Method

1. Collect `g_expert_freq` over a calibration corpus (routing log exists).
2. Write `packed_experts_hot/layer_XX.bin` with top-N first; keep cold tail
   in original file or second extent.
3. Runtime: translate expert id → hot file offset via small remap table.
4. Measure random vs sequential pread throughput for 320 × 1.7 MiB patterns.

## Risks

- Large packaging change; migration UX.
- Hot set drifts by workload — needs periodic rebuild or multi-profile.

## Verdict

Pending. P3 — only after pin/lm_head/split-io.
