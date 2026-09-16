# 05 — Matrix-vector kernel occupancy remap

Status: **DEPRIORITIZED by code review; no numbered experiment measurements found.**
Reconstructed from the archive and Git history on 2026-09-16; no new tests run.

## Question and proposed approach

Could remapping the matrix-vector kernel's work across GPU threads improve
occupancy and decode throughput? The initial index in `9d0e0f2` proposes
column-parallel dequantization and matrix-vector multiplication, claiming only
32 of 256 lanes were active. Later index revisions shorten this to "kernel remap"
and reject that premise. No concrete alternative kernel patch was recovered.

## Evidence and outcome

- The later campaign index records that
  the v3 kernel already used an 8×8 arrangement and identifies dispatch/round-trip
  overhead, rather than occupancy, as the reason to deprioritize this route.
- The inspected `dequant_matvec_4bit_v3` in `metal_infer/shaders.metal` assigns
  eight SIMD groups to eight output rows (`ROWS_PER_TG=8`); this supports the
  need to check the actual kernel instead of assuming only one group works.
  The index's "8×8" shorthand is not a measured GPU occupancy percentage.
- [The historical status board](../STATUS.md) labels the idea DEAD on that
  rationale. This is a prioritization verdict, **not a measured negative result**.
- [The campaign analysis](../12-pin-default-on/campaign-analysis/ANALYSIS.md),
  section 3, lists prior matvec improvements separately from this unpursued remap.
- No dedicated exp/05 implementation commit or raw A/B output was found in the
  inspected repository history. Earlier kernel commits such as `67f5c81`
  (multi-row matmulN) and `68c9976` (large single-token matvec dispatch) are
  different work, not evidence that experiment 05 was benchmarked.

## What would justify revisiting

A concrete mapping change with a different bottleneck hypothesis, supported by
the current kernel and actual model dimensions. Dispatch-bound historical
behavior does not prove all future occupancy changes useless. Use a bounded
probe if warranted; no performance gain or loss can be attributed to 05 today.
