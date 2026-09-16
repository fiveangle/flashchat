# 01 — Zero-copy expert pin cache

Branch: `exp/01-pin-zero-copy` (rebased on main@d507059, the serve-segfault fix)

## Hypothesis

The pin cache's hit path was self-defeating: `expert_pin_lookup` memcpy'd the
whole expert (1.69 MiB) into the Metal pool buffer while holding `g_pin_mu`,
serially on the decode thread. Zero-copy: page-aligned arena, each slot
wrapped once in a persistent `newBufferWithBytesNoCopy` MTLBuffer, hit binds
the slot buffer directly as the GPU expert source. See top of NOTES for the
eviction-while-in-flight safety argument (decode-thread phase ordering).

## Method

- `tools/quick_decode_bench.sh <dir> 150 3` per pin size; PIN off baseline
  = 17.33 tok/s median (../00-baseline/NOTES.md).
- Correctness: temp-0 fixed prompts, output text compared PIN vs NOPIN.

## Observations

| config | hit rate | decode tok/s (median) |
|---|---|---|
| off (baseline) | — | 17.33 |
| 4 GB (2,427 slots) | 75.3% | 19.18 |
| 8 GB (4,854 slots) | 88.2% | 18.98 |
| 12 GB | 88.7% | 18.77 |
| 16 GB | 88.7% | 18.48 |

- 8GB run layer phases: expert_io 0.375 -> 0.268 ms, total_layer 1.364 ->
  1.227 ms.
- Hit-rate curve saturates ~88%: the residual misses are the cold tail of the
  routing distribution (LFU can't hold them all); on 32GB those misses are
  page-cache-warm anyway, so bigger arenas buy nothing and cost RAM.
- Sizes 4-16GB are within run noise of each other on 32GB warm (~19 tok/s);
  +10% vs off. On 16GB (misses = real SSD reads) the same hit rates are worth
  much more — recommended size there: whatever fits after OS + non-expert
  weights, ~3-4 GB.
- Correctness: PIN vs NOPIN temp-0 outputs **byte-identical** (weights are
  the same bytes, only the source buffer differs). Coherent text.

## Verdict

KEEP. +10% decode on 32GB (17.33 -> ~19 tok/s) with a 4 GB arena, zero-copy
on hits, no quality change. Recommend default pin sizing stays configurable;
measure 16GB on hardware when available.
