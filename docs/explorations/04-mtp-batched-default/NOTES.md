# 04 — MTP batched verify as default (16GB byte-amortization lever)

Branch: `exp/04-mtp-batched-default` (probe-only; no engine changes yet)

## Hypothesis

Batched verify (fused_batched_forward_N, union expert reads) amortizes expert
I/O across B positions — the biggest bytes/token lever for 16GB machines.
Artifacts exist and the path works today behind FLASHCHAT_MTP_BATCH_VERIFY=1
with MTP>=2 from the config file (no FLASHCHAT_MTP env exists; CLI --mtp only
shadows).

## Observations (32GB Mac17,2; 17GiB ram_pressure = 16GB emulation)

Working end-to-end via serve: acceptance 75.9% per-pos (B=2), refwd 0.
BUT net speed multiplier **0.76x — slower than plain decode**:
- warm 32GB: 14.46 tok/s (MTP) vs ~18+ (no MTP)
- 16GB-emulated: 16.03 tok/s (MTP+fuse+pin3G) vs 21.59 (fuse+pin3G alone)

Root causes seen in the economics line:
1. Margin gate (margin_min=4.0) fell back to *sequential production verify*
   on 39/58 iters — 2 full forwards per step. Low confidence should SKIP
   drafting (fall to plain forward), never pay double.
2. Draft cost: predictor forward + full lm_head (286 MB read per draft).
3. Even emulated, this machine's SSD is ~2x a base M4's — the union savings
   don't cover the extra forwards here. On a real 16GB base M4 the I/O term
   doubles, but 0.76 is too far from 1.0 for that alone to flip it.

## Required fixes before re-evaluation

1. Margin gate -> skip-spec (draft only when margin >= threshold; else plain
   step). Makes MTP never-worse-than-baseline by construction.
2. Try B=3..4: draft cost amortizes; per-pos acceptance curve needed.
3. Enable the per-iter debug timings (they logged 0.0 — g_server_debug off)
   to attribute draft vs verify vs refwd.
4. Re-test on real 16GB base-M4 hardware (user's target).

## Verdict

NOT READY. The mechanism and artifacts are healthy (76% single-pos
acceptance is a strong signal), but the economics at B=2 with the current
fallback policy are negative even in the I/O-bound regime. The skip-spec fix
plus deeper B is the concrete path; park until a 16GB machine is available
for truth testing.

## 16GB emulation context (ram_pressure 17GiB, this session)

Quick-bench 100-token decode under emulation:
- base 15.69 -> fuse 18.28 -> fuse+pin3GB 21.59 tok/s (+37.6% over base)
(absolute numbers flattered by this machine's faster SSD/RAM vs base M4;
relative multipliers are the signal)
