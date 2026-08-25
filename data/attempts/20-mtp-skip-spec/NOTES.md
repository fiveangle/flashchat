# 20 — MTP skip-spec + deeper batch (fix exp/04)

Date: 2026-08-24  
Status: **NOT STARTED** (exp/04 parked at 0.76× net)

## Prior result (exp/04)

- Artifacts healthy; single-pos acceptance **75.9%** at B=2.
- Net speed **0.76×** vs plain decode (even under 16GB emu).
- Root cause: margin gate fell back to *sequential double production
  verify* on 39/58 iters instead of skipping speculation.
- Draft also pays full lm_head (286 MB) per draft step.

## Required fixes

1. **Skip-spec:** if draft margin < threshold → plain single forward
   (never worse than baseline by construction).
2. Try **B=3..4** so draft cost amortizes; record per-pos acceptance curve.
3. Ensure draft path uses resident MTP experts + GPU experts (flags exist,
   default on) and does not thrash lm_head (pairs with attempt 15).
4. Re-enable per-iter debug timings (`g_server_debug`) for attribution.
5. Truth-test on **real 16GB base M4**, not only ram_pressure on 32GB.

## Expected

If skip-spec works and B≥3 holds ~70%+ accept: **1.2–1.6×** on I/O-bound
16GB — the remaining lever called out in explorations README to cross 15
tok/s after fuse+pin.

## Verdict

Pending. Do not default MTP until net ≥ 1.05× on target hardware.
