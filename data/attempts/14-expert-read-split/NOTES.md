# 14 — Split expert I/O (gate+up wave, down overlapped)

Date: 2026-08-24  
Status: **KEEP** — default candidate `FLASHCHAT_EXPERT_SPLIT_IO=1`

## Implementation

- Layout already splits at `down_w_off` (66.7% gate+up / 33.3% down).
- Wave1: pread gate+up (or pin hit) → encode gate+up+SwiGLU (+ shared) → commit.
- Concurrent: pread down suffix on GCD.
- Wave2: wait down → encode down_proj + combine on next cmd buffer (queue-ordered).
- All-pin-hit layers skip split (single whole-expert path, no extra RTT).
- Flag: `FLASHCHAT_EXPERT_SPLIT_IO=1` (currently opt-in).

## Results (interleaved, fuse on, lm_head mlock on)

| config | median tok/s | total_layer | expert_io | cmd3_encode |
|--------|-------------:|------------:|----------:|------------:|
| pin OFF | 17.61 | 1.337 | 0.462 | 0.030 |
| pin OFF + split | **18.75** (+6.5%) | 1.250 | 0.529* | 0.238 |
| pin 4G | ~23.0 (session base) | ~1.00 | ~0.17 | 0.020 |
| pin 4G + split | **25.23** (+~10% vs pin4) | **0.909** | 0.174 | 0.112 |

\* expert_io accounting now includes wave2 wait nested inside the GPU window;
wall-clock layer time is what matters (down).

## Correctness

First 50+ generated token_ids **byte-identical** vs baseline at temp-0
(pin off, pin on, split on/off).

## Verdict

**KEEP.** Recommend default-on after config-chain wiring. Works with pin
(miss tail still benefits) and without. Largest new code win this session.
