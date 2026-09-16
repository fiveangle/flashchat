# 15 — lm_head RAM residency (mlock)

Date: 2026-08-24  
Status: **KEEP** — default ON (`FLASHCHAT_LM_HEAD_MLOCK=1`)

## Implementation

- After `open_weights`, compute span of `lm_head.weight/scales/biases`
  (first **272.8 MiB** of q4 `model_weights.bin`).
- Page-align, touch once, `mlock`; fail-soft to `MADV_WILLNEED`.
- Also changed weight-file `madvise` from `MADV_SEQUENTIAL` → `MADV_RANDOM`
  (decode access is random per layer; sequential advice was wrong).
- Timing: `lm_head_avg` printed under `--timing`.

## Results (pin 4G + mlock experts, fuse on)

| config | median tok/s | lm_head_avg ms |
|--------|-------------:|---------------:|
| lm_head mlock ON | **22.16 / 23.87** (two blocks) | 2.66–2.68 |
| lm_head mlock OFF | **19.48** | 2.88 |

Even on a warm 32GB machine, locking lm_head is **+15–20%** vs unlocked
under pin-cache pressure (pin arena competes for RAM; unlocked lm_head
pages get reclaimed). Token stream identical.

## Verdict

**KEEP default on.** On 16GB this should matter more (pair with pin 3G).
Wire through config menu when packaging defaults.
