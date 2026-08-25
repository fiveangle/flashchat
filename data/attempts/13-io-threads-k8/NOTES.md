# 13 — I/O concurrency for K=8

Date: 2026-08-24  
Status: **NEUTRAL for decode** (pool default raised to 8; no decode gain)

## Change

- `NUM_IO_THREADS` hardcoded 4 → runtime `FLASHCHAT_IO_THREADS` (default **8**, max 16).
- Prefill CSR / legacy pthread pool uses the new width.
- Decode miss path already uses GCD one-block-per-expert (unaffected).

## Results (pin OFF, miss path, 150 tok × 3)

| config | median tok/s | expert_io ms |
|--------|-------------:|-------------:|
| IO=4 | 17.42 | 0.462 |
| IO=8 | 17.31 | 0.465 |

Within noise. Decode does not use the pthread pool for expert misses.

## Verdict

Keep default 8 (matches K and does not hurt). No further decode work here.
Prefill TTFT not re-measured this session — optional follow-up.
