# 09 — Predictive expert prefetch

Branch: `exp/09-expert-prefetch` (from main @ cdb14b8). Instrumentation-only;
no prefetch engine change was built — the ceiling measurement killed the
cheap variants first (that was the point of Stage 0).

## Hypothesis

Per layer, decode is serial: GPU phases run with the SSD idle, expert reads
run with the GPU idle. A predictor of the next expert set lets the SSD work
during GPU phases — the last free overlap source on cold-cache machines.
Wrong prefetches cost only bandwidth; outputs stay byte-identical (safe,
unlike MTP).

## Method: measure the ceiling BEFORE building

Added prefetch-ceiling instrumentation to the existing overlap counters
(reported by mtp_overlap_report, enabled with FLASHCHAT_MTP_OVERLAP=1):
per pin-MISS expert fetch, was that expert in the previous token's set at
the same layer? Runs via serve + bench prompt, 128 tokens.

## Observations (32GB Mac17,2; 16GB = ram_pressure 17GiB)

| config | pin hit rate | miss fetches | predictable | ceiling |
|---|---|---|---|---|
| pin off | 0% (all fetches) | 32,640 | 10,705 | 32.8% |
| pin 4GB (32GB) | 77% | 7,520 | 425 | **5.7%** |
| pin 3GB (16GB emu) | 69.9% | 9,610 | 790 | **8.2%** |

16GB-emu decode for context: 11.16 tok/s (vs ~19-21 warm) — the miss pool
is where the loss lives, and it is NOT prev-token predictable.

## Why the cheap predictors are dead (the key insight)

- The 33% of fetches that repeat token-to-token ARE the hot set — and the
  pin cache already holds exactly that set (LFU). Pin misses = what's left
  after removing everything a frequency/prev-token predictor could cover.
- Top-N hot-set prefetcher: would prefetch precisely the pinned set —
  zero coverage of misses by construction.
- F_RDADVISE on prev-set: same 5.7-8.2% ceiling, same death.
- Cold-tail activations are content-driven: the only signal that carries
  them is the hidden state itself, i.e. a learned pre-gated router probe.

## What a learned predictor could still be worth (parked, quantified)

Miss pool ~23-30% of reads at 3-4GB pin. A strong linear probe (70-90%
top-8 recall is the literature range for MoE routing probes) covers
~60% of misses -> ~15-18% of total expert reads. On a cold 16GB decode
(~11 tok/s emu baseline) that's worth roughly +10%. Cost: routing-log
training pipeline (g_routing_log exists), per-layer probe storage +
eval, prefetch buffer management. Same hardware gate as MTP: build it
when a real 16GB machine is available to validate.

## Verdict

Prev-token / frequency prefetch: **DEAD — do not build** (5.7-8.2% of a
pool that is itself only ~25% of reads). Learned pre-gated probe: parked
with a quantified ~10%-on-16GB upside behind the hardware gate.

Keep the instrumentation (zero-cost unless FLASHCHAT_MTP_OVERLAP=1) — it
is the measurement harness for any future prefetch/predictor work, and it
doubles as pin-cache sizing evidence (miss-fraction vs pin size).
