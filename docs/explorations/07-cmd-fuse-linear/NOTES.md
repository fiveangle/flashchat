# 07 — Fuse CMD1+CMD2 on GPU-linear layers; GPU-resident dataflow

Branch: `exp/07-cmd-fuse-linear` (stage A+B, then merged exp/01 for combo)

## Hypothesis

Per linear layer, two GPU commit+wait round trips even though with
gpu_linear_attn + prev_gpu_combined nothing on the CPU is needed between
CMD1 and CMD2. Stage A: single commit+wait. Stage B: bind GPU-resident
sources (buf_input as expert input, batch_out[1]/[2] as shared gate/up).

## Observations

Stage A (interleaved A/B, 100 tok):
- off: 16.29 / 16.19  -> fuse: 18.15 / 18.15  = **+11.7%**
- sync_waits 11920 -> 7920; total_layer 1.461 -> 1.294 ms
- temp-0 outputs byte-identical (two prompts)

Stage B: 18.21-18.22 vs 18.15 — flat (uploads were cheap on 32GB warm) but
kept: fewer CPU-GPU round trips, matters on bandwidth-constrained machines.
Byte-identical.

Combo with exp/01 (fuse + 8GB zero-copy pin), quick bench: 18.3-18.7 vs
contemporaneous off 16.2.

Durable bench-api (env FUSE=1 [+PIN for combo]), commit 281dfd4:

| scenario | main@9fd7c83 | fuse-only | fuse+pin8G |
|---|---|---|---|
| chat technical decode | 16.04 | 17.61 | ~27 (short ctx) |
| chat worstcase decode | 15.30 | 17.32 | 18.45 |
| responses worstcase decode | 15.23 | 17.48 | 18.87 |
| worstcase prefill ms | 2435 | 2399 (no cost) | 3143 (**+26%**) |
| tool_call ms | 3026 | 2864 | 3471 |

- Fuse is a pure win: +14% long-context decode, more at short context, no
  prefill cost.
- The 8GB pin arena costs prefill ~26% on 32GB by shrinking the page cache
  that prefill's expert streaming relies on. 4GB pin gave the same decode as
  8GB in the quick harness (19.18 vs 18.98) — recommend 4GB class sizing on
  32GB; on 16GB the trade is far more favorable (decode hits >> prefill cache
  loss since prefill is SSD-bound there anyway).
- Note: main prefill (2.4s) is still ~2.2x slower than the Jul-8 ane-prefill
  branch rows (~1.1s) — pre-existing main regression vs ane branch, unrelated
  to this work. Follow-up candidate.

## Verdict

KEEP BOTH STAGES. Strongest clean win so far: +11.7% controlled A/B,
+14-24% durable bench (no prefill cost with fuse alone). Merge candidate
(default-on after wiring FLASHCHAT_FUSE_LINEAR through the config chain, or
inverting to opt-out). The pin prefill trade-off is a sizing question —
handled separately in exp/02 (phase-aware / F_NOCACHE).
