# Decode-Speed Explorations

Systematic exploration of decode tok/s improvements for SSD-streamed MoE
inference. Each path gets a git branch `exp/<NN>-<slug>` and a directory
`docs/explorations/<NN>-<slug>/` containing:

- `NOTES.md` — hypothesis, method, observations, comparison, verdict
- raw bench output (quick harness runs, bench_api rows, timing dumps)

Measurement protocol:

- Inner loop: `tools/quick_decode_bench.sh <dir> [tokens] [runs]` — fixed
  prompt, 150 tokens, 3 runs, median decode tok/s + layer-phase medians.
- Sign-off: `bash tests/bench_api.sh --model-id Qwen-Qwen36-35B-A3B` +
  `make bench-report` (no regression), on this 32GB machine. 16GB targets
  are reasoned about via bytes/token math until hardware is available.

Baseline (see `00-BASELINE.md`): 32GB M4 Pro-class, warm cache, pin cache off,
MTP off, K=8: ~12.4 tok/s decode, per-layer 2.02ms (cmd1_wait 0.605,
cmd2_wait 0.599, expert_io 0.771 — strictly serial phases).

## Paths

| # | Branch | Hypothesis | Status | Result |
|---|--------|-----------|--------|--------|
| 00 | main | baseline capture | done | 17.3 tok/s quick / 15.2-16.0 bench @32GB |
| — | main d507059 | serve segfault fix (blocking bench since 86c8846) | done | bench resurrected |
| 01 | exp/01-pin-zero-copy | zero-copy pin slot buffers | done, KEEP | +10% quick A/B; 88% hits @8GB; byte-identical |
| 03 | exp/03-adaptive-k | routing-mass K truncation | done, opt-in | +4.6%; router flat (K 6.92 @0.90) |
| 07 | exp/07-cmd-fuse-linear | single commit+wait on linear layers | done, KEEP | +11.7% A/B; bench +14-24% decode, no prefill cost; combo w/ pin up to +80% short-ctx |
| 02 | exp/02-ram-discipline | F_NOCACHE + decode-aware admission + post-prefill pin growth | next | pin steals page cache -> prefill -26% @8GB; fix sizing/phase |
| 04 | exp/04-mtp-batched-default | batched verify (union expert I/O) as default | pending | biggest 16GB lever |
| 05 | exp/05-matvec-occupancy | kernel remap | deprioritized | v3 kernel already 8x8; cost is dispatch/RTT not occupancy |
| 06 | exp/06-io-threads-prefill | NUM_IO_THREADS 4->8 | pending | prefill TTFT |
| 08 | exp/08-expert-read-split | gate+up wave, down overlapped | pending | |
| 09 | exp/09-expert-prefetch-spec | pre-gated expert prefetch | pending | |

Measurement protocol notes: ambient machine drift is +-6% across this
session — ALWAYS interleave A/B runs; never compare across hours. The
durable bench (tests/bench_api.sh) uses short contexts for technical/
creative, so its decode numbers run higher than the 150-token quick
harness at long context; compare like with like.


Merge policy: experiments land on their branch with measurements; merges to
main only after bench_api sign-off; each merge re-baselines the next path.
