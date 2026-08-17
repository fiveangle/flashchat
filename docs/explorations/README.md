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
| 00 | main | baseline capture | done | 12.4 tok/s @32GB |
| 01 | exp/01-pin-zero-copy | bind pin-cache slots as GPU buffers (no memcpy on hit) | | |
| 02 | exp/02-ram-discipline | F_NOCACHE expert streams + decode-aware pin admission + post-prefill pin growth | | |
| 03 | exp/03-adaptive-k | route by weight mass (~98%), not fixed K=8 | | |
| 04 | exp/04-mtp-batched-default | batched verify (union expert I/O) as default MTP mode | | |
| 05 | exp/05-matvec-occupancy | column-parallel dequant GEMV kernels (32/256 lanes active today) | | |
| 06 | exp/06-io-threads-prefill | NUM_IO_THREADS 4→8 (prefill waves of MAX_K=16) | | |
| 07 | exp/07-cmd-fuse-linear | single commit+wait for CMD1+CMD2 on linear layers | | |
| 08 | exp/08-expert-read-split | gate+up pread wave, down wave overlapped with GPU gate/up | | |
| 09 | exp/09-expert-prefetch-spec | pre-gated expert prefetch from routing signal | | |

Merge policy: experiments land on their branch with measurements; merges to
main only after bench_api sign-off; each merge re-baselines the next path.
