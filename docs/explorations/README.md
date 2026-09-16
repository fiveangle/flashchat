# Decode-Speed Explorations

Systematic exploration of decode tok/s improvements for SSD-streamed MoE
inference. Each experiment has one canonical directory,
`docs/explorations/<index>-<experiment>/`, containing:

- `NOTES.md` — hypothesis, method, observations, comparison, verdict
- raw bench output (quick harness runs, bench_api rows, timing dumps)

Raw output retains the paths printed when it was captured; those historical paths
do not define the current archive layout.

Use `exp/<index>:` commit headlines and the next unused integer index; consult
both this archive and Git history. Preserve negative and inconclusive findings.
There is no second archive or mirrored notes tree. See [AGENTS.md](../../AGENTS.md)
for benchmark scope, noise handling, and commit policy.

[STATUS.md](STATUS.md) preserves the later campaign's status board, including
experiments 12–23 and reserved proposals 22 and 24. Shared August 24 analysis,
cost model, and session results are retained as
[campaign background](12-pin-default-on/campaign-analysis/ANALYSIS.md) alongside
experiment 12, the first experiment in that campaign. Their scope spans the
campaign, not just experiment 12.

Measurement protocol:

- Inner loop: `tools/quick_decode_bench.sh <dir> [tokens] [runs]` — fixed
  prompt, 150 tokens, 3 runs, median decode tok/s + layer-phase medians.
- Sign-off: `bash tests/bench_api.sh --model-id Qwen-Qwen36-35B-A3B` +
  `make bench-report` (interpret under AGENTS.md's noise policy), on this 32GB machine. 16GB targets
  are reasoned about via bytes/token math until hardware is available.

Baseline (see [00-baseline/NOTES.md](00-baseline/NOTES.md)): 32GB M4 Pro-class, warm cache, pin cache off,
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
| 02 | exp/02-ram-discipline | F_NOCACHE + decode-aware admission + post-prefill pin growth | done | MLOCK=+15-24% under pressure, +6% warm; NOCACHE + DECAY dead (measured); post-prefill growth already satisfied by lazy init |
| 04 | exp/04-mtp-batched-default | batched verify (union expert I/O) as default | probed, PARKED | works, 76% acceptance, but 0.76x net at B=2; needs skip-spec gate + B>=3 + 16GB hw |
| 10 | main | prefill "regression" vs Jul-8 fast rows | done, NO REGRESSION | fast rows were a dirty WIP tree; no clean commit reproduces them; ANE fully engaged; see 10 NOTES |
| 11 | exp/11-prefill-ane-crossover | hybrid ANE/GPU prefill by chunk size | done, KEEP | GPU path below 512-token chunks: -37% short cold prefill, bit-faithful; ANE keeps long-chunk overlap win |
| 14 | exp/14-expert-read-split | gate+up wave, down overlapped (shipped v12) | done, KEEP | +6.5% pin-off, +10% over shipped defaults → 25.2 tok/s |
| 15 | exp/15-lm-head-resident | lm_head mlock headroom-guarded (shipped v12) | done, KEEP | +15–20% under pin pressure |
| 21 | exp/21-io-event-gating | pin reserve (pread-into-slot) + event-gated CMD3 probe | done, KEEP (21a) / DEAD (21b) | 21a +2.5% shipped defaults; 21b −4%/−42% removed — see 21 NOTES |

## 16GB emulation result (ram_pressure 17GiB on the 32GB dev box)

100-token decode: base 15.69 -> fuse 18.28 -> fuse+pin3GB **21.59 tok/s
(+37.6%)**. Absolute numbers are flattered by this machine's faster
SSD/RAM vs a base M4; the multipliers are the signal. Extrapolated to the
user-reported 7-9 tok/s base on a real 16GB M4: fuse ~+16%, pin ~+18%,
adaptive-K ~+5% -> roughly 10-13 tok/s without MTP; MTP (once fixed) is
the remaining lever to cross 15.

## Session state (2026-08-17)

- main: serve segfault fixed (d507059), bench-api resurrected, baseline rows.
- exp/07 branch contains: exp/01 (merged) + stages A/B. Recommended merge
  order to main: exp/07 (contains 01), then rebase exp/03 (opt-in knob).
  Both byte-identical; fuse-only shows no prefill cost in durable rows.
- Before shipping defaults: wire FLASHCHAT_FUSE_LINEAR / pin sizing through
  the config chain (menu + config.sh sites + schema bump) per AGENTS.md.

## Recovered proposal dispositions (2026-09-16)

These entries previously lacked dedicated notes. Reconstruction uses repository
documents, commit messages/diffs, and, for 08, a surviving worktree note plus its
session record. No new performance tests were run. "Not found" describes the
inspected evidence, not proof that no private experiment ever existed.

| # | Notes | Disposition |
|---|-------|-------------|
| 05 | [Matrix-vector occupancy](05-matvec-occupancy/NOTES.md) | Deprioritized by code review; no measured negative result recovered |
| 06 | [Prefill I/O threads](06-io-threads-prefill/NOTES.md) | Implemented through 13; decode neutral, prefill benefit unmeasured |
| 08 | [Split expert reads](08-expert-read-split/NOTES.md) | Early negative prototypes recovered; later implementation shipped as 14 |
| 22 | [GPU tail fusion](22-gpu-tail-fusion/NOTES.md) | Design only; estimated benefit is not a measurement |
| 24 | [Idle page-cache warmer](24-idle-page-cache-warmer/NOTES.md) | Design only; no measured benefit recovered |

The old pending entry for 09 is superseded by
[its measured prefetch notes](09-expert-prefetch/NOTES.md). Experiment 14's
pre-landing "opt-in" wording is also corrected in its notes using `5af6a43`.

### Other-checkout recovery

[25 — recovered wonderment campaign](25-wonderment-recovered/NOTES.md) preserves
the unique `flashchat-work` commit `a399937`, its original notes, and 39 API rows
absent from this tree. Its original index 10 collided with our prefill-regression
investigation. The summary maps recovered work to 01/02/07/08 and later commits;
25 is an archival assignment, not a newly run experiment.

Measurement protocol notes: ambient machine drift is +-6% across this
session — ALWAYS interleave A/B runs; never compare across hours. The
durable bench (tests/bench_api.sh) uses short contexts for technical/
creative, so its decode numbers run higher than the 150-token quick
harness at long context; compare like with like.


Historical measurements above describe their original workloads, not universal
performance expectations. Follow AGENTS.md for current validation; do not
automatically re-baseline after every merge.
