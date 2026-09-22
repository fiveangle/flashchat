# HTTP responsiveness validation — 2026-09-16

Validated on Mac17,2 (Apple M5, 32 GiB), using Qwen-Qwen3-Coder-Next-q4.

## Live behavior

- `tests/test_server_live.py` passed with a short prompt and with
  `--prompt-repetitions 40` (409 new prompt tokens, 30 restored context tokens).
- Worst sampled health latency was 0.7 ms and 1.5 ms respectively while the
  real inference worker was active. Phase, occupied context, prefill progress,
  and generated-token counts advanced, and final context remained available.
- Overlapping generation requests returned `503 server_busy`.
- The actual menu renderer was inspected against live status during prefill
  and generation. Small occupied contexts now show exact token counts instead
  of rounding to `0k`.
- All 20 API smoke checks passed: both endpoint families, streaming and
  non-streaming, forced tool calls, and tool-result follow-ups.
- Six transport tests and seven meter tests passed. The transport tests cover
  slow uploads, output backpressure, disconnect recovery, half-close behavior,
  framing errors, exclusive admission, and shutdown.
- Build, 35 tool-template checks, and 124 Python unit tests passed.
- Fixed-layout follow-up: inspected the actual menu renderer with stopped,
  idle, preparing, prompt-processing, generating, and unavailable snapshots.
  Every state retains the same seven rows (including context usage), and idle
  clears stale request counters. The meter tests enforce row order and content;
  shell syntax and diff whitespace checks also passed. No engine changes were
  needed for this presentation-only follow-up.

Live testing used the user's q8 context cache, 8 GiB expert pinning, and
4096-token prefill chunks. Test logs, sessions, and prompt-cache artifacts
were isolated under `debug/http-responsive/`.

## Canonical A/B benchmark

Baseline: sources extracted from committed `b803619` into
`debug/http-responsive/baseline/`, then built with the same compiler and flags.
Candidate: the current uncommitted HTTP/progress implementation.
Both used the canonical `tests/bench_api.sh`, the resolved model registry,
one warmup, two measured repeats, and 128 maximum output tokens. Benchmark
defaults were 4 GiB expert pinning, fp32 context cache, 64K context window,
batch prefill enabled, and MTP disabled. Both health preflights passed.

| Scenario | Baseline tok/s | Updated tok/s | Baseline prefill ms | Updated prefill ms |
| --- | ---: | ---: | ---: | ---: |
| Chat technical | 10.941 | 11.038 | 3355.5 | 3408.5 |
| Responses technical | 11.063 | 11.232 | 3353 | 3417 |
| Chat creative | 11.024 | 10.972 | 2437.5 | 2480.5 |
| Responses creative | 11.529 | 11.712 | 2602 | 2681 |
| Chat mixed code/prose | 9.118 | 8.986 | 3730.5 | 3664 |
| Responses mixed code/prose | 9.383 | 9.210 | 3736 | 3746 |

No Coder-Next scenario crossed the 5% regression threshold. Decode changes
were -1.8% to +1.6%; prefill latency changes were -1.8% to +3.0%.
Both forced-tool checks passed. Their single-shot durations (17784 ms and
4257 ms) are preserved but are not used to claim a speedup: unlike the six
streaming scenarios, that check has no warmup or repeats to control cache
state and response variability.

All 37 new rows (11 smoke, 13 baseline, 13 candidate) are retained in
`assets/api_perf_log.tsv`. Both A/B runs carry commit `b803619` because the
candidate changes were uncommitted; distinguish them by timestamp:

- Baseline: `2026-09-16T12:13:41Z` through `2026-09-16T12:17:46Z`.
- Candidate: `2026-09-16T12:18:35Z` through `2026-09-16T12:22:28Z`.
- Baseline binary SHA-256: `fce330bc0e955fa9dcea527b2675d577e798bfa410243bd4f2a039a5af079e42`.
- Candidate binary SHA-256: `f464be4a651f57083112613c16f74e6ddf49d353dea7521a28cf0b4fc53b2d8e`.

The Coder-Next-only report returned zero regressions. Full `make bench-report`
still returns failure for three existing historical comparisons on other
model variants (`e237d9a` to `f9a56c8`); none involve these new measurements.

## External state and cleanup

All validation servers exited. The user's config file was not modified.

The first live launch inherited the user's profiler destination:
`/Volumes/usr/Users/speedster/dev/flashchat_logs/pread.tsv`. Before its shutdown
flush, the original file was preserved; after shutdown it was restored with
its original timestamp and verified byte-for-byte. SHA-256:
`f2cc0615470c37b7aaf836f0c8065dae4e5d130d1e886f6c690a32ec0ff41e80`.
Subsequent profiling went to the project test directory.

The canonical benchmark exercised normal model-local cache maintenance.
These generated cache files were written/refreshed:

- `/Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb/flashchat/q4/system_prompt_cache/40151882ca0c04f9-v2.fcache`
- `/Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb/flashchat/q4/system_prompt_cache/6517233bbcdcf8c3-v2.fcache`

Its cache maintenance also removed the old generated cache:
`/Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb/flashchat/q4/system_prompt_cache/40151882ca0c04f9-v1.fcache`.
This is reproducible prompt-processing state, not model weights or conversation
data; the server regenerates the current cache format as needed.
