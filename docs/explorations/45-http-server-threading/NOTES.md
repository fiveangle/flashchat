# 45 — Dedicated HTTP thread for API responsiveness

Date: 2026-09-16 implementation and validation; registered 2026-09-22.
Branch: `exp/45-http-server-threading` (documentation only; the feature already shipped).
Commits: `5f3fdcd` (HTTP/inference split), `fccc672` (disconnect cancellation on that split).
Status: **Closed.** Shipped before it was given an exploration ID.
Verdict: **Keep.** Health and status stay responsive during inference. No material decode or prefill change.

## Hypothesis

Serving HTTP on the inference thread made `/health`, model listing, and a second request wait behind generation. A dedicated HTTP thread can answer those while one inference worker owns the model, without changing tokens/sec.

## What shipped

`5f3fdcd` replaced single-threaded HTTP/request execution with an HTTP event thread and one inference worker.

- The HTTP thread accepts bounded requests, serves `/health`, `/v1`, `/v1/models`, and preflight, and relays inference output.
- The inference thread owns model state, caches, and accelerator work. It publishes a short status snapshot under a mutex that does no inference or network I/O.
- A second generation request gets HTTP 503 `server_busy`. There is no queue and no concurrent model execution.
- OpenCode title requests are answered on the HTTP thread and do not take the inference slot.
- `fccc672` propagates client disconnect from the HTTP thread so the worker can stop between prefill layers, generated tokens, and tool-correction steps. That cancellation is a follow-up on this split, not a separate throughput experiment.

Queueing, priority, and multi-conversation batching are still proposal 36 in the [exploration register](../../END_TO_END_SPEED_OPPORTUNITIES.md). They are not part of this result.

## Measurements

Validated 2026-09-16 on Mac17,2 (Apple M5, 32 GiB), Qwen-Qwen3-Coder-Next-q4. Full method and provenance: [HTTP_RESPONSIVENESS_VALIDATION.md](HTTP_RESPONSIVENESS_VALIDATION.md).

**Responsiveness (the target).** While the inference worker was active, worst sampled `/health` latency was **0.7 ms** on a short prompt and **1.5 ms** with 409 new prompt tokens (30 restored context tokens). Phase, occupied context, prefill progress, and generated-token counts advanced. Overlapping generation returned `503 server_busy`. No paired baseline health latency was recorded: the previous server ran HTTP on the inference thread, so concurrent health was not an independent measurement. These figures are absolute candidate latencies, not a percentage speedup.

**Tokens/sec (not the target).** Canonical `tests/bench_api.sh` A/B against sources from `b803619`, same compiler and flags, one warmup, two repeats, 128 max output tokens. Benchmark defaults: 4 GiB expert pinning, fp32 context cache, 64K window, batch prefill on, MTP off. Both health preflights passed.

| Scenario | Baseline tok/s | Updated tok/s | Baseline prefill ms | Updated prefill ms |
| --- | ---: | ---: | ---: | ---: |
| Chat technical | 10.941 | 11.038 | 3355.5 | 3408.5 |
| Responses technical | 11.063 | 11.232 | 3353 | 3417 |
| Chat creative | 11.024 | 10.972 | 2437.5 | 2480.5 |
| Responses creative | 11.529 | 11.712 | 2602 | 2681 |
| Chat mixed code/prose | 9.118 | 8.986 | 3730.5 | 3664 |
| Responses mixed code/prose | 9.383 | 9.210 | 3736 | 3746 |

Decode changes were **−1.8% to +1.6%**. Prefill latency changes were **−1.8% to +3.0%**. No scenario crossed 5%. Treat this as **neutral** for tokens/sec and prefill time. Forced-tool durations are preserved in the validation doc and are not a speed claim (no warmup or repeats).

Both A/B runs are stamped `b803619` because the candidate was uncommitted at measurement time. Distinguish them by timestamp (baseline `2026-09-16T12:13:41Z`–`12:17:46Z`, candidate `2026-09-16T12:18:35Z`–`12:22:28Z`) and binary SHA-256 in the validation doc. All 37 new rows remain in `assets/api_perf_log.tsv`.

## Correctness

Live server tests, 20 API smoke checks, six transport tests, seven meter tests, 35 tool-template checks, and 124 Python unit tests passed on the validation day. Transport coverage includes slow uploads, output backpressure, disconnect recovery, half-close, framing errors, exclusive admission, and shutdown. The user's config was not modified.

## Conclusion

Keep the dedicated HTTP thread. The improvement is API responsiveness during inference (sub-2 ms sampled health, live progress, immediate busy rejection), not generation throughput. Do not cite 45 as a tokens/sec win, and do not reopen it to chase decode speed. Queueing, priority, and multi-conversation batching stay with 36 and still need evidence of concurrent demand.
