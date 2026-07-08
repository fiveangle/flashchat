# Prefill Transient Buffer Release Benchmark Log

Date: 2026-07-08

## Objective

Measure whether releasing buffers that are only needed for batched prefill can return useful memory before decode starts, without hurting prefill or token generation.

This benchmark is intentionally small and repeatable. It is meant to capture what was tried and what was observed so future optimization work does not have to rediscover the same details.

## Change Under Test

`metal_infer/infer.m` now tracks the large reusable Metal buffers that grow during batched prefill and releases them after prompt prefill completes, before token generation begins.

The release path covers:

- `gpu_dequant_matmulN` input/output scratch buffers.
- `gpu_dequant_matmulN_batch` per-slot input/output scratch buffers.
- Batched MoE CSR staging buffers.
- Batched delta/attention prefill buffers in `MetalCtx`.
- Excess `buf_output` capacity above the decode-time vocabulary-sized output buffer.

`FLASHCHAT_PREFILL_RELEASE=0` disables the release for A/B measurement. The default behavior is release enabled. The release is skipped while MTP is active because that path may reuse more prefill-era state.

## Test Configuration

- Model: `Qwen-Qwen36-35B-A3B`
- Quantization: q4 expert weights
- KV cache: q8
- Context window: 262144 tokens
- Active experts: K=8
- MTP: disabled
- Batched prefill: enabled
- Prefill chunk: 1024
- ANE prefill: enabled
- Expert pin cache: 4.0 GiB cap, auto fraction 0.50, mlock off
- System prompt cache: enabled, disk hit during both runs
- Request: 1231 prompt tokens, 16 generated tokens, deterministic sampler request

Raw files are under:

- `debug/prefill-release-bench-20260708/raw-release-off-captured.log`
- `debug/prefill-release-bench-20260708/raw-release-on-captured.log`
- `debug/prefill-release-bench-20260708/server-release-off-captured.log`
- `debug/prefill-release-bench-20260708/server-release-on-captured.log`
- `debug/prefill-release-bench-20260708/request.json`

## Commands

Build and syntax checks:

```bash
rtk make -B infer
rtk bash -n flashchat
```

Release disabled:

```bash
rtk env FLASHCHAT_SERVER_LOG=/Users/speedster/dev/flashchat/debug/prefill-release-bench-20260708/server-release-off-direct.log FLASHCHAT_SERVER_DEBUG=1 FLASHCHAT_SERVER_HTTP_LOG=0 FLASHCHAT_BATCH_PREFILL=1 FLASHCHAT_PREFILL_CHUNK=1024 FLASHCHAT_PREFILL_DEBUG=1 FLASHCHAT_PREFILL_RELEASE=0 FLASHCHAT_ANE_PREFILL=1 FLASHCHAT_EXPERT_PIN_MAX_GB=4 FLASHCHAT_EXPERT_PIN_AUTO_FRAC=0.50 FLASHCHAT_EXPERT_PIN_MLOCK=0 FLASHCHAT_SYSTEM_PROMPT=/Users/speedster/.config/flashchat/system.md FLASHCHAT_MODEL_CONFIG=/Users/speedster/dev/flashchat/assets/model_configs.json FLASHCHAT_ACTIVE_EXPERTS=8 FLASHCHAT_CONTEXT_WINDOW=262144 FLASHCHAT_KV_QUANT=q8 ./metal_infer/infer --serve 9999 --config /Users/speedster/.config/flashchat/config --model /Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
```

Release enabled:

```bash
rtk env FLASHCHAT_SERVER_LOG=/Users/speedster/dev/flashchat/debug/prefill-release-bench-20260708/server-release-on-direct.log FLASHCHAT_SERVER_DEBUG=1 FLASHCHAT_SERVER_HTTP_LOG=0 FLASHCHAT_BATCH_PREFILL=1 FLASHCHAT_PREFILL_CHUNK=1024 FLASHCHAT_PREFILL_DEBUG=1 FLASHCHAT_PREFILL_RELEASE=1 FLASHCHAT_ANE_PREFILL=1 FLASHCHAT_EXPERT_PIN_MAX_GB=4 FLASHCHAT_EXPERT_PIN_AUTO_FRAC=0.50 FLASHCHAT_EXPERT_PIN_MLOCK=0 FLASHCHAT_SYSTEM_PROMPT=/Users/speedster/.config/flashchat/system.md FLASHCHAT_MODEL_CONFIG=/Users/speedster/dev/flashchat/assets/model_configs.json FLASHCHAT_ACTIVE_EXPERTS=8 FLASHCHAT_CONTEXT_WINDOW=262144 FLASHCHAT_KV_QUANT=q8 ./metal_infer/infer --serve 9999 --config /Users/speedster/.config/flashchat/config --model /Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
```

The captured evidence files were produced by the same direct `infer --serve` launch shape with stdout/stderr redirected to `raw-*.log`, a `/health` wait loop, one request, then `SIGINT`.

## Results

| Mode | Transient buffers before decode | Release action | Prefill | Decode | Expert pin result |
| --- | ---: | --- | ---: | ---: | --- |
| Release disabled | 401.43 MiB | Disabled by `FLASHCHAT_PREFILL_RELEASE=0` | 32790 ms | 16 tokens in 3298 ms, 4.85 tok/s | 3266 / 5440 hits, 60.0%; pinned 2174 / 2427; evictions 0 |
| Release enabled | 401.43 MiB | Released 401.43 MiB | 32850 ms | 16 tokens in 3477 ms, 4.60 tok/s | 3266 / 5440 hits, 60.0%; pinned 2174 / 2427; evictions 0 |

Representative memory lines:

```text
[prefill-mem] before transient release rss=1.31 GiB free_reclaimable=8.27 GiB transient=401.43 MiB
[prefill-mem] transient release disabled by FLASHCHAT_PREFILL_RELEASE=0

[prefill-mem] before transient release rss=1.40 GiB free_reclaimable=7.90 GiB transient=401.43 MiB
[prefill-mem] released 401.43 MiB of transient buffers | rss 1.40 -> 1.40 GiB | free_reclaimable 7.90 -> 7.90 GiB
```

Representative timing lines:

```text
[serve] chatcmpl-2 prefill=1231 tokens in 32790ms
[serve] chatcmpl-2 generated=16 tokens in 3298ms (4.85 tok/s, experts 6.0 MiB/s, 0.7 MiB/s/expert, TTFT 32.8s)

[serve] chatcmpl-2 prefill=1231 tokens in 32850ms
[serve] chatcmpl-2 generated=16 tokens in 3477ms (4.60 tok/s, experts 5.9 MiB/s, 0.7 MiB/s/expert, TTFT 32.9s)
```

## Interpretation

The direct retained-buffer accounting shows a clear 401.43 MiB reduction in prefill-only Metal buffer retention before decode. Prefill time was effectively unchanged in this small A/B run: 32.79 s disabled versus 32.85 s enabled.

The RSS and `host_free_ram_bytes()` readings did not visibly move immediately after setting the Objective-C buffer references to `nil`. That does not mean the release is useless; it means the macOS/Metal allocator did not immediately reflect the freed resources in these process-level counters at this measurement point. The engine-side retained-buffer total is still lower, and that should reduce pressure on later allocations and leave the allocator/OS more room to reuse memory.

Decode speed was slightly lower in the enabled run, but this single 16-token decode is too small to treat as a regression. The expert pin stats were identical in both captured runs.

The direct server path was used for the captured runs because the `flashchat serve` wrapper briefly reported ready and then left a stale PID during an early attempt. Direct `infer --serve` stayed stable and gave cleaner stdout/stderr evidence.

## Follow-Ups

- Run `make bench-api` or `make bench-api --model-id Qwen-Qwen36-35B-A3B` before treating this as production-ready hot-path work.
- Add `FLASHCHAT_PREFILL_RELEASE` as a first-class config/debug setting if we want users to toggle it outside one-off environment tests.
- Consider logging transient-buffer bytes to the persistent server log, not only stderr, so background server runs preserve this evidence.
- If the goal is proving memory return to the OS, add a second delayed measurement after one or more decode tokens and possibly include `MTLDevice currentAllocatedSize` if available on the local SDK.
