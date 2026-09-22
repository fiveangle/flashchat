# 46 — Release prefill-only Metal buffers before decode

Date: 2026-07-08. Registered 2026-09-22.
Commit: `86c8846` (same commit as the slot-cap control surface, 47).
Status: **Closed.** Shipped before it had an exploration ID.
Verdict: **Keep.** Engine accounting released 401.43 MiB of prefill-only buffers. Speed was not established.

## Hypothesis

Batched prefill grows large Metal scratch buffers from the chunk size. Decode does not need that capacity. Releasing those buffers after the prompt, before generation, should return memory without changing prefill or decode time.

## What the commit did

`86c8846` tracks the reusable prefill Metal buffers and nils them after prompt processing, in both `serve_loop` and the CLI path. Release runs only when batched prefill is on. It drops:

- `gpu_dequant_matmulN` input/output scratch, and the eight batched matmul slots
- batched MoE CSR staging (`x`, `gate`, `up`, `act`, `out`)
- batched delta and full-attention buffers on `MetalCtx`
- `buf_output` capacity above one vocabulary-sized decode buffer

`FLASHCHAT_PREFILL_RELEASE=0`, `off`, or `false` skips the release. MTP also skips it, because that path may still need prefill-era state. At landing the toggle was environment-only. `b513986` (2026-09-16) later made `PREFILL_RELEASE` a config setting, default on. That later wiring is not part of the July measurement.

## Measurement

One direct `infer --serve` pair on 2026-07-08, Qwen3.6-35B-A3B q4, q8 context, 262144-token window, K=8, MTP off, batched prefill, chunk 1024, ANE prefill on, 4 GiB pin cap, mlock off. Request: 1231 prompt tokens, 16 generated tokens. System-prompt cache hit on both runs. Captured raw logs lived under `debug/prefill-release-bench-20260708/` and were not committed. Retained write-up: [PREFILL_RELEASE_BENCHMARK_LOG.md](PREFILL_RELEASE_BENCHMARK_LOG.md).

| Mode | Transient buffers | Prefill | Decode |
| --- | ---: | ---: | --- |
| Release off | 401.43 MiB retained | 32790 ms | 16 tokens in 3298 ms, 4.85 tok/s |
| Release on | 401.43 MiB released | 32850 ms | 16 tokens in 3477 ms, 4.60 tok/s |

Prefill time changed by +60 ms. Decode was 4.85→4.60 tok/s on this single 16-token run. The original log correctly refuses to call that a regression. Expert-pin stats matched (3266/5440 hits, 60.0%, 0 evictions).

Process RSS and reclaimable free RAM did not move at the measurement point (release-on: RSS 1.40→1.40 GiB, free 7.90→7.90 GiB). The 401.43 MiB figure is the engine's own buffer-length sum after the Objective-C references were cleared, not an observed return to the OS.

## Conclusion

Keep the release. It drops prefill-only Metal retention before decode. Do not cite a tokens/sec change: the only pair is one 16-token run, and prefill time was effectively unchanged. A later delayed RSS or `currentAllocatedSize` reading would be required before claiming memory returned to the OS. Do not reopen this to chase decode speed.
