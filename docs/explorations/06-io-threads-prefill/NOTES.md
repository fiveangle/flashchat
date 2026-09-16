# 06 — Increase the prompt-processing I/O thread pool

Status: **FOLLOWED UP AS 13; prompt-processing benefit remains unmeasured.**
Reconstructed on 2026-09-16; no new tests run.

## Hypothesis

The original campaign proposed increasing `NUM_IO_THREADS` from 4 to 8 to
reduce prompt-processing time to the first generated token. The initial index
in `9d0e0f2` specifically mentions `MAX_K=16` prefill waves; the later analysis
focuses on eight active experts. In either case, the hypothesis was that a
four-worker pool could serialize reads that might overlap.

## What was actually done

The original [index](../README.md) left 06 pending. The later
[campaign analysis, section 4.2](../12-pin-default-on/campaign-analysis/ANALYSIS.md)
distinguishes two paths: prompt processing uses the pthread pool, whereas decode
misses already dispatch independent reads through Grand Central Dispatch.

[Experiment 13](../13-io-threads-k8/NOTES.md) implemented a configurable pool
width, default 8 and maximum 16. Commit `5af6a43` records that change alongside
the split-read and model-output-weight residency work.

Its archived 150-token, three-run decode probes with pinning off reported:

| Pool width | Median decode | Expert I/O time |
|---|---:|---:|
| 4 | 17.42 tokens/s | 0.462 ms |
| 8 | 17.31 tokens/s | 0.465 ms |

These are neutral within noise and do not test the proposed prompt-processing
gain: decode does not use that pool. Experiment 13 explicitly says prompt
processing was not remeasured. Raw runs remain alongside its notes.

## Conclusion

Do not repeat the decode comparison to answer the prefill question. The proposed
configuration change landed through 13, but no isolated time-to-first-token
result for 06 was found. Revisit only with a specific prompt-processing workload
and evidence that pool concurrency is a relevant bottleneck.
