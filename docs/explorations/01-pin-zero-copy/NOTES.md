# 01 — Zero-copy expert pin cache

Branch: `exp/01-pin-zero-copy` (from main @ 9fd7c83)

## Hypothesis

The pin cache's hit path is self-defeating: `expert_pin_lookup` memcpy's the
whole expert (1.69 MiB) into the Metal pool buffer **while holding g_pin_mu**,
serially on the decode thread, before the miss preads are even dispatched.
Eight hits/layer ≈ 13 MB of serial memcpy per layer ≈ 0.7ms — the same order
as the pread it avoids. So even with a large pin cache the win is capped.

If instead each pin slot is a page-aligned arena region wrapped once in a
persistent `newBufferWithBytesNoCopy` MTLBuffer, a hit binds the slot buffer
directly as the expert source for `gpu_encode_experts_batched` — **zero
copies**. Lookup becomes a mutex + array read. Expert I/O phase should
collapse toward 0 for hit fractions of ~50%.

GPU-safety argument for eviction-while-in-flight: admits/evictions only run
in `async_pread_wait` (layer L+1 expert phase), which is strictly after
`finalize_deferred_experts` completed CMD3(L) (CMD1(L+1) waitUntilCompleted
implies queue-serialized CMD3(L) done). No slot bound by CMD3(L) can be
evicted while the GPU still reads it.

## Method

- A/B with `tools/quick_decode_bench.sh`, PIN off vs PIN_GB=8 on 32GB warm.
- Watch `expert_io` phase and `[expert-pin]` hit stats.

## Observations

(to be filled after measurement)

## Verdict

(pending)
