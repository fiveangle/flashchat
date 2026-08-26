# 21 — Expert I/O restructuring: pread-into-slot + event-gated CMD3

Date: 2026-08-25
Status: **MEASURED** — two sub-attempts, both byte-identical; verdicts below.

Machine: 32GB MBP (Mac17,2). Model: Qwen3.6-35B-A3B q4. Fuse on, MTP off,
lm_head mlock on, split-io on (schema v12 defaults). TM idle, no thermal
warnings. All runs interleaved per protocol.

Binaries: `bin/infer-base` (pristine ab27fb0 build), `bin/infer-21b-events`
(final tree with FLASHCHAT_EXPERT_IO_EVENTS flag).

---

## Part A — pread misses directly into reserved pin slots ("pread-into-slot")

### Hypothesis
`expert_pin_admit` memcpys the full 1.69 MiB expert from pool buffer into the
pin arena under `g_pin_mu` after every miss read (~18% of loads at pin-on).
Reserving the slot up front and pointing the pread at slot memory removes the
copy entirely (~2.5 MB/layer of pure memcpy off the critical path).

### Implementation (`infer.m`)
- `expert_pin_reserve_slot()` — same LFU victim policy as admit; arms the
  gid<->slot mapping immediately; returns the slot's zero-copy MTLBuffer.
- `expert_pin_release_reservation()` — unmaps on failed reads so a garbage
  slot never serves a future hit.
- Misses pread into the reserved slot; admission step becomes a no-op for them.
- Each reservation joins the batch's protected-slot set immediately (a sibling
  reservation must not evict a half-loaded slot).
- Failure semantics: short read -> release reservation; expert contributes 0
  (weight zeroing paths unchanged).

### Result (interleaved, PIN=4GB, 150 tok x3)
| binary | median tok/s | runs |
|--------|-------------:|------|
| base | 21.61 | 21.32 / 21.81 / 21.61 |
| 21a slot-reserve | **23.35** (+8.0%) | 23.15 / 23.35 / 23.39 |

Raw: `ab-slot/`. Byte-identical verified (100 tok, pin on): `verify_identical.sh`.

### Verdict A: **KEEP**

---

## Part B — event-gated CMD3 (`FLASHCHAT_EXPERT_IO_EVENTS=1`)

### Hypothesis
Even with split-I/O, the CPU blocks on wave1/wave2 preads *before* CMD3 can be
encoded+committed, so GPU/SSD/CPU run serially per layer (~0.4 ms/layer of
GPU-idle CPU work: topk + wave waits + encode). With an `MTLSharedEvent`:
1. miss preads enqueue and return instantly;
2. GCD completion handlers validate/retry, release failed reservations, zero
   failed experts' combine weights, then signal v1 (gate+up ready) / v2 (down ready);
3. CMD3 encodes IMMEDIATELY with `encodeWaitForEvent` gates where expert data
   is consumed, commits without any CPU wait;
4. shared-expert kernels move ahead of the gates (independent of routed data);
5. both waves issue concurrently (QD~16 instead of serialized 8+8).
The GPU blocks on the event instead of the CPU; the CPU runs ahead encoding
the next layer while SSD and GPU work overlap.

### Implementation
- `MetalCtx.io_event` + monotonic value pairs (v1=2n, v2=2n+1 per launch).
- `async_pread_start_events()` — heap EventPreadCtx; dispatch_group_notify
  handlers do validation/retry/admission-release and signal the event.
- Encode site branches: `[cmd encodeWaitForEvent:]` between encoders; single
  commit for what used to be phase-A + phase-B buffers.
- Combine uses NEW kernel `moe_combine_residual_guarded` (isfinite guard per
  term) ONLY in event mode — see "shader lesson" below.
- CPU finalize path gained an equivalent finite-scan guard (last-layer combine).
- Timing: handler-side ns accumulate into `g_event_io_ns`; per-layer expert_io
  bucket measures start->encode-end in ev mode (approximate, noted).

### Shader lesson (important — cost half a session)
First attempt modified `moe_combine_residual` in place to add the isfinite
guard. That changed Metal codegen for the DEFAULT path too → low-bit drift →
topk order flips among near-equal logits → visible output divergence at pin-on
(~token 35+). Fix: original kernel restored byte-for-byte; guarded variant is
a SEPARATE kernel bound only when events are active. Rule going forward:
**never edit a hot-path kernel in place for a failure-path concern — add a
variant and bind it conditionally.** (Also found en route: an earlier
two-pass restructure of `async_pread_start` diverged at pin-off even though
data flow was "identical"; restored to the original single-loop structure with
the reservation hook gated inside it — divergence gone. Keep hot loops'
operation ORDER stable unless proven otherwise.)

### Correctness (120-token outputs, text-diff vs pristine base binary)
| config | result |
|--------|--------|
| EV=0, PIN=0 | IDENTICAL |
| EV=0, PIN=4 | IDENTICAL |
| EV=1, PIN=0 | IDENTICAL |
| EV=1, PIN=4 | IDENTICAL |

Traced to root cause earlier via FNV traces (RT/GS/HP/LD lines): loaded expert
bytes were always identical; divergence entered through combine numerics.

### Performance (interleaved, 150 tok x3, `ab-events/`)

| config | EV=0 median | EV=1 median | delta |
|--------|------------:|------------:|------:|
| PIN=4GB | 22.17 / 22.10 / 24.37 → **22.10** | 21.32 / 21.58 / 22.57 → **21.58** | **−4%** |
| PIN=0 | 19.74 / 19.73 / 19.79 → **19.74** | 12.35 / 11.39 / 10.85 → **11.39** | **−42%** |

### Verdict B: **DEAD as implemented — do not ship; code removed from tree**

Mechanism of the loss (important for anyone re-attempting GPU-side I/O waits):

1. **Bandwidth contention kills the split-I/O overlap.** The shipped split path
   issues wave1 (gate+up) alone, computes gate/up/SwiGLU on GPU while wave2
   (down) reads *after* phase-A commit — down bytes ride free under GPU compute.
   Event-gating issues BOTH waves up front, so wave2 competes with wave1 for
   SSD bandwidth exactly when wave1 latency is on the critical path (GPU is
   blocked at v1). At PIN=0 (all misses) this collapses overlap entirely: −42%.
2. **CPU run-ahead does not help** because the bottleneck is GPU-side
   serialization on the event wait: encoding ahead just queues buffers behind
   a stalled CMD3. There is no independent GPU work to fill the gap (routing
   for layer L+1 needs L's output).
3. Even at high pin hit-rate (few misses), handler dispatch + event-signal RTT
   costs more than the CPU spin-wait it replaced: −4%.

What WOULD be required to revisit (parked): issue wave1-only before commit,
enqueue wave2 after phase-A commit (preserving old overlap), and use events
only to replace the *wave2* CPU wait. Expected ceiling ~+2-4%; not worth the
complexity today given attempt 14 already banks the same overlap.

### Final shipped state (this attempt)

Tree contains **Part A only**: `expert_pin_reserve_slot` /
`expert_pin_release_reservation` + reservation hook inside the original
single-loop `async_pread_start` + reserved-slot handling in both wave-wait
paths (+100 lines). `shaders.metal` untouched vs HEAD.

Correctness of final binary (`bin/infer-final`), 120-token text diff vs
pristine base: IDENTICAL at PIN=0 and PIN=4.

Final sign-off A/B: `ab-final-summary.txt` + `ab-final/`.


---

## Durable sign-off (tests/bench_api.sh @ commit 2561405 tree + this change)

Qwen-Qwen36-35B-A3B q4, as-shipped defaults, healthy preflight:

| scenario | prior (e85192f) | current | decode delta |
|----------|----------------:|--------:|-------------:|
| chat_stream:technical    | 18.86 tok/s | 25.57 tok/s | +35.6% |
| chat_stream:creative     | 20.11 tok/s | 23.85 tok/s | +18.6% |
| chat_stream:worstcase    | 16.86 tok/s | 20.24 tok/s | +20.0% |
| responses_stream:technical | 19.31 tok/s | 26.39 tok/s | +36.7% |
| responses_stream:creative | 21.68 tok/s | 25.62 tok/s | +18.2% |
| responses_stream:worstcase | 16.67 tok/s | 20.88 tok/s | +25.2% |

Prefill also improved +15–40% (ANE crossover + split-io defaults now in the
durable path). Rows appended to assets/api_perf_log.tsv (preserved per policy).

Isolated attribution of THIS attempt (interleaved A/B, same session,
`ab-final/`): base median 24.71 → final 25.32 tok/s at PIN=4 (**+2.5%**,
wins all 3 pairs); PIN=0 neutral by design (19.17 → 19.16).

`make bench-report` flags 3 marginal ▼ items — ALL on other models whose
latest rows predate this session (e237d9a/f9a56c8 comparisons; e.g. a
42 s tool-call test ±2 s). Not touched by this change; pre-existing stale
cross-model noise. No regression on any benchmarked-today model.

## Follow-ups not attempted (documented for future sessions)

- Attempt 22 design (GPU tail fusion: final norm + lm_head into last CMD3,
  logits-only readback) — sketched in session, est. +1–3%, NOT implemented.
- Idle-time page-cache warmer (attempt 24 design): background LFU prefetch of
  unpinned hot experts between requests; targets 16GB boxes where page cache
  is the only tier. NOT implemented.
- F_RDADVISE before miss waves: untested; likely noise-level next to 21a.
- iogpu.wired_limit_mb guidance for 16GB: doc-only candidate, needs hardware.
