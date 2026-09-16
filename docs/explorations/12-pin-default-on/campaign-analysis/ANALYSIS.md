# Deep Decode-Speed Analysis — 2026-08-24

Machine context: 32GB Mac17,2 (MBP M-class). Target: 16GB Mac Mini M4
usability threshold ≈ 15 tok/s on Qwen3.6-35B-A3B q4.

This document is a code-level + historical synthesis. It does **not** claim
new measured wins until the experiment dirs under `docs/explorations/` are filled.

---

## 1. Cost model (the wall)

Qwen3.6-35B-A3B q4 (from registry + `model_config.h` layout math):

| Quantity | Value |
|----------|------:|
| layers | 40 |
| experts / layer | 256 |
| active K | 8 |
| expert size | 1,769,472 B (1.6875 MiB, 108 × 16KB pages) |
| expert bytes / token | 320 × 1.69 MiB ≈ **540 MB** |
| lm_head (q4) | **286 MB** / token |
| gate+up fraction of expert | 66.7% (contiguous prefix) |
| down fraction of expert | 33.3% (contiguous suffix) |

Per-token working set ≈ 0.8+ GB when cold. On a 16GB machine the OS +
non-expert weights + KV leave only a few GB for pin/page-cache, so most of
that set is real SSD traffic every token.

Serial layer phases (baseline 00, warm 32GB, pin off):

```
cmd1_wait 0.62 + cmd2_wait 0.33 + expert_io 0.38 ≈ 1.32 of 1.36 ms/layer
```

GPU and SSD almost never work at the same time. **Overlap is the only free
speed left once bytes/token is fixed.**

Cold-SSD ceilings (sequential GB/s assumptions):

| SSD GB/s | expert ms | lm_head ms | tok/s I/O-only |
|---------:|----------:|-----------:|---------------:|
| 3 | 189 | 95 | 3.5 |
| 5 | 113 | 57 | 5.9 |
| 7 | 81 | 41 | 8.2 |

Pin hit-rate effect on expert I/O alone @5 GB/s:

| hit% | expert ms | expert-only tok/s |
|-----:|----------:|------------------:|
| 0 | 113 | 8.8 |
| 50 | 57 | 17.7 |
| 70 | 34 | 29.4 |
| 88 | 14 | 73.6 |

**Implication:** on 16GB, every percent of pin hit rate and every MB of
lm_head residency is worth more than almost any kernel tweak.

---

## 2. What the engine already does well

Pipeline (`fused_layer_forward`, header comment + live code):

1. **CMD1** — attention projections (+ full GPU linear-attn chain when enabled)
2. **CPU** — RoPE / softmax / or nothing if GPU linear
3. **CMD2** — o_proj + residual + RMSNorm + router + shared gate/up
4. **CPU** — softmax + top-K
5. **I/O** — parallel `pread` of K experts (GCD / pin zero-copy)
6. **CMD3** — K expert forwards + shared SwiGLU/down + GPU combine + next-layer norm
   (async commit; next layer’s CMD1 queues behind it)

Already landed wins (see `docs/explorations/README.md`):

| Path | Gain | Notes |
|------|------|-------|
| pin zero-copy (01) | +10% warm | memcpy-under-mutex removed |
| fuse linear CMD1+CMD2 (07) | +12–14% | default-on via `FUSE_LINEAR=1` |
| pin mlock (02) | +15–24% under pressure | anonymous pin pages were being compressed |
| adaptive-K (03) | +4.6% | opt-in; router is flat |
| ANE/GPU prefill crossover (11) | −37% short TTFT | decode-neutral |
| prefill buffer release | RAM reclaim | feeds pin/page-cache |
| 2MB-aligned expert buffers | 3.6× DMA vs 16KB | already in metal_setup |
| GPU-side MoE combine | kills deferred CPU wait | already default |
| matvec v3 / matmulN | +14% on big matvecs | default |

Dead / parked (do not rebuild casually): F_NOCACHE, pin-decay, prev-token
prefetch, matvec occupancy remap, MTP@B=2 without skip-spec.

---

## 3. Critical live-config finding

User config on this machine (`~/.config/flashchat/config`):

```
EXPERT_PIN_MAX_GB="0"          # ← DISABLES the entire pin cache
EXPERT_PIN_MAX_EXPERTS="2560"
EXPERT_PIN_MLOCK="1"
FUSE_LINEAR="1"
ADAPTIVE_K_MASS="0.95"
MTP="0"
```

In `expert_pin_init` (`infer.m` ~7486):

```c
if (cap_gb <= 0.0) return;   // disabled
```

So **all measured pin wins are currently off** on the interactive path,
even though MLOCK and slot targets are set. Durable `bench-api` rows at
e85192f (18–21 tok/s short-ctx) reflect fuse + warm page cache, not pin.

This is the single largest “free” lever before any new code.

---

## 4. Structural gaps still in the code

### 4.1 Double-buffer is commented, not live (decode path)

Header promises `buf_multi_expert_data / data_B`. Only set A exists.
`main.m` still has a real layer-N / layer-N+1 double-buffer for the
synthetic full-forward benchmark, but production decode **cannot** prefetch
layer L+1’s experts until routing at L+1 finishes — true cross-layer
double-buffer needs a predictor (exp/09 killed cheap ones).

What *is* live: `async_pread_start` after top-K, overlapping only:

- a few `memcpy`s of h_post / shared gate-up when not GPU-resident
- or **nothing** on the fuse+GPU-resident path

So the “async pread” window is microseconds on the hot path. Expert I/O
and CMD3 remain almost fully serial. See attempt 16.

### 4.2 NUM_IO_THREADS = 4 while production K = 8

```c
#define NUM_IO_THREADS 4  // 4 threads for K=4 experts (one per expert)
```

Decode uses GCD `dispatch_group_async` (one block per miss), which is
better, but the pthread pool path and the comment still encode a K=4
assumption. Prefill CSR waves use the pool. Worth forcing 8 concurrent
preads and measuring cold miss latency (attempt 13).

### 4.3 Expert layout enables split-read for free (exp/08 never built)

Packed expert layout (`model_config.h`):

```
[gate_w|s|b | up_w|s|b | down_w|s|b]
 0 ............... 1,179,648 .... 1,769,472
        66.7%              33.3%
```

gate+up is one contiguous prefix. Pipeline:

1. `pread` gate+up only (1.125 MiB)
2. encode gate+up+SwiGLU on GPU **while** `pread` down (0.562 MiB)
3. encode down when ready

On cold 16GB this hides ~1/3 of expert I/O behind GPU matvecs that already
run. Conservative estimate: 10–20% of miss-path layer time. Byte-identical.
See attempt 14.

### 4.4 lm_head is 286 MB/token and unprotected

`lm_head_forward` is a full vocab matvec every token. Warm: ~4 ms (in
page cache). Cold 16GB: **~40–95 ms** — can dominate the entire decode
budget even if experts are free.

No pin/mlock/priority residency for lm_head today. Non-expert mmap relies
on OS luck under expert thrash. Attempt 15: mlock (or explicit pin) the
lm_head weight/scale/bias tensors; optionally keep a small “hot vocab”
working set if a draft/spec path only needs top-k (harder, quality risk).

### 4.5 Fuse stops at linear layers

`fuse_cmd12` requires `gpu_linear_attn_will_run`. Full-attn layers still
pay CMD1 wait → CPU RoPE/KV append → CMD2. GPU attention already defers
into CMD2 when `seq_len >= 32`, but the commit boundary remains. Attempt 17.

### 4.6 MTP economics broken at B=2

exp/04: 76% acceptance but **0.76× net** because low-margin drafts fall
back to *two* production forwards. Fix: skip-spec (draft only when margin
high; else plain step) + B≥3 + measure on real 16GB. This is still the
only documented path that can push a base-M4 16GB box past ~15 tok/s after
fuse+pin+mlock (~10–13 extrapolated). Attempt 20.

### 4.7 Speculative decoding without a draft model

Prompt-lookup / n-gram speculation (PLD, REST, etc.) drafts from recent
context with **zero** extra weights. On code/chat with repetition, 20–40%
token acceptance is common in the literature; rejected drafts cost one
verify forward. MoE verify can use union expert I/O (already instrumented).
Untouched in this repo. Attempt 18.

### 4.8 Hot-expert disk locality

Experts are packed by index 0..255 per layer. LFU hot set is scattered →
random SSD reads. A background repack that co-locates the top-N experts
per layer (or a secondary hot file) converts misses to sequential DMA.
High engineering cost; attempt 19.

### 4.9 Adaptive-K quality vs speed

Router is flat (K≈6.92 at mass 0.90). +4.6% already measured. Not default
because computation changes. For 16GB usability, shipping mass=0.90 as
default (or profile-specific) is a product decision, not a research one.

---

## 5. Ranked roadmap (what to try next)

Priority scores combine: expected tok/s delta on **16GB cold**, engineering
risk, and whether prior work already bounded the upside.

| Pri | Attempt | Expected 16GB delta | Risk | Effort |
|----:|---------|---------------------:|------|--------|
| P0 | **12 pin-default-on** (config only) | +15–35% (restore 01+02) | none | minutes |
| P0 | **15 lm-head-resident** | +10–40% cold (stops thrash) | low | small |
| P1 | **13 io-threads / GCD depth** | +2–8% miss path | low | small |
| P1 | **14 expert-read-split** | +8–18% miss path | med | medium |
| P1 | **20 MTP skip-spec + B≥3** | possibly 1.2–1.6× if fixed | high | large |
| P2 | **16 earlier I/O overlap** | +3–10% if any window found | med | medium |
| P2 | **17 full-attn fuse** | +3–8% | med | medium |
| P2 | **18 ngram / prompt-lookup** | workload-dependent 0–30% | med | medium |
| P3 | **19 hot-expert repack** | +5–15% cold miss | high | large |
| — | learned router probe (09) | ~+10% | high | large |
| — | lower default K / adaptive default | +5–15% | quality | config |

**Do not expect kernel micro-opts alone to hit 15 tok/s on 16GB.** The
bytes/token math forbids it. The winning stack is:

```
fuse (done) + pin 3–4GB mlock (config) + lm_head resident
  + split-read overlap + (later) fixed MTP or n-gram spec
```

Extrapolated from prior 16GB emulation (fuse+pin3G → +37.6% over base):

| Stack | ~tok/s on real base-M4 16GB (est.) |
|-------|-------------------------------------:|
| base (user-reported) | 7–10 |
| + fuse + pin3G mlock | 10–13 |
| + lm_head resident + split-read | 12–15 |
| + working skip-spec MTP or PLD | 15–20 (if acceptance holds) |

---

## 6. Out-of-the-box ideas examined and rejected (for now)

| Idea | Why not first |
|------|----------------|
| ANE for decode experts | ANE path is prefill-oriented; decode experts are tiny matvecs; dispatch overhead dominates |
| Metal 4 TensorOps matmul2d | AGENTS.md: not production-ready; buffer-index and tile limits |
| mmap experts instead of pread | random 1.7MB touches; pread+aligned DMA already tuned 3.6× |
| Quantize experts to 2-bit | quality / format project; not a hot-path tweak |
| CPU expert compute | GPU already faster; bandwidth still paid |
| Bigger K parallelism on GPU | compute not the bottleneck when cold |
| Cross-layer expert prefetch without predictor | routing unknown; exp/09 ceiling killed cheap signals |
| F_NOCACHE / decay | measured dead in exp/02 |
| Holding pin mutex during memcpy | already fixed by zero-copy (01) |

---

## 7. Immediate actions (this session recommendations)

1. **Turn pin on** for interactive use: `EXPERT_PIN_MAX_GB=4` (32GB) or `3`
   with `MLOCK=1` (16GB). Confirm with quick bench before any code change.
2. **Instrument lm_head stall**: add timing bucket for lm_head wall time and
   RSS of weight file under pressure (feeds attempt 15).
3. **Implement attempt 14 (split-read)** behind `FLASHCHAT_EXPERT_SPLIT_IO=1`
   — highest-upside untried code path with clean layout support.
4. **Bump IO concurrency** to 8 for K=8 (attempt 13) — one-line + measure.
5. Park MTP until skip-spec is designed; do not default MTP on.

---

## 8. Files touched by this analysis (read-only)

- `metal_infer/infer.m` (hot path, pin, async pread, fuse, lm_head, MTP)
- `metal_infer/shaders.metal` (kernels)
- `metal_infer/main.m` (legacy double-buffer benchmark path)
- `metal_infer/model_config.h` (expert layout)
- `docs/explorations/**` (exp 00–11)
- `docs/POTENTIAL_OPTIMIZATIONS.md`
- `assets/api_perf_log.tsv` (healthy rows @ e85192f)
- `~/.config/flashchat/config` (pin disabled)

No engine code was modified in this analysis pass.
