# Flashchat Speed-Attempt Archive

Every decode/prefill speed investigation lands here (or under
`docs/explorations/` for the 2026-08-17 systematic campaign) so future work
does not rediscover dead ends.

## Layout

| Path | What |
|------|------|
| `docs/explorations/` | Original exp/00–11 campaign (branches + NOTES + raw benches) |
| `docs/POTENTIAL_OPTIMIZATIONS.md` | Code-review candidates (prefill RAM, pin slots, MTP KV) |
| `docs/PINNED_EXPERT_CACHE_SIZING_LOG.md` | Slot-based pin sizing control surface |
| `docs/PREFILL_RELEASE_BENCHMARK_LOG.md` | Post-prefill buffer release A/B |
| `data/2026-08-24-deep-speed-analysis/` | This session’s full hot-path analysis + ranked roadmap |
| `data/attempts/<NN>-<slug>/` | One directory per new attempt (NOTES + raw) |

## Protocol (mandatory)

1. Read this index + the matching NOTES before starting.
2. Interleave A/B runs (±6% ambient drift on the 32GB box).
3. Inner loop: `tools/quick_decode_bench.sh data/attempts/<slug> 150 3`
4. Sign-off: `bash tests/bench_api.sh --model-id Qwen-Qwen36-35B-A3B` + `make bench-report`
5. Verdict in NOTES: KEEP / PARK / DEAD + why + numbers.

## Status board (2026-08-24)

### Shipped / KEEP (already in main or recommended default)

| # | Name | Result |
|---|------|--------|
| 01 | pin zero-copy | +10% decode @4GB pin, 32GB warm |
| 02 | pin mlock | +15–24% under 16GB pressure; +6% warm |
| 03 | adaptive-K mass | +4.6% opt-in (flat router) |
| 07 | fuse CMD1+CMD2 linear | +11.7% A/B, default-on |
| 11 | ANE/GPU prefill crossover | −37% short cold prefill |
| — | prefill buffer release | RAM → pin/page-cache (landed) |

### DEAD (do not rebuild without new evidence)

| # | Name | Why |
|---|------|-----|
| 02 | F_NOCACHE | unstable / page cache helps misses |
| 02 | pin decode-decay | no hit-rate or speed effect |
| 05 | matvec occupancy remap | v3 already 8×8; cost is RTT |
| 09 | prev-token / frequency prefetch | 5.7–8.2% of miss pool only |

### PARKED (needs fix or 16GB hardware)

| # | Name | Blocker |
|---|------|---------|
| 04 | MTP batched verify | 0.76× net @B=2; need skip-spec + B≥3 |
| 09 | learned router probe | ~10% 16GB upside; needs training pipeline |

### This session (2026-08-24) — measured

| # | slug | Verdict | Result |
|---|------|---------|--------|
| 12 | pin A/B | no change | re-validated shipped default (pin was already ON by default); local pin-off state was personal |
| 13 | io-threads-k8 | NEUTRAL | decode already GCD; default pool 8 |
| 14 | expert-read-split | **KEEP** | +6.5% pin-off, +~10% over shipped defaults → **25.2 tok/s**; schema v12 default |
| 15 | lm-head-resident | **KEEP** | mlock default on (schema v12), headroom-guarded; +15–20% under pin pressure |

### This session (2026-08-25) — measured

| # | slug | Verdict | Result |
|---|------|---------|--------|
| 23 | opencode-context-pin-sizing | **MEASURED — retain 8 GiB for this workload** | Real ~16k OpenCode: 8 GiB 8.63–8.64 tok/s; 4 GiB 5.52–5.58 (72.7% hits, 6,920 evictions); 12 GiB max safely resolved 9.13 GiB and 8.14–8.37, no win. Warm q8 64k vs 262k neutral; fp32 slower. |

### Still open

| # | slug | Priority |
|---|------|----------|
| 16 | earlier-io-overlap | P3 (mostly superseded by 14) |
| 17 | full-attn-fuse | P2 |
| 18 | ngram-spec | P2 |
| 19 | expert-hot-repack | P3 |
| 20 | mtp-skip-spec | P1 for 16GB >15 tok/s |

See `2026-08-24-deep-speed-analysis/ANALYSIS.md` + `SESSION_RESULTS.md`.
