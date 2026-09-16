# Flashchat Speed-Attempt Archive

Historical campaign status board. All numbered investigations now live under
`docs/explorations/<index>-<experiment>/`. See [README.md](README.md) for the
archive index and [AGENTS.md](../../AGENTS.md) for current working rules.

## Layout

| Path | What |
|------|------|
| `docs/explorations/<index>-<experiment>/` | All experiments (NOTES + raw results) |
| `docs/POTENTIAL_OPTIMIZATIONS.md` | Code-review candidates (prefill RAM, pin slots, MTP KV) |
| `docs/PINNED_EXPERT_CACHE_SIZING_LOG.md` | Slot-based pin sizing control surface |
| `docs/PREFILL_RELEASE_BENCHMARK_LOG.md` | Post-prefill buffer release A/B |
| `12-pin-default-on/campaign-analysis/` | Shared August 24 hot-path analysis + ranked roadmap |

## Reading the historical results

1. Read this index + the matching NOTES before starting.
2. These measurements had ±6% ambient drift on the 32GB box; small deltas are
   not reliable universal gains or regressions.
3. Inner-loop output belongs in `docs/explorations/<index>-<experiment>/`.
4. Use the Make-based benchmark and shipping-target policy in AGENTS.md.
5. Record verdicts in NOTES: KEEP / PARK / DEAD / INCONCLUSIVE + evidence.

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
| 05 | matvec occupancy remap | Deprioritized by review, not measured dead; see [05 notes](05-matvec-occupancy/NOTES.md) |
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
| 21a | pread-into-slot (pin reserve) | **KEEP** | +2.5% @pin-on shipped defaults, all interleaved pairs win; neutral pin-off; byte-identical; bench-api decode +18–37% vs e85192f |
| 21b | event-gated CMD3 (MTLSharedEvent) | **DEAD — removed** | −4% pin-on, −42% pin-off: issuing wave2 with wave1 competes for SSD bandwidth and destroys split-I/O overlap; GPU-side event wait serializes what CPU-spin overlapped. See NOTES "shader lesson" before touching hot kernels. |
| 23 | opencode-context-pin-sizing | **MEASURED — retain 8 GiB for this workload** | Real ~16k OpenCode: 8 GiB 8.63–8.64 tok/s; 4 GiB 5.52–5.58 (72.7% hits, 6,920 evictions); 12 GiB max safely resolved 9.13 GiB and 8.14–8.37, no win. Warm q8 64k vs 262k neutral; fp32 slower. |

### Still open

| # | slug | Priority |
|---|------|----------|
| 16 | earlier-io-overlap | P3 (mostly superseded by 14) |
| 17 | full-attn-fuse | P2 |
| 18 | ngram-spec | P2 |
| 19 | expert-hot-repack | P3 |
| 20 | mtp-skip-spec | P1 for 16GB >15 tok/s |
| 22 | gpu-tail-fusion (final norm+lm_head in last CMD3) | P2 design sketched in 21 NOTES |
| 24 | idle page-cache warmer | P2 design sketched in 21 NOTES (16GB lever) |

See [analysis](12-pin-default-on/campaign-analysis/ANALYSIS.md) and
[session results](12-pin-default-on/campaign-analysis/SESSION_RESULTS.md).
