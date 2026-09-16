# Session results — 2026-08-24 (unattended continuation)

Machine: 32GB Mac17,2. Model: Qwen3.6-35B-A3B q4. Fuse on, MTP off.
TM idle. All work archived in the repository (no Desktop/tmp dependency).

## desktop-tmp-stuff-cleanup

No LaunchAgent/cron matching that name found. No cleanup job that would
touch `/Users/speedster/dev/flashchat`. All artifacts written only under
the repository. Experiments are now under `docs/explorations/`; shared analysis
is alongside experiment 12 in `campaign-analysis/`. **Safe.**

## Stack progression (quick_decode_bench 150 tok median)

| Step | tok/s | Δ vs prior | Notes |
|------|------:|-----------:|-------|
| Pin-off local test state | 17.4 | — | personal config, NOT shipped default |
| Shipped default (pin 4G) | ~23 | — | already the product baseline |
| + lm_head mlock (new code, default on) | 22.2–23.9 vs **19.5 off** | +15–20% under pin pressure | stabilizes decode |
| + expert split I/O (new code) | **25.2** | +~10% over shipped default | lossless |

Pin-off miss path: 17.6 → 18.8 with split alone (+6.5%).

**Correction:** the "+34% from enabling pin" seen mid-session was against
the dev box's transient pin-off config. The shipped default
(`EXPERT_PIN_MAX_GB=4`) already provided that; attempt 12 is archived as a
re-validation only.

## Attempt verdicts

| # | Verdict | Notes |
|---|---------|-------|
| 12 pin A/B | no change | re-validated shipped default; see NOTES correction |
| 13 io-threads | NEUTRAL | Default pool 8; decode uses GCD already |
| 14 expert-split-io | **KEEP** | Default ON (schema v12); lossless |
| 15 lm-head mlock | **KEEP** | Default ON (schema v12); headroom-guarded |

## Code changes (committed together — intertwined in infer.m)

`metal_infer/infer.m`:

1. lm_head mlock with free-RAM + 512 MiB headroom guard (default on)
2. Expert split I/O: gate+up wave / down overlap on misses (default on)
3. Weight mmap MADV_SEQUENTIAL → MADV_RANDOM (decode access is random)
4. IO pool width configurable via FLASHCHAT_IO_THREADS (default 8)
5. Phased GPU encode helpers (gate_up_swiglu / down)
6. lm_head timing bucket under --timing
7. Pin-arena mlock now also respects the shared headroom guard (skips +
   logs instead of risking swap pressure)

Config wiring (schema v11 → v12): `LM_HEAD_MLOCK=1`, `EXPERT_SPLIT_IO=1`
defaults added through lib/config.sh (defaults/migrate/env/getter/default
file), flashchat (show/signature/serve bridge), config wizard advanced
menu. EXPERT_PIN_MLOCK default 0 → 1. Existing user values are never
rewritten (append-only backfill).

## Still open (not built this session)

- 16 earlier I/O window (mostly superseded by 14)
- 17 full-attn fuse
- 18 n-gram / prompt-lookup speculation
- 19 hot-expert disk repack
- 20 MTP skip-spec (still the 16GB >15 tok/s lever)
- `make bench-api` sign-off on a clean system for the new defaults
- 16GB hardware validation

## Extrapolated 16GB base-M4 path

```
7–10 user base
→ shipped defaults (pin+mlock+fuse)     ~10–13   (prior emu)
→ + lm_head mlock                      ~11–14
→ + split I/O                          ~12–15
→ + MTP/PLD if fixed                   15–20
```
