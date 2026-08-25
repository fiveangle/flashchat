# 12 — Expert pin cache A/B (config-only; NOT a product change)

Date: 2026-08-24  
Status: **MEASURED — no code change** (pin cache was already the shipped default)

## Context correction (post-session)

The dev box's interactive `~/.config/flashchat/config` had
`EXPERT_PIN_MAX_GB=0` as a **personal transient test state**, not a system
default. The shipped default (`FLASHCHAT_DEFAULT_EXPERT_PIN_MAX_GB`) is
already `4`. This A/B therefore re-validated the existing default against
that local pin-off state; it did **not** discover a production win. Kept
here so future agents do not re-flag "pin disabled" from a personal config.

## What was measured

Interleaved quick_decode_bench (150 tok × 3 runs), fuse on, MTP off.
TM idle (`tmutil status Running=0`). Machine: 32GB Mac17,2.

Order: off1 → on4g1 → off2 → on3g1 → on4g2

## Results (median decode tok/s)

| config | median | best | expert_io ms | total_layer ms | hit rate |
|--------|-------:|-----:|-------------:|---------------:|---------:|
| PIN off (1) | 17.20 | 17.45 | 0.452 | 1.369 | — |
| PIN 4G+mlock (1) | **22.98** | 23.16 | 0.207 | 1.009 | 81.8% (2427 slots) |
| PIN off (2) | 17.65 | 17.66 | 0.473 | 1.336 | — |
| PIN 3G+mlock | **22.82** | 22.86 | 0.203 | 1.015 | 74.5% (1820 slots) |
| PIN 4G+mlock (2) | **23.81** | 23.89 | 0.168 | 0.971 | 81.8% |

Off average median ≈ **17.4 tok/s**  
On 4G average median ≈ **23.4 tok/s** → **+34%**  
On 3G median **22.82** → **+31%**

expert_io: 0.46 → 0.17–0.21 ms (−55 to −63%)

Ambient off drift was only ~0.45 tok/s between blocks — win is far outside noise.

## Pin enable lines (confirmed)

```
[expert-pin] enabled (zero-copy): 2427 experts pinnable (4.00 GiB) ...
[expert-pin] hits 39266 / 48000 (81.8%), pinned 2427/2427, evictions 3706

[expert-pin] enabled (zero-copy): 1820 experts pinnable (3.00 GiB) ...
[expert-pin] hits 35767 / 48000 (74.5%), pinned 1820/1820, evictions 4553
```

Note: `requested 0 experts` because `FLASHCHAT_EXPERT_PIN_MAX_EXPERTS` was
not passed in env (user config has 2560 but quick bench uses env override
path for GB only). GB cap alone is sufficient.

## Verdict

**No product change.** Shipped default was already pin-on 4 GB. The A/B
re-confirms the default's value (+34% vs a local pin-off test state) and
the mlock benefit that exp/02 measured. No commit for this attempt beyond
this archive.

### Follow-ups (optional, separate)

1. Footgun: `MAX_GB=0` + nonzero `MAX_EXPERTS` silently disables cache —
   warn in wizard, or allow slots alone to enable (schema behavior change).

## Raw

- `off1/` `off2/` `on3g1/` `on4g1/` `on4g2/`
