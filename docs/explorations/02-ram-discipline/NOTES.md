# 02 — RAM discipline (F_NOCACHE / decode-decay / mlock)

Branch: `exp/02-ram-discipline` (from main @ cdb14b8)

## Hypothesis (three sub-knobs)

1. FLASHCHAT_EXPERT_NOCACHE=1 — F_NOCACHE on packed-expert fds so the
   ~540MB/token miss-stream stops evicting the hot non-expert weight mmap
   and KV state.
2. FLASHCHAT_PIN_DECODE_DECAY=1 — halve g_expert_freq at decode start so
   prefill-trained LFU victims give way to decode-hot experts.
3. FLASHCHAT_EXPERT_PIN_MLOCK=1 (existing knob) — wire the pin arena so
   hits are truly RAM-resident (anonymous pages get compressed under
   pressure, making "hits" silently slow).

## Method

tools/ram_discipline_matrix.sh: fresh server per config (cold pin), 549-token
prompt, 80-token decode, temp 0, one request. 16GB emulation =
tools/ram_pressure 17GiB on the 32GB dev box. Three rounds (r1/r2/r3; r3 =
full interleaved matrix) to bound ±7% ambient drift.

## Observations (decode tok/s)

| config | r1 | r2 | r3 |
|---|---|---|---|
| control pin3G | 11.13 | 10.37 | 10.55 |
| +NOCACHE | 10.20 | — | 10.73 |
| +MLOCK | 12.84 | 12.28 | 12.81 |
| +MLOCK+NOCACHE | — | 13.45 | 11.87 |
| +MLOCK+NOCACHE+DECAY | — | — | 11.92 |
| pin4G+MLOCK+NOCACHE | — | — | 12.87 |
| 32GB WARM pin4G | 21.44 | | |
| 32GB WARM pin4G+MLOCK | 22.78 | | |

Pin hit rate identical (70.9% @3GB, 76.8% @4GB) across all knobs within a
pin size — the knobs change hit SPEED, not hit RATE (except pin size).

## Verdicts

- **MLOCK: KEEP — the real win.** +15–24% under 16GB emulation, +6% on warm
  32GB, consistent across every round. Mechanism: macOS compresses the pin
  arena's anonymous dirty pages under pressure; pin "hits" then fault through
  the compressor. mlock makes hits true RAM reads. The knob already exists
  and is config-wired (default 0); RECOMMENDATION: set EXPERT_PIN_MLOCK=1
  whenever the pin cache is enabled and the pin size leaves the system
  headroom (~3–4GB on 16GB machines). Consider defaulting to 1 with a
  wizard-guided size.
- **NOCACHE: DO NOT ENABLE.** Unstable across rounds (−8%, +9.5%, −7% on
  top of mlock); the emulation cannot resolve an effect, and the page cache
  evidently serves part of the miss stream usefully. Env knob kept for
  future re-test on real 16GB hardware, documented dead here.
- **DECAY: DO NOT ENABLE.** No hit-rate or speed effect in any round
  (prompt experts and decode experts overlap enough that prefill training
  isn't hurting admission).
- Pin 4GB vs 3GB under emulation: no decode difference (locked arena grows
  at the expense of miss-serving cache) — on a real 16GB machine the
  optimum needs hardware testing; 3GB remains the emulated recommendation.

## Config takeaway for low-RAM machines (best measured)

FLASHCHAT_EXPERT_PIN_MAX_GB=3, AUTO_FRAC=0.95, MLOCK=1 (fuse already
default-on). Emulated decode: 10.5 -> 12.8 tok/s (+22%) from mlock alone on
top of the earlier fuse+pin gains.
