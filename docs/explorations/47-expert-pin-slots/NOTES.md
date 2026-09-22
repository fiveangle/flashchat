# 47 — Expert pin cache sized in whole experts

Date: 2026-07-08. Registered 2026-09-22.
Commit: `86c8846` (same commit as prefill-buffer release, 46).
Status: **Closed.** Shipped before it had an exploration ID.
Verdict: **Keep** as a control surface. No throughput measurement.

## Hypothesis

The pin cache already stored complete experts, but the user cap was a GiB guess that the runtime rounded down. A slot count is the real unit, especially on small-memory machines.

## What the commit did

`86c8846` added `EXPERT_PIN_MAX_EXPERTS` / `FLASHCHAT_EXPERT_PIN_MAX_EXPERTS` and bumped the config schema to v9. Wiring went through `lib/config.sh` (default, migration, env override, getter), `flashchat` serve-time export and `config --show`, and the config wizard.

Resolution, still the current rule:

- Empty slot value: derive the slot count from `EXPERT_PIN_MAX_GB`, as before.
- Positive integer: that many complete experts, then clamp.
- `EXPERT_PIN_MAX_GB` remains the hard ceiling and the off switch. `0` disables the cache even if a slot count is set.
- `EXPERT_PIN_AUTO_FRAC` still caps the arena from reclaimable free RAM at first expert read.
- The arena cannot exceed the model's expert count.
- Allocation is `slots * active_expert_size()`. No partial expert.

The wizard guidance is computed from the user config and registry architecture, not from built runtime artifacts. For Qwen3.6-35B-A3B q4 it reported: recommended 640 experts (1.1 GiB), about two generated tokens of K=8 choices across 40 layers; smaller 320 (540.0 MiB); larger 1280 (2.1 GiB); maximum useful 10240 (16.9 GiB). A token can touch up to `layers * K` = 320 expert activations.

## Validation

Not a speed test. Isolated config migration reached schema 9 and left `EXPERT_PIN_MAX_EXPERTS` empty, with `max_gb=4` and `auto_frac=0.5`.

Native cap check: `FLASHCHAT_EXPERT_PIN_MAX_GB=1` and `FLASHCHAT_EXPERT_PIN_MAX_EXPERTS=2560` logged:

```text
[expert-pin] enabled: 606 experts pinnable (1022.62 MiB, 1.00 GiB), free RAM 10.32 GiB, GB cap 1.00 GiB, requested 2560 experts, pool total 10240 experts
```

The GiB ceiling won, as intended. Retained write-up: [PINNED_EXPERT_CACHE_SIZING_LOG.md](PINNED_EXPERT_CACHE_SIZING_LOG.md).

## Conclusion

Keep the slot cap. It makes the existing whole-expert arena explicit. It is not a tokens/sec result and must not be added to the pin-cache speed gains in 01, 02, or 12. The original follow-up — benchmark 160/320/640/960 experts on a real 16 GB machine — was not run here. Decode-frequency decay, proposed in the same July review, was later measured and rejected as 02.
