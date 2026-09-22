# Pinned Expert Cache Sizing Log

Date: 2026-07-08

## Objective

Move the expert pin-cache sizing model away from only accepting an arbitrary GiB value and toward a cache capacity expressed as complete expert slots.

The motivation is especially important for small-memory targets. If the useful unit of caching is one complete packed expert, the control surface should be able to say "pin N experts" directly instead of asking the user to estimate a GiB value and letting the runtime round it down.

## Change Under Test

Added `EXPERT_PIN_MAX_EXPERTS` / `FLASHCHAT_EXPERT_PIN_MAX_EXPERTS`.

Behavior:

- Empty value: keep the previous behavior and derive slots from `EXPERT_PIN_MAX_GB`.
- Positive integer: use that as the requested number of complete expert slots.
- `EXPERT_PIN_MAX_GB` is always the hard GiB ceiling and off switch.
- `EXPERT_PIN_AUTO_FRAC` still caps allocation based on reclaimable free RAM at first expert read.
- The runtime still caps the cache at the total number of experts in the model.
- `EXPERT_PIN_MAX_GB=0` disables the cache even when `EXPERT_PIN_MAX_EXPERTS` is set.

The actual arena remains `slots * active_expert_size()`, so there are no partial expert allocations.

## Files Touched

- `metal_infer/infer.m`: native expert-pin sizing and log output.
- `lib/config.sh`: schema v9, default, migration/backfill, env override, getter, default config generation.
- `flashchat`: `config --show`, server signature, and serve-time env bridge.
- `modelmgr/tui/config_wizard.py`: advanced setting exposed in the config wizard with model-specific slot guidance.

## Validation

Build and syntax:

```bash
rtk bash -n flashchat
rtk make -B infer
```

Isolated config migration smoke:

```bash
HOME="$PWD/debug/fresh-envs/.../home" bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get EXPERT_PIN_MAX_EXPERTS'
```

Observed:

```text
Config migrated to schema v9: backfilled ... missing key(s)
schema=9
max_gb=4
max_experts=
auto_frac=0.5
EXPERT_PIN_MAX_EXPERTS=""
```

Native proof that expert slots are capped by the GiB limit:

```bash
rtk env FLASHCHAT_EXPERT_PIN_MAX_GB=1 FLASHCHAT_EXPERT_PIN_MAX_EXPERTS=2560 FLASHCHAT_EXPERT_PIN_AUTO_FRAC=1 FLASHCHAT_EXPERT_PIN_MLOCK=0 FLASHCHAT_ACTIVE_EXPERTS=8 FLASHCHAT_CONTEXT_WINDOW=2048 FLASHCHAT_KV_QUANT=q8 FLASHCHAT_BATCH_PREFILL=0 ./metal_infer/infer --config /Users/speedster/.config/flashchat/config --model /Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 --prompt hi --tokens 1
```

Observed:

```text
[expert-pin] enabled: 606 experts pinnable (1022.62 MiB, 1.00 GiB), free RAM 10.32 GiB, GB cap 1.00 GiB, requested 2560 experts, pool total 10240 experts
```

Wizard guidance for `Qwen-Qwen36-35B-A3B` q4:

```text
Recommended: 640 experts (1.1 GiB), enough for about two generated tokens worth of K=8 expert choices.
Smaller: 320 experts (540.0 MiB); larger: 1280 experts (2.1 GiB).
Maximum useful value for this model: 10240 experts (16.9 GiB).
```

## Interpretation

This is a control-surface improvement more than a performance benchmark. The cache already stored complete experts internally, but the user-visible cap was memory-sized. The new slot cap makes the runtime behavior explicit and easier to tune for constrained systems, and the wizard now shows concrete slot counts instead of asking the user to guess.

For the current q4 Qwen3.6-35B-A3B path, a generated token with K=8 across 40 layers can touch up to 320 expert activations. The wizard now converts that into concrete RAM-costed choices. It derives this guidance only from the user config and model registry architecture/variant metadata, not from built runtime artifacts.

## Follow-Ups

- Once a 16 GB target machine is available, benchmark slot counts around 160, 320, 640, and 960 experts rather than arbitrary GiB values.
- Consider deriving a model-specific recommended slot count from `num_layers * num_experts_per_tok` for first-run defaults on low-memory machines.
