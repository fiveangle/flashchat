"""Emit the engine-facing resolved registry view.

The C engine (metal_infer/model_config.h) parses a flat legacy-schema JSON:
one entry per (model, variant) with the architecture keys inlined. This
module renders that view from the hierarchical manifests so the engine
needs zero changes. Constraints imposed by the C strstr-based parser:

- an entry is located by the first occurrence of its quoted id anywhere in
  the file, so models must come before `default_model`/`server_defaults`
  and no resolved id may appear (quoted, exact) earlier in the file;
- per-entry keys are scoped to the entry object, names must match the
  legacy schema exactly.
"""
from __future__ import annotations

import json

from . import paths
from .manifest import Manifest
from .registry import Registry, resolved_id

# Global server defaults forwarded verbatim to the engine (legacy
# top-level "server_defaults" object).
SERVER_DEFAULTS = {"mtp_default_predictions": 0}

# Architecture keys copied into each flat entry, in emission order.
_ARCH_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "rms_norm_eps",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "intermediate_size",
    "full_attention_interval",
    "max_context",
    "linear_num_value_heads",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "partial_rotary_factor",
    "rope_theta",
)

# Vestigial legacy-schema strings: the C parser reads scripts{} into g_cfg
# but nothing executes them, and the files no longer exist (the step
# implementations live in modelmgr/steps/). Emitted only so the resolved
# view stays field-identical to the schema the engine was built against.
_SCRIPTS = {
    "native_bf16": {
        "extract_weights": "scripts/compile_native_qwen.py",
        "repack_experts": "scripts/compile_native_qwen.py",
    },
    "mlx_quantized": {
        "extract_weights": "scripts/extract_weights.py",
        "repack_experts": "scripts/repack_experts.py",
    },
}


def flat_entry(manifest: Manifest, variant_name: str) -> dict:
    variant = manifest.variant(variant_name)
    entry: dict = {"name": variant.display_name or manifest.name}

    benchmark = manifest.benchmark if variant.benchmark is None else variant.benchmark
    if not benchmark:
        # Legacy schema: benchmark defaults to true and is only written
        # when disabled; same for thinking_capable below.
        entry["benchmark"] = False

    entry["hf_repo"] = manifest.hf_repo
    for key in _ARCH_KEYS:
        if key in manifest.architecture:
            entry[key] = manifest.architecture[key]
    if not manifest.thinking_capable:
        entry["thinking_capable"] = False

    entry["quantization"] = {"bits": variant.bits, "group_size": variant.group_size}
    entry["scripts"] = dict(_SCRIPTS[manifest.source_format])
    entry["special_tokens"] = dict(manifest.special_tokens)
    entry["default_sampling_profile"] = manifest.default_sampling_profile
    entry["sampling_profiles"] = manifest.sampling_profiles

    if manifest.source_format == "native_bf16":
        entry["source_format"] = "native_bf16"
    if manifest.mtp:
        entry["mtp_default_predictions"] = manifest.mtp["default_predictions"]
        if manifest.mtp.get("num_hidden_layers"):
            entry["mtp_num_hidden_layers"] = manifest.mtp["num_hidden_layers"]
    return entry


def render(registry: Registry, include_all: bool = False) -> dict:
    """Build the resolved view: enabled models only, or the shipped set.

    include_all=True renders the SHIPPED view (generated
    assets/model_configs.json fallback): every shipped model, no user
    manifests, and the suggested default — per-user state must never leak
    into a file committed to the repo.
    """
    models: dict = {}
    for manifest in registry.manifests.values():
        if include_all:
            if manifest.user_defined:
                continue
        elif not registry.is_enabled(manifest.id):
            continue
        for variant_name in manifest.variants:
            rid = resolved_id(manifest, variant_name)
            if rid in models:
                raise ValueError(f"resolved id collision: '{rid}'")
            models[rid] = flat_entry(manifest, variant_name)

    if include_all:
        default = next((m for m in registry.manifests.values() if m.suggested_default),
                       None)
    else:
        default = registry.default_model()
    default_id = ""
    if default is not None:
        default_id = resolved_id(default, default.default_variant)
        if default_id not in models and models:
            default_id = next(iter(models))

    view = {
        "models": models,
        "default_model": default_id,
        "server_defaults": dict(SERVER_DEFAULTS),
    }
    _check_strstr_safety(view)
    return view


def _check_strstr_safety(view: dict) -> None:
    """Reject any rendering the C parser would mis-scope.

    load_model_config() finds an entry by strstr of '"<id>"', so the first
    occurrence of each quoted id must be its own key in "models".
    """
    text = json.dumps(view, indent=2)
    for rid in view["models"]:
        if len(rid) > 63:
            raise ValueError(f"resolved id too long for engine (char[64]): '{rid}'")
        needle = f'"{rid}"'
        first = text.find(needle)
        key_pos = text.find(needle + ":")
        if key_pos == -1:
            key_pos = text.find(needle + " :")
        if first == -1 or first != key_pos:
            raise ValueError(
                f"resolved id '{rid}' appears before its model entry; "
                f"the engine's strstr lookup would mis-scope it"
            )


def write(registry: Registry, path: str | None = None, include_all: bool = False) -> str:
    out = path or paths.resolved_registry_path()
    paths.write_json_atomic(out, render(registry, include_all=include_all))
    return out
