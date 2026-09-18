"""Add-model wizard core: HF repo -> user manifest in models.d/.

Replaces the old add_model_to_registry which mutated the shipped registry
JSON. The shipped files stay read-only; user-added models live in
~/.config/flashchat/models.d/ and shadow shipped ids.
"""
from __future__ import annotations

import json
import math
import os

from . import paths
from .manifest import parse_manifest
from .registry import Registry

SUPPORTED_MODEL_TYPES = {"qwen3_5_moe", "qwen3_next"}

_NATIVE_MOE_ARTIFACTS = {
    "model_weights.bin": {"step": "compile_native:non_experts",
                          "companions": ["model_weights.json"]},
    "packed_experts/": {"step": "compile_native:experts",
                        "per_layer": "moe_layers", "layout": "layout.json"},
    "packed_mtp_experts/": {"step": "compile_native:mtp_experts",
                            "per_layer": "mtp_layers", "layout": "layout.json",
                            "required_if": "mtp"},
    "vocab.bin": {"from_shared": "vocab.bin"},
    "bf16/": {"from_shared": "bf16/", "optional": True},
}
_MLX_ARTIFACTS = {
    "expert_index.json": {"step": "generate_expert_index"},
    "model_weights.bin": {"step": "extract_weights",
                          "companions": ["model_weights.json"]},
    "packed_experts/": {"step": "repack_experts",
                        "per_layer": "moe_layers", "layout": "layout.json"},
    "vocab.bin": {"from_shared": "vocab.bin"},
}


class AddModelError(RuntimeError):
    pass


def load_generation_config(hf_repo: str, cache_dir: str, override: str | None = None) -> dict:
    """Read this model's generation settings, or an explicitly supplied file."""
    try:
        if override:
            filename = override
        else:
            from .steps.download import download_file
            filename = download_file(hf_repo, "generation_config.json", cache_dir)
        with open(filename) as f:
            value = json.load(f)
    except Exception as exc:
        raise AddModelError(
            f"Cannot read generation settings for {hf_repo}: {exc}. "
            "Supply a generation settings JSON file with --generation-config FILE "
            "(or the file prompt in Add a Model).") from exc
    if not isinstance(value, dict):
        raise AddModelError("generation settings must be a JSON object")
    return value


def sampling_profile(hf_repo: str, generation_config: dict | None,
                     thinking_capable: bool | None) -> dict:
    """Map model-specific generation settings to the supported engine controls."""
    if not isinstance(generation_config, dict):
        raise AddModelError("This model needs its own generation settings; provide generation_config.json.")
    greedy = generation_config.get("do_sample") is False
    required = () if greedy else ("temperature", "top_p", "top_k")
    missing = [key for key in required if key not in generation_config]
    if missing:
        raise AddModelError(
            "Generation settings are missing " + ", ".join(missing) +
            "; supply these explicitly in a generation settings JSON file.")
    values = {
        "temperature": 0.0 if greedy else generation_config["temperature"],
        "top_p": 1.0 if greedy else generation_config["top_p"],
        "top_k": 1 if greedy else generation_config["top_k"],
        "min_p": generation_config.get("min_p", 0.0),
        "presence_penalty": generation_config.get("presence_penalty", 0.0),
        "repetition_penalty": generation_config.get("repetition_penalty", 1.0),
    }
    bounds = {"temperature": (0, 5), "top_p": (0.01, 1), "top_k": (1, 1024),
              "min_p": (0, 1), "presence_penalty": (-2, 2), "repetition_penalty": (0.01, 10)}
    for key, value in values.items():
        lo, hi = bounds[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not lo <= value <= hi or
                (key == "top_k" and not isinstance(value, int))):
            raise AddModelError(f"Unsupported {key}={value!r}; expected {'integer' if key == 'top_k' else 'number'} in {lo}..{hi}.")
    return {
        "label": "Model default",
        "description": f"Imported generation settings for {hf_repo}; omitted penalties are disabled.",
        **values,
        "reasoning": int(thinking_capable is True),
    }


def derive_manifest(hf_repo: str, hf_config: dict, registry: Registry,
                    variants: list | None = None,
                    thinking_capable: bool | None = None,
                    generation_config: dict | None = None) -> dict:
    """Build a manifest dict from a HuggingFace config.json."""
    model_type = hf_config.get("model_type", "")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise AddModelError(
            f"model type '{model_type}' is not supported "
            f"(supported: {', '.join(sorted(SUPPORTED_MODEL_TYPES))})")

    # MLX-community models nest architecture under text_config; original
    # Qwen configs keep it at the top level.
    c = hf_config.get("text_config", hf_config)
    moe = c.get("moe_config", {})
    la = c.get("linear_attention", {})

    architecture = {
        "hidden_size": c.get("hidden_size", 2048),
        "num_hidden_layers": c.get("num_hidden_layers", 40),
        "num_attention_heads": c.get("num_attention_heads", 16),
        "num_key_value_heads": c.get("num_key_value_heads", 2),
        "head_dim": c.get("head_dim", 256),
        "vocab_size": c.get("vocab_size", 248320),
        "rms_norm_eps": c.get("rms_norm_eps", 1e-06),
        "num_experts": moe.get("num_experts", c.get("num_experts", 256)),
        "num_experts_per_tok": moe.get("num_experts_per_tok",
                                       c.get("num_experts_per_tok", 8)),
        "moe_intermediate_size": moe.get("moe_intermediate_size",
                                         c.get("moe_intermediate_size", 512)),
        "shared_expert_intermediate_size": moe.get(
            "shared_expert_intermediate_size",
            c.get("shared_expert_intermediate_size", 512)),
        "full_attention_interval": c.get("full_attention_interval", 4),
        "max_context": c.get("max_position_embeddings", 0),
        "linear_num_value_heads": la.get("num_value_heads",
                                         c.get("linear_num_value_heads", 32)),
        "linear_num_key_heads": la.get("num_key_heads",
                                       c.get("linear_num_key_heads", 16)),
        "linear_key_head_dim": la.get("key_head_dim", c.get("linear_key_head_dim", 128)),
        "linear_value_head_dim": la.get("value_head_dim",
                                        c.get("linear_value_head_dim", 128)),
        "linear_conv_kernel_dim": la.get("conv_kernel_dim",
                                         c.get("linear_conv_kernel_dim", 4)),
        "partial_rotary_factor": c.get("partial_rotary_factor", 0.25),
        "rope_theta": c.get("rope_theta",
                            c.get("rope_parameters", {}).get("rope_theta", 10000000.0)),
    }
    if int(architecture["num_experts"]) == 0:
        raise AddModelError("dense models (num_experts=0) are not supported")

    # Quantization metadata present => pre-quantized source; absent =>
    # native BF16 that we compile ourselves (q4 + q8 variants on offer).
    qc = hf_config.get("quantization_config", hf_config.get("quantization", {}))
    native = not bool(qc)
    mtp_layers = c.get("mtp_num_hidden_layers", hf_config.get("mtp_num_hidden_layers", 0))

    profile = sampling_profile(hf_repo, generation_config, thinking_capable)
    # Token IDs retain the existing tokenizer-family fallback. Sampling settings
    # must come from the individual model, never this architecture template.
    template = None
    for m in registry.manifests.values():
        if not m.user_defined and int(m.architecture.get("vocab_size", 0)) == int(architecture["vocab_size"]):
            template = m
            break
    if template is None:
        template = next(m for m in registry.manifests.values() if not m.user_defined)

    model_id = hf_repo.replace("/", "-").replace(".", "").lower()
    if native:
        artifacts = json.loads(json.dumps(_NATIVE_MOE_ARTIFACTS))
        offered = variants or ["q4", "q8"]
        shared = {"vocab.bin": {"step": "export_tokenizer"}}
        if mtp_layers:
            shared["bf16/mtp_weights.bin"] = {
                "step": "compile_native:bf16_mtp", "optional": True,
                "companions": ["bf16/mtp_weights.json"]}
        else:
            artifacts.pop("bf16/", None)
    else:
        artifacts = _MLX_ARTIFACTS
        bits = int(qc.get("bits", 4) or 4)
        offered = variants or [f"q{bits}"]
        shared = {"vocab.bin": {"step": "export_tokenizer"}}

    manifest = {
        "schema": 1,
        "id": model_id,
        "name": hf_repo.rsplit("/", 1)[-1],
        "hf_repo": hf_repo,
        "source_format": "native_bf16" if native else "mlx_quantized",
        "default_variant": offered[0],
        "architecture": architecture,
        "special_tokens": dict(template.special_tokens),
        "default_sampling_profile": "model-default",
        "sampling_profiles": {"model-default": profile},
        "thinking_capable": bool(thinking_capable) if thinking_capable is not None else True,
        "shared_artifacts": shared,
        "variants": {},
    }
    if mtp_layers:
        manifest["mtp"] = {"default_predictions": 1, "num_hidden_layers": mtp_layers}
    for vname in offered:
        bits = int(vname[1:]) if vname.startswith("q") and vname[1:].isdigit() else 4
        manifest["variants"][vname] = {
            "quantization": {"bits": bits,
                             "group_size": int(qc.get("group_size", 64) or 64)},
            "artifacts": json.loads(json.dumps(artifacts)),
        }
    return manifest


def save_user_manifest(manifest_dict: dict) -> str:
    """Validate and write a user manifest; returns the path."""
    parsed = parse_manifest(manifest_dict, user_defined=True)
    registry = Registry.load()
    if parsed.id in registry.manifests and not registry.manifests[parsed.id].user_defined:
        raise AddModelError(f"model id '{parsed.id}' collides with a shipped model")
    path = os.path.join(paths.user_manifest_dir(), f"{parsed.id}.json")
    paths.write_json_atomic(path, manifest_dict)
    return path
