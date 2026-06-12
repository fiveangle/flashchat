"""Add-model wizard core: HF repo -> user manifest in models.d/.

Replaces the old add_model_to_registry which mutated the shipped registry
JSON. The shipped files stay read-only; user-added models live in
~/.config/flashchat/models.d/ and shadow shipped ids.
"""

import json
import os

from . import paths
from .manifest import parse_manifest
from .registry import Registry

SUPPORTED_MODEL_TYPES = {"qwen3_5_moe"}

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
_NATIVE_DENSE_ARTIFACTS = {
    "model_weights.bin": {"step": "compile_native:non_experts",
                          "companions": ["model_weights.json"]},
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


def derive_manifest(hf_repo: str, hf_config: dict, registry: Registry,
                    variants: list | None = None) -> dict:
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
        architecture["intermediate_size"] = c.get("intermediate_size", 0)

    # Quantization metadata present => pre-quantized source; absent =>
    # native BF16 that we compile ourselves (q4 + q8 variants on offer).
    qc = hf_config.get("quantization_config", hf_config.get("quantization", {}))
    native = not bool(qc)
    dense = int(architecture["num_experts"]) == 0
    mtp_layers = c.get("mtp_num_hidden_layers", hf_config.get("mtp_num_hidden_layers", 0))

    # Borrow special tokens + sampling profiles from a shipped model with the
    # same vocab size (they are tokenizer-family-wide, not model-specific).
    template = None
    for m in registry.manifests.values():
        if not m.user_defined and int(m.architecture.get("vocab_size", 0)) == int(architecture["vocab_size"]):
            template = m
            break
    if template is None:
        template = next(m for m in registry.manifests.values() if not m.user_defined)

    model_id = hf_repo.replace("/", "-").replace(".", "").lower()
    if native:
        artifacts = _NATIVE_DENSE_ARTIFACTS if dense else _NATIVE_MOE_ARTIFACTS
        offered = variants or ["q4", "q8"]
        shared = {"vocab.bin": {"step": "export_tokenizer"}}
        if mtp_layers:
            shared["bf16/mtp_weights.bin"] = {
                "step": "compile_native:bf16_mtp", "optional": True,
                "companions": ["bf16/mtp_weights.json"]}
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
        "default_sampling_profile": template.default_sampling_profile,
        "sampling_profiles": json.loads(json.dumps(template.sampling_profiles)),
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
