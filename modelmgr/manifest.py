"""Declarative per-model manifests.

One manifest JSON per base model (assets/models/*.json shipped, plus
~/.config/flashchat/models.d/*.json user-added). A manifest declares the
HF source repo, the engine architecture, the variants (q4/q8/...) and the
artifacts each variant needs, with the step that produces each artifact.
Adding a model means writing one of these files; code is only needed for
architecturally novel models (hooks).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Step names a manifest may reference. The implementations live in
# modelmgr.steps; the names are fixed here so manifests validate
# even before the step modules are importable (steps need the venv).
KNOWN_STEPS = {
    "export_tokenizer",
    "generate_expert_index",
    "extract_weights",
    "repack_experts",
    "compile_native:non_experts",
    "compile_native:experts",
    "compile_native:mtp_experts",
    "compile_native:bf16_mtp",
}

SOURCE_FORMATS = {"mlx_quantized", "native_bf16"}

# The C registry parser stores ids in char[64] and looks entries up by
# strstr of the quoted id, so ids must be short and quote-unique.
MAX_MODEL_ID_LEN = 63
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")
_VARIANT_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


class ManifestError(ValueError):
    pass


@dataclass
class ArtifactSpec:
    """One declared artifact inside shared_artifacts or a variant.

    Exactly one of `step` / `from_shared` is set. A trailing '/' on the
    relpath means a directory artifact (e.g. packed_experts/).
    """

    relpath: str
    step: str | None = None
    from_shared: str | None = None
    optional: bool = False
    required_if: str | None = None  # "mtp": required only when model has MTP
    companions: list = field(default_factory=list)
    per_layer: str | None = None    # "moe_layers" | "mtp_layers"
    layout: str | None = None       # layout json inside a directory artifact

    @property
    def is_dir(self) -> bool:
        return self.relpath.endswith("/")


@dataclass
class Variant:
    name: str
    bits: int
    group_size: int
    artifacts: dict  # relpath -> ArtifactSpec
    legacy_ids: list = field(default_factory=list)
    display_name: str | None = None
    benchmark: bool | None = None  # None -> inherit model-level


@dataclass
class Manifest:
    id: str
    name: str
    hf_repo: str
    source_format: str
    architecture: dict
    special_tokens: dict
    variants: dict  # name -> Variant
    shared_artifacts: dict  # relpath -> ArtifactSpec
    default_variant: str
    sampling_profiles: dict
    default_sampling_profile: str
    thinking_capable: bool = True
    benchmark: bool = True
    suggested_default: bool = False
    mtp: dict | None = None  # {"default_predictions": N[, "num_hidden_layers": N]}
    source_path: str | None = None  # file this was loaded from
    user_defined: bool = False

    @property
    def num_experts_per_tok(self) -> int:
        return int(self.architecture.get("num_experts_per_tok", 0) or 0)

    @property
    def max_context(self) -> int:
        """Model's trained max context (max_position_embeddings); 0 if unknown."""
        return int(self.architecture.get("max_context", 0) or 0)

    @property
    def mtp_artifacts_required(self) -> bool:
        # Mirrors the old bash model_mtp_artifacts_required(): native models
        # with an MTP head carry mtp_num_hidden_layers; quantized community
        # exports have no extractable MTP tensors.
        return bool(self.mtp and int(self.mtp.get("num_hidden_layers", 0)) > 0)

    def variant(self, name: str) -> Variant:
        try:
            return self.variants[name]
        except KeyError:
            raise ManifestError(f"model '{self.id}' has no variant '{name}'") from None


def _parse_artifact(relpath: str, raw: dict, where: str) -> ArtifactSpec:
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: artifact '{relpath}' must be an object")
    step = raw.get("step")
    from_shared = raw.get("from_shared")
    if bool(step) == bool(from_shared):
        raise ManifestError(
            f"{where}: artifact '{relpath}' needs exactly one of 'step' or 'from_shared'"
        )
    if step and step not in KNOWN_STEPS:
        raise ManifestError(f"{where}: artifact '{relpath}' references unknown step '{step}'")
    per_layer = raw.get("per_layer")
    if per_layer not in (None, "moe_layers", "mtp_layers"):
        raise ManifestError(f"{where}: artifact '{relpath}' has invalid per_layer '{per_layer}'")
    required_if = raw.get("required_if")
    if required_if not in (None, "mtp"):
        raise ManifestError(f"{where}: artifact '{relpath}' has invalid required_if '{required_if}'")
    return ArtifactSpec(
        relpath=relpath,
        step=step,
        from_shared=from_shared,
        optional=bool(raw.get("optional", False)),
        required_if=required_if,
        companions=list(raw.get("companions", [])),
        per_layer=per_layer,
        layout=raw.get("layout"),
    )


_REQUIRED_ARCH_KEYS = (
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
    "full_attention_interval",
    "linear_num_value_heads",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "partial_rotary_factor",
    "rope_theta",
)


def parse_manifest(data: dict, source_path: str | None = None, user_defined: bool = False) -> Manifest:
    where = source_path or "<manifest>"
    if data.get("schema") != 1:
        raise ManifestError(f"{where}: unsupported manifest schema {data.get('schema')!r}")

    for key in ("id", "name", "hf_repo", "source_format", "architecture", "variants",
                "default_variant", "special_tokens", "sampling_profiles",
                "default_sampling_profile"):
        if key not in data:
            raise ManifestError(f"{where}: missing required key '{key}'")

    model_id = data["id"]
    if not _ID_RE.match(model_id) or len(model_id) > MAX_MODEL_ID_LEN:
        raise ManifestError(f"{where}: invalid model id '{model_id}'")
    if data["source_format"] not in SOURCE_FORMATS:
        raise ManifestError(f"{where}: invalid source_format '{data['source_format']}'")

    arch = data["architecture"]
    for key in _REQUIRED_ARCH_KEYS:
        if key not in arch:
            raise ManifestError(f"{where}: architecture missing '{key}'")
    if int(arch.get("num_experts", 0)) == 0:
        raise ManifestError(f"{where}: dense models (num_experts=0) are not supported")

    shared = {
        rel: _parse_artifact(rel, raw, where)
        for rel, raw in data.get("shared_artifacts", {}).items()
    }
    for rel, spec in shared.items():
        if spec.from_shared:
            raise ManifestError(f"{where}: shared artifact '{rel}' cannot use from_shared")

    mtp = data.get("mtp")
    if mtp is not None and "default_predictions" not in mtp:
        raise ManifestError(f"{where}: mtp block requires default_predictions")

    variants = {}
    for vname, vraw in data["variants"].items():
        if not _VARIANT_RE.match(vname):
            raise ManifestError(f"{where}: invalid variant name '{vname}'")
        quant = vraw.get("quantization") or {}
        if "bits" not in quant or "group_size" not in quant:
            # The engine refuses to run without these; fail at validation
            # time, not at engine startup.
            raise ManifestError(f"{where}: variant '{vname}' requires quantization.bits and group_size")
        artifacts = {
            rel: _parse_artifact(rel, raw, f"{where}:{vname}")
            for rel, raw in vraw.get("artifacts", {}).items()
        }
        for rel, spec in artifacts.items():
            if spec.from_shared:
                target = spec.from_shared
                if target not in shared and not any(
                    s.relpath == target or s.relpath.startswith(target.rstrip("/") + "/")
                    for s in shared.values()
                ):
                    raise ManifestError(
                        f"{where}: variant '{vname}' artifact '{rel}' references "
                        f"unknown shared artifact '{target}'"
                    )
        variants[vname] = Variant(
            name=vname,
            bits=int(quant["bits"]),
            group_size=int(quant["group_size"]),
            artifacts=artifacts,
            legacy_ids=list(vraw.get("legacy_ids", [])),
            display_name=vraw.get("display_name"),
            benchmark=vraw.get("benchmark"),
        )

    if data["default_variant"] not in variants:
        raise ManifestError(f"{where}: default_variant '{data['default_variant']}' not in variants")
    if data["default_sampling_profile"] not in data["sampling_profiles"]:
        raise ManifestError(
            f"{where}: default_sampling_profile '{data['default_sampling_profile']}' not in sampling_profiles"
        )

    return Manifest(
        id=model_id,
        name=data["name"],
        hf_repo=data["hf_repo"],
        source_format=data["source_format"],
        architecture=arch,
        special_tokens=data["special_tokens"],
        variants=variants,
        shared_artifacts=shared,
        default_variant=data["default_variant"],
        sampling_profiles=data["sampling_profiles"],
        default_sampling_profile=data["default_sampling_profile"],
        thinking_capable=bool(data.get("thinking_capable", True)),
        benchmark=bool(data.get("benchmark", True)),
        suggested_default=bool(data.get("suggested_default", False)),
        mtp=mtp,
        source_path=source_path,
        user_defined=user_defined,
    )
