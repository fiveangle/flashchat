"""Artifact directories with integrity manifests.

Every artifact dir (shared/, q4/, q8/) carries a `.flashchat_artifacts.json`
recording per-file size + sha256 + the producing step. Hashes are computed
while the bytes are written (hash-on-write) so multi-GB expert packs never
need a second read pass. Writers stream to `<name>.partial` and rename on
success, so an interrupted step can never leave a file that quick-verify
would accept.

Verification levels:
- quick: existence + size + the legacy dimension cross-checks (cheap stats
  and small JSON reads; safe to run on every status render)
- deep: full sha256 re-hash against the manifest
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .manifest import ArtifactSpec, Manifest, Variant

MANIFEST_NAME = ".flashchat_artifacts.json"
MANIFEST_SCHEMA = 1
_CHUNK = 8 * 1024 * 1024

# Files the engine creates/manages at runtime inside artifact dirs; never
# manifested, never flagged as unexpected.
ENGINE_MANAGED = ("system_prompt_cache",)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str, progress=None) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
            if progress:
                progress(len(chunk))
    return h.hexdigest(), size


class HashingWriter:
    """File-like sink that hashes while writing and registers the artifact
    entry on close. Use via ArtifactDir.open()."""

    def __init__(self, artifact_dir: "ArtifactDir", relpath: str, step: str | None,
                 step_version: int | None, from_shared: bool = False):
        self._dir = artifact_dir
        self.relpath = relpath
        self._step = step
        self._step_version = step_version
        self._from_shared = from_shared
        self._hash = hashlib.sha256()
        self._size = 0
        self._final_path = os.path.join(artifact_dir.root, relpath)
        self._partial_path = self._final_path + ".partial"
        os.makedirs(os.path.dirname(self._final_path), exist_ok=True)
        self._f = open(self._partial_path, "wb")
        self._closed = False

    def write(self, data) -> int:
        n = self._f.write(data)
        self._hash.update(data)
        self._size += len(data)
        return n

    def tell(self) -> int:
        return self._f.tell()

    def close(self) -> None:
        if self._closed:
            return
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        os.replace(self._partial_path, self._final_path)
        self._dir.stage(
            self.relpath,
            size=self._size,
            sha256=self._hash.hexdigest(),
            step=self._step,
            step_version=self._step_version,
            from_shared=self._from_shared,
        )
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._f.close()
        if os.path.exists(self._partial_path):
            os.unlink(self._partial_path)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


class ArtifactDir:
    """One artifact directory (shared/ or a variant dir) and its manifest."""

    def __init__(self, root: str, model_id: str | None = None, variant: str | None = None):
        self.root = root
        self.model_id = model_id
        self.variant = variant
        self._entries = None  # lazy

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, MANIFEST_NAME)

    @property
    def entries(self) -> dict:
        if self._entries is None:
            if os.path.isfile(self.manifest_path):
                with open(self.manifest_path) as f:
                    self._entries = json.load(f).get("files", {})
            else:
                self._entries = {}
        return self._entries

    def open(self, relpath: str, step: str | None = None,
             step_version: int | None = None, from_shared: bool = False) -> HashingWriter:
        return HashingWriter(self, relpath, step, step_version, from_shared)

    def stage(self, relpath: str, size: int, sha256: str | None, step: str | None = None,
              step_version: int | None = None, from_shared: bool = False) -> None:
        # sha256 may be None for entries adopted from a pre-manifest tree
        # (migration records size+provenance immediately; the hash baseline
        # is computed on demand — deep verify reports such files "unhashed").
        entry = {"size": size, "sha256": sha256, "completed": utc_now()}
        if step:
            entry["step"] = step
        if step_version is not None:
            entry["step_version"] = step_version
        if from_shared:
            entry["from_shared"] = True
        self.entries[relpath] = entry

    def commit(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        data = {"schema": MANIFEST_SCHEMA, "files": self.entries}
        if self.model_id:
            data["model"] = self.model_id
        if self.variant:
            data["variant"] = self.variant
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp, self.manifest_path)

    def forget(self, relpath_prefix: str) -> None:
        """Drop manifest entries for a deleted/regenerated artifact."""
        for key in [k for k in self.entries if k == relpath_prefix
                    or k.startswith(relpath_prefix.rstrip("/") + "/")]:
            del self.entries[key]

    def backfill(self, relpath: str, step: str | None = None,
                 step_version: int | None = None, progress=None) -> None:
        """Hash an existing file into the manifest (migration, legacy files)."""
        digest, size = sha256_file(os.path.join(self.root, relpath), progress=progress)
        self.stage(relpath, size=size, sha256=digest, step=step, step_version=step_version)

    # -- per-file checks ---------------------------------------------------

    def quick_check(self, relpath: str) -> str:
        """'ok' | 'missing' | 'size-mismatch' | 'unhashed'(present, no entry)"""
        path = os.path.join(self.root, relpath)
        if not os.path.exists(path):  # follows symlinks: a dangling link is missing
            return "missing"
        entry = self.entries.get(relpath)
        if entry is None:
            return "unhashed"
        if os.path.getsize(path) != entry["size"]:
            return "size-mismatch"
        return "ok"

    def deep_check(self, relpath: str, progress=None) -> str:
        """'ok' | 'missing' | 'unhashed' | 'hash-mismatch'"""
        path = os.path.join(self.root, relpath)
        if not os.path.exists(path):
            return "missing"
        entry = self.entries.get(relpath)
        if entry is None or entry.get("sha256") is None:
            return "unhashed"
        digest, size = sha256_file(path, progress=progress)
        if size != entry["size"] or digest != entry["sha256"]:
            return "hash-mismatch"
        return "ok"


# ---------------------------------------------------------------------------
# Legacy dimension cross-checks (ports of runtime_weights_match_model /
# runtime_packed_experts_match_model / runtime_mtp_match_model from the bash
# implementation). These catch "file exists but was extracted for a different
# model/quantization" — cheap JSON reads, part of quick verification.
# ---------------------------------------------------------------------------

_INT_FIELDS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "num_experts",
    "num_experts_per_tok", "moe_intermediate_size",
    "shared_expert_intermediate_size", "full_attention_interval",
    "linear_num_value_heads", "linear_num_key_heads",
    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
)
_FLOAT_FIELDS = ("rms_norm_eps", "partial_rotary_factor", "rope_theta")

# Tensors every MTP head needs regardless of architecture. Quantized
# companions (.scales/.biases) are required per-tensor only when the weight
# is stored quantized (dtype U32) — q4 builds made with MTP_BF16 keep the
# MTP head in BF16 and legitimately have no companions.
MTP_BASE_TENSORS = (
    "mtp.fc.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.norm.weight",
)
# MoE MTP heads have an expert router.
MTP_MOE_TENSORS = ("mtp.layers.0.mlp.gate.weight",)


def expert_pack_size(hidden: int, moe: int, bits: int, group_size: int) -> int:
    values_per_word = 32 // bits
    gate_w = moe * (hidden // values_per_word) * 4
    gate_s = moe * (hidden // group_size) * 2
    down_w = hidden * (moe // values_per_word) * 4
    down_s = hidden * (moe // group_size) * 2
    return gate_w + gate_s + gate_s + gate_w + gate_s + gate_s + down_w + down_s + down_s


def weights_config_matches(manifest: Manifest, variant: Variant, weights_json_path: str) -> bool:
    try:
        with open(weights_json_path) as f:
            config = json.load(f).get("config", {})
    except (OSError, json.JSONDecodeError):
        return False
    if not config:
        return False
    arch = manifest.architecture
    for field in _INT_FIELDS:
        expected = int(arch.get(field, 0) or 0)
        actual = int(config[field]) if config.get(field) is not None else -1
        if expected != actual:
            return False
    for field in _FLOAT_FIELDS:
        expected = float(arch.get(field, 0.0) or 0.0)
        actual = float(config[field]) if config.get(field) is not None else -1.0
        if abs(expected - actual) > max(1e-9, abs(expected) * 1e-9):
            return False
    quant = config.get("quantization", {})
    return (variant.bits, variant.group_size) == (
        int(quant.get("bits", 4) or 4),
        int(quant.get("group_size", 64) or 64),
    )


def packed_layout_matches(manifest: Manifest, variant: Variant, layout_path: str,
                          expected_layers: int, packed_name: str) -> bool:
    try:
        with open(layout_path) as f:
            layout = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    arch = manifest.architecture
    hidden = int(arch.get("hidden_size") or 0)
    moe = int(arch.get("moe_intermediate_size") or 0)
    num_experts = int(arch.get("num_experts") or 0)
    if not hidden or not moe or not num_experts or variant.bits not in (4, 8):
        return False
    artifact = layout.get("artifact")
    if artifact and artifact != packed_name:
        return False
    return (
        int(layout.get("expert_size") or 0)
        == expert_pack_size(hidden, moe, variant.bits, variant.group_size)
        and int(layout.get("num_experts") or 0) == num_experts
        and (expected_layers <= 0 or int(layout.get("num_layers") or 0) == expected_layers)
    )


def mtp_tensors_present(weights_json_path: str) -> bool:
    try:
        with open(weights_json_path) as f:
            tensors = json.load(f).get("tensors", {})
    except (OSError, json.JSONDecodeError):
        return False
    required = MTP_BASE_TENSORS + MTP_MOE_TENSORS
    for name in required:
        if name not in tensors:
            return False
        if tensors[name].get("dtype") == "U32":
            base = name[:-len(".weight")]
            if f"{base}.scales" not in tensors or f"{base}.biases" not in tensors:
                return False
    return True


def source_mtp_tensors_present(model_path: str) -> bool | None:
    """Whether the ORIGINAL checkpoint ships an MTP/nextn head, by scanning the
    safetensors weight map for mtp.*/nextn tensors. Returns None when it cannot be
    determined (no index present — e.g. the model isn't downloaded) so callers can
    fall back to the manifest's declared capability. This is the authoritative
    source-side check, independent of whether addmodel captured the MTP config."""
    if not model_path:
        return None
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    try:
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    except (OSError, json.JSONDecodeError):
        return None
    return any("mtp" in k.lower() or "nextn" in k.lower() for k in weight_map)


def template_supports_thinking(model_path: str) -> bool | None:
    """Whether the model's chat template emits <think> reasoning blocks. Reads
    tokenizer_config.json's chat_template and checks for the <think> marker:
    Qwen3 thinking models have it; non-thinking variants (e.g. Qwen3-Coder)
    do not and have dropped the enable_thinking flag entirely. Returns None when
    undeterminable (no tokenizer_config / template — e.g. the model isn't
    downloaded) so callers can stay conservative and not warn."""
    if not model_path:
        return None
    tcp = os.path.join(model_path, "tokenizer_config.json")
    try:
        with open(tcp) as f:
            ct = json.load(f).get("chat_template")
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(ct, list):  # some repos ship a list of named templates
        ct = " ".join((d or {}).get("template", "") for d in ct)
    if not ct:
        return None
    return "<think>" in ct


def packed_layer_files(packed_dir: str, num_layers: int) -> list:
    """Missing layer_NN.bin files (empty list when complete)."""
    missing = []
    for i in range(num_layers):
        if not os.path.isfile(os.path.join(packed_dir, f"layer_{i:02d}.bin")):
            missing.append(i)
    return missing


# ---------------------------------------------------------------------------
# Per-artifact verification combining manifest specs with the checks above.
# ---------------------------------------------------------------------------


@dataclass
class ArtifactStatus:
    relpath: str
    # ok | missing | incomplete | invalid | size-mismatch | hash-mismatch | unhashed | skipped
    state: str
    required: bool
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        return self.state in ("ok", "unhashed", "skipped") or not self.required


def _layers_for(manifest: Manifest, spec: ArtifactSpec) -> int:
    if spec.per_layer == "mtp_layers":
        return int((manifest.mtp or {}).get("num_hidden_layers", 0))
    return int(manifest.architecture.get("num_hidden_layers", 0))


def spec_required(manifest: Manifest, spec: ArtifactSpec, want_optional: bool = False) -> bool:
    if spec.optional and not want_optional:
        return False
    if spec.required_if == "mtp" and not manifest.mtp_artifacts_required:
        return False
    return True


def check_artifact(manifest: Manifest, variant: Variant, adir: ArtifactDir,
                   spec: ArtifactSpec, deep: bool = False,
                   want_optional: bool = False, want_mtp: bool = False,
                   progress=None) -> ArtifactStatus:
    required = spec_required(manifest, spec, want_optional)
    rel = spec.relpath

    if spec.is_dir:
        packed_dir = os.path.join(adir.root, rel.rstrip("/"))
        if not os.path.isdir(packed_dir):
            return ArtifactStatus(rel, "missing", required)
        if spec.layout:
            layers = _layers_for(manifest, spec)
            layout_path = os.path.join(packed_dir, spec.layout)
            if not packed_layout_matches(manifest, variant, layout_path, layers, rel.rstrip("/")):
                return ArtifactStatus(rel, "invalid", required, "layout mismatch")
            missing = packed_layer_files(packed_dir, layers)
            if missing:
                return ArtifactStatus(
                    rel, "missing", required,
                    f"{len(missing)}/{layers} layer files missing",
                )
        if deep:
            for key in sorted(adir.entries):
                if key.startswith(rel.rstrip("/") + "/"):
                    state = adir.deep_check(key, progress=progress)
                    if state not in ("ok", "unhashed"):
                        return ArtifactStatus(rel, state, required, key)
        return ArtifactStatus(rel, "ok", required)

    state = adir.deep_check(rel, progress=progress) if deep else adir.quick_check(rel)
    if state != "ok" and state != "unhashed":
        return ArtifactStatus(rel, state, required)

    # Semantic cross-checks beyond size/hash.
    if rel == "model_weights.bin":
        weights_json = os.path.join(adir.root, "model_weights.json")
        if not os.path.isfile(weights_json):
            return ArtifactStatus(rel, "invalid", required, "model_weights.json missing")
        if not weights_config_matches(manifest, variant, weights_json):
            return ArtifactStatus(rel, "invalid", required, "config mismatch with model")
        if want_mtp and manifest.mtp_artifacts_required and not mtp_tensors_present(weights_json):
            # Weights are usable for plain inference; MTP needs a rebuild.
            return ArtifactStatus(rel, "incomplete", required,
                                  "MTP tensors absent — rebuild weights to enable MTP")
    for companion in spec.companions:
        if not os.path.exists(os.path.join(adir.root, companion)):
            return ArtifactStatus(rel, "invalid", required, f"companion {companion} missing")
    return ArtifactStatus(rel, state, required)


def variant_status(manifest: Manifest, variant_name: str, snapshot: str,
                   deep: bool = False, want_optional: bool = False,
                   want_mtp: bool = False, progress=None) -> list:
    """Status of every artifact a variant declares (shared deps included
    implicitly via the from_shared links materialized in the variant dir)."""
    from . import paths

    variant = manifest.variant(variant_name)
    adir = ArtifactDir(paths.variant_dir(snapshot, variant_name),
                       model_id=manifest.id, variant=variant_name)
    out = []
    for spec in variant.artifacts.values():
        out.append(check_artifact(manifest, variant, adir, spec, deep=deep,
                                  want_optional=want_optional, want_mtp=want_mtp,
                                  progress=progress))
    return out


def shared_status(manifest: Manifest, snapshot: str, deep: bool = False,
                  want_optional: bool = False, progress=None) -> list:
    from . import paths

    adir = ArtifactDir(paths.shared_dir(snapshot), model_id=manifest.id, variant="shared")
    # Shared artifacts have no variant-specific dims; use any variant for the
    # quant-independent checks (vocab.bin, bf16 MTP carry no layout.json).
    any_variant = next(iter(manifest.variants.values()))
    out = []
    for spec in manifest.shared_artifacts.values():
        out.append(check_artifact(manifest, any_variant, adir, spec, deep=deep,
                                  want_optional=want_optional, progress=progress))
    return out


def variant_ready(manifest: Manifest, variant_name: str, snapshot: str,
                  want_optional: bool = False, want_mtp: bool = False) -> bool:
    return all(
        s.satisfied
        for s in variant_status(manifest, variant_name, snapshot,
                                want_optional=want_optional, want_mtp=want_mtp)
    )
