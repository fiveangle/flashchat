"""Build synthetic HF-cache snapshot trees for tests.

Creates miniature (bytes, not gigabytes) artifact trees that satisfy the
quick-verify checks: model_weights.json configs echo the manifest
architecture, packed layout.json carries the real computed expert sizes,
and integrity manifests are written through the same hash-on-write path
production uses.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import paths
from modelmgr.artifacts import (
    MTP_BASE_TENSORS,
    MTP_DENSE_TENSORS,
    MTP_MOE_TENSORS,
    ArtifactDir,
    expert_pack_size,
)
from modelmgr.steps import step_version


def weights_config_for(manifest, variant) -> dict:
    config = {k: manifest.architecture[k] for k in manifest.architecture}
    config["quantization"] = {"bits": variant.bits, "group_size": variant.group_size}
    return config


def make_weights_json(manifest, variant) -> dict:
    tensors = {"embed_tokens.weight": {"offset": 0, "size": 16}}
    if manifest.mtp_artifacts_required:
        required = MTP_BASE_TENSORS + (
            MTP_DENSE_TENSORS if manifest.is_dense else MTP_MOE_TENSORS)
        for name in required:
            tensors[name] = {"offset": 0, "size": 16, "dtype": "BF16"}
    return {
        "model": "fixture",
        "num_tensors": len(tensors),
        "tensors": tensors,
        "config": weights_config_for(manifest, variant),
    }


def write_artifact(adir: ArtifactDir, relpath: str, data: bytes,
                   step: str | None = None, from_shared: bool = False) -> None:
    version = step_version(step) if step else None
    with adir.open(relpath, step=step, step_version=version, from_shared=from_shared) as w:
        w.write(data)


def populate_shared(manifest, snapshot: str, want_optional: bool = True) -> ArtifactDir:
    adir = ArtifactDir(paths.shared_dir(snapshot), manifest.id, "shared")
    for spec in manifest.shared_artifacts.values():
        if spec.optional and not want_optional:
            continue
        write_artifact(adir, spec.relpath, b"shared:" + spec.relpath.encode(), step=spec.step)
        for companion in spec.companions:
            write_artifact(adir, companion, b"companion", step=spec.step)
    adir.commit()
    return adir


def populate_variant(manifest, variant_name: str, snapshot: str,
                     want_optional: bool = False) -> ArtifactDir:
    variant = manifest.variant(variant_name)
    vdir_path = paths.variant_dir(snapshot, variant_name)
    adir = ArtifactDir(vdir_path, manifest.id, variant_name)
    shared = paths.shared_dir(snapshot)

    for spec in variant.artifacts.values():
        if spec.optional and not want_optional:
            continue
        if spec.from_shared:
            target = os.path.join(shared, spec.from_shared.rstrip("/"))
            link = os.path.join(vdir_path, spec.relpath.rstrip("/"))
            os.makedirs(os.path.dirname(link), exist_ok=True)
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
            sdir = ArtifactDir(shared)
            for key, entry in sdir.entries.items():
                if key == spec.from_shared or key.startswith(spec.from_shared.rstrip("/") + "/"):
                    adir.stage(key, size=entry["size"], sha256=entry["sha256"],
                               from_shared=True)
            continue
        if spec.is_dir:
            layers = (int((manifest.mtp or {}).get("num_hidden_layers", 0))
                      if spec.per_layer == "mtp_layers"
                      else int(manifest.architecture["num_hidden_layers"]))
            base = spec.relpath.rstrip("/")
            for i in range(layers):
                write_artifact(adir, f"{base}/layer_{i:02d}.bin", b"\0" * 32, step=spec.step)
            if spec.layout:
                arch = manifest.architecture
                layout = {
                    "artifact": base,
                    "expert_size": expert_pack_size(
                        arch["hidden_size"], arch["moe_intermediate_size"],
                        variant.bits, variant.group_size),
                    "num_layers": layers,
                    "num_experts": arch["num_experts"],
                    "components": [],
                }
                write_artifact(adir, f"{base}/{spec.layout}",
                               json.dumps(layout).encode(), step=spec.step)
            continue
        if spec.relpath == "model_weights.bin":
            write_artifact(adir, "model_weights.bin", b"\0" * 64, step=spec.step)
            write_artifact(adir, "model_weights.json",
                           json.dumps(make_weights_json(manifest, variant)).encode(),
                           step=spec.step)
            continue
        write_artifact(adir, spec.relpath, b"artifact:" + spec.relpath.encode(), step=spec.step)
        for companion in spec.companions:
            write_artifact(adir, companion, b"companion", step=spec.step)
    adir.commit()
    return adir


def _write_plain(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def make_legacy_snapshot(root: str, manifest, variants=None, with_blobs: bool = True,
                         with_bf16: bool = False, vocab_content=None) -> str:
    """Replicate the pre-migration layout: full artifact set per variant dir,
    duplicated vocab.bin (and optionally bf16/), no shared/, no integrity
    manifests."""
    repo_dir = paths.repo_root_dir(root, manifest.hf_repo)
    snapshot = os.path.join(repo_dir, "snapshots", "fixturehash0000")
    os.makedirs(snapshot, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    with open(os.path.join(repo_dir, "refs", "main"), "w") as f:
        f.write("fixturehash0000")
    if with_blobs:
        blob_dir = os.path.join(repo_dir, "blobs")
        os.makedirs(blob_dir, exist_ok=True)
        blob = os.path.join(blob_dir, "blob0")
        with open(blob, "w") as f:
            f.write("fake safetensors bytes")
        link = os.path.join(snapshot, "model-00001-of-00001.safetensors")
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(blob, snapshot), link)
        with open(os.path.join(snapshot, "config.json"), "w") as f:
            json.dump({"model_type": "fixture"}, f)

    for vname in (variants or list(manifest.variants)):
        variant = manifest.variant(vname)
        vdir = paths.variant_dir(snapshot, vname)
        content = vocab_content(vname) if vocab_content else b"legacy vocab content"
        _write_plain(os.path.join(vdir, "vocab.bin"), content)
        _write_plain(os.path.join(vdir, "model_weights.bin"), b"\0" * 64)
        _write_plain(os.path.join(vdir, "model_weights.json"),
                     json.dumps(make_weights_json(manifest, variant)).encode())
        os.makedirs(os.path.join(vdir, "system_prompt_cache"), exist_ok=True)
        if with_bf16 and manifest.mtp_artifacts_required:
            _write_plain(os.path.join(vdir, "bf16", "mtp_weights.bin"), b"bf16 mtp weights")
            _write_plain(os.path.join(vdir, "bf16", "mtp_weights.json"), b"{}")
        for spec in variant.artifacts.values():
            if not spec.is_dir or spec.from_shared:
                continue
            layers = (int((manifest.mtp or {}).get("num_hidden_layers", 0))
                      if spec.per_layer == "mtp_layers"
                      else int(manifest.architecture["num_hidden_layers"]))
            base = spec.relpath.rstrip("/")
            for i in range(layers):
                _write_plain(os.path.join(vdir, base, f"layer_{i:02d}.bin"), b"\0" * 32)
            if spec.layout:
                arch = manifest.architecture
                layout = {
                    "artifact": base,
                    "expert_size": expert_pack_size(
                        arch["hidden_size"], arch["moe_intermediate_size"],
                        variant.bits, variant.group_size),
                    "num_layers": layers,
                    "num_experts": arch["num_experts"],
                    "components": [],
                }
                _write_plain(os.path.join(vdir, base, spec.layout),
                             json.dumps(layout).encode())
    return snapshot


def make_snapshot(root: str, manifest, variants=None, with_blobs: bool = True,
                  want_optional: bool = False) -> str:
    """Create <root>/<models--Org--Name>/snapshots/<hash> with artifacts."""
    repo_dir = paths.repo_root_dir(root, manifest.hf_repo)
    snapshot = os.path.join(repo_dir, "snapshots", "fixturehash0000")
    os.makedirs(snapshot, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    with open(os.path.join(repo_dir, "refs", "main"), "w") as f:
        f.write("fixturehash0000")
    if with_blobs:
        blob_dir = os.path.join(repo_dir, "blobs")
        os.makedirs(blob_dir, exist_ok=True)
        blob = os.path.join(blob_dir, "blob0")
        with open(blob, "w") as f:
            f.write("fake safetensors bytes")
        link = os.path.join(snapshot, "model-00001-of-00001.safetensors")
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(blob, snapshot), link)
        with open(os.path.join(snapshot, "config.json"), "w") as f:
            json.dump({"model_type": "fixture"}, f)
    populate_shared(manifest, snapshot, want_optional=True)
    for vname in (variants or list(manifest.variants)):
        populate_variant(manifest, vname, snapshot, want_optional=want_optional)
    return snapshot
