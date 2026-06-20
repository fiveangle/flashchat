"""Model status computation: what exists where, for every known model.

Single source for the status views in `flashchat models`, the config
wizard, the manage TUI, and the launch-time `ensure` checks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import configfile, offload, paths
from .artifacts import variant_status
from .manifest import Manifest
from .registry import Registry, resolved_id


@dataclass
class VariantStatus:
    name: str
    ready: bool
    offloaded: bool = False
    local_bytes: int = 0
    offload_bytes: int = 0
    missing: list = field(default_factory=list)   # ArtifactStatus that fail
    offload_missing: list = field(default_factory=list)
    resolved_id: str = ""


@dataclass
class ModelStatus:
    manifest: Manifest
    enabled: bool
    snapshot: str | None          # local snapshot path
    originals_local: bool
    originals_bytes: int
    originals_offloaded: bool
    offload_originals_bytes: int
    archive: str                  # none | originals | full
    offload_snapshot: str | None  # snapshot path inside the offload dir
    variants: dict = field(default_factory=dict)  # name -> VariantStatus

    @property
    def any_ready(self) -> bool:
        return any(v.ready for v in self.variants.values())

    def summary_line(self, variant_name: str) -> str:
        v = self.variants[variant_name]
        if v.ready:
            return "ready"
        if v.offloaded:
            return "offloaded"
        source_available = self.originals_local or self.archive in ("originals", "full")
        if not self.snapshot and not self.offload_snapshot and not source_available:
            return "not downloaded"
        broken = [s for s in (v.missing + v.offload_missing)
                  if s.state not in ("missing", "incomplete")]
        if broken:
            return f"needs attention ({broken[0].relpath}: {broken[0].state})"
        incomplete = [s for s in v.missing if s.state == "incomplete"]
        if incomplete and len(incomplete) == len(v.missing):
            return f"usable, MTP needs rebuild ({incomplete[0].relpath})"
        if source_available:
            return "not extracted"
        return "not downloaded"


def hf_cache_dir() -> str:
    return os.path.expanduser(
        configfile.get("HUGGINGFACE_CACHE_DIR", paths.DEFAULT_HF_CACHE))


def offload_dir() -> str:
    return configfile.get("OFFLOAD_DIR", "")


def model_status(registry: Registry, manifest: Manifest,
                 cache_dir: str | None = None,
                 offload_root: str | None = None,
                 check_offload: bool = True) -> ModelStatus:
    cache_dir = cache_dir or hf_cache_dir()
    offload_root = offload_root if offload_root is not None else offload_dir()

    snapshot = paths.snapshot_dir(cache_dir, manifest.hf_repo)
    originals_bytes = offload.blobs_size(snapshot) if snapshot else 0

    archive = "none"
    offload_snapshot = None
    offload_originals_bytes = 0
    if check_offload and offload_root:
        root = os.path.expanduser(offload_root)
        if os.path.isdir(root):
            archive = offload.archive_state(manifest, root)
            offload_snapshot = paths.snapshot_dir(root, manifest.hf_repo)
            offload_originals_bytes = offload.blobs_size(offload_snapshot) if offload_snapshot else 0

    status = ModelStatus(
        manifest=manifest,
        enabled=registry.is_enabled(manifest.id),
        snapshot=snapshot,
        originals_local=originals_bytes > 0,
        originals_bytes=originals_bytes,
        originals_offloaded=offload_originals_bytes > 0,
        offload_originals_bytes=offload_originals_bytes,
        archive=archive,
        offload_snapshot=offload_snapshot,
    )
    for vname in manifest.variants:
        ready = False
        offloaded_ready = False
        missing = []
        offload_missing = []
        local_bytes = 0
        offload_bytes = 0
        if snapshot:
            states = variant_status(manifest, vname, snapshot)
            missing = [s for s in states if not s.satisfied]
            ready = not missing
            local_bytes = paths.dir_size_bytes(paths.variant_dir(snapshot, vname)) if ready else 0
        if not ready and offload_snapshot:
            offload_states = variant_status(manifest, vname, offload_snapshot)
            offload_missing = [s for s in offload_states if not s.satisfied]
            offloaded_ready = not offload_missing
            offload_bytes = paths.dir_size_bytes(paths.variant_dir(offload_snapshot, vname)) if offloaded_ready else 0
        status.variants[vname] = VariantStatus(
            name=vname, ready=ready, offloaded=offloaded_ready,
            local_bytes=local_bytes, offload_bytes=offload_bytes,
            missing=missing, offload_missing=offload_missing,
            resolved_id=resolved_id(manifest, vname))
    return status


def all_statuses(registry: Registry, enabled_only: bool = False,
                 check_offload: bool = True) -> list:
    out = []
    for manifest in registry.manifests.values():
        if enabled_only and not registry.is_enabled(manifest.id):
            continue
        out.append(model_status(registry, manifest, check_offload=check_offload))
    return out


def selected_model(registry: Registry):
    """(manifest, variant_name) from config, with legacy fallback."""
    values = configfile.load()
    base = values.get("MODEL_BASE")
    if base and base in registry.manifests:
        manifest = registry.manifests[base]
        vname = values.get("MODEL_VARIANT") or manifest.default_variant
        if vname in manifest.variants:
            return manifest, vname
    hit = registry.lookup_legacy(values.get("MODEL", ""))
    if hit:
        return hit
    default = registry.default_model()
    if default is None:
        return None
    return default, default.default_variant
