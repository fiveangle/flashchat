"""Registry: shipped manifests + user manifests + per-user state overlay.

Shipped manifests (assets/models/*.json) are read-only. User-added models
live in ~/.config/flashchat/models.d/ and shadow shipped ids. Per-user
state (which models are enabled, the default model, migration progress)
lives in ~/.config/flashchat/models.state.json — never in the shipped
files.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

from . import paths
from .manifest import Manifest, ManifestError, parse_manifest

STATE_SCHEMA = 1
CURRENT_LAYOUT_VERSION = 1


@dataclass
class RegistryState:
    enabled: dict = field(default_factory=dict)        # model id -> bool
    default_model: str | None = None
    layout_version: int = 0
    never_offload: list = field(default_factory=list)  # model ids the user said "never" for

    @classmethod
    def load(cls) -> "RegistryState":
        path = paths.state_file_path()
        if not os.path.isfile(path):
            return cls()
        data = paths.read_json(path)
        return cls(
            enabled=dict(data.get("enabled", {})),
            default_model=data.get("default_model"),
            layout_version=int(data.get("layout_version", 0)),
            never_offload=list(data.get("never_offload", [])),
        )

    def save(self) -> None:
        paths.write_json_atomic(
            paths.state_file_path(),
            {
                "schema": STATE_SCHEMA,
                "enabled": self.enabled,
                "default_model": self.default_model,
                "layout_version": self.layout_version,
                "never_offload": self.never_offload,
            },
        )


def resolved_id(manifest: Manifest, variant_name: str) -> str:
    """Engine-facing flat id for a (model, variant).

    The first legacy id wins so existing config files, API model ids, and
    user muscle memory keep working. New models derive an id matching the
    historical convention (repo with separators stripped + bits suffix).
    """
    variant = manifest.variant(variant_name)
    if variant.legacy_ids:
        return variant.legacy_ids[0]
    base = manifest.hf_repo.replace("/", "-").replace(".", "")
    return f"{base}-{variant_name}"


class Registry:
    def __init__(self, manifests: dict, state: RegistryState):
        self.manifests = manifests  # id -> Manifest, insertion-ordered
        self.state = state
        self._legacy: dict = {}  # legacy/resolved id -> (model id, variant name)
        for m in manifests.values():
            for vname, variant in m.variants.items():
                for lid in variant.legacy_ids:
                    if lid in self._legacy:
                        raise ManifestError(
                            f"legacy id '{lid}' claimed by both "
                            f"{self._legacy[lid][0]} and {m.id}"
                        )
                    self._legacy[lid] = (m.id, vname)
                self._legacy.setdefault(resolved_id(m, vname), (m.id, vname))

    @classmethod
    def load(cls) -> "Registry":
        manifests: dict = {}
        for path in sorted(glob.glob(os.path.join(paths.SHIPPED_MANIFEST_DIR, "*.json"))):
            m = parse_manifest(paths.read_json(path), source_path=path)
            if m.id in manifests:
                raise ManifestError(f"duplicate model id '{m.id}' ({path})")
            manifests[m.id] = m
        user_dir = paths.user_manifest_dir()
        if os.path.isdir(user_dir):
            for path in sorted(glob.glob(os.path.join(user_dir, "*.json"))):
                m = parse_manifest(paths.read_json(path), source_path=path, user_defined=True)
                manifests[m.id] = m  # user manifests shadow shipped ids
        return cls(manifests, RegistryState.load())

    def get(self, model_id: str) -> Manifest:
        try:
            return self.manifests[model_id]
        except KeyError:
            raise ManifestError(f"unknown model '{model_id}'") from None

    def lookup_legacy(self, legacy_or_resolved_id: str):
        """Map an old flat registry id (or a resolved id) to (Manifest, variant name)."""
        hit = self._legacy.get(legacy_or_resolved_id)
        if hit is None:
            return None
        model_id, vname = hit
        return self.manifests[model_id], vname

    def enabled_models(self) -> list:
        return [m for m in self.manifests.values() if self.is_enabled(m.id)]

    def is_enabled(self, model_id: str) -> bool:
        return bool(self.state.enabled.get(model_id, False))

    def default_model(self) -> Manifest | None:
        """The user's default if set and enabled, else the shipped suggestion."""
        if self.state.default_model and self.state.default_model in self.manifests:
            return self.manifests[self.state.default_model]
        for m in self.manifests.values():
            if m.suggested_default:
                return m
        return next(iter(self.manifests.values()), None)
