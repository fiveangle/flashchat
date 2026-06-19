"""One-time migration: legacy artifact layout -> layout v1.

Legacy layout kept a full artifact set per quant variant
(flashchat/q4, flashchat/q8) with vocab.bin and bf16/ MTP weights
duplicated between them, and no integrity manifests. Layout v1 moves
variant-independent artifacts to flashchat/shared/, symlinks them back
into the variant dirs, and adopts every artifact into
.flashchat_artifacts.json (size + step provenance immediately; sha256
baselines optionally now or lazily on first deep verify).

After migration nothing reads the old layout — there is no dual-layout
support. The pass covers the HF cache and the offload dir in one run and
is idempotent per snapshot.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import configfile, paths
from .artifacts import ArtifactDir, sha256_file
from .manifest import Manifest
from .registry import CURRENT_LAYOUT_VERSION, Registry
from .steps import step_version


@dataclass
class DedupAction:
    snapshot: str
    relpath: str            # e.g. "vocab.bin" or "bf16/"
    keep_from: str          # variant whose copy becomes the shared canonical
    duplicates: list        # other variants whose copies are replaced by links
    bytes_reclaimed: int


@dataclass
class SnapshotPlan:
    manifest: Manifest
    snapshot: str
    location: str           # "local" | "offload"
    variants_present: list = field(default_factory=list)
    dedups: list = field(default_factory=list)
    adopt_files: int = 0
    adopt_bytes: int = 0
    conflicts: list = field(default_factory=list)

    @property
    def bytes_reclaimed(self) -> int:
        return sum(d.bytes_reclaimed for d in self.dedups)


@dataclass
class MigrationPlan:
    snapshots: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return all(not s.dedups and not s.adopt_files for s in self.snapshots)

    @property
    def bytes_reclaimed(self) -> int:
        return sum(s.bytes_reclaimed for s in self.snapshots)

    @property
    def adopt_bytes(self) -> int:
        return sum(s.adopt_bytes for s in self.snapshots)


def _is_real(path: str) -> bool:
    return os.path.lexists(path) and not os.path.islink(path)


def inventory(registry: Registry, cache_dir: str, offload_dir: str | None = None):
    """[(manifest, snapshot, location)] for every known model found on disk."""
    found = []
    roots = [(os.path.expanduser(cache_dir), "local")]
    if offload_dir and os.path.isdir(os.path.expanduser(offload_dir)):
        roots.append((os.path.expanduser(offload_dir), "offload"))
    for root, location in roots:
        for manifest in registry.manifests.values():
            snapshot = paths.snapshot_dir(root, manifest.hf_repo)
            if snapshot and os.path.isdir(paths.flashchat_dir(snapshot)):
                found.append((manifest, snapshot, location))
    return found


def _shared_dedup_targets(manifest: Manifest):
    """Shared artifacts that legacy variants duplicated: relpath -> spec."""
    out = {}
    for rel, spec in manifest.shared_artifacts.items():
        if rel == "vocab.bin":
            out["vocab.bin"] = spec
        elif rel.startswith("bf16/"):
            out["bf16/"] = spec
    return out


def plan_snapshot(manifest: Manifest, snapshot: str, location: str) -> SnapshotPlan:
    plan = SnapshotPlan(manifest=manifest, snapshot=snapshot, location=location)
    plan.variants_present = [
        v for v in manifest.variants
        if os.path.isdir(paths.variant_dir(snapshot, v))
    ]
    if not plan.variants_present:
        return plan

    shared = paths.shared_dir(snapshot)
    for rel in _shared_dedup_targets(manifest):
        shared_target = os.path.join(shared, rel.rstrip("/"))
        holders = []
        for v in plan.variants_present:
            cand = os.path.join(paths.variant_dir(snapshot, v), rel.rstrip("/"))
            if _is_real(cand):
                holders.append(v)
        if not holders:
            continue
        if os.path.exists(shared_target):
            # canonical already exists -> variants just need links
            keep_from = "shared"
            duplicates = holders
        else:
            keep_from = holders[0]
            duplicates = holders[1:]

        # Identity check before deleting anything: vocab/bf16 are small
        # enough to hash outright.
        if duplicates and keep_from != "shared":
            base = os.path.join(paths.variant_dir(snapshot, keep_from), rel.rstrip("/"))
            mismatched = [
                v for v in duplicates
                if not _trees_identical(
                    base, os.path.join(paths.variant_dir(snapshot, v), rel.rstrip("/")))
            ]
            if mismatched:
                plan.conflicts.append(
                    f"{rel} differs between variants {keep_from} and "
                    f"{'/'.join(mismatched)} — left in place, flag for regeneration")
                duplicates = [v for v in duplicates if v not in mismatched]

        reclaimed = sum(
            _tree_size(os.path.join(paths.variant_dir(snapshot, v), rel.rstrip("/")))
            for v in duplicates
        )
        plan.dedups.append(DedupAction(
            snapshot=snapshot, relpath=rel, keep_from=keep_from,
            duplicates=duplicates, bytes_reclaimed=reclaimed))

    # Adoption: count unmanifested files the integrity manifests will absorb.
    dirs = [shared] + [paths.variant_dir(snapshot, v) for v in plan.variants_present]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        adir = ArtifactDir(d)
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [n for n in dirnames if n != "system_prompt_cache"]
            for name in filenames:
                if name.startswith(".flashchat") or name == ".DS_Store":
                    continue
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    continue
                rel = os.path.relpath(full, d)
                if rel not in adir.entries:
                    plan.adopt_files += 1
                    plan.adopt_bytes += os.path.getsize(full)
    return plan


def _tree_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return paths.dir_size_bytes(path)


def _trees_identical(a: str, b: str) -> bool:
    if os.path.isfile(a) and os.path.isfile(b):
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        return sha256_file(a)[0] == sha256_file(b)[0]
    if os.path.isdir(a) and os.path.isdir(b):
        la = sorted(os.path.relpath(os.path.join(dp, f), a)
                    for dp, _, fs in os.walk(a) for f in fs if f != ".DS_Store")
        lb = sorted(os.path.relpath(os.path.join(dp, f), b)
                    for dp, _, fs in os.walk(b) for f in fs if f != ".DS_Store")
        if la != lb:
            return False
        return all(_trees_identical(os.path.join(a, r), os.path.join(b, r)) for r in la)
    return False


def build_plan(registry: Registry, cache_dir: str, offload_dir: str | None = None) -> MigrationPlan:
    plan = MigrationPlan()
    for manifest, snapshot, location in inventory(registry, cache_dir, offload_dir):
        plan.snapshots.append(plan_snapshot(manifest, snapshot, location))
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _step_for(manifest: Manifest, variant_name: str | None, rel: str) -> str | None:
    """Map an on-disk relpath back to the step that would have produced it."""
    specs = (manifest.shared_artifacts if variant_name is None
             else manifest.variant(variant_name).artifacts)
    for spec_rel, spec in specs.items():
        if spec.from_shared:
            continue
        if rel == spec_rel or rel in spec.companions:
            return spec.step
        if spec.is_dir and rel.startswith(spec_rel.rstrip("/") + "/"):
            return spec.step
    return None


def _adopt_dir(manifest: Manifest, variant_name: str | None, root: str,
               hash_now: bool, progress=None) -> int:
    """Record every untracked artifact file into the dir's manifest."""
    if not os.path.isdir(root):
        return 0
    adir = ArtifactDir(root, manifest.id, variant_name or "shared")
    adopted = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [n for n in dirnames if n != "system_prompt_cache"]
        for name in sorted(filenames):
            if name.startswith(".flashchat") or name == ".DS_Store" or name.endswith(".partial"):
                continue
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                continue
            rel = os.path.relpath(full, root)
            if rel in adir.entries:
                continue
            step = _step_for(manifest, variant_name, rel)
            digest = None
            if hash_now:
                if progress:
                    progress("hash", 0, 0, rel)
                digest, _ = sha256_file(full)
            adir.stage(rel, size=os.path.getsize(full), sha256=digest,
                       step=step, step_version=step_version(step) if step else None)
            adopted += 1
    adir.commit()
    return adopted


def execute_snapshot(plan: SnapshotPlan, hash_now: bool = False, progress=None) -> None:
    from .steps.materialize import materialize

    manifest, snapshot = plan.manifest, plan.snapshot
    shared = paths.shared_dir(snapshot)

    for action in plan.dedups:
        rel = action.relpath.rstrip("/")
        shared_target = os.path.join(shared, rel)
        if action.keep_from != "shared":
            src = os.path.join(paths.variant_dir(snapshot, action.keep_from), rel)
            os.makedirs(os.path.dirname(shared_target), exist_ok=True)
            os.rename(src, shared_target)
            if progress:
                progress("migrate", 0, 0, f"moved {action.keep_from}/{rel} -> shared/")
        for v in action.duplicates:
            dup = os.path.join(paths.variant_dir(snapshot, v), rel)
            if os.path.isdir(dup):
                import shutil
                shutil.rmtree(dup)
            elif os.path.lexists(dup):
                os.unlink(dup)

    # Adopt shared first so materialize can mirror hash entries down.
    _adopt_dir(manifest, None, shared, hash_now, progress)
    for v in plan.variants_present:
        vdir = paths.variant_dir(snapshot, v)
        _adopt_dir(manifest, v, vdir, hash_now, progress)
        materialize(manifest, v, shared, vdir, progress=progress)


def migrate_user_config(registry: Registry) -> dict:
    """Derive MODEL_BASE/MODEL_VARIANT from the legacy MODEL key; returns
    the changes applied (empty when nothing to do)."""
    values = configfile.load()
    if not values or "MODEL_BASE" in values:
        return {}
    legacy_model = values.get("MODEL", "")
    hit = registry.lookup_legacy(legacy_model) if legacy_model else None
    if hit is None:
        return {}
    manifest, variant_name = hit
    changes = {
        "MODEL_BASE": manifest.id,
        "MODEL_VARIANT": variant_name,
        "CONFIG_SCHEMA_VERSION": "3",
    }
    configfile.update(changes)
    return changes


def bootstrap_state(registry: Registry, cache_dir: str, offload_dir: str | None) -> None:
    """Enable every model with artifacts on disk; set default from config."""
    state = registry.state
    for manifest, _snapshot, _location in inventory(registry, cache_dir, offload_dir):
        state.enabled[manifest.id] = True
    values = configfile.load()
    hit = registry.lookup_legacy(values.get("MODEL", "")) if values else None
    if hit is not None:
        state.default_model = hit[0].id
    state.layout_version = CURRENT_LAYOUT_VERSION
    state.save()


def needed(registry: Registry) -> bool:
    return registry.state.layout_version < CURRENT_LAYOUT_VERSION


def run(registry: Registry, cache_dir: str, offload_dir: str | None = None,
        hash_now: bool = False, progress=None) -> MigrationPlan:
    plan = build_plan(registry, cache_dir, offload_dir)
    for snap_plan in plan.snapshots:
        execute_snapshot(snap_plan, hash_now=hash_now, progress=progress)
    migrate_user_config(registry)
    bootstrap_state(registry, cache_dir, offload_dir)
    return plan
