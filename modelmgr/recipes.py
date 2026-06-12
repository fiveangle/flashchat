"""Recipe planning: manifest -> ordered list of steps for a (model, variant).

The plan is derived, never hand-written: walk the variant's declared
artifacts, check each against its integrity manifest, and emit the steps
whose outputs are missing/invalid/outdated. Valid artifacts are skipped, so
re-running a recipe converges to a no-op (idempotency).
"""

import os
from dataclasses import dataclass, field

from . import paths
from .artifacts import ArtifactDir, check_artifact, spec_required
from .manifest import Manifest
from .steps import step_version


@dataclass
class PlannedStep:
    step: str
    scope: str               # "shared" or the variant name
    artifacts: list          # relpaths the step produces
    reason: str              # missing | invalid | outdated-step | forced
    detail: str = ""


@dataclass
class Plan:
    model_id: str
    variant_name: str
    snapshot: str
    steps: list = field(default_factory=list)
    needs_download: bool = False

    @property
    def empty(self) -> bool:
        return not self.steps and not self.needs_download


def _artifact_needs_build(manifest: Manifest, variant, adir: ArtifactDir, spec,
                          want_optional: bool, force: bool):
    """(needs, reason, detail) for one declared artifact."""
    if not spec_required(manifest, spec, want_optional):
        return False, "", ""
    if force:
        return True, "forced", ""
    status = check_artifact(manifest, variant, adir, spec, want_optional=want_optional)
    if status.state in ("missing", "incomplete", "size-mismatch", "hash-mismatch", "invalid"):
        return True, "missing" if status.state == "missing" else "invalid", status.detail
    if spec.step:
        recorded = adir.entries.get(spec.relpath, {}).get("step_version")
        # Directory artifacts record versions on their member files.
        if spec.is_dir:
            members = [v for k, v in adir.entries.items()
                       if k.startswith(spec.relpath.rstrip("/") + "/")]
            if members:
                recorded = members[0].get("step_version")
        if recorded is not None and recorded != step_version(spec.step):
            return True, "outdated-step", f"built by v{recorded}, current v{step_version(spec.step)}"
    return False, "", ""


def _source_blobs_present(manifest: Manifest, snapshot: str) -> bool:
    """True when the original model files needed for extraction are local."""
    if not os.path.isdir(snapshot):
        return False
    for name in os.listdir(snapshot):
        if name.endswith(".safetensors"):
            # HF snapshots symlink into blobs/; a dangling link means the
            # blob was removed/offloaded even though the link survives.
            if os.path.exists(os.path.join(snapshot, name)):
                return True
    return False


def plan(manifest: Manifest, variant_name: str, snapshot: str,
         want_optional: bool = False, force: bool = False) -> Plan:
    variant = manifest.variant(variant_name)
    shared_adir = ArtifactDir(paths.shared_dir(snapshot), manifest.id, "shared")
    variant_adir = ArtifactDir(paths.variant_dir(snapshot, variant_name),
                               manifest.id, variant_name)
    result = Plan(model_id=manifest.id, variant_name=variant_name, snapshot=snapshot)

    # Shared artifacts the variant depends on (directly via from_shared, or
    # all shared when none are referenced — shared is small).
    needed_shared = []
    for spec in manifest.shared_artifacts.values():
        needs, reason, detail = _artifact_needs_build(
            manifest, variant, shared_adir, spec, want_optional, force)
        if needs:
            needed_shared.append((spec, reason, detail))

    # Variant artifacts, in manifest declaration order.
    needed_variant = []
    materialize = []
    for spec in variant.artifacts.values():
        if spec.from_shared:
            if not spec_required(manifest, spec, want_optional):
                continue
            status = check_artifact(manifest, variant, variant_adir, spec,
                                    want_optional=want_optional)
            if force or not status.satisfied or status.state == "missing":
                materialize.append(spec.relpath)
            continue
        needs, reason, detail = _artifact_needs_build(
            manifest, variant, variant_adir, spec, want_optional, force)
        if needs:
            needed_variant.append((spec, reason, detail))

    # Group consecutive artifacts produced by the same step.
    def emit(scope, items):
        by_step: dict = {}
        for spec, reason, detail in items:
            entry = by_step.setdefault(spec.step, PlannedStep(
                step=spec.step, scope=scope, artifacts=[], reason=reason, detail=detail))
            entry.artifacts.append(spec.relpath)
        result.steps.extend(by_step.values())

    emit("shared", needed_shared)
    emit(variant_name, needed_variant)
    if materialize:
        result.steps.append(PlannedStep(
            step="materialize_shared", scope=variant_name,
            artifacts=materialize, reason="missing"))

    # Any real (non-materialize) build step reads the original model files.
    builds = [s for s in result.steps if s.step != "materialize_shared"]
    if builds and not _source_blobs_present(manifest, snapshot):
        result.needs_download = True
    return result
