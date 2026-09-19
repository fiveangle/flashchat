"""Interactive build flow: plan preview -> informed confirm -> run -> offers.

Shared by onboarding, the config wizard, manage, and the launch-time
`ensure` repair path, so every entry point behaves identically.
"""

import os

from .. import configfile, offload, paths, recipes, runner
from ..artifacts import variant_ready
from ..manifest import Manifest
from ..registry import Registry
from ..steps.download import download_snapshot
from ..status import hf_cache_dir, offload_dir
from . import common


def want_optional() -> bool:
    """BF16 MTP artifacts are wanted when the user enabled MTP_BF16."""
    return configfile.get("MTP_BF16") == "1"


def want_mtp() -> bool:
    return configfile.mtp_enabled()


def ensure_variant_built(registry: Registry, manifest: Manifest, variant_name: str,
                         assume_yes: bool = False, force: bool = False,
                         include_optional: bool = False,
                         source_snapshot: str | None = None) -> bool:
    """Build local runtime artifacts, optionally reading source weights elsewhere."""
    cache_dir = hf_cache_dir()
    local_snapshot = paths.snapshot_dir(cache_dir, manifest.hf_repo)
    optional = include_optional or want_optional()
    source = source_snapshot
    if not local_snapshot and not force and source is None:
        restored = _offer_runtime_restore(manifest, variant_name, cache_dir, optional)
        if restored is not None:
            return restored
    if not local_snapshot:
        source = source or _offer_build_source(
            manifest, cache_dir, None, prefer_offload=False)
        if not source:
            return False
        local_snapshot = paths.snapshot_dir(cache_dir, manifest.hf_repo)
        if not local_snapshot and os.path.commonpath([
                os.path.abspath(source), os.path.abspath(cache_dir)]) == os.path.abspath(cache_dir):
            local_snapshot = source
        if not local_snapshot:
            local_snapshot = os.path.join(
                paths.repo_root_dir(cache_dir, manifest.hf_repo),
                "snapshots", os.path.basename(os.path.normpath(source)))

    plan = recipes.plan(manifest, variant_name, local_snapshot,
                        want_optional=optional, want_mtp=want_mtp(),
                        force=force)
    if plan.needs_download and source is None:
        source = _offer_build_source(manifest, cache_dir, local_snapshot,
                                     prefer_offload=include_optional)
        if not source:
            return False
    source = source or local_snapshot

    if plan.needs_download:
        if not recipes.source_blobs_present(manifest, source):
            print(common.red("selected build source does not contain original model files"))
            return False
        plan.needs_download = False

    if plan.empty:
        return True

    if source != local_snapshot:
        print(common.dim(f"Using original model files from: {source}"))
        print(common.dim(f"Writing runtime artifacts to: {local_snapshot}"))

    total = runner.plan_total_bytes(manifest, variant_name, plan)
    print(f"\nPlan {common.bold(manifest.name)} [{variant_name}]:")
    for i, step in enumerate(plan.steps, 1):
        est = runner.estimate_step(manifest, variant_name, step.step)
        why = f" ({step.reason}{': ' + step.detail if step.detail else ''})"
        print(f"  {i}. {step.step} -> {', '.join(step.artifacts)}"
              f" — {est.description}{common.dim(why)}")
    if total:
        ok, free = runner.free_space_ok(local_snapshot, total)
        print(f"\n  Disk needed: ~{paths.human_bytes(total)}"
              f" (free: {paths.human_bytes(free)})")
        if not ok:
            print(common.red("not enough free space"))
            return False
    if not assume_yes and not common.confirm("Proceed?"):
        return False

    progress = common.ProgressLine()
    try:
        changed_scopes = runner.execute_plan(
            manifest, variant_name, source, plan, progress=progress,
            options={"cache_dir": cache_dir}, output_snapshot=local_snapshot)
    finally:
        progress.finish()

    if not variant_ready(manifest, variant_name, local_snapshot,
                         want_optional=optional, want_mtp=want_mtp()):
        print(common.red("build finished but local verification failed — "
                         "check 'flashchat manage' for details"))
        return False
    print(common.green(f"{manifest.name} [{variant_name}] local artifacts are ready."))

    od = offload_dir()
    if changed_scopes and od and offload.archive_state(manifest, od) == "full":
        sync_progress = common.ProgressLine()
        try:
            synced = offload.sync_artifact_scopes(
                manifest, local_snapshot, od, sorted(changed_scopes),
                progress=sync_progress)
            if synced:
                print(common.green(
                    f"offload artifacts updated ({paths.human_bytes(synced)})"))
        except offload.OffloadError as e:
            print(common.yellow(f"offload artifact sync deferred: {e}"))
        finally:
            sync_progress.finish()
    elif source == local_snapshot:
        offer_offload_model(registry, manifest, local_snapshot, assume_yes=assume_yes)
    return True


def _offer_runtime_restore(manifest: Manifest, variant_name: str,
                           cache_dir: str, optional: bool) -> bool | None:
    """Restore a ready archived variant; None means continue to the build flow."""
    od = offload_dir()
    snapshot = paths.snapshot_dir(od, manifest.hf_repo) if od else None
    if not snapshot or not variant_ready(
            manifest, variant_name, snapshot,
            want_optional=optional, want_mtp=want_mtp()):
        return None
    plan = offload.plan_restore(manifest, cache_dir, od, "runtime", variant_name)
    print(f"\nReady runtime files for {manifest.name} [{variant_name}] "
          f"are available in the offload archive at {od}.")
    print(f"Restore size: {paths.human_bytes(plan.needed_bytes)} "
          f"(free: {paths.human_bytes(plan.free_bytes)}). "
          "Original model files stay in the archive.")
    if not common.confirm("Restore runtime files locally?", default=True):
        return False
    progress = common.ProgressLine()
    try:
        offload.restore_runtime_only(manifest, cache_dir, od, progress=progress,
                                     variant_name=variant_name)
    except offload.OffloadError as e:
        print(common.red(f"restore failed: {e}"))
        return False
    finally:
        progress.finish()
    local_snapshot = paths.snapshot_dir(cache_dir, manifest.hf_repo)
    if not local_snapshot or not variant_ready(
            manifest, variant_name, local_snapshot,
            want_optional=optional, want_mtp=want_mtp()):
        print(common.red("restore finished but local verification failed — "
                         "check 'flashchat manage' for details"))
        return False
    print(common.green(f"{manifest.name} [{variant_name}] local artifacts are ready."))
    return True


def _offer_build_source(manifest: Manifest, cache_dir: str,
                        local_snapshot: str | None,
                        prefer_offload: bool = False,
                        allow_download: bool = True) -> str | None:
    od = offload_dir()
    if od and offload.archive_state(manifest, od) == "full":
        offload_snapshot = paths.snapshot_dir(od, manifest.hf_repo)
        if offload_snapshot:
            print(f"\nOriginal model files for {manifest.name} are available in "
                  f"the offload copy at {od}.")
            if common.confirm("Generate artifacts directly from the offload copy?",
                              default=prefer_offload):
                return offload_snapshot

    if not allow_download:
        print(common.yellow(
            "Original model source files are not available locally. "
            "Use restore/download explicitly before building this artifact."))
        return None
    return _offer_download_or_restore(manifest, cache_dir, local_snapshot)

def _offer_download_or_restore(manifest: Manifest, cache_dir: str,
                               local_snapshot: str | None) -> str | None:
    od = offload_dir()
    if od and offload.archive_state(manifest, od) == "originals":
        print(f"\nOriginals for {manifest.name} are archived on {od}.")
        if common.confirm("Restore original source blobs locally?", default=False):
            progress = common.ProgressLine()
            try:
                offload.restore_originals(manifest, cache_dir, od,
                                          progress=progress)
                return paths.snapshot_dir(cache_dir, manifest.hf_repo)
            except offload.OffloadError as e:
                print(common.red(f"restore failed: {e}"))
            finally:
                progress.finish()

    print(f"\n{manifest.name} needs the original model from HuggingFace "
          f"({manifest.hf_repo}).")
    print("  n) cancel")
    print("  l) download originals to the local HuggingFace cache")
    if od:
        print(f"  o) download originals to offload storage ({od})")
    choice = common.prompt("Download originals where?", "n").lower()
    if choice in ("", "n", "no", "cancel"):
        return None
    if choice in ("l", "local"):
        dest_cache = cache_dir
    elif choice in ("o", "offload") and od:
        dest_cache = od
    else:
        print(common.red("download cancelled: choose 'l' for local or 'o' for offload"))
        return None

    progress = common.ProgressLine()
    try:
        snapshot = download_snapshot(manifest.hf_repo, dest_cache, progress=progress)
    finally:
        progress.finish()
    if dest_cache == od:
        return paths.snapshot_dir(od, manifest.hf_repo) or snapshot
    return snapshot

def offer_offload_model(registry: Registry, manifest: Manifest, snapshot: str,
                        assume_yes: bool = False) -> None:
    """Post-build offer: offload the model and remove redundant originals."""
    od = offload_dir()
    if not od:
        return
    if manifest.id in registry.state.never_offload:
        return
    size = offload.blobs_size(snapshot)
    if not size:
        return
    report = offload.preflight(od, needed_bytes=0)
    if not report.ok:
        return
    print(f"\nThe original model files ({paths.human_bytes(size)}) are no longer "
          f"needed for inference — the runtime artifacts are self-contained.")
    print(f"Flashchat can sync the full model directory to {od}, record "
          f"lightweight offload metadata, and remove only the local original "
          f"blob files.")
    reply = common.prompt("Offload model now? [y/N/never]", "n").lower()
    if reply == "never":
        registry.state.never_offload.append(manifest.id)
        registry.state.save()
        return
    if reply in ("y", "yes"):
        progress = common.ProgressLine()
        try:
            moved = offload.offload_model(manifest, snapshot, od, progress=progress)
            print(common.green(f"offloaded {paths.human_bytes(moved)} to {od}"))
        except offload.OffloadError as e:
            print(common.red(f"offload failed: {e}"))
        finally:
            progress.finish()
