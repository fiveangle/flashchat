"""Interactive build flow: plan preview -> informed confirm -> run -> offers.

Shared by onboarding, the config wizard, manage, and the launch-time
`ensure` repair path, so every entry point behaves identically.
"""

import os

from .. import configfile, offload, paths, recipes, runner
from ..manifest import Manifest
from ..registry import Registry
from ..steps.download import download_snapshot
from ..status import hf_cache_dir, offload_dir
from . import common


def want_optional() -> bool:
    """BF16 MTP artifacts are wanted when the user enabled MTP_BF16."""
    return configfile.get("MTP_BF16", "0") == "1"


def want_mtp() -> bool:
    return configfile.mtp_enabled()


def ensure_variant_built(registry: Registry, manifest: Manifest, variant_name: str,
                         assume_yes: bool = False, force: bool = False) -> bool:
    """Walk download -> build -> offload-offer for one (model, variant).
    Returns True when the variant verifies ready at the end."""
    cache_dir = hf_cache_dir()
    snapshot = paths.snapshot_dir(cache_dir, manifest.hf_repo)

    plan = (recipes.plan(manifest, variant_name, snapshot, want_optional=want_optional(),
                         want_mtp=want_mtp(), force=force)
            if snapshot else None)

    if snapshot is None or plan.needs_download:
        restored_or_downloaded = _offer_download_or_restore(
            registry, manifest, cache_dir, assume_yes)
        if not restored_or_downloaded:
            return False
        snapshot = restored_or_downloaded
        if snapshot is None:
            print(common.red("download did not produce a snapshot"))
            return False
        plan = recipes.plan(manifest, variant_name, snapshot,
                            want_optional=want_optional(), want_mtp=want_mtp(),
                            force=force)

    if plan.empty:
        return True

    total = runner.plan_total_bytes(manifest, variant_name, plan)
    print(f"\nPlan for {common.bold(manifest.name)} [{variant_name}]:")
    for i, step in enumerate(plan.steps, 1):
        est = runner.estimate_step(manifest, variant_name, step.step)
        why = f" ({step.reason}{': ' + step.detail if step.detail else ''})"
        print(f"  {i}. {step.step} -> {', '.join(step.artifacts)}"
              f" — {est.description}{common.dim(why)}")
    if total:
        ok, free = runner.free_space_ok(snapshot, total)
        print(f"\n  Disk needed: ~{paths.human_bytes(total)}"
              f"  (free: {paths.human_bytes(free)})")
        if not ok:
            print(common.red("  Not enough free space — offload another model "
                             "first ('flashchat manage') or free up disk."))
            return False
    if not assume_yes and not common.confirm("Proceed?"):
        return False

    progress = common.ProgressLine()
    try:
        runner.execute_plan(manifest, variant_name, snapshot, plan, progress=progress,
                            options={"cache_dir": cache_dir})
    finally:
        progress.finish()

    from ..artifacts import variant_ready
    if not variant_ready(manifest, variant_name, snapshot, want_optional=False,
                         want_mtp=want_mtp()):
        print(common.red("build finished but verification failed — "
                         "check 'flashchat manage' for details"))
        return False
    print(common.green(f"{manifest.name} [{variant_name}] is ready."))

    offer_offload_originals(registry, manifest, snapshot, assume_yes=assume_yes)
    return True


def _offer_download_or_restore(registry: Registry, manifest: Manifest,
                               cache_dir: str, assume_yes: bool) -> str | None:
    od = offload_dir()
    if od and offload.archive_state(manifest, od) != "none":
        print(f"\nOriginals for {manifest.name} are archived on {od}.")
        if assume_yes or common.confirm("Restore them from the archive?"):
            progress = common.ProgressLine()
            try:
                offload.restore_originals(
                    manifest, paths.snapshot_dir(cache_dir, manifest.hf_repo) or "",
                    od, progress=progress)
                return paths.snapshot_dir(cache_dir, manifest.hf_repo)
            except offload.OffloadError:
                # fall through to a full restore (archive may be full-tree)
                try:
                    offload.restore_full(manifest, cache_dir, od, progress=progress)
                    return paths.snapshot_dir(cache_dir, manifest.hf_repo)
                except offload.OffloadError as e:
                    print(common.red(f"restore failed: {e}"))
            finally:
                progress.finish()
    print(f"\n{manifest.name} needs the original model from HuggingFace "
          f"({manifest.hf_repo}).")
    if not assume_yes and not common.confirm("Download now?"):
        return None
    progress = common.ProgressLine()
    try:
        return download_snapshot(manifest.hf_repo, cache_dir, progress=progress)
    finally:
        progress.finish()


def offer_offload_originals(registry: Registry, manifest: Manifest, snapshot: str,
                            assume_yes: bool = False) -> None:
    """Post-build offer: archive the now-redundant originals."""
    od = offload_dir()
    if not od:
        return
    if manifest.id in registry.state.never_offload:
        return
    size = offload.blobs_size(snapshot)
    if not size:
        return
    report = offload.preflight(od, needed_bytes=size)
    if not report.ok:
        return
    print(f"\nThe original model files ({paths.human_bytes(size)}) are no longer "
          f"needed for inference — the runtime artifacts are self-contained.")
    print(f"They can be archived to {od} and restored any time "
          f"(needed again only to re-extract artifacts). After verification, "
          f"the local original blob files will be removed.")
    reply = common.prompt("Archive originals now? [y/N/never]", "n").lower()
    if reply == "never":
        registry.state.never_offload.append(manifest.id)
        registry.state.save()
        return
    if reply in ("y", "yes"):
        progress = common.ProgressLine()
        try:
            moved = offload.offload_originals(manifest, snapshot, od, progress=progress)
            print(common.green(f"archived {paths.human_bytes(moved)} to {od}"))
        except offload.OffloadError as e:
            print(common.red(f"offload failed: {e}"))
        finally:
            progress.finish()
