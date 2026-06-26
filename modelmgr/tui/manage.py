"""[M]anage: manual outlier-fixing for model storage.

The automation (onboarding, ensure, post-build offers) handles the normal
lifecycle; this menu exists for everything flashchat does not do on its
own: per-artifact verification against hash baselines, building/repairing
artifacts (creating what is missing and rebuilding what fails verification),
deleting variants, and manual archive moves.
"""

import os
import shutil

from .. import configfile, offload, paths, recipes, resolved
from ..artifacts import ArtifactDir, shared_status, variant_status
from ..registry import Registry, resolved_id
from ..server import guard_model_not_serving
from ..status import hf_cache_dir, model_status, offload_dir
from . import build, common


def run(registry: Registry) -> None:
    while True:
        manifests = list(registry.manifests.values())
        common.heading("Manage models")
        for i, m in enumerate(manifests, 1):
            status = model_status(registry, m)
            ready = [v for v, vs in status.variants.items() if vs.ready]
            desc = f"ready: {', '.join(ready)}" if ready else "no variants built"
            if not status.snapshot:
                desc = "not local" + (f" (archive: {status.archive})"
                                      if status.archive != "none" else "")
            # hf_repo disambiguates models that share a display name
            # (e.g. the native and mlx-community Qwen3.6-35B-A3B).
            print(f"  {i}) {m.name} ({m.hf_repo}) — {desc}")
        choice = common.select_number(len(manifests), "Manage which model (empty to exit)")
        if choice is None:
            return
        _model_menu(registry, manifests[int(choice) - 1])


def _state_lines(registry: Registry, manifest) -> "tuple":
    status = model_status(registry, manifest)
    print(f"\n=== Manage: {common.bold(manifest.name)} ===")
    print(f"snapshot:  {status.snapshot or common.dim('not local')}")
    originals = (f"local ({paths.human_bytes(status.originals_bytes)})"
                 if status.originals_local else "not local")
    print(f"originals: {originals} | archive: {status.archive}"
          + (f" ({offload_dir()})" if offload_dir() else " (no offload dir configured)"))
    if status.snapshot:
        shared = shared_status(manifest, status.snapshot, want_optional=False)
        if shared:
            line = "  shared/   " + "  ".join(
                f"{_artifact_label('shared', s.relpath)}: {_st(s)}" for s in shared)
            print(line)
        for vname in manifest.variants:
            states = variant_status(manifest, vname, status.snapshot)
            line = f"  {vname + '/':9} " + "  ".join(
                f"{_artifact_label(vname, s.relpath)}: {_st(s)}" for s in states)
            print(line)
    return status


def _artifact_label(scope: str, relpath: str) -> str:
    if relpath in ("bf16/mtp_weights.bin", "bf16/"):
        return "MTP BF16 weights"
    return relpath


def _scoped_artifact_label(scope: str, relpath: str) -> str:
    return f"{scope}/{_artifact_label(scope, relpath)}"


def _st(s) -> str:
    if s.state == "ok":
        return common.green("OK")
    if s.state == "unhashed":
        return common.yellow("OK*")  # valid, no hash baseline yet
    if s.state == "missing" and not s.required:
        return common.dim("absent (optional)")
    color = common.yellow if s.state in ("missing", "incomplete") else common.red
    return color(s.state + (f" ({s.detail})" if s.detail else ""))


def _model_menu(registry: Registry, manifest) -> None:
    while True:
        status = _state_lines(registry, manifest)
        print("""
 [v] re-check (quick verify)      [h] deep verify (full hash; offers to repair)
 [b] build / repair               create missing + rebuild broken artifacts
 [o] offload model                sync to offload storage + remove source blobs
 [l] restore from archive...      [d] select local components to delete (to reclaim disk space)
 [x] delete entire model          remove ALL local files and artifacts for this model
 [q] back""")
        choice = common.prompt("Action", "q").lower()
        if choice in ("q", ""):
            return
        handler = {
            "v": lambda: None,  # loop re-renders
            "h": lambda: _deep_verify(registry, manifest, status),
            "b": lambda: _build_repair(registry, manifest, status),
            "o": lambda: _offload_model(registry, manifest, status),
            "l": lambda: _restore_menu(registry, manifest),
            "d": lambda: _delete_components(registry, manifest, status),
            "x": lambda: _delete_local(registry, manifest, status),
        }.get(choice)
        if handler:
            handler()


def _deep_verify(registry, manifest, status) -> None:
    if not status.snapshot:
        print("model is not local")
        return
    progress = common.ProgressLine()
    total_bad = 0
    unhashed = []
    corrupt = []  # (scope, rel, spec) for rebuildable corrupt artifacts
    for scope, states in [("shared", shared_status(manifest, status.snapshot, deep=True,
                                                   progress=None))] + [
            (v, variant_status(manifest, v, status.snapshot, deep=True, progress=None))
            for v in manifest.variants]:
        for s in states:
            label = f"{scope}/{s.relpath}"
            if s.state in ("hash-mismatch", "size-mismatch", "invalid"):
                print(common.red(f"  CORRUPT: {label} — {s.state} {s.detail}"))
                total_bad += 1
                spec = (manifest.shared_artifacts.get(s.relpath) if scope == "shared"
                        else manifest.variant(scope).artifacts.get(s.relpath))
                if spec is not None and spec.step:
                    corrupt.append((scope, s.relpath, spec))
            elif s.state == "unhashed":
                unhashed.append((scope, s.relpath))
    progress.finish()
    if total_bad == 0:
        print(common.green("deep verify passed — all hashed artifacts match"))
    elif corrupt:
        # The only thing the old standalone "regenerate" did, now driven by the
        # check that actually detected the problem.
        print(common.yellow(
            f"\n{len(corrupt)} corrupt artifact(s) can be rebuilt "
            "(they will be DELETED, then regenerated)."))
        if common.confirm("Rebuild them now?", default=False):
            err = guard_model_not_serving(
                [resolved_id(manifest, v) for v in manifest.variants])
            if err:
                print(common.red(err))
            else:
                variants = sorted({sc for sc, _r, _s in corrupt if sc != "shared"}
                                  or {manifest.default_variant})
                if any(sc == "shared" for sc, _r, _s in corrupt) \
                        and manifest.default_variant not in variants:
                    variants.append(manifest.default_variant)
                _rebuild_artifacts(registry, manifest, status, corrupt, variants)
    if unhashed:
        print(f"{len(unhashed)} artifact(s) have no hash baseline yet "
              "(adopted from the pre-hash layout).")
        if common.confirm("Compute hash baselines now?", default=False):
            _backfill_hashes(manifest, status.snapshot, progress=common.ProgressLine())


def _backfill_hashes(manifest, snapshot, progress=None) -> None:
    dirs = [(paths.shared_dir(snapshot), "shared")] + [
        (paths.variant_dir(snapshot, v), v) for v in manifest.variants]
    for root, scope in dirs:
        if not os.path.isdir(root):
            continue
        adir = ArtifactDir(root, manifest.id, scope)
        for rel, entry in sorted(adir.entries.items()):
            if entry.get("sha256") is None and not entry.get("from_shared"):
                if progress:
                    progress("hash", 0, 0, f"{scope}/{rel}")
                adir.backfill(rel, step=entry.get("step"),
                              step_version=entry.get("step_version"))
        adir.commit()
    if progress:
        progress.finish()
    print(common.green("hash baselines recorded"))


def _rebuild_artifacts(registry, manifest, status, targets, variants) -> None:
    """Executor: delete the given (scope, rel, spec) artifacts, then (re)build the
    named variants. The recipe planner sees the holes and heals them plus any
    dependents. Callers are responsible for the serving guard and confirmation."""
    for scope, rel, spec in targets:
        root = (paths.shared_dir(status.snapshot) if scope == "shared"
                else paths.variant_dir(status.snapshot, scope))
        target_path = os.path.join(root, rel.rstrip("/"))
        if os.path.isdir(target_path) and not os.path.islink(target_path):
            shutil.rmtree(target_path)
        elif os.path.lexists(target_path):
            os.unlink(target_path)
        adir = ArtifactDir(root, manifest.id, scope if scope != "shared" else "shared")
        adir.forget(rel)
        for companion in spec.companions:
            cpath = os.path.join(root, companion)
            if os.path.lexists(cpath):
                os.unlink(cpath)
            adir.forget(companion)
        adir.commit()
    source_snapshot = status.snapshot
    if not recipes.source_blobs_present(manifest, status.snapshot):
        source_snapshot = build._offer_build_source(
            manifest, hf_cache_dir(), status.snapshot, prefer_offload=True,
            allow_download=True)
    if not source_snapshot:
        return
    for v in variants:
        build.ensure_variant_built(
            registry, manifest, v, assume_yes=True, include_optional=True,
            source_snapshot=source_snapshot)


def _build_repair(registry, manifest, status) -> None:
    """Bring the model to ready: create anything MISSING and rebuild anything that
    fails verification. Replaces the old build-missing / regenerate split — there is
    no 'variant vs artifact' choice; you pick what to fix, it fixes all of it."""
    if not status.snapshot:
        print("model is not local")
        return
    not_ready = [v for v in manifest.variants if not status.variants[v].ready]
    if not_ready:
        candidates = not_ready
    else:
        candidates = list(manifest.variants)
    if len(candidates) > 1 and not_ready:
        for i, v in enumerate(not_ready, 1):
            print(f"  {i}) {v} — {status.summary_line(v)}")
        print(f"  {len(candidates) + 1}) all of them")
        choice = common.select_number(len(candidates) + 1, "Build/repair which")
        if choice is None:
            return
        chosen = (candidates if int(choice) == len(candidates) + 1
                  else [candidates[int(choice) - 1]])
    else:
        chosen = candidates

    # Partition unsatisfied, buildable artifacts into create (absent) vs rebuild
    # (present-but-broken). Only specs with a build `step` can be (re)generated.
    create = []                 # display labels for missing artifacts
    rebuild_targets = []        # (scope, rel, spec) to delete-then-rebuild
    rebuild_labels = []         # display labels with the failure reason

    def _consider(scope, s, spec):
        if spec is None or not spec.step or s.satisfied:
            return
        if s.state == "missing":
            create.append(_scoped_artifact_label(scope, s.relpath))
        else:
            rebuild_targets.append((scope, s.relpath, spec))
            detail = s.state + (f": {s.detail}" if s.detail else "")
            rebuild_labels.append(f"{_scoped_artifact_label(scope, s.relpath)} ({detail})")

    for v in chosen:
        var = manifest.variant(v)
        for s in variant_status(manifest, v, status.snapshot,
                                want_optional=True, want_mtp=build.want_mtp()):
            _consider(v, s, var.artifacts.get(s.relpath))
    for s in shared_status(manifest, status.snapshot, want_optional=True):
        _consider("shared", s, manifest.shared_artifacts.get(s.relpath))

    if not create and not rebuild_targets:
        print("nothing buildable is missing or broken for the selected variant(s)")
        return

    print("\nbuild / repair plan:")
    if create:
        print("  create (missing): " + ", ".join(sorted(set(create))))
    if rebuild_labels:
        print("  rebuild (broken): " + ", ".join(rebuild_labels))
    print("  (re)build variant(s): " + ", ".join(chosen))
    if rebuild_targets:
        print(common.yellow("  the 'broken' artifacts above will be DELETED, then rebuilt"))
    if not common.confirm("Proceed?", default=False):
        return
    err = guard_model_not_serving([resolved_id(manifest, v) for v in manifest.variants])
    if err:
        print(common.red(err))
        return
    _rebuild_artifacts(registry, manifest, status, rebuild_targets, chosen)


def _offload_model(registry, manifest, status) -> None:
    if not status.snapshot or not status.originals_local:
        print("no local source blobs to offload")
        return
    od = offload_dir()
    if not od:
        print("no OFFLOAD_DIR configured — set it in './flashchat config'")
        return
    print(f"This syncs the full local Hugging Face model directory to {od}.")
    print("After rsync completes, Flashchat records lightweight file metadata "
          "in the offload journal and removes only the local Hugging Face "
          "safetensors source blobs. Runtime artifacts stay local and usable.")
    print("system_prompt_cache directories are skipped and never written to the "
          "offload copy because they may contain prompt-derived user data.")
    if not status.any_ready:
        print(common.yellow(
            "no runtime variant is built yet — offloading the model now "
            "means they must be restored before extraction"))
        if not common.confirm("Archive anyway?", default=False):
            return
    elif not common.confirm("Offload model and remove local source blobs?", default=False):
        return
    progress = common.ProgressLine()
    try:
        moved = offload.offload_model(manifest, status.snapshot, od, progress=progress)
        print(common.green(f"offloaded {paths.human_bytes(moved)}"))
        resolved.write(registry)
    except offload.OffloadError as e:
        print(common.red(str(e)))
    finally:
        progress.finish()


def _restore_menu(registry, manifest) -> None:
    od = offload_dir()
    if not od or offload.archive_state(manifest, od) == "none":
        print("no archive found for this model")
        return
    print("  1) restore originals only")
    print("  2) restore runtime artifacts only")
    print("  3) restore everything")
    choice = common.select_number(3, "Restore what")
    if choice is None:
        return
    cache = hf_cache_dir()
    progress = common.ProgressLine()
    try:
        if choice == 1:
            snapshot = paths.snapshot_dir(cache, manifest.hf_repo) or ""
            n = offload.restore_originals(manifest, snapshot, od, progress=progress)
        elif choice == 2:
            n = offload.restore_runtime_only(manifest, cache, od, progress=progress)
        else:
            n = offload.restore_full(manifest, cache, od, progress=progress)
        print(common.green(f"restored {paths.human_bytes(n)}"))
        resolved.write(registry)
    except offload.OffloadError as e:
        print(common.red(str(e)))
    finally:
        progress.finish()


def _delete_components(registry, manifest, status) -> None:
    if not status.snapshot:
        print(common.yellow("no local components exist for this model — nothing to reclaim"))
        if status.archive != "none":
            print(common.dim("archive/offload storage is remote backup state; use restore if you want files local again"))
        return

    components = []
    originals = offload.list_blob_files(status.snapshot)
    originals_size = sum(os.path.getsize(target) for _rel, target in originals
                         if os.path.isfile(target))
    if originals_size:
        components.append({
            "kind": "originals",
            "label": "original Hugging Face source blobs",
            "detail": "needed to rebuild artifacts; runtime stays usable",
            "size": originals_size,
            "ids": [resolved_id(manifest, v) for v in manifest.variants],
        })

    for v in ("q4", "q8"):
        if v not in manifest.variants:
            continue
        root = paths.variant_dir(status.snapshot, v)
        if os.path.isdir(root):
            components.append({
                "kind": "variant",
                "variant": v,
                "label": f"{v} runtime artifacts",
                "detail": f"removes local flashchat/{v}/ runtime files",
                "size": paths.dir_size_bytes(root),
                "ids": [resolved_id(manifest, v)],
            })

    if not components:
        print("no local source blobs or q4/q8 runtime artifacts exist")
        return

    print(common.yellow("\nDelete local components to reclaim disk space"))
    print(common.dim("Offload/archive copies are not changed by this action."))
    for i, item in enumerate(components, 1):
        print(f"  {i}) {common.bold(item['label'])} "
              f"{common.green(paths.human_bytes(item['size']))}")
        print(f"     {common.dim(item['detail'])}")

    choice = common.select_number(len(components), "Delete which component")
    if choice is None:
        return
    item = components[int(choice) - 1]

    err = guard_model_not_serving(item["ids"])
    if err:
        print(common.red(err))
        return
    if not common.confirm_exact(
            manifest.id,
            f"deleting {item['label']} ({paths.human_bytes(item['size'])})"):
        return

    if item["kind"] == "originals":
        removed = 0
        for _rel, target in originals:
            if os.path.isfile(target):
                removed += os.path.getsize(target)
                os.unlink(target)
        print(common.green(
            f"deleted original source blobs ({paths.human_bytes(removed)})"))
    else:
        shutil.rmtree(paths.variant_dir(status.snapshot, item["variant"]))
        print(common.green(f"deleted {item['variant']} runtime artifacts "
                           f"({paths.human_bytes(item['size'])})"))
    resolved.write(registry)

def _delete_local(registry, manifest, status) -> None:
    if not status.snapshot:
        print("model is not local")
        return
    err = guard_model_not_serving([resolved_id(manifest, v) for v in manifest.variants])
    if err:
        print(common.red(err))
        return
    repo_root = os.path.dirname(os.path.dirname(status.snapshot))
    size = paths.dir_size_bytes(repo_root)
    if status.archive == "none":
        print(common.yellow(
            f"there is NO archive of this model — deleting removes "
            f"{paths.human_bytes(size)} permanently (re-download to get it back)"))
    if not common.confirm_exact(manifest.id, f"deleting {paths.human_bytes(size)}"):
        return
    shutil.rmtree(repo_root)
    print(common.green("deleted local model"))
    resolved.write(registry)
