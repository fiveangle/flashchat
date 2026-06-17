"""[M]anage: manual outlier-fixing for model storage.

The automation (onboarding, ensure, post-build offers) handles the normal
lifecycle; this menu exists for everything flashchat does not do on its
own: per-artifact verification against hash baselines, building/repairing
artifacts (creating what is missing and rebuilding what fails verification),
deleting variants, and manual archive moves.
"""

import os
import shutil

from .. import configfile, offload, paths, resolved
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
                f"{s.relpath}: {_st(s)}" for s in shared)
            print(line)
        for vname in manifest.variants:
            states = variant_status(manifest, vname, status.snapshot)
            line = f"  {vname + '/':9} " + "  ".join(
                f"{s.relpath}: {_st(s)}" for s in states)
            print(line)
    return status


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
 [a] archive originals            DELETE local source blobs (copy kept in archive)
 [c] copy model to offload storage (non-destructive backup)
 [l] restore from archive...      [d] delete a variant (frees its q4/q8 disk)
 [x] delete local model           remove ALL local files for this model
 [q] back""")
        choice = common.prompt("Action", "q").lower()
        if choice in ("q", ""):
            return
        handler = {
            "v": lambda: None,  # loop re-renders
            "h": lambda: _deep_verify(registry, manifest, status),
            "b": lambda: _build_repair(registry, manifest, status),
            "a": lambda: _archive_originals(registry, manifest, status),
            "c": lambda: _full_offload(registry, manifest, status),
            "l": lambda: _restore_menu(registry, manifest),
            "d": lambda: _delete_variant(registry, manifest, status),
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
    for v in variants:
        build.ensure_variant_built(registry, manifest, v, assume_yes=True)


def _build_repair(registry, manifest, status) -> None:
    """Bring the model to ready: create anything MISSING and rebuild anything that
    fails verification. Replaces the old build-missing / regenerate split — there is
    no 'variant vs artifact' choice; you pick what to fix, it fixes all of it."""
    if not status.snapshot:
        print("model is not local")
        return
    not_ready = [v for v in manifest.variants if not status.variants[v].ready]
    if not not_ready:
        print(common.green("all variants are built and verified — nothing to repair"))
        return
    if len(not_ready) > 1:
        for i, v in enumerate(not_ready, 1):
            print(f"  {i}) {v} — {status.summary_line(v)}")
        print(f"  {len(not_ready) + 1}) all of them")
        choice = common.select_number(len(not_ready) + 1, "Build/repair which")
        if choice is None:
            return
        chosen = (not_ready if int(choice) == len(not_ready) + 1
                  else [not_ready[int(choice) - 1]])
    else:
        chosen = not_ready

    # Partition unsatisfied, buildable artifacts into create (absent) vs rebuild
    # (present-but-broken). Only specs with a build `step` can be (re)generated.
    create = []                 # display labels for missing artifacts
    rebuild_targets = []        # (scope, rel, spec) to delete-then-rebuild
    rebuild_labels = []         # display labels with the failure reason

    def _consider(scope, s, spec):
        if spec is None or not spec.step or s.satisfied:
            return
        if s.state == "missing":
            create.append(f"{scope}/{s.relpath}")
        else:
            rebuild_targets.append((scope, s.relpath, spec))
            detail = s.state + (f": {s.detail}" if s.detail else "")
            rebuild_labels.append(f"{scope}/{s.relpath} ({detail})")

    for v in chosen:
        var = manifest.variant(v)
        for s in variant_status(manifest, v, status.snapshot):
            _consider(v, s, var.artifacts.get(s.relpath))
    for s in shared_status(manifest, status.snapshot, want_optional=False):
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


def _archive_originals(registry, manifest, status) -> None:
    if not status.snapshot or not status.originals_local:
        print("no local originals to archive")
        return
    od = offload_dir()
    if not od:
        print("no OFFLOAD_DIR configured — set it in './flashchat config'")
        return
    print("This archives only the original Hugging Face safetensors source blobs "
          f"to {od}.")
    print("After each blob is copied and journaled, the local blob file is "
          "deleted. Built Flashchat runtime artifacts stay local, so ready "
          "variants remain usable; rebuilding or regenerating artifacts may "
          "require restoring the originals.")
    if not status.any_ready:
        print(common.yellow(
            "no runtime variant is built yet — archiving the originals now "
            "means they must be restored before extraction"))
        if not common.confirm("Archive anyway?", default=False):
            return
    elif not common.confirm("Archive originals and remove local source blobs?", default=False):
        return
    progress = common.ProgressLine()
    try:
        moved = offload.offload_originals(manifest, status.snapshot, od, progress=progress)
        print(common.green(f"archived {paths.human_bytes(moved)}"))
    except offload.OffloadError as e:
        print(common.red(str(e)))
    finally:
        progress.finish()


def _full_offload(registry, manifest, status) -> None:
    if not status.snapshot:
        print("model is not local")
        return
    err = guard_model_not_serving([resolved_id(manifest, v) for v in manifest.variants])
    if err:
        print(common.red(err))
        return
    od = offload_dir()
    if not od:
        print("no OFFLOAD_DIR configured")
        return
    size = paths.dir_size_bytes(os.path.dirname(os.path.dirname(status.snapshot)))
    print(f"This archives the local Hugging Face model directory "
          f"({paths.human_bytes(size)}) to {od}.")
    print("Local files are not deleted, so the model remains usable after the "
          "archive is refreshed.")
    print("system_prompt_cache directories are skipped and never written to the "
          "archive because they may contain prompt-derived user data.")
    if not common.confirm("Archive full model copy now?", default=False):
        return
    progress = common.ProgressLine()
    try:
        moved = offload.offload_full(manifest, status.snapshot, od, progress=progress)
        print(common.green(f"archived {paths.human_bytes(moved)}"))
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


def _delete_variant(registry, manifest, status) -> None:
    if not status.snapshot:
        print("model is not local")
        return
    present = [v for v in manifest.variants
               if os.path.isdir(paths.variant_dir(status.snapshot, v))]
    if not present:
        print("no variant artifacts exist")
        return
    for i, v in enumerate(present, 1):
        size = paths.dir_size_bytes(paths.variant_dir(status.snapshot, v))
        print(f"  {i}) {v} ({paths.human_bytes(size)})")
    choice = common.select_number(len(present), "Delete which variant")
    if choice is None:
        return
    v = present[int(choice) - 1]
    err = guard_model_not_serving([resolved_id(manifest, v)])
    if err:
        print(common.red(err))
        return
    if not common.confirm_exact(manifest.id, f"deleting the {v} artifacts"):
        return
    shutil.rmtree(paths.variant_dir(status.snapshot, v))
    print(common.green(f"deleted {v} artifacts"))
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
