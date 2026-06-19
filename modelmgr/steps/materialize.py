"""materialize_shared step: expose shared artifacts inside a variant dir.

The engine expects everything under <snapshot>/flashchat/q<bits>/, so
variant dirs carry relative symlinks to the shared canonicals
(q4/vocab.bin -> ../shared/vocab.bin). Symlinks work on APFS and HFS+
(both case variants); filesystems without symlink support (exFAT, some
SMB servers) fall back to real copies with a warning — the only case
where dedup costs disk space, and only when running directly from such a
volume.
"""
from __future__ import annotations

import os
import shutil

from . import StepContext
from ..artifacts import ArtifactDir


def _link_or_copy(target: str, link: str) -> str:
    """Returns 'symlink' or 'copy' (fallback on fs without symlink support)."""
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.lexists(link):
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.isdir(link):
            shutil.rmtree(link)
        else:
            os.unlink(link)
    rel_target = os.path.relpath(target, os.path.dirname(link))
    try:
        os.symlink(rel_target, link)
        return "symlink"
    except OSError:
        if os.path.isdir(target):
            shutil.copytree(target, link)
        else:
            shutil.copyfile(target, link)
        return "copy"


def materialize(manifest, variant_name: str, shared_dir: str, variant_dir: str,
                only: list | None = None, progress=None) -> list:
    """Create the from_shared links a variant declares; returns [(rel, how)]."""
    variant = manifest.variant(variant_name)
    vdir = ArtifactDir(variant_dir, manifest.id, variant_name)
    sdir = ArtifactDir(shared_dir, manifest.id, "shared")
    results = []
    for spec in variant.artifacts.values():
        if not spec.from_shared:
            continue
        if only is not None and spec.relpath not in only:
            continue
        target = os.path.join(shared_dir, spec.from_shared.rstrip("/"))
        if not os.path.exists(target):
            if spec.optional:
                continue
            raise FileNotFoundError(f"shared artifact missing: {target}")
        link = os.path.join(variant_dir, spec.relpath.rstrip("/"))
        how = _link_or_copy(target, link)
        if how == "copy" and progress:
            progress("materialize_shared", 0, 0,
                     f"warning: {spec.relpath}: filesystem does not support "
                     f"symlinks, made a real copy")
        # Mirror the shared hash entries into the variant manifest so
        # verification works locally without cross-referencing dirs.
        for key, entry in sdir.entries.items():
            if key == spec.from_shared or key.startswith(spec.from_shared.rstrip("/") + "/"):
                vdir.stage(key if not spec.is_dir else key,
                           size=entry["size"], sha256=entry["sha256"], from_shared=True)
        results.append((spec.relpath, how))
    vdir.commit()
    return results


def run(ctx: StepContext, planned=None) -> None:
    if ctx.dry_run:
        ctx.report("materialize_shared", 0, 1, "dry run")
        return
    only = planned.artifacts if planned is not None else None
    results = materialize(ctx.manifest, ctx.variant_name, ctx.shared_dir,
                          ctx.variant_dir, only=only, progress=ctx.progress)
    ctx.report("materialize_shared", 1, 1,
               ", ".join(f"{rel} ({how})" for rel, how in results) or "nothing to do")


def get_runner(name: str):
    return run
