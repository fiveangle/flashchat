"""Offload storage: archive model data to external (NAS/USB) volumes safely.

Design rules learned from the failures of the old `mv`-based implementation:

- **Preflight by probing, not guessing**: before any transfer the
  destination is checked by actually creating/fsyncing/removing a probe
  file and a probe symlink. Permission problems on SMB/USB surface
  immediately with the exact failing syscall instead of mid-transfer.
- **Never `mv` across filesystems**: every file is streamed (8 MiB chunks,
  sha256 computed during the read), written to `<name>.partial`, fsynced,
  renamed, journaled. Operations that free local space delete source bytes
  only after the operation's file set is journaled done.
- **Resume for free**: a killed transfer re-runs and skips files the
  journal marks done (size-checked).
- **No duplicate bytes on the archive**: symlinks (HF blob links,
  from_shared variant links) are recorded in the journal as links, never
  expanded to copies. Restore recreates them; on filesystems without
  symlink support the *local* restore side still gets real links because
  restores always target the local APFS cache.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field

from . import paths
from .artifacts import sha256_file
from .manifest import Manifest

JOURNAL_NAME = ".flashchat_offload.json"
JOURNAL_SCHEMA = 1
_CHUNK = 8 * 1024 * 1024

# Sidecar files HF puts beside snapshots that are worthless on the archive.
_SKIP_NAMES = {".DS_Store", ".lock"}
_FULL_ARCHIVE_SKIP_DIRS = {"system_prompt_cache"}


class OffloadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


@dataclass
class PreflightReport:
    dest: str
    ok: bool
    writable: bool = False
    symlinks: bool = False
    free_bytes: int = 0
    needed_bytes: int = 0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def preflight(dest: str, needed_bytes: int = 0) -> PreflightReport:
    report = PreflightReport(dest=dest, ok=False, needed_bytes=needed_bytes)
    if not dest:
        report.errors.append("no offload directory configured")
        return report
    dest = os.path.expanduser(dest)
    if not os.path.isabs(dest):
        report.errors.append(f"offload directory must be an absolute path: {dest}")
        return report
    if not os.path.isdir(dest):
        parent = os.path.dirname(dest.rstrip("/"))
        if os.path.isdir(parent):
            try:
                os.makedirs(dest, exist_ok=True)
            except OSError as e:
                report.errors.append(f"cannot create {dest}: {e.strerror} ({e.__class__.__name__})")
                return report
        else:
            report.errors.append(
                f"offload directory does not exist (volume unmounted?): {dest}")
            return report

    # Writability: actually write+fsync+remove a probe file. This is the
    # check that catches the classic NAS/USB permission failures early.
    try:
        fd, probe = tempfile.mkstemp(prefix=".flashchat_probe_", dir=dest)
        try:
            os.write(fd, b"probe")
            os.fsync(fd)
        finally:
            os.close(fd)
            os.unlink(probe)
        report.writable = True
    except OSError as e:
        report.errors.append(
            f"destination is not writable: {e.strerror} on {dest} "
            f"(check share permissions / volume mount options)")
        return report

    # Symlink capability: probe, don't guess from fs names.
    probe_link = os.path.join(dest, ".flashchat_probe_link")
    try:
        if os.path.lexists(probe_link):
            os.unlink(probe_link)
        os.symlink("probe-target", probe_link)
        os.unlink(probe_link)
        report.symlinks = True
    except OSError:
        report.warnings.append(
            "filesystem does not support symlinks (exFAT/SMB?): archives still "
            "dedup via the journal, but running models directly from this "
            "volume will use real copies")

    try:
        st = os.statvfs(dest)
        report.free_bytes = st.f_bavail * st.f_frsize
    except OSError as e:
        report.errors.append(f"cannot stat volume: {e.strerror}")
        return report
    if needed_bytes and report.free_bytes < needed_bytes:
        report.errors.append(
            f"not enough free space: need {paths.human_bytes(needed_bytes)}, "
            f"only {paths.human_bytes(report.free_bytes)} free on {dest}")
        return report

    report.ok = True
    return report


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class Journal:
    """Per-destination-repo transfer record enabling resume + verified delete."""

    def __init__(self, dest_repo_dir: str):
        self.path = os.path.join(dest_repo_dir, JOURNAL_NAME)
        if os.path.isfile(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {"schema": JOURNAL_SCHEMA, "files": {}, "links": {}}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def file_done(self, rel: str, size: int) -> bool:
        entry = self.data["files"].get(rel)
        return bool(entry and entry.get("done") and entry.get("size") == size)

    def mark_file(self, rel: str, size: int, sha256: str | None) -> None:
        self.data["files"][rel] = {"size": size, "sha256": sha256, "done": True}

    def mark_link(self, rel: str, target: str) -> None:
        self.data["links"][rel] = target


# ---------------------------------------------------------------------------
# Transfer engine
# ---------------------------------------------------------------------------


def copy_file_verified(src: str, dst: str, progress=None) -> tuple[str, int]:
    """Streamed copy hashing during the read; .partial + fsync + rename."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    partial = dst + ".partial"
    import hashlib

    h = hashlib.sha256()
    size = 0
    with open(src, "rb") as fin, open(partial, "wb") as fout:
        while True:
            chunk = fin.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            fout.write(chunk)
            size += len(chunk)
            if progress:
                progress(len(chunk))
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(partial, dst)
    try:
        shutil.copystat(src, dst)
    except OSError:
        pass  # xattr/perm loss on FAT/SMB is acceptable
    return h.hexdigest(), size


def _walk_tree(root: str, skip_dirs=None):
    """Yield (relpath, kind) for files and symlinks under root; kind in
    {'file', 'link'}. Directories are implicit."""
    skip_dirs = set(skip_dirs or ())
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(n for n in dirnames if n not in skip_dirs)
        for name in sorted(filenames):
            if name in _SKIP_NAMES or name.endswith(".partial"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            yield rel, ("link" if os.path.islink(full) else "file")


def _rel_contains_component(rel: str, component: str) -> bool:
    return component in rel.split(os.sep)


def _prune_archive_dirs(dest_root: str, journal: Journal, skip_dirs) -> None:
    """Remove sensitive/private directories from an existing archive refresh."""
    skip_dirs = set(skip_dirs or ())
    if not skip_dirs:
        return

    changed = False
    for bucket in ("files", "links"):
        for rel in list(journal.data[bucket]):
            if any(_rel_contains_component(rel, dirname) for dirname in skip_dirs):
                del journal.data[bucket][rel]
                changed = True

    if os.path.isdir(dest_root):
        for dirpath, dirnames, _filenames in os.walk(dest_root):
            for dirname in list(dirnames):
                if dirname in skip_dirs:
                    shutil.rmtree(os.path.join(dirpath, dirname), ignore_errors=True)
                    dirnames.remove(dirname)

    if changed:
        journal.save()


def _adopt_existing(src: str, dst: str, rel: str, journal: Journal,
                    progress=None) -> bool:
    """Adopt a pre-journal archive copy (e.g. from the old mv-based offload):
    if dst already exists with matching size AND matching content hash,
    journal it instead of rewriting it. Content is verified for real because
    the archive may later be used as the source of truth — a size match alone
    is never grounds for skipping the copy."""
    if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
        return False
    if progress:
        progress("verify-existing", 0, 0, rel)
    src_digest, src_size = sha256_file(src)
    dst_digest, _dst_size = sha256_file(dst)
    if src_digest != dst_digest:
        return False  # stale/corrupt archive copy: fall through to recopy
    journal.mark_file(rel, src_size, src_digest)
    journal.save()
    return True


def transfer_tree(src_root: str, dest_root: str, journal: Journal,
                  progress=None, relpaths=None, skip_dirs=None) -> int:
    """Copy a tree into dest_root with journaled resume; returns bytes copied.

    Symlinks are journaled (relative targets preserved), never expanded —
    this is what keeps shared artifacts single-copy on any filesystem.
    Destination files from a pre-journal archive are adopted (hash-verified,
    not rewritten) rather than re-copied.
    """
    copied = 0
    entries = list(_walk_tree(src_root, skip_dirs=skip_dirs))
    if relpaths is not None:
        wanted = set(relpaths)
        entries = [e for e in entries if e[0] in wanted]
    total_files = len(entries)
    for i, (rel, kind) in enumerate(entries):
        src = os.path.join(src_root, rel)
        if kind == "link":
            journal.mark_link(rel, os.readlink(src))
            journal.save()
            continue
        size = os.path.getsize(src)
        dst = os.path.join(dest_root, rel)
        if journal.file_done(rel, size):
            if os.path.isfile(dst) and os.path.getsize(dst) == size:
                continue  # resumed: already transferred
        elif _adopt_existing(src, dst, rel, journal, progress=progress):
            continue  # legacy archive copy verified in place
        if progress:
            progress("offload", i + 1, total_files, rel)
        digest, size = copy_file_verified(src, dst)
        journal.mark_file(rel, size, digest)
        journal.save()
        copied += size
    return copied


def restore_tree(dest_root: str, src_root: str, journal: Journal,
                 progress=None, relpaths=None, verify: bool = True) -> int:
    """Copy files back from the archive and recreate journaled links."""
    restored = 0
    files = journal.data["files"]
    if relpaths is not None:
        wanted = set(relpaths)
        files = {k: v for k, v in files.items() if k in wanted}
    for i, (rel, entry) in enumerate(sorted(files.items())):
        src = os.path.join(dest_root, rel)
        dst = os.path.join(src_root, rel)
        if os.path.isfile(dst) and os.path.getsize(dst) == entry["size"]:
            continue
        if progress:
            progress("restore", i + 1, len(files), rel)
        digest, size = copy_file_verified(src, dst)
        if verify and entry.get("sha256") and digest != entry["sha256"]:
            os.unlink(dst)
            raise OffloadError(
                f"restored file failed hash verification: {rel} "
                f"(archive may be corrupt)")
        restored += size
    for rel, target in sorted(journal.data["links"].items()):
        if relpaths is not None and rel not in set(relpaths):
            continue
        link = os.path.join(src_root, rel)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(target, link)
    return restored


# ---------------------------------------------------------------------------
# Original-blob helpers (port of resolve_model_blob_target & friends)
# ---------------------------------------------------------------------------


def list_blob_files(snapshot: str) -> list:
    """[(snapshot_relpath, resolved_target_abspath)] for original weights."""
    out = []
    if not os.path.isdir(snapshot):
        return out
    for name in sorted(os.listdir(snapshot)):
        if not (name == "model.safetensors" or
                (name.startswith("model") and name.endswith(".safetensors"))):
            continue
        path = os.path.join(snapshot, name)
        target = os.path.realpath(path) if os.path.islink(path) else path
        if os.path.isfile(target):
            out.append((name, target))
    return out


def blobs_size(snapshot: str) -> int:
    return sum(os.path.getsize(t) for _rel, t in list_blob_files(snapshot))


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def dest_repo_dir(offload_dir: str, manifest: Manifest) -> str:
    return paths.repo_root_dir(os.path.expanduser(offload_dir.rstrip("/")), manifest.hf_repo)


def offload_originals(manifest: Manifest, snapshot: str, offload_dir: str,
                      progress=None) -> int:
    """Archive original safetensors blobs and remove them locally.

    The snapshot's blob symlinks stay in place (dangling) so the HF layout
    survives; recipes treat dangling links as 'needs download/restore'.
    """
    blob_files = list_blob_files(snapshot)
    if not blob_files:
        return 0
    needed = sum(os.path.getsize(t) for _r, t in blob_files)
    report = preflight(offload_dir, needed_bytes=needed)
    if not report.ok:
        raise OffloadError("; ".join(report.errors))

    dest = dest_repo_dir(offload_dir, manifest)
    journal = Journal(dest)
    moved = 0
    for i, (rel, target) in enumerate(blob_files):
        blob_rel = os.path.join("blobs", os.path.basename(target))
        blob_dst = os.path.join(dest, blob_rel)
        size = os.path.getsize(target)
        if not journal.file_done(blob_rel, size) and \
           not _adopt_existing(target, blob_dst, blob_rel, journal, progress=progress):
            if progress:
                progress("offload", i + 1, len(blob_files), rel)
            digest, size = copy_file_verified(target, blob_dst)
            journal.mark_file(blob_rel, size, digest)
        if os.path.islink(os.path.join(snapshot, rel)):
            journal.mark_link(os.path.join("snapshots", os.path.basename(snapshot), rel),
                              os.readlink(os.path.join(snapshot, rel)))
        journal.save()
        moved += size
    # Every blob journaled done -> delete local copies.
    for _rel, target in blob_files:
        os.unlink(target)
    return moved


def restore_originals(manifest: Manifest, snapshot: str, offload_dir: str,
                      progress=None) -> int:
    dest = dest_repo_dir(offload_dir, manifest)
    journal = Journal(dest)
    blob_entries = {rel: e for rel, e in journal.data["files"].items()
                    if rel.startswith("blobs/")}
    if not blob_entries:
        raise OffloadError(f"no archived originals found under {dest}")
    repo_root = os.path.dirname(os.path.dirname(snapshot))
    restored = restore_tree(dest, repo_root, journal, progress=progress,
                            relpaths=list(blob_entries))
    return restored


def offload_full(manifest: Manifest, snapshot: str, offload_dir: str,
                 progress=None) -> int:
    """Archive the whole repo tree (originals + runtime artifacts) while
    keeping local files in place. Snapshot/variant symlinks travel as journal
    entries. system_prompt_cache is intentionally never archived because it can
    contain prompt-derived user data."""
    repo_root = os.path.dirname(os.path.dirname(snapshot))
    dest = dest_repo_dir(offload_dir, manifest)
    # Free-space requirement excludes files the archive already holds at
    # matching size (resumed transfers and adoptable pre-journal copies).
    needed = 0
    for rel, kind in _walk_tree(repo_root, skip_dirs=_FULL_ARCHIVE_SKIP_DIRS):
        if kind != "file":
            continue
        size = os.path.getsize(os.path.join(repo_root, rel))
        dst = os.path.join(dest, rel)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == size):
            needed += size
    report = preflight(offload_dir, needed_bytes=needed)
    if not report.ok:
        raise OffloadError("; ".join(report.errors))
    journal = Journal(dest)
    _prune_archive_dirs(dest, journal, _FULL_ARCHIVE_SKIP_DIRS)
    return transfer_tree(repo_root, dest, journal, progress=progress,
                         skip_dirs=_FULL_ARCHIVE_SKIP_DIRS)


def restore_full(manifest: Manifest, cache_dir: str, offload_dir: str,
                 progress=None) -> int:
    dest = dest_repo_dir(offload_dir, manifest)
    journal = Journal(dest)
    if not journal.data["files"]:
        raise OffloadError(f"no archive found under {dest}")
    repo_root = paths.repo_root_dir(os.path.expanduser(cache_dir), manifest.hf_repo)
    return restore_tree(dest, repo_root, journal, progress=progress)


def restore_runtime_only(manifest: Manifest, cache_dir: str, offload_dir: str,
                         progress=None) -> int:
    """Bring back only flashchat runtime artifacts (no original blobs)."""
    dest = dest_repo_dir(offload_dir, manifest)
    journal = Journal(dest)
    runtime = [rel for rel in journal.data["files"]
               if "/flashchat/" in rel or rel.startswith("flashchat/")]
    if not runtime:
        raise OffloadError(f"no archived runtime artifacts found under {dest}")
    runtime_links = [rel for rel in journal.data["links"]
                     if "/flashchat/" in rel or rel.startswith("flashchat/")]
    repo_root = paths.repo_root_dir(os.path.expanduser(cache_dir), manifest.hf_repo)
    return restore_tree(dest, repo_root, journal, progress=progress,
                        relpaths=runtime + runtime_links)


def archive_state(manifest: Manifest, offload_dir: str) -> str:
    """'none' | 'originals' | 'full' — what the archive holds for a model."""
    if not offload_dir:
        return "none"
    dest = dest_repo_dir(offload_dir, manifest)
    if not os.path.isfile(os.path.join(dest, JOURNAL_NAME)):
        # Legacy full-tree offloads (old `mv`-based manage) have no journal.
        if os.path.isdir(os.path.join(dest, "snapshots")):
            return "full"
        return "none"
    journal = Journal(dest)
    files = journal.data["files"]
    if any("/flashchat/" in rel or not rel.startswith("blobs/") and "snapshots/" in rel
           for rel in files):
        return "full"
    if any(rel.startswith("blobs/") for rel in files):
        return "originals"
    return "none"
