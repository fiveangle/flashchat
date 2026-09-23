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
- **Never delete archive bytes implicitly**: pushes to the archive never
  mirror-delete at repo level (an interrupted restore or a deleted local
  component must not erase the only good copy), and local artifact scopes
  that are absent or fail verification are excluded from the push.
- **Restores are inventory-driven and preflighted**: the file set comes from
  the journal, or from walking a legacy unjournaled archive; local free space
  is checked before the first byte is written.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from . import paths
from .artifacts import sha256_file
from .manifest import Manifest

JOURNAL_NAME = ".flashchat_offload.json"
PENDING_NAME = "offload_pending.json"
JOURNAL_SCHEMA = 1
_CHUNK = 8 * 1024 * 1024

# Sidecar files HF puts beside snapshots that are worthless on the archive.
_SKIP_NAMES = {".DS_Store", ".lock", JOURNAL_NAME}
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
        self.data.setdefault("files", {})
        self.data.setdefault("links", {})
        self.data.setdefault("dirty_scopes", [])

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def file_done(self, rel: str, size: int) -> bool:
        entry = self.data["files"].get(rel)
        return bool(entry and entry.get("done") and entry.get("size") == size)

    def mark_file(self, rel: str, size: int, sha256: str | None,
                  mtime_ns: int | None = None,
                  present_in_offload: bool = True) -> None:
        entry = {"size": size, "sha256": sha256, "done": True,
                 "present_in_offload": present_in_offload}
        if mtime_ns is not None:
            entry["mtime_ns"] = mtime_ns
        self.data["files"][rel] = entry

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


def _record_archive_tree(dest_root: str, journal: Journal, skip_dirs=None) -> int:
    total = 0
    journal.data["files"] = {}
    journal.data["links"] = {}
    journal.data["dirty_scopes"] = []
    for rel, kind in _walk_tree(dest_root, skip_dirs=skip_dirs):
        full = os.path.join(dest_root, rel)
        if kind == "link":
            journal.mark_link(rel, os.readlink(full))
            continue
        st = os.stat(full)
        journal.mark_file(rel, st.st_size, None, st.st_mtime_ns,
                          present_in_offload=True)
        total += st.st_size
    journal.save()
    return total


def _pending_path() -> str:
    return os.path.join(paths.config_dir(), PENDING_NAME)


def _load_pending() -> dict:
    path = _pending_path()
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_pending(data: dict) -> None:
    path = _pending_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def pending_scopes(manifest: Manifest) -> list[str]:
    data = _load_pending()
    return sorted(set(data.get(manifest.id, [])))


def _set_pending_scopes(manifest: Manifest, scopes: list[str]) -> None:
    data = _load_pending()
    scopes = sorted(set(scopes))
    if scopes:
        data[manifest.id] = scopes
    elif manifest.id in data:
        del data[manifest.id]
    _save_pending(data)


def scope_fingerprint(scope_root: str) -> dict:
    """Stat-only identity of an artifact scope, using rsync's own quick check
    (size plus whole-second mtime, link targets): equal fingerprints mean a
    sync would transfer nothing. Never reads file contents."""
    out = {}
    if not os.path.isdir(scope_root):
        return out
    for rel, kind in _walk_tree(scope_root, skip_dirs=_FULL_ARCHIVE_SKIP_DIRS):
        full = os.path.join(scope_root, rel)
        if kind == "link":
            out[rel] = ("link", os.readlink(full))
        else:
            st = os.stat(full)
            out[rel] = ("file", st.st_size, int(st.st_mtime))
    return out


def scope_fingerprints(snapshot: str, scopes) -> dict:
    return {s: scope_fingerprint(_scope_dir(snapshot, s)) for s in scopes}


def reconcile_pending_scopes(manifest: Manifest, snapshot: str | None,
                             offload_dir: str) -> list[str]:
    """Pending scopes whose offload copy still differs from the local one.
    Scopes that already match are cleared, so a flag left by a rebuild that
    changed nothing stops asking. Costs a stat walk of the pending scopes
    only while something is pending."""
    pending = pending_scopes(manifest)
    if not pending or not snapshot or not offload_dir \
            or archive_state(manifest, offload_dir) != "full":
        return pending
    dest = dest_repo_dir(offload_dir, manifest)
    dest_snapshot = os.path.join(dest, "snapshots", os.path.basename(snapshot))
    local = scope_fingerprints(snapshot, pending)
    archived = scope_fingerprints(dest_snapshot, pending)
    stale = [s for s in pending if local[s] != archived[s]]
    if stale != pending:
        _set_pending_scopes(manifest, stale)
        journal = Journal(dest)
        journal.data["dirty_scopes"] = stale
        journal.save()
    return stale


def mark_artifact_scopes_dirty(manifest: Manifest, offload_dir: str,
                               scopes: list[str]) -> None:
    scopes = sorted(set(scopes))
    if not scopes:
        return
    _set_pending_scopes(manifest, pending_scopes(manifest) + scopes)
    if not offload_dir or archive_state(manifest, offload_dir) != "full":
        return
    dest = dest_repo_dir(offload_dir, manifest)
    if not os.path.isdir(dest):
        return
    journal = Journal(dest)
    journal.data["dirty_scopes"] = sorted(
        set(journal.data.get("dirty_scopes", []) + scopes))
    journal.save()


def sync_artifact_scopes(manifest: Manifest, snapshot: str, offload_dir: str,
                         scopes: list[str], progress=None) -> int:
    if archive_state(manifest, offload_dir) != "full":
        raise OffloadError("no full offload archive found for this model")
    dest = dest_repo_dir(offload_dir, manifest)
    repo_root = os.path.dirname(os.path.dirname(snapshot))
    problems = local_scope_problems(manifest, snapshot)
    bad = {s: problems[s] for s in scopes
           if s in problems and os.path.isdir(_scope_dir(snapshot, s))}
    if bad:
        raise OffloadError(
            "local artifacts fail verification, refusing to overwrite the "
            "offload copy: " + "; ".join(f"{s}: {r}" for s, r in sorted(bad.items())))
    total = 0
    for scope in sorted(set(scopes)):
        rel = os.path.join("snapshots", os.path.basename(snapshot),
                           "flashchat", scope)
        src = os.path.join(repo_root, rel)
        dst = os.path.join(dest, rel)
        if not os.path.isdir(src):
            continue
        # Verified complete above, so mirroring this one scope is safe.
        _rsync_tree(src, dst, skip_dirs=_FULL_ARCHIVE_SKIP_DIRS,
                    progress=progress, delete=True)
        total += paths.dir_size_bytes(src)
    journal = Journal(dest)
    _record_archive_tree(dest, journal, skip_dirs=_FULL_ARCHIVE_SKIP_DIRS)
    remaining = [s for s in pending_scopes(manifest) if s not in set(scopes)]
    _set_pending_scopes(manifest, remaining)
    journal.data["dirty_scopes"] = remaining
    journal.save()
    return total


def _rsync_tree(src_root: str, dest_root: str, skip_dirs=None, progress=None,
                delete: bool = False, exclude_paths=None) -> None:
    """Push src_root into dest_root. `delete` mirror-deletes receiver files
    absent from the source; only pass it for a scope that verified complete
    locally. `exclude_paths` are src-relative dirs anchored at the root;
    rsync also protects excluded receiver paths from --delete."""
    os.makedirs(dest_root, exist_ok=True)
    cmd = ["rsync", "-a"]
    if delete:
        cmd.append("--delete")
    for dirname in sorted(set(skip_dirs or ())):
        cmd.extend(["--exclude", dirname + "/"])
    for rel in sorted(set(exclude_paths or ())):
        cmd.extend(["--exclude", "/" + rel.strip("/") + "/"])
    cmd.extend([src_root.rstrip("/") + "/", dest_root.rstrip("/") + "/"])
    if progress:
        progress("rsync", 0, 0, os.path.basename(src_root.rstrip("/")) or src_root)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        raise OffloadError("rsync is required for offload operations") from e
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip()
        raise OffloadError(f"rsync failed: {detail or e}") from e


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


def restore_tree(dest_root: str, src_root: str, files: dict, links: dict,
                 progress=None, verify: bool = True) -> int:
    """Copy files back from the archive and recreate archived links. Never
    deletes local files. Smallest files go first so an interrupted restore
    still lands metadata (layout.json, vocab.bin, indexes) before bulk data."""
    restored = 0
    order = sorted(files.items(), key=lambda kv: (kv[1]["size"], kv[0]))
    for i, (rel, entry) in enumerate(order):
        src = os.path.join(dest_root, rel)
        dst = os.path.join(src_root, rel)
        if os.path.isfile(dst) and os.path.getsize(dst) == entry["size"]:
            continue
        if progress:
            progress("restore", i + 1, len(order), rel)
        digest, size = copy_file_verified(src, dst)
        if verify and entry.get("sha256") and digest != entry["sha256"]:
            os.unlink(dst)
            raise OffloadError(
                f"restored file failed hash verification: {rel} "
                f"(archive may be corrupt)")
        restored += size
    for rel, target in sorted(links.items()):
        link = os.path.join(src_root, rel)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            if os.path.islink(link) and os.readlink(link) == target:
                continue
            if not os.path.islink(link):
                continue  # never replace a real local file with a link
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


def _scope_dir(snapshot: str, scope: str) -> str:
    return paths.shared_dir(snapshot) if scope == "shared" \
        else paths.variant_dir(snapshot, scope)


_BROKEN_STATES = ("missing", "incomplete", "invalid", "size-mismatch", "hash-mismatch")


def local_scope_problems(manifest: Manifest, snapshot: str) -> dict:
    """{scope: reason} for local artifact scopes that must not be pushed to
    the archive: absent entirely, or present but failing quick verification
    (e.g. a half-finished restore). Pushing those would overwrite or, with
    mirroring, erase a good archived copy."""
    from .artifacts import shared_status, variant_status

    out = {}
    scopes = [("shared", lambda: shared_status(manifest, snapshot))]
    scopes += [(v, (lambda v=v: variant_status(manifest, v, snapshot)))
               for v in manifest.variants]
    for scope, status in scopes:
        if not os.path.isdir(_scope_dir(snapshot, scope)):
            out[scope] = "not present locally"
            continue
        bad = [s for s in status() if s.state in _BROKEN_STATES
               and not (s.state == "missing" and not s.required)]
        if bad:
            out[scope] = ", ".join(
                f"{s.relpath} {s.state}" + (f" ({s.detail})" if s.detail else "")
                for s in bad)
    return out


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


def offload_model(manifest: Manifest, snapshot: str, offload_dir: str,
                  progress=None) -> int:
    """Sync the whole model repo to offload storage, then remove local source blobs."""
    if not os.path.isdir(snapshot):
        return 0
    repo_root = os.path.dirname(os.path.dirname(snapshot))
    blob_files = list_blob_files(snapshot)
    needed = 0
    dest = dest_repo_dir(offload_dir, manifest)
    # Absent or broken local scopes stay out of the push so they can neither
    # overwrite nor (historically, via --delete) erase the archived copy.
    excluded = [os.path.relpath(_scope_dir(snapshot, scope), repo_root)
                for scope in local_scope_problems(manifest, snapshot)]
    for rel, kind in _walk_tree(repo_root, skip_dirs=_FULL_ARCHIVE_SKIP_DIRS):
        if any(rel.startswith(prefix + os.sep) for prefix in excluded):
            continue
        if kind != "file":
            continue
        src = os.path.join(repo_root, rel)
        dst = os.path.join(dest, rel)
        size = os.path.getsize(src)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == size):
            needed += size
    report = preflight(offload_dir, needed_bytes=needed)
    if not report.ok:
        raise OffloadError("; ".join(report.errors))
    journal = Journal(dest)
    _rsync_tree(repo_root, dest, skip_dirs=_FULL_ARCHIVE_SKIP_DIRS,
                progress=progress, exclude_paths=excluded)
    _prune_archive_dirs(dest, journal, _FULL_ARCHIVE_SKIP_DIRS)
    archived_bytes = _record_archive_tree(dest, journal,
                                          skip_dirs=_FULL_ARCHIVE_SKIP_DIRS)
    _set_pending_scopes(manifest, [])
    removed = 0
    for _rel, target in blob_files:
        blob_rel = os.path.join("blobs", os.path.basename(target))
        blob_dst = os.path.join(dest, blob_rel)
        if os.path.isfile(blob_dst) and os.path.getsize(blob_dst) == os.path.getsize(target):
            removed += os.path.getsize(target)
            os.unlink(target)
        else:
            raise OffloadError(
                f"offload copy missing or incomplete for source blob: {blob_rel}")
    return archived_bytes if not removed else removed


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


# Keep the boot volume usable after a restore: leave free space proportional
# to the restore (capped), so tiny restores still fit on a nearly-full disk.
RESTORE_HEADROOM = 2 * 1024 ** 3


def _is_runtime_rel(rel: str) -> bool:
    return "flashchat" in rel.split(os.sep)


_RESTORE_SELECTORS = {
    "originals": lambda rel: not _is_runtime_rel(rel),
    "runtime": _is_runtime_rel,
    "full": lambda rel: True,
}


def archive_inventory(dest: str) -> tuple[dict, dict]:
    """({rel: {size, sha256}}, {rel: link_target}) for an archived repo.

    The journal is authoritative when it has entries. Legacy archives (the old
    mv/rsync offload, no journal) are walked directly instead of being treated
    as empty — the tree itself is the record."""
    journal = Journal(dest)
    skip = _FULL_ARCHIVE_SKIP_DIRS

    def keep(rel):
        return not any(_rel_contains_component(rel, d) for d in skip)

    if journal.data["files"] or journal.data["links"]:
        return ({r: e for r, e in journal.data["files"].items() if keep(r)},
                {r: t for r, t in journal.data["links"].items() if keep(r)})
    files, links = {}, {}
    if not os.path.isdir(dest):
        return files, links
    for rel, kind in _walk_tree(dest, skip_dirs=skip):
        full = os.path.join(dest, rel)
        if kind == "link":
            links[rel] = os.readlink(full)
        else:
            files[rel] = {"size": os.path.getsize(full), "sha256": None}
    return files, links


def _local_free_bytes(path: str) -> int:
    probe = os.path.abspath(path)
    while not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    st = os.statvfs(probe)
    return st.f_bavail * st.f_frsize


@dataclass
class RestorePlan:
    what: str
    dest: str
    repo_root: str
    files: dict
    links: dict
    needed_bytes: int   # bytes still to copy (already-present files skipped)
    free_bytes: int

    @property
    def headroom_bytes(self) -> int:
        return min(RESTORE_HEADROOM, self.needed_bytes)

    @property
    def fits(self) -> bool:
        return self.needed_bytes + self.headroom_bytes <= self.free_bytes

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.needed_bytes + self.headroom_bytes - self.free_bytes)


def plan_restore(manifest: Manifest, cache_dir: str, offload_dir: str,
                 what: str, variant_name: str | None = None) -> RestorePlan:
    """What a restore of `what` (originals|runtime|full) would copy locally."""
    dest = dest_repo_dir(offload_dir, manifest)
    repo_root = paths.repo_root_dir(os.path.expanduser(cache_dir), manifest.hf_repo)
    select = _RESTORE_SELECTORS[what]
    if variant_name is not None:
        if what != "runtime":
            raise ValueError("variant selection requires a runtime restore")
        manifest.variant(variant_name)
        snapshot = paths.snapshot_dir(os.path.expanduser(offload_dir), manifest.hf_repo)
        if not snapshot:
            raise OffloadError("no archived snapshot found for this model")
        runtime = os.path.relpath(paths.flashchat_dir(snapshot), dest)
        prefixes = tuple(os.path.join(runtime, scope) + os.sep
                         for scope in ("shared", variant_name))
        select = lambda rel: rel.startswith(prefixes)
    files, links = archive_inventory(dest)
    files = {r: e for r, e in files.items() if select(r)}
    links = {r: t for r, t in links.items() if select(r)}
    needed = 0
    for rel, entry in files.items():
        dst = os.path.join(repo_root, rel)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == entry["size"]):
            # Counted in full even when replacing a wrong-size file: the old
            # copy lives until the verified .partial is renamed over it.
            needed += entry["size"]
    return RestorePlan(what, dest, repo_root, files, links, needed,
                       _local_free_bytes(repo_root))


_RESTORE_LABELS = {"originals": "archived originals",
                   "runtime": "archived runtime artifacts",
                   "full": "archive"}


def _restore(manifest: Manifest, cache_dir: str, offload_dir: str, what: str,
             progress=None, variant_name: str | None = None) -> int:
    plan = plan_restore(manifest, cache_dir, offload_dir, what, variant_name)
    if not plan.files and not plan.links:
        raise OffloadError(f"no {_RESTORE_LABELS[what]} found under {plan.dest}")
    if not plan.fits:
        raise OffloadError(
            f"not enough local disk space to restore: need "
            f"{paths.human_bytes(plan.needed_bytes)} (+"
            f"{paths.human_bytes(plan.headroom_bytes)} headroom), only "
            f"{paths.human_bytes(plan.free_bytes)} free — free "
            f"{paths.human_bytes(plan.shortfall_bytes)} and retry "
            f"(already-restored files are skipped)")
    return restore_tree(plan.dest, plan.repo_root, plan.files, plan.links,
                        progress=progress)


def restore_originals(manifest: Manifest, cache_dir: str, offload_dir: str,
                      progress=None) -> int:
    return _restore(manifest, cache_dir, offload_dir, "originals", progress)


def restore_runtime_only(manifest: Manifest, cache_dir: str, offload_dir: str,
                         progress=None, variant_name: str | None = None) -> int:
    """Bring back only flashchat runtime artifacts (no original blobs)."""
    return _restore(manifest, cache_dir, offload_dir, "runtime", progress,
                    variant_name)


def restore_full(manifest: Manifest, cache_dir: str, offload_dir: str,
                 progress=None) -> int:
    return _restore(manifest, cache_dir, offload_dir, "full", progress)


def archive_state(manifest: Manifest, offload_dir: str) -> str:
    """'none' | 'originals' | 'full' — what the archive holds for a model."""
    if not offload_dir:
        return "none"
    dest = dest_repo_dir(offload_dir, manifest)
    if os.path.isdir(os.path.join(dest, "snapshots")):
        return "full"
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
