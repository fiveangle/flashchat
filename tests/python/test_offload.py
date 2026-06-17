"""Offload subsystem: preflight probing, journaled transfer/resume, restores."""

import os
import shutil
import stat
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import offload, paths, recipes
from modelmgr.registry import Registry

from treebuilder import make_snapshot


class OffloadBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, "hub")
        self.dest = os.path.join(self.tmp.name, "offload")
        os.makedirs(self.cache)
        os.makedirs(self.dest)
        registry = Registry.load()
        self.moe = registry.get("qwen3.6-35b-a3b")

    def tearDown(self):
        self.tmp.cleanup()


class TestPreflight(OffloadBase):
    def test_ok_destination(self):
        report = offload.preflight(self.dest, needed_bytes=1024)
        self.assertTrue(report.ok)
        self.assertTrue(report.writable)
        self.assertTrue(report.symlinks)
        self.assertGreater(report.free_bytes, 0)

    def test_unconfigured(self):
        report = offload.preflight("")
        self.assertFalse(report.ok)
        self.assertIn("no offload directory configured", report.errors[0])

    def test_missing_volume(self):
        report = offload.preflight(os.path.join(self.tmp.name, "no", "such", "mount"))
        self.assertFalse(report.ok)
        self.assertIn("unmounted", report.errors[0])

    def test_creates_leaf_dir_under_existing_parent(self):
        report = offload.preflight(os.path.join(self.dest, "models"))
        self.assertTrue(report.ok)

    def test_read_only_destination_reports_permission_problem(self):
        ro = os.path.join(self.tmp.name, "readonly")
        os.makedirs(ro)
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        try:
            report = offload.preflight(ro)
            self.assertFalse(report.ok)
            self.assertIn("not writable", report.errors[0])
        finally:
            os.chmod(ro, stat.S_IRWXU)

    def test_insufficient_space(self):
        report = offload.preflight(self.dest, needed_bytes=1 << 60)
        self.assertFalse(report.ok)
        self.assertIn("not enough free space", report.errors[0])


class TestOriginals(OffloadBase):
    def test_offload_and_restore_originals(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        blob_link = os.path.join(snapshot, "model-00001-of-00001.safetensors")
        self.assertTrue(os.path.exists(blob_link))
        self.assertFalse(recipes.plan(self.moe, "q4", snapshot, force=True).needs_download)

        moved = offload.offload_originals(self.moe, snapshot, self.dest)
        self.assertGreater(moved, 0)
        # local blob gone, snapshot link dangling, archive holds the bytes
        self.assertTrue(os.path.islink(blob_link))
        self.assertFalse(os.path.exists(blob_link))
        self.assertEqual(offload.archive_state(self.moe, self.dest), "originals")
        self.assertTrue(recipes.plan(self.moe, "q4", snapshot, force=True).needs_download)

        restored = offload.restore_originals(self.moe, snapshot, self.dest)
        self.assertEqual(restored, moved)
        self.assertTrue(os.path.exists(blob_link))
        self.assertFalse(recipes.plan(self.moe, "q4", snapshot, force=True).needs_download)

    def test_offload_originals_noop_when_no_blobs(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"], with_blobs=False)
        self.assertEqual(offload.offload_originals(self.moe, snapshot, self.dest), 0)

    def test_runtime_artifacts_never_touched(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_originals(self.moe, snapshot, self.dest)
        from modelmgr.artifacts import variant_ready
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))


class TestFullOffload(OffloadBase):
    def test_full_roundtrip_preserves_links_without_duplicating_bytes(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4", "q8"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)

        copied = offload.offload_full(self.moe, snapshot, self.dest)
        self.assertTrue(os.path.exists(repo_root), "full archive must keep local files")
        self.assertEqual(offload.archive_state(self.moe, self.dest), "full")
        from modelmgr.artifacts import variant_ready
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))

        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        journal = offload.Journal(dest_repo)
        # vocab.bin variant links journaled as links, not stored as bytes
        link_rels = [r for r in journal.data["links"] if r.endswith("q4/vocab.bin")]
        self.assertTrue(link_rels)
        self.assertFalse(any(r.endswith("q4/vocab.bin") for r in journal.data["files"]))

        shutil.rmtree(repo_root)
        restored = offload.restore_full(self.moe, self.cache, self.dest)
        self.assertEqual(restored, copied)
        new_snapshot = paths.snapshot_dir(self.cache, self.moe.hf_repo)
        self.assertEqual(new_snapshot, snapshot)
        link = os.path.join(paths.variant_dir(snapshot, "q4"), "vocab.bin")
        self.assertTrue(os.path.islink(link))
        self.assertTrue(os.path.exists(link))
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))

    def test_full_archive_skips_system_prompt_cache(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        cache_file = os.path.join(paths.variant_dir(snapshot, "q4"),
                                  "system_prompt_cache", "secret.fcache")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            f.write("prompt-derived cache")
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        old_rel = os.path.join(os.path.relpath(paths.variant_dir(snapshot, "q4"), repo_root),
                               "system_prompt_cache", "old.fcache")
        old_dst = os.path.join(dest_repo, old_rel)
        os.makedirs(os.path.dirname(old_dst), exist_ok=True)
        with open(old_dst, "w") as f:
            f.write("old prompt-derived cache")
        old_journal = offload.Journal(dest_repo)
        old_journal.mark_file(old_rel, os.path.getsize(old_dst), "old")
        old_journal.save()

        offload.offload_full(self.moe, snapshot, self.dest)

        self.assertTrue(os.path.isfile(cache_file), "local cache must remain")
        journal = offload.Journal(dest_repo)
        self.assertFalse(any("system_prompt_cache" in rel
                             for rel in journal.data["files"]))
        self.assertFalse(any("system_prompt_cache" in rel
                             for rel in journal.data["links"]))
        self.assertFalse(os.path.exists(os.path.join(
            dest_repo, os.path.relpath(cache_file, repo_root))))
        self.assertFalse(os.path.exists(old_dst))

    def test_restore_runtime_only_skips_blobs(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        offload.offload_full(self.moe, snapshot, self.dest)
        shutil.rmtree(repo_root)
        offload.restore_runtime_only(self.moe, self.cache, self.dest)
        self.assertTrue(os.path.isfile(
            os.path.join(paths.variant_dir(snapshot, "q4"), "model_weights.bin")))
        self.assertFalse(os.path.isdir(os.path.join(repo_root, "blobs")))


class TestLegacyArchiveAdoption(OffloadBase):
    """Pre-journal archives (from the old mv-based offload) must be adopted
    by hash verification, never blindly re-copied — and never blindly
    trusted on size alone, since the archive may be restored later."""

    def _make_legacy_archive(self, repo_root):
        """Simulate an old mv-style archive: identical tree, no journal."""
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        shutil.copytree(repo_root, dest_repo, symlinks=True)
        return dest_repo

    def test_identical_legacy_archive_adopted_without_recopy(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = self._make_legacy_archive(repo_root)

        weights_dst = os.path.join(
            dest_repo, os.path.relpath(snapshot, repo_root),
            "flashchat", "q4", "model_weights.bin")
        before = os.stat(weights_dst).st_mtime_ns

        copied = offload.offload_full(self.moe, snapshot, self.dest)
        self.assertEqual(copied, 0, "identical archive must be adopted, not re-copied")
        self.assertEqual(os.stat(weights_dst).st_mtime_ns, before,
                         "adopted file must not be rewritten")
        journal = offload.Journal(dest_repo)
        self.assertTrue(journal.data["files"], "adoption must populate the journal")
        for entry in journal.data["files"].values():
            self.assertTrue(entry.get("sha256"), "adopted entries carry verified hashes")
        self.assertTrue(os.path.exists(repo_root), "full archive keeps local files")

    def test_stale_same_size_archive_copy_is_recopied(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = self._make_legacy_archive(repo_root)
        victim = os.path.join(
            dest_repo, os.path.relpath(snapshot, repo_root),
            "flashchat", "q4", "model_weights.bin")
        with open(victim, "r+b") as f:
            f.write(b"X")  # same size, different content

        copied = offload.offload_full(self.moe, snapshot, self.dest)
        self.assertGreater(copied, 0, "stale copy must be replaced")
        journal = offload.Journal(dest_repo)
        rel = os.path.relpath(victim, dest_repo)
        from modelmgr.artifacts import sha256_file
        self.assertEqual(journal.data["files"][rel]["sha256"], sha256_file(victim)[0])

    def test_offload_originals_adopts_existing_blobs(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        shutil.copytree(os.path.join(repo_root, "blobs"),
                        os.path.join(dest_repo, "blobs"))
        blob_dst = next(os.path.join(dest_repo, "blobs", n)
                        for n in os.listdir(os.path.join(dest_repo, "blobs")))
        before = os.stat(blob_dst).st_mtime_ns

        moved = offload.offload_originals(self.moe, snapshot, self.dest)
        self.assertGreater(moved, 0)
        self.assertEqual(os.stat(blob_dst).st_mtime_ns, before,
                         "existing identical blob must be adopted, not rewritten")
        self.assertEqual(offload.archive_state(self.moe, self.dest), "originals")


class TestTransferEngine(OffloadBase):
    def test_resume_skips_completed_files(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = os.path.join(self.dest, "repo")
        journal = offload.Journal(dest_repo)
        first = offload.transfer_tree(repo_root, dest_repo, journal, progress=None)
        self.assertGreater(first, 0)
        second = offload.transfer_tree(repo_root, dest_repo, offload.Journal(dest_repo))
        self.assertEqual(second, 0)

    def test_interrupted_file_recopied(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = os.path.join(self.dest, "repo")
        journal = offload.Journal(dest_repo)
        offload.transfer_tree(repo_root, dest_repo, journal)
        # simulate a crash that left a journaled file missing on dest
        victim = next(r for r in journal.data["files"]
                      if r.endswith("model_weights.bin"))
        os.unlink(os.path.join(dest_repo, victim))
        recopied = offload.transfer_tree(repo_root, dest_repo, offload.Journal(dest_repo))
        self.assertGreater(recopied, 0)
        self.assertTrue(os.path.isfile(os.path.join(dest_repo, victim)))

    def test_no_partial_files_left(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = os.path.join(self.dest, "repo")
        offload.transfer_tree(repo_root, dest_repo, offload.Journal(dest_repo))
        partials = [f for dp, _, fs in os.walk(dest_repo) for f in fs
                    if f.endswith(".partial")]
        self.assertEqual(partials, [])

    def test_corrupt_archive_detected_on_restore(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        offload.offload_full(self.moe, snapshot, self.dest)
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        journal = offload.Journal(dest_repo)
        victim = next(r for r in journal.data["files"] if r.endswith("model_weights.bin"))
        with open(os.path.join(dest_repo, victim), "r+b") as f:
            f.write(b"X")  # same size, corrupted content
        shutil.rmtree(repo_root)
        with self.assertRaisesRegex(offload.OffloadError, "hash verification"):
            offload.restore_full(self.moe, self.cache, self.dest)


if __name__ == "__main__":
    unittest.main()
