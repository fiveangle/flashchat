"""Offload subsystem: preflight probing, journaled transfer/resume, restores."""

import os
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
        self.assertFalse(os.path.exists(repo_root))
        self.assertEqual(offload.archive_state(self.moe, self.dest), "full")

        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        journal = offload.Journal(dest_repo)
        # vocab.bin variant links journaled as links, not stored as bytes
        link_rels = [r for r in journal.data["links"] if r.endswith("q4/vocab.bin")]
        self.assertTrue(link_rels)
        self.assertFalse(any(r.endswith("q4/vocab.bin") for r in journal.data["files"]))

        restored = offload.restore_full(self.moe, self.cache, self.dest)
        self.assertEqual(restored, copied)
        new_snapshot = paths.snapshot_dir(self.cache, self.moe.hf_repo)
        self.assertEqual(new_snapshot, snapshot)
        link = os.path.join(paths.variant_dir(snapshot, "q4"), "vocab.bin")
        self.assertTrue(os.path.islink(link))
        self.assertTrue(os.path.exists(link))
        from modelmgr.artifacts import variant_ready
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))

    def test_restore_runtime_only_skips_blobs(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        offload.offload_full(self.moe, snapshot, self.dest)
        offload.restore_runtime_only(self.moe, self.cache, self.dest)
        self.assertTrue(os.path.isfile(
            os.path.join(paths.variant_dir(snapshot, "q4"), "model_weights.bin")))
        self.assertFalse(os.path.isdir(os.path.join(repo_root, "blobs")))


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
        with self.assertRaisesRegex(offload.OffloadError, "hash verification"):
            offload.restore_full(self.moe, self.cache, self.dest)


if __name__ == "__main__":
    unittest.main()
