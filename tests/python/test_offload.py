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
        self.old_config_dir = os.environ.get("FLASHCHAT_CONFIG_DIR")
        os.environ["FLASHCHAT_CONFIG_DIR"] = os.path.join(self.tmp.name, "config")
        os.makedirs(self.cache)
        os.makedirs(self.dest)
        registry = Registry.load()
        self.moe = registry.get("qwen3.6-35b-a3b")

    def tearDown(self):
        if self.old_config_dir is None:
            os.environ.pop("FLASHCHAT_CONFIG_DIR", None)
        else:
            os.environ["FLASHCHAT_CONFIG_DIR"] = self.old_config_dir
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
    def test_offload_model_syncs_tree_and_removes_local_source_blobs(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4", "q8"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        blob_link = os.path.join(snapshot, "model-00001-of-00001.safetensors")

        copied = offload.offload_model(self.moe, snapshot, self.dest)
        self.assertTrue(os.path.exists(repo_root), "full archive must keep local files")
        self.assertEqual(offload.archive_state(self.moe, self.dest), "full")
        self.assertTrue(os.path.islink(blob_link))
        self.assertFalse(os.path.exists(blob_link))
        from modelmgr.artifacts import variant_ready
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))

        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        journal = offload.Journal(dest_repo)
        # vocab.bin variant links journaled as links, not stored as bytes
        link_rels = [r for r in journal.data["links"] if r.endswith("q4/vocab.bin")]
        self.assertTrue(link_rels)
        self.assertFalse(any(r.endswith("q4/vocab.bin") for r in journal.data["files"]))
        weights_rel = next(r for r in journal.data["files"]
                           if r.endswith("q4/model_weights.bin"))
        self.assertIsNone(journal.data["files"][weights_rel]["sha256"])
        self.assertTrue(journal.data["files"][weights_rel]["present_in_offload"])

        shutil.rmtree(repo_root)
        restored = offload.restore_full(self.moe, self.cache, self.dest)
        self.assertGreater(restored, copied)
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

        offload.offload_model(self.moe, snapshot, self.dest)

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
        offload.offload_model(self.moe, snapshot, self.dest)
        shutil.rmtree(repo_root)
        offload.restore_runtime_only(self.moe, self.cache, self.dest)
        self.assertTrue(os.path.isfile(
            os.path.join(paths.variant_dir(snapshot, "q4"), "model_weights.bin")))
        self.assertFalse(os.path.isdir(os.path.join(repo_root, "blobs")))

    def test_restore_full_supports_legacy_unjournaled_archive(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        shutil.copytree(repo_root, dest_repo, symlinks=True)
        shutil.rmtree(repo_root)

        restored = offload.restore_full(self.moe, self.cache, self.dest)

        self.assertGreater(restored, 0)
        self.assertTrue(os.path.isfile(
            os.path.join(paths.variant_dir(snapshot, "q4"), "model_weights.bin")))
        self.assertTrue(os.path.exists(
            os.path.join(snapshot, "model-00001-of-00001.safetensors")))

    def test_dirty_artifact_scope_can_be_synced_later(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)
        local_weights = os.path.join(paths.variant_dir(snapshot, "q4"),
                                     "model_weights.bin")
        with open(local_weights, "ab") as f:
            f.write(b"changed")

        offload.mark_artifact_scopes_dirty(self.moe, self.dest, ["q4"])
        self.assertEqual(offload.pending_scopes(self.moe), ["q4"])
        dest_weights = os.path.join(
            offload.dest_repo_dir(self.dest, self.moe),
            "snapshots", os.path.basename(snapshot), "flashchat", "q4",
            "model_weights.bin")
        self.assertNotEqual(os.path.getsize(local_weights),
                            os.path.getsize(dest_weights))

        synced = offload.sync_artifact_scopes(
            self.moe, snapshot, self.dest, ["q4"])
        self.assertGreater(synced, 0)
        self.assertEqual(os.path.getsize(local_weights),
                         os.path.getsize(dest_weights))
        self.assertEqual(offload.pending_scopes(self.moe), [])

    def test_dirty_artifact_scope_is_remembered_when_offload_unavailable(self):
        offload.mark_artifact_scopes_dirty(
            self.moe, os.path.join(self.tmp.name, "missing-offload"), ["shared"])
        self.assertEqual(offload.pending_scopes(self.moe), ["shared"])


class TestLightweightOffload(OffloadBase):
    """The default offload path trusts successful rsync plus lightweight metadata."""

    def _make_legacy_archive(self, repo_root):
        """Simulate an old mv-style archive: identical tree, no journal."""
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        shutil.copytree(repo_root, dest_repo, symlinks=True)
        return dest_repo

    def test_existing_archive_refreshed_without_hashing(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = self._make_legacy_archive(repo_root)

        weights_dst = os.path.join(
            dest_repo, os.path.relpath(snapshot, repo_root),
            "flashchat", "q4", "model_weights.bin")
        copied = offload.offload_model(self.moe, snapshot, self.dest)
        self.assertGreater(copied, 0)
        journal = offload.Journal(dest_repo)
        self.assertTrue(journal.data["files"], "adoption must populate the journal")
        for entry in journal.data["files"].values():
            self.assertIsNone(entry.get("sha256"))
            self.assertTrue(entry.get("present_in_offload"))
        self.assertTrue(os.path.exists(repo_root), "full archive keeps local files")

    def test_stale_same_size_archive_copy_is_left_to_rsync_metadata_policy(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = self._make_legacy_archive(repo_root)
        victim = os.path.join(
            dest_repo, os.path.relpath(snapshot, repo_root),
            "flashchat", "q4", "model_weights.bin")
        with open(victim, "r+b") as f:
            f.write(b"X")  # same size, different content

        offload.offload_model(self.moe, snapshot, self.dest)
        journal = offload.Journal(dest_repo)
        rel = os.path.relpath(victim, dest_repo)
        self.assertIn(rel, journal.data["files"])
        self.assertIsNone(journal.data["files"][rel]["sha256"])

    def test_source_blobs_removed_after_existing_archive_confirmed_by_size(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        shutil.copytree(os.path.join(repo_root, "blobs"),
                        os.path.join(dest_repo, "blobs"))
        blob_dst = next(os.path.join(dest_repo, "blobs", n)
                        for n in os.listdir(os.path.join(dest_repo, "blobs")))
        moved = offload.offload_model(self.moe, snapshot, self.dest)
        self.assertGreater(moved, 0)
        self.assertTrue(os.path.isfile(blob_dst))
        blob_link = os.path.join(snapshot, "model-00001-of-00001.safetensors")
        self.assertFalse(os.path.exists(blob_link))
        self.assertEqual(offload.archive_state(self.moe, self.dest), "full")


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

    def test_restore_trusts_lightweight_journal_entries_without_hashes(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        offload.offload_model(self.moe, snapshot, self.dest)
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        journal = offload.Journal(dest_repo)
        victim = next(r for r in journal.data["files"] if r.endswith("model_weights.bin"))
        with open(os.path.join(dest_repo, victim), "r+b") as f:
            f.write(b"X")  # same size, corrupted content
        shutil.rmtree(repo_root)
        restored = offload.restore_full(self.moe, self.cache, self.dest)
        self.assertGreater(restored, 0)


if __name__ == "__main__":
    unittest.main()
