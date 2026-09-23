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
        # Restores preflight local free space; don't depend on the host disk.
        self._orig_free = offload._local_free_bytes
        offload._local_free_bytes = lambda _p: 1 << 50

    def tearDown(self):
        offload._local_free_bytes = self._orig_free
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

        restored = offload.restore_originals(self.moe, self.cache, self.dest)
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
        # a real rebuild re-records the artifact; unrecorded drift is refused
        from modelmgr.artifacts import ArtifactDir
        adir = ArtifactDir(paths.variant_dir(snapshot, "q4"), self.moe.id, "q4")
        adir.backfill("model_weights.bin")
        adir.commit()

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

    def _dest_journal_dirty(self):
        return offload.Journal(
            offload.dest_repo_dir(self.dest, self.moe)).data["dirty_scopes"]

    def test_reconcile_clears_scope_whose_offload_copy_matches(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)
        cache_dir = os.path.join(paths.variant_dir(snapshot, "q4"), "system_prompt_cache")
        os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, "local-only.fcache"), "wb") as f:
            f.write(b"runtime cache, never archived")
        offload.mark_artifact_scopes_dirty(self.moe, self.dest, ["q4"])

        self.assertEqual(offload.reconcile_pending_scopes(self.moe, snapshot, self.dest), [])
        self.assertEqual(offload.pending_scopes(self.moe), [])
        self.assertEqual(self._dest_journal_dirty(), [])

    def test_reconcile_keeps_scope_whose_offload_copy_differs(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)
        with open(os.path.join(paths.variant_dir(snapshot, "q4"),
                               "model_weights.bin"), "ab") as f:
            f.write(b"changed")
        offload.mark_artifact_scopes_dirty(self.moe, self.dest, ["q4", "shared"])

        self.assertEqual(offload.reconcile_pending_scopes(self.moe, snapshot, self.dest), ["q4"])
        self.assertEqual(offload.pending_scopes(self.moe), ["q4"])
        self.assertEqual(self._dest_journal_dirty(), ["q4"])

    def test_reconcile_leaves_flag_alone_when_offload_unavailable(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        missing = os.path.join(self.tmp.name, "missing-offload")
        offload.mark_artifact_scopes_dirty(self.moe, missing, ["q4"])
        self.assertEqual(offload.reconcile_pending_scopes(self.moe, snapshot, missing), ["q4"])
        self.assertEqual(offload.pending_scopes(self.moe), ["q4"])


class TestLaunchOffloadSyncOffer(OffloadBase):
    """The launch-time offer asks only about a real difference, and only on a terminal."""

    def _offer(self, snapshot, tty=True):
        from unittest import mock
        from modelmgr import ensure
        with mock.patch.object(ensure, "offload_dir", return_value=self.dest), \
                mock.patch.object(ensure.sys.stdin, "isatty", return_value=tty), \
                mock.patch("builtins.input", return_value="n") as ask:
            ensure._offer_pending_offload_sync(self.moe, snapshot)
        return ask.call_count

    def test_stale_flag_is_cleared_without_asking(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)
        offload.mark_artifact_scopes_dirty(self.moe, self.dest, ["q4"])
        self.assertEqual(self._offer(snapshot), 0)
        self.assertEqual(offload.pending_scopes(self.moe), [])

    def test_real_difference_is_offered(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)
        with open(os.path.join(paths.variant_dir(snapshot, "q4"),
                               "model_weights.bin"), "ab") as f:
            f.write(b"changed")
        offload.mark_artifact_scopes_dirty(self.moe, self.dest, ["q4"])
        self.assertEqual(self._offer(snapshot), 1)
        self.assertEqual(self._offer(snapshot, tty=False), 0)
        self.assertEqual(offload.pending_scopes(self.moe), ["q4"])


class TestRunnerDirtyScopes(OffloadBase):
    """A build marks the offload copy stale only when it changed local files."""

    def _run(self, snapshot, step_fn):
        from unittest import mock
        from modelmgr import runner
        plan = recipes.Plan(self.moe.id, "q4", snapshot, steps=[
            recipes.PlannedStep("export_tokenizer", "q4", ["model_weights.json"], "forced")])
        with mock.patch.object(runner, "load_step", return_value=step_fn), \
                mock.patch.object(runner.configfile, "get", return_value=self.dest):
            return runner.execute_plan(self.moe, "q4", snapshot, plan)

    def test_step_that_changes_nothing_is_not_marked(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)
        self.assertEqual(self._run(snapshot, lambda ctx, planned: None), set())
        self.assertEqual(offload.pending_scopes(self.moe), [])

    def test_step_that_rewrites_a_file_is_marked(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)

        def rewrite(ctx, planned):
            target = os.path.join(ctx.variant_dir, "model_weights.json")
            with open(target, "a") as f:
                f.write(" ")

        self.assertEqual(self._run(snapshot, rewrite), {"q4"})
        self.assertEqual(offload.pending_scopes(self.moe), ["q4"])


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


class TestRestoreAndPushSafety(OffloadBase):
    """Regressions from a 397B restore that ran the boot volume out of space
    mid-copy, left packed_experts/ without layout.json, and was one [o]ffload
    away from rsync --delete erasing the only complete copy on the archive."""

    def _legacy_archive(self, variants=("q4",)):
        snapshot = make_snapshot(self.cache, self.moe, variants=list(variants))
        repo_root = paths.repo_root_dir(self.cache, self.moe.hf_repo)
        dest_repo = offload.dest_repo_dir(self.dest, self.moe)
        shutil.copytree(repo_root, dest_repo, symlinks=True)
        return snapshot, repo_root, dest_repo

    def _packed_dir(self, root_snapshot):
        return os.path.join(paths.variant_dir(root_snapshot, "q4"), "packed_experts")

    def test_runtime_only_restore_from_legacy_unjournaled_archive(self):
        snapshot, repo_root, _dest = self._legacy_archive()
        shutil.rmtree(paths.flashchat_dir(snapshot))
        restored = offload.restore_runtime_only(self.moe, self.cache, self.dest)
        self.assertGreater(restored, 0)
        from modelmgr.artifacts import variant_ready
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))
        self.assertTrue(os.path.isfile(os.path.join(repo_root, "blobs", "blob0")),
                        "local originals untouched")

    def test_originals_only_restore_from_legacy_unjournaled_archive(self):
        snapshot, repo_root, _dest = self._legacy_archive()
        shutil.rmtree(os.path.join(repo_root, "blobs"))
        offload.restore_originals(self.moe, self.cache, self.dest)
        self.assertTrue(os.path.exists(
            os.path.join(snapshot, "model-00001-of-00001.safetensors")))

    def test_restore_never_deletes_local_only_files(self):
        snapshot, _repo_root, _dest = self._legacy_archive()
        local_only = os.path.join(paths.variant_dir(snapshot, "q4"),
                                  "system_prompt_cache", "keep.fcache")
        os.makedirs(os.path.dirname(local_only))
        with open(local_only, "w") as f:
            f.write("local")
        offload.restore_full(self.moe, self.cache, self.dest)
        self.assertTrue(os.path.isfile(local_only))

    def test_restore_refuses_before_copying_when_disk_too_small(self):
        snapshot, _repo_root, _dest = self._legacy_archive()
        shutil.rmtree(paths.flashchat_dir(snapshot))
        offload._local_free_bytes = lambda _p: 1024
        with self.assertRaises(offload.OffloadError) as ctx:
            offload.restore_runtime_only(self.moe, self.cache, self.dest)
        self.assertIn("not enough local disk space", str(ctx.exception))
        self.assertFalse(os.path.isdir(paths.flashchat_dir(snapshot)),
                         "nothing may be written when the preflight fails")

    def test_plan_skips_files_already_restored(self):
        snapshot, _repo_root, _dest = self._legacy_archive()
        self.assertEqual(
            offload.plan_restore(self.moe, self.cache, self.dest, "runtime").needed_bytes, 0)
        layer = os.path.join(self._packed_dir(snapshot), "layer_00.bin")
        size = os.path.getsize(layer)
        os.unlink(layer)
        plan = offload.plan_restore(self.moe, self.cache, self.dest, "runtime")
        self.assertEqual(plan.needed_bytes, size)

    def test_missing_layout_json_reads_as_incomplete_not_mismatch(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        os.unlink(os.path.join(self._packed_dir(snapshot), "layout.json"))
        from modelmgr.artifacts import variant_status
        st = next(s for s in variant_status(self.moe, "q4", snapshot)
                  if s.relpath.startswith("packed_experts"))
        self.assertEqual(st.state, "incomplete")
        self.assertIn("layout.json missing", st.detail)

    def test_offload_with_interrupted_local_restore_keeps_archive_intact(self):
        snapshot, _repo_root, dest_repo = self._legacy_archive()
        # Simulate the disk-full restore: last layer + layout.json never landed,
        # shared vocab.bin never landed.
        local_packed = self._packed_dir(snapshot)
        layers = sorted(n for n in os.listdir(local_packed) if n.startswith("layer_"))
        os.unlink(os.path.join(local_packed, layers[-1]))
        os.unlink(os.path.join(local_packed, "layout.json"))
        os.unlink(os.path.join(paths.shared_dir(snapshot), "vocab.bin"))
        problems = offload.local_scope_problems(self.moe, snapshot)
        self.assertIn("q4", problems)
        self.assertIn("shared", problems)

        offload.offload_model(self.moe, snapshot, self.dest)

        arch_snapshot = os.path.join(dest_repo, os.path.relpath(
            snapshot, paths.repo_root_dir(self.cache, self.moe.hf_repo)))
        arch_packed = self._packed_dir(arch_snapshot)
        self.assertTrue(os.path.isfile(os.path.join(arch_packed, "layout.json")))
        self.assertTrue(os.path.isfile(os.path.join(arch_packed, layers[-1])))
        self.assertTrue(os.path.isfile(
            os.path.join(paths.shared_dir(arch_snapshot), "vocab.bin")))

    def test_offload_never_deletes_archived_blobs_missing_locally(self):
        snapshot, repo_root, dest_repo = self._legacy_archive()
        extra = os.path.join(dest_repo, "blobs", "blob-only-on-archive")
        with open(extra, "w") as f:
            f.write("archived original shard")
        offload.offload_model(self.moe, snapshot, self.dest)
        self.assertTrue(os.path.isfile(extra))

    def test_scope_sync_refuses_broken_local_scope(self):
        snapshot = make_snapshot(self.cache, self.moe, variants=["q4"])
        offload.offload_model(self.moe, snapshot, self.dest)
        os.unlink(os.path.join(self._packed_dir(snapshot), "layout.json"))
        with self.assertRaises(offload.OffloadError) as ctx:
            offload.sync_artifact_scopes(self.moe, snapshot, self.dest, ["q4"])
        self.assertIn("refusing to overwrite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
