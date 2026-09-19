"""Interactive build flow edge cases."""

import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr.registry import Registry
from modelmgr import artifacts, offload, paths
from modelmgr.tui import build, common

from treebuilder import make_snapshot, populate_shared, populate_variant


class TestBuildFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        registry = Registry.load()
        self.registry = registry
        self.moe = registry.get("qwen3.6-35b-a3b")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_local_model_restores_only_selected_archived_runtime(self):
        manifest = self.registry.get("qwen3-next-80b-a3b-instruct")
        cache = os.path.join(self.tmp.name, "hub")
        archive = os.path.join(self.tmp.name, "archive")
        source = make_snapshot(archive, manifest, variants=["q4", "q8"])
        with patch.object(build, "hf_cache_dir", return_value=cache), \
                patch.object(build, "offload_dir", return_value=archive), \
                patch.object(common, "confirm", return_value=True), \
                patch.object(build, "want_optional", return_value=False), \
                patch.object(build, "want_mtp", return_value=False), \
                patch.object(build.runner, "execute_plan") as execute, \
                patch.object(build, "download_snapshot") as download, \
                redirect_stdout(StringIO()):
            self.assertTrue(build.ensure_variant_built(
                self.registry, manifest, "q4"))
        local = paths.snapshot_dir(cache, manifest.hf_repo)
        self.assertTrue(artifacts.variant_ready(manifest, "q4", local))
        self.assertFalse(os.path.exists(paths.variant_dir(local, "q8")))
        self.assertFalse(os.path.exists(os.path.join(
            paths.repo_root_dir(cache, manifest.hf_repo), "blobs")))
        self.assertFalse(os.path.exists(os.path.join(local, "config.json")))
        self.assertTrue(os.path.isfile(os.path.join(
            source, "model-00001-of-00001.safetensors")))
        self.assertTrue(artifacts.variant_ready(manifest, "q8", source))
        execute.assert_not_called()
        download.assert_not_called()

    def test_archive_build_without_local_snapshot_keeps_selected_source(self):
        for selection in ("prompt", "explicit", "download"):
            with self.subTest(selection=selection):
                root = os.path.join(self.tmp.name, selection)
                cache = os.path.join(root, "hub")
                archive = os.path.join(root, "archive")
                source = make_snapshot(archive, self.moe, variants=["q4"])
                shutil.rmtree(paths.flashchat_dir(source))

                def execute(manifest, variant, snapshot, plan, **kwargs):
                    self.assertEqual(snapshot, source)
                    self.assertFalse(plan.needs_download)
                    output = kwargs["output_snapshot"]
                    self.assertEqual(output, os.path.join(
                        paths.repo_root_dir(cache, manifest.hf_repo),
                        "snapshots", os.path.basename(source)))
                    populate_shared(manifest, output)
                    populate_variant(manifest, variant, output)
                    return set()

                with patch.object(build, "hf_cache_dir", return_value=cache), \
                        patch.object(build, "offload_dir", return_value=archive), \
                        patch.object(common, "confirm", return_value=selection != "download") as confirm, \
                        patch.object(common, "prompt", return_value="o"), \
                        patch.object(build, "download_snapshot", return_value=source), \
                        patch.object(build, "want_optional", return_value=False), \
                        patch.object(build.runner, "execute_plan", side_effect=execute) as run, \
                        redirect_stdout(StringIO()):
                    self.assertTrue(build.ensure_variant_built(
                        self.registry, self.moe, "q4", assume_yes=True,
                        source_snapshot=source if selection == "explicit" else None))
                    self.assertEqual(confirm.call_count, 0 if selection == "explicit" else 1)
                run.assert_called_once()
                local = paths.snapshot_dir(cache, self.moe.hf_repo)
                self.assertTrue(artifacts.variant_ready(self.moe, "q4", local))
                self.assertFalse(os.path.exists(paths.flashchat_dir(source)))
                self.assertFalse(os.path.exists(os.path.join(
                    paths.repo_root_dir(cache, self.moe.hf_repo), "blobs")))

    def test_declined_or_out_of_space_restore_does_not_build_or_download(self):
        archive = os.path.join(self.tmp.name, "archive")
        make_snapshot(archive, self.moe, variants=["q4"])
        for accept in (False, True):
            with self.subTest(accept=accept):
                cache = os.path.join(self.tmp.name, "hub")
                with patch.object(build, "hf_cache_dir", return_value=cache), \
                        patch.object(build, "offload_dir", return_value=archive), \
                        patch.object(build, "want_optional", return_value=False), \
                        patch.object(common, "confirm", return_value=accept), \
                        patch.object(offload, "_local_free_bytes", return_value=0), \
                        patch.object(build, "_offer_build_source") as source, \
                        redirect_stdout(StringIO()):
                    self.assertFalse(build.ensure_variant_built(
                        self.registry, self.moe, "q4", assume_yes=True))
                source.assert_not_called()
                self.assertFalse(os.path.exists(cache))

    def test_ready_archive_without_originals_cannot_be_used_for_forced_build(self):
        cache = os.path.join(self.tmp.name, "hub")
        archive = os.path.join(self.tmp.name, "archive")
        source = make_snapshot(archive, self.moe, variants=["q4"], with_blobs=False)
        with patch.object(build, "hf_cache_dir", return_value=cache), \
                patch.object(build, "want_optional", return_value=False), \
                patch.object(build.runner, "execute_plan") as execute, \
                redirect_stdout(StringIO()):
            self.assertFalse(build.ensure_variant_built(
                self.registry, self.moe, "q4", assume_yes=True,
                force=True, source_snapshot=source))
        execute.assert_not_called()
        self.assertFalse(os.path.exists(cache))

    def test_download_returned_snapshot_is_used_when_cache_scan_misses(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])

        old_snapshot_dir = build.paths.snapshot_dir
        old_download_snapshot = build.download_snapshot
        old_hf_cache_dir = build.hf_cache_dir
        old_offload_dir = build.offload_dir
        old_confirm = common.confirm
        old_prompt = common.prompt
        try:
            common.confirm = lambda *a, **k: True
            common.prompt = lambda *a, **k: "l"
            build.paths.snapshot_dir = lambda cache, repo: None
            build.download_snapshot = lambda repo, cache, progress=None: snapshot
            build.hf_cache_dir = lambda: self.tmp.name
            build.offload_dir = lambda: ""
            with redirect_stdout(StringIO()):
                self.assertTrue(
                    build.ensure_variant_built(
                        self.registry, self.moe, "q4", assume_yes=True))
        finally:
            common.confirm = old_confirm
            common.prompt = old_prompt
            build.paths.snapshot_dir = old_snapshot_dir
            build.download_snapshot = old_download_snapshot
            build.hf_cache_dir = old_hf_cache_dir
            build.offload_dir = old_offload_dir
            common.confirm = old_confirm
            common.prompt = old_prompt
    def test_build_source_can_use_full_offload_snapshot_directly(self):
        local_snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"],
                                       with_blobs=False)
        offload_root = os.path.join(self.tmp.name, "offload")
        offload_snapshot = make_snapshot(offload_root, self.moe, variants=["q4"])
        old_offload_dir = build.offload_dir
        old_confirm = common.confirm
        try:
            build.offload_dir = lambda: offload_root
            common.confirm = lambda *a, **k: True
            with redirect_stdout(StringIO()):
                chosen = build._offer_build_source(
                    self.moe, self.tmp.name, local_snapshot, prefer_offload=True)
            self.assertEqual(chosen, offload_snapshot)
            self.assertFalse(os.path.exists(
                os.path.join(local_snapshot, "model-00001-of-00001.safetensors")))
        finally:
            build.offload_dir = old_offload_dir
            common.confirm = old_confirm


    def test_assume_yes_does_not_auto_download_originals(self):
        old_download_snapshot = build.download_snapshot
        old_hf_cache_dir = build.hf_cache_dir
        old_offload_dir = build.offload_dir
        old_confirm = common.confirm
        old_prompt = common.prompt
        calls = []
        try:
            build.download_snapshot = lambda *a, **k: calls.append(a) or None
            build.hf_cache_dir = lambda: self.tmp.name
            build.offload_dir = lambda: ""
            common.confirm = lambda *a, **k: False
            common.prompt = lambda *a, **k: "n"
            with redirect_stdout(StringIO()):
                ok = build.ensure_variant_built(
                    self.registry, self.moe, "q4", assume_yes=True)
            self.assertFalse(ok)
            self.assertEqual(calls, [])
        finally:
            build.download_snapshot = old_download_snapshot
            build.hf_cache_dir = old_hf_cache_dir
            build.offload_dir = old_offload_dir
            common.confirm = old_confirm
            common.prompt = old_prompt

    def test_download_to_offload_returns_offload_snapshot(self):
        offload_root = os.path.join(self.tmp.name, "offload")
        downloaded = make_snapshot(offload_root, self.moe, variants=["q4"])
        old_download_snapshot = build.download_snapshot
        old_offload_dir = build.offload_dir
        old_prompt = common.prompt
        try:
            build.download_snapshot = lambda repo, cache, progress=None: downloaded
            build.offload_dir = lambda: offload_root
            common.prompt = lambda *a, **k: "o"
            with redirect_stdout(StringIO()):
                chosen = build._offer_download_or_restore(
                    self.moe, self.tmp.name, None)
            self.assertEqual(chosen, downloaded)
        finally:
            build.download_snapshot = old_download_snapshot
            build.offload_dir = old_offload_dir
            common.prompt = old_prompt


if __name__ == "__main__":
    unittest.main()
