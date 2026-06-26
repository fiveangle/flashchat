"""Manage TUI repair behavior."""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr.registry import Registry
from modelmgr import paths
from modelmgr.artifacts import ArtifactDir
from modelmgr.status import model_status
from modelmgr.tui import common, manage

from treebuilder import make_snapshot


class TestManageRepair(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = Registry.load()
        self.manifest = self.registry.get("qwen3-next-80b-a3b-instruct")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ready_model_can_offer_optional_mtp_bf16_build(self):
        snapshot = make_snapshot(self.tmp.name, self.manifest, variants=["q4", "q8"])
        shutil.rmtree(os.path.join(paths.shared_dir(snapshot), "bf16"))
        adir = ArtifactDir(paths.shared_dir(snapshot), self.manifest.id, "shared")
        adir.forget("bf16/")
        adir.commit()
        status = model_status(
            self.registry, self.manifest, cache_dir=self.tmp.name,
            offload_root="", check_offload=False)
        original_confirm = common.confirm
        common.confirm = lambda *args, **kwargs: False
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                manage._build_repair(self.registry, self.manifest, status)
        finally:
            common.confirm = original_confirm
        text = out.getvalue()
        self.assertIn("build / repair plan", text)
        self.assertIn("shared/MTP BF16 weights", text)
        self.assertNotIn("nothing to repair", text)

    def test_rebuild_uses_empty_journal_full_offload_for_all_variants(self):
        local_snapshot = make_snapshot(
            self.tmp.name, self.manifest, variants=["q4", "q8"],
            with_blobs=False)
        offload_root = os.path.join(self.tmp.name, "offload")
        offload_snapshot = make_snapshot(
            offload_root, self.manifest, variants=["q4", "q8"])
        journal = os.path.join(
            paths.repo_root_dir(offload_root, self.manifest.hf_repo),
            ".flashchat_offload.json")
        with open(journal, "w") as f:
            f.write('{"schema":1,"files":{},"links":{},"dirty_scopes":["q4","shared"]}')
        status = model_status(
            self.registry, self.manifest, cache_dir=self.tmp.name,
            offload_root=offload_root, check_offload=True)
        self.assertEqual(status.archive, "full")

        original_offload_dir = manage.build.offload_dir
        original_ensure = manage.build.ensure_variant_built
        original_confirm = common.confirm
        calls = []
        try:
            manage.build.offload_dir = lambda: offload_root
            common.confirm = lambda *a, **k: True
            def fake_ensure(*args, **kwargs):
                calls.append((args[2], kwargs.get("source_snapshot")))
                return True
            manage.build.ensure_variant_built = fake_ensure
            with contextlib.redirect_stdout(io.StringIO()):
                manage._rebuild_artifacts(
                    self.registry, self.manifest, status, [], ["q4", "q8"])
        finally:
            manage.build.offload_dir = original_offload_dir
            manage.build.ensure_variant_built = original_ensure
            common.confirm = original_confirm
        self.assertEqual(calls, [("q4", offload_snapshot),
                                 ("q8", offload_snapshot)])
        self.assertFalse(os.path.exists(
            os.path.join(local_snapshot, "model-00001-of-00001.safetensors")))

    def test_rebuild_download_menu_defaults_to_cancel(self):
        make_snapshot(self.tmp.name, self.manifest, variants=["q4", "q8"],
                      with_blobs=False)
        status = model_status(
            self.registry, self.manifest, cache_dir=self.tmp.name,
            offload_root="", check_offload=False)
        original_download = manage.build.download_snapshot
        calls = []
        try:
            manage.build.download_snapshot = lambda *a, **k: calls.append(a) or None
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                manage._rebuild_artifacts(
                    self.registry, self.manifest, status, [], ["q4", "q8"])
        finally:
            manage.build.download_snapshot = original_download
        self.assertEqual(calls, [])
        self.assertIn("download originals to the local HuggingFace cache", out.getvalue())

    def test_delete_components_menu_names_reclaimable_parts(self):
        snapshot = make_snapshot(self.tmp.name, self.manifest, variants=["q4", "q8"])
        status = model_status(
            self.registry, self.manifest, cache_dir=self.tmp.name,
            offload_root="", check_offload=False)
        original_select = common.select_number
        try:
            common.select_number = lambda *a, **k: None
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                manage._delete_components(self.registry, self.manifest, status)
        finally:
            common.select_number = original_select
        text = out.getvalue()
        self.assertIn("Delete local components to reclaim disk space", text)
        self.assertIn("original Hugging Face source blobs", text)
        self.assertIn("q4 runtime artifacts", text)
        self.assertIn("q8 runtime artifacts", text)
        self.assertIn("Offload/archive copies are not changed", text)

    def test_delete_components_explains_no_local_snapshot(self):
        status = model_status(
            self.registry, self.manifest, cache_dir=self.tmp.name,
            offload_root="", check_offload=False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            manage._delete_components(self.registry, self.manifest, status)
        text = out.getvalue()
        self.assertIn("no local components exist", text)
        self.assertIn("nothing to reclaim", text)


if __name__ == "__main__":
    unittest.main()
