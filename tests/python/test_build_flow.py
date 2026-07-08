"""Interactive build flow edge cases."""

import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr.registry import Registry
from modelmgr.tui import build, common

from treebuilder import make_snapshot


class TestBuildFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        registry = Registry.load()
        self.registry = registry
        self.moe = registry.get("qwen3.6-35b-a3b")

    def tearDown(self):
        self.tmp.cleanup()

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
