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
from modelmgr.tui import build

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
        try:
            build.paths.snapshot_dir = lambda cache, repo: None
            build.download_snapshot = lambda repo, cache, progress=None: snapshot
            build.hf_cache_dir = lambda: self.tmp.name
            build.offload_dir = lambda: ""
            with redirect_stdout(StringIO()):
                self.assertTrue(
                    build.ensure_variant_built(
                        self.registry, self.moe, "q4", assume_yes=True))
        finally:
            build.paths.snapshot_dir = old_snapshot_dir
            build.download_snapshot = old_download_snapshot
            build.hf_cache_dir = old_hf_cache_dir
            build.offload_dir = old_offload_dir


if __name__ == "__main__":
    unittest.main()
