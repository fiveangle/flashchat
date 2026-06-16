"""Model status labels across local cache and offload archive."""

import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "python"))

from modelmgr import status
from modelmgr.tui import status_view
from modelmgr.registry import Registry

from treebuilder import make_snapshot


class TestVariantStatusLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, "hub")
        self.offload = os.path.join(self.tmp.name, "offload")
        os.makedirs(self.cache)
        os.makedirs(self.offload)
        self.registry = Registry.load()
        self.manifest = self.registry.get("qwen3.6-35b-a3b")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_local_artifacts_show_offloaded_when_archive_has_variant(self):
        make_snapshot(self.cache, self.manifest, variants=["q4"])
        make_snapshot(self.offload, self.manifest, variants=["q4", "q8"])

        s = status.model_status(self.registry, self.manifest, self.cache, self.offload)

        self.assertEqual(s.summary_line("q4"), "ready")
        self.assertEqual(s.summary_line("q8"), "offloaded")
        self.assertGreater(s.variants["q4"].local_bytes, 0)
        self.assertGreater(s.variants["q8"].offload_bytes, 0)

    def test_missing_variant_with_local_originals_is_not_extracted(self):
        make_snapshot(self.cache, self.manifest, variants=["q4"], with_blobs=True)

        s = status.model_status(self.registry, self.manifest, self.cache, self.offload)

        self.assertTrue(s.originals_local)
        self.assertEqual(s.summary_line("q8"), "not extracted")

    def test_missing_variant_with_archived_originals_is_not_extracted(self):
        make_snapshot(self.offload, self.manifest, variants=["q4"], with_blobs=True)

        s = status.model_status(self.registry, self.manifest, self.cache, self.offload)

        self.assertTrue(s.originals_offloaded)
        self.assertGreater(s.offload_originals_bytes, 0)
        self.assertEqual(s.summary_line("q4"), "offloaded")
        self.assertEqual(s.summary_line("q8"), "not extracted")
        self.assertIn("source", status_view.variant_glyph(s, "q8"))
        self.assertIn("B", status_view.variant_glyph(s, "q8"))

    def test_missing_variant_without_source_tensors_is_not_downloaded(self):
        s = status.model_status(self.registry, self.manifest, self.cache, self.offload)

        self.assertEqual(s.summary_line("q4"), "not downloaded")
        self.assertEqual(s.summary_line("q8"), "not downloaded")


if __name__ == "__main__":
    unittest.main()
