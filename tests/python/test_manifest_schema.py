"""Manifest schema validation: shipped manifests parse, bad ones fail loudly."""

import copy
import glob
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import paths
from modelmgr.manifest import ManifestError, parse_manifest


def load_shipped():
    out = {}
    for path in sorted(glob.glob(os.path.join(paths.SHIPPED_MANIFEST_DIR, "*.json"))):
        out[path] = paths.read_json(path)
    return out


class TestShippedManifests(unittest.TestCase):
    def test_all_shipped_manifests_parse(self):
        shipped = load_shipped()
        self.assertEqual(len(shipped), 5)
        ids = set()
        for path, data in shipped.items():
            m = parse_manifest(data, source_path=path)
            self.assertNotIn(m.id, ids)
            ids.add(m.id)

    def test_legacy_ids_globally_unique(self):
        seen = {}
        for path, data in load_shipped().items():
            m = parse_manifest(data, source_path=path)
            for variant in m.variants.values():
                for lid in variant.legacy_ids:
                    self.assertNotIn(lid, seen, f"legacy id {lid} duplicated in {path} and {seen.get(lid)}")
                    seen[lid] = path
        # All shipped registry ids must be claimed somewhere.
        legacy = paths.read_json(
            os.path.join(os.path.dirname(__file__), "fixtures", "model_configs_legacy.json")
        )
        self.assertEqual(set(seen), set(legacy["models"]))

    def test_exactly_one_suggested_default(self):
        suggested = [
            parse_manifest(d, source_path=p).id
            for p, d in load_shipped().items()
            if d.get("suggested_default")
        ]
        self.assertEqual(len(suggested), 1, suggested)

    def test_mtp_required_only_for_native(self):
        for path, data in load_shipped().items():
            m = parse_manifest(data, source_path=path)
            if m.mtp_artifacts_required:
                self.assertEqual(m.source_format, "native_bf16", path)


class TestManifestValidation(unittest.TestCase):
    def setUp(self):
        path = sorted(glob.glob(os.path.join(paths.SHIPPED_MANIFEST_DIR, "*35b-a3b.json")))[0]
        self.good = paths.read_json(path)

    def _expect_error(self, mutate, pattern):
        data = copy.deepcopy(self.good)
        mutate(data)
        with self.assertRaisesRegex(ManifestError, pattern):
            parse_manifest(data)

    def test_unknown_step_rejected(self):
        def mutate(d):
            d["variants"]["q4"]["artifacts"]["model_weights.bin"]["step"] = "no_such_step"
        self._expect_error(mutate, "unknown step")

    def test_dangling_from_shared_rejected(self):
        def mutate(d):
            d["variants"]["q4"]["artifacts"]["vocab.bin"]["from_shared"] = "nope.bin"
        self._expect_error(mutate, "unknown shared artifact")

    def test_step_and_from_shared_mutually_exclusive(self):
        def mutate(d):
            d["variants"]["q4"]["artifacts"]["vocab.bin"]["step"] = "export_tokenizer"
        self._expect_error(mutate, "exactly one of")

    def test_missing_quantization_rejected(self):
        def mutate(d):
            del d["variants"]["q4"]["quantization"]["bits"]
        self._expect_error(mutate, "quantization.bits")

    def test_bad_default_variant_rejected(self):
        def mutate(d):
            d["default_variant"] = "q2"
        self._expect_error(mutate, "default_variant")

    def test_dense_requires_intermediate_size(self):
        def mutate(d):
            d["architecture"]["num_experts"] = 0
            d["architecture"].pop("intermediate_size", None)
        self._expect_error(mutate, "intermediate_size")

    def test_invalid_id_rejected(self):
        def mutate(d):
            d["id"] = "Has Spaces"
        self._expect_error(mutate, "invalid model id")

    def test_wrong_schema_rejected(self):
        def mutate(d):
            d["schema"] = 2
        self._expect_error(mutate, "unsupported manifest schema")


if __name__ == "__main__":
    unittest.main()
