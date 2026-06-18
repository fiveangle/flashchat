"""Hash-on-write, partial-file discipline, and tamper detection."""

import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import paths
from modelmgr.artifacts import ArtifactDir, sha256_file, variant_ready, variant_status
from modelmgr.registry import Registry

from treebuilder import make_snapshot


class TestHashOnWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.adir = ArtifactDir(self.tmp.name, "test-model", "q4")

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_commit_verify(self):
        with self.adir.open("weights.bin", step="extract_weights", step_version=1) as w:
            w.write(b"hello ")
            w.write(b"world")
        self.adir.commit()

        fresh = ArtifactDir(self.tmp.name)
        self.assertEqual(fresh.quick_check("weights.bin"), "ok")
        self.assertEqual(fresh.deep_check("weights.bin"), "ok")
        entry = fresh.entries["weights.bin"]
        self.assertEqual(entry["size"], 11)
        digest, _size = sha256_file(os.path.join(self.tmp.name, "weights.bin"))
        self.assertEqual(entry["sha256"], digest)
        self.assertEqual(entry["step"], "extract_weights")

    def test_no_partial_left_after_close(self):
        with self.adir.open("a.bin") as w:
            w.write(b"x")
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "a.bin.partial")))

    def test_abort_on_exception_leaves_nothing(self):
        try:
            with self.adir.open("b.bin") as w:
                w.write(b"partial bytes")
                raise RuntimeError("interrupted")
        except RuntimeError:
            pass
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "b.bin")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "b.bin.partial")))
        self.assertNotIn("b.bin", self.adir.entries)

    def test_flip_byte_caught_by_deep_not_quick(self):
        with self.adir.open("c.bin") as w:
            w.write(b"AAAA")
        self.adir.commit()
        path = os.path.join(self.tmp.name, "c.bin")
        with open(path, "r+b") as f:
            f.write(b"B")  # same size, different content
        fresh = ArtifactDir(self.tmp.name)
        self.assertEqual(fresh.quick_check("c.bin"), "ok")
        self.assertEqual(fresh.deep_check("c.bin"), "hash-mismatch")

    def test_truncate_caught_by_quick(self):
        with self.adir.open("d.bin") as w:
            w.write(b"AAAA")
        self.adir.commit()
        with open(os.path.join(self.tmp.name, "d.bin"), "wb") as f:
            f.write(b"AA")
        self.assertEqual(ArtifactDir(self.tmp.name).quick_check("d.bin"), "size-mismatch")

    def test_unmanifested_file_reported_unhashed(self):
        with open(os.path.join(self.tmp.name, "legacy.bin"), "wb") as f:
            f.write(b"old")
        self.assertEqual(self.adir.quick_check("legacy.bin"), "unhashed")

    def test_backfill_hashes_legacy_file(self):
        with open(os.path.join(self.tmp.name, "legacy.bin"), "wb") as f:
            f.write(b"old")
        self.adir.backfill("legacy.bin", step="extract_weights", step_version=1)
        self.adir.commit()
        self.assertEqual(ArtifactDir(self.tmp.name).deep_check("legacy.bin"), "ok")

    def test_forget_drops_directory_entries(self):
        with self.adir.open("packed/layer_00.bin") as w:
            w.write(b"e")
        with self.adir.open("other.bin") as w:
            w.write(b"o")
        self.adir.forget("packed/")
        self.assertNotIn("packed/layer_00.bin", self.adir.entries)
        self.assertIn("other.bin", self.adir.entries)


class TestVariantVerification(unittest.TestCase):
    """End-to-end verification over a synthetic snapshot tree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        registry = Registry.load()
        self.moe = registry.get("qwen3.6-35b-a3b")        # native MoE, q4+q8, MTP
        self.dense = registry.get("qwen3.6-27b")          # dense, no expert dirs

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_tree_verifies(self):
        for manifest in (self.moe, self.dense):
            snapshot = make_snapshot(self.tmp.name, manifest)
            for vname in manifest.variants:
                self.assertTrue(
                    variant_ready(manifest, vname, snapshot),
                    f"{manifest.id}:{vname} should verify",
                )

    def test_symlinked_shared_artifact_passes(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        vocab = os.path.join(paths.variant_dir(snapshot, "q4"), "vocab.bin")
        self.assertTrue(os.path.islink(vocab))
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))

    def test_dangling_shared_link_fails(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        os.unlink(os.path.join(paths.shared_dir(snapshot), "vocab.bin"))
        self.assertFalse(variant_ready(self.moe, "q4", snapshot))

    def test_wrong_quant_config_detected(self):
        # q8 weights manifest copied into the q4 dir must fail the
        # dimension/quant cross-check even though files exist.
        import json

        from treebuilder import make_weights_json

        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        wrong = make_weights_json(self.moe, self.moe.variant("q8"))
        with open(os.path.join(paths.variant_dir(snapshot, "q4"), "model_weights.json"), "w") as f:
            json.dump(wrong, f)
        states = {s.relpath: s for s in variant_status(self.moe, "q4", snapshot)}
        self.assertEqual(states["model_weights.bin"].state, "invalid")

    def test_missing_expert_layer_detected(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        os.unlink(os.path.join(paths.variant_dir(snapshot, "q4"),
                               "packed_experts", "layer_07.bin"))
        states = {s.relpath: s for s in variant_status(self.moe, "q4", snapshot)}
        self.assertEqual(states["packed_experts/"].state, "missing")
        self.assertIn("1/40", states["packed_experts/"].detail)

    def test_missing_mtp_tensors_detected_for_native(self):
        import json

        snapshot = make_snapshot(self.tmp.name, self.dense, variants=["q4"])
        weights_json = os.path.join(paths.variant_dir(snapshot, "q4"), "model_weights.json")
        with open(weights_json) as f:
            data = json.load(f)
        data["tensors"] = {"embed_tokens.weight": {"offset": 0, "size": 16}}
        with open(weights_json, "w") as f:
            json.dump(data, f)
        states = {s.relpath: s for s in variant_status(self.dense, "q4", snapshot)}
        self.assertEqual(states["model_weights.bin"].state, "incomplete")
        self.assertIn("MTP", states["model_weights.bin"].detail)
        self.assertFalse(states["model_weights.bin"].satisfied)

    def test_optional_bf16_absence_is_satisfied(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        bf16_states = [s for s in variant_status(self.moe, "q4", snapshot)
                       if s.relpath == "bf16/"]
        self.assertTrue(all(s.satisfied for s in bf16_states))


if __name__ == "__main__":
    unittest.main()
