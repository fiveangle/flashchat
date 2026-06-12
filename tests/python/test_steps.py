"""Step registry wiring, materialize behavior, and tokenizer step round-trip."""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import paths
from modelmgr.artifacts import ArtifactDir, variant_ready
from modelmgr.registry import Registry
from modelmgr.steps import STEP_TABLE, StepContext, load_step
from modelmgr.steps.materialize import materialize

from treebuilder import make_snapshot


class TestStepRegistry(unittest.TestCase):
    def test_every_step_resolves_to_a_runner(self):
        for name in STEP_TABLE:
            runner = load_step(name)
            self.assertTrue(callable(runner), name)

    def test_manifest_known_steps_subset_of_table(self):
        from modelmgr.manifest import KNOWN_STEPS
        self.assertTrue(KNOWN_STEPS <= set(STEP_TABLE),
                        KNOWN_STEPS - set(STEP_TABLE))


class TestMaterialize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.moe = Registry.load().get("qwen3.6-35b-a3b")

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_relative_symlinks_and_mirrors_hashes(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        vdir = paths.variant_dir(snapshot, "q4")
        os.unlink(os.path.join(vdir, "vocab.bin"))

        results = materialize(self.moe, "q4", paths.shared_dir(snapshot), vdir)
        self.assertIn(("vocab.bin", "symlink"), results)
        link = os.path.join(vdir, "vocab.bin")
        self.assertTrue(os.path.islink(link))
        self.assertFalse(os.path.isabs(os.readlink(link)))
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))
        entry = ArtifactDir(vdir).entries["vocab.bin"]
        self.assertTrue(entry.get("from_shared"))
        shared_entry = ArtifactDir(paths.shared_dir(snapshot)).entries["vocab.bin"]
        self.assertEqual(entry["sha256"], shared_entry["sha256"])

    def test_replaces_stale_real_file_with_link(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        vdir = paths.variant_dir(snapshot, "q4")
        link = os.path.join(vdir, "vocab.bin")
        os.unlink(link)
        with open(link, "wb") as f:
            f.write(b"stale duplicate")
        materialize(self.moe, "q4", paths.shared_dir(snapshot), vdir)
        self.assertTrue(os.path.islink(link))

    def test_only_filter_limits_targets(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        vdir = paths.variant_dir(snapshot, "q4")
        os.unlink(os.path.join(vdir, "vocab.bin"))
        results = materialize(self.moe, "q4", paths.shared_dir(snapshot), vdir,
                              only=["vocab.bin"])
        self.assertEqual([r[0] for r in results], ["vocab.bin"])


class TestTokenizerStep(unittest.TestCase):
    def test_step_runner_writes_hashed_vocab(self):
        registry = Registry.load()
        manifest = registry.get("qwen3.6-35b-a3b")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = os.path.join(tmp, "snap")
            os.makedirs(snapshot)
            tok = {
                "model": {"vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]},
                "added_tokens": [{"id": 2, "content": "<eos>"}],
            }
            with open(os.path.join(snapshot, "tokenizer.json"), "w") as f:
                json.dump(tok, f)
            shared = os.path.join(snapshot, "flashchat", "shared")
            ctx = StepContext(manifest=manifest, variant_name=None, snapshot=snapshot,
                              shared_dir=shared, variant_dir=None)
            load_step("export_tokenizer")(ctx)
            adir = ArtifactDir(shared)
            self.assertEqual(adir.deep_check("vocab.bin"), "ok")
            self.assertEqual(adir.entries["vocab.bin"]["step"], "export_tokenizer")
            with open(os.path.join(shared, "vocab.bin"), "rb") as f:
                self.assertEqual(f.read(4), b"BPET")


if __name__ == "__main__":
    unittest.main()
