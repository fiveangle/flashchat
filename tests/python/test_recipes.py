"""Recipe planning: skip-if-valid, regeneration triggers, download detection."""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import paths, recipes
from modelmgr.artifacts import ArtifactDir
from modelmgr.registry import Registry

from treebuilder import make_snapshot


class TestRecipePlanning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        registry = Registry.load()
        self.moe = registry.get("qwen3.6-35b-a3b")
        self.dense = registry.get("qwen3.6-27b")

    def tearDown(self):
        self.tmp.cleanup()

    def steps_of(self, plan):
        return [s.step for s in plan.steps]

    def test_valid_tree_plans_nothing(self):
        for manifest in (self.moe, self.dense):
            snapshot = make_snapshot(self.tmp.name, manifest)
            for vname in manifest.variants:
                plan = recipes.plan(manifest, vname, snapshot)
                self.assertTrue(plan.empty, f"{manifest.id}:{vname}: {plan.steps}")

    def test_fresh_snapshot_plans_full_recipe(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=[])
        # shared was populated; wipe it to simulate a brand new download
        import shutil
        shutil.rmtree(paths.flashchat_dir(snapshot))
        plan = recipes.plan(self.moe, "q4", snapshot)
        steps = self.steps_of(plan)
        self.assertIn("export_tokenizer", steps)
        self.assertIn("compile_native:non_experts", steps)
        self.assertIn("compile_native:experts", steps)
        self.assertIn("compile_native:mtp_experts", steps)
        self.assertIn("materialize_shared", steps)
        # optional bf16 not planned unless asked for
        self.assertNotIn("compile_native:bf16_mtp", steps)
        self.assertFalse(plan.needs_download)
        # shared steps come before variant steps, materialize last
        self.assertEqual(steps[-1], "materialize_shared")
        self.assertLess(steps.index("export_tokenizer"),
                        steps.index("compile_native:non_experts"))

    def test_optional_bf16_planned_when_wanted(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        import shutil
        shutil.rmtree(os.path.join(paths.shared_dir(snapshot), "bf16"), ignore_errors=True)
        sdir = ArtifactDir(paths.shared_dir(snapshot))
        sdir.forget("bf16/")
        sdir.commit()
        self.assertTrue(recipes.plan(self.moe, "q4", snapshot).empty)
        plan = recipes.plan(self.moe, "q4", snapshot, want_optional=True)
        self.assertIn("compile_native:bf16_mtp", self.steps_of(plan))

    def test_dense_model_never_plans_expert_steps(self):
        snapshot = make_snapshot(self.tmp.name, self.dense, variants=[])
        import shutil
        shutil.rmtree(paths.flashchat_dir(snapshot))
        steps = self.steps_of(recipes.plan(self.dense, "q4", snapshot))
        self.assertNotIn("compile_native:experts", steps)
        self.assertNotIn("compile_native:mtp_experts", steps)
        self.assertIn("compile_native:non_experts", steps)

    def test_deleted_weights_replans_only_that_step(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        vdir = paths.variant_dir(snapshot, "q4")
        os.unlink(os.path.join(vdir, "model_weights.bin"))
        plan = recipes.plan(self.moe, "q4", snapshot)
        self.assertEqual(self.steps_of(plan), ["compile_native:non_experts"])

    def test_stale_step_version_triggers_rebuild(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        adir = ArtifactDir(paths.variant_dir(snapshot, "q4"))
        adir.entries["model_weights.bin"]["step_version"] = 0  # ancient
        adir.commit()
        plan = recipes.plan(self.moe, "q4", snapshot)
        self.assertEqual(self.steps_of(plan), ["compile_native:non_experts"])
        self.assertEqual(plan.steps[0].reason, "outdated-step")

    def test_force_replans_everything(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        plan = recipes.plan(self.moe, "q4", snapshot, force=True)
        self.assertIn("compile_native:non_experts", self.steps_of(plan))
        self.assertIn("export_tokenizer", self.steps_of(plan))

    def test_missing_shared_link_plans_materialize_only(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=["q4"])
        os.unlink(os.path.join(paths.variant_dir(snapshot, "q4"), "vocab.bin"))
        plan = recipes.plan(self.moe, "q4", snapshot)
        self.assertEqual(self.steps_of(plan), ["materialize_shared"])
        self.assertEqual(plan.steps[0].artifacts, ["vocab.bin"])

    def test_needs_download_when_blobs_absent(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=[], with_blobs=False)
        import shutil
        shutil.rmtree(paths.flashchat_dir(snapshot))
        plan = recipes.plan(self.moe, "q4", snapshot)
        self.assertTrue(plan.needs_download)

    def test_dangling_blob_symlink_counts_as_absent(self):
        snapshot = make_snapshot(self.tmp.name, self.moe, variants=[])
        import shutil
        shutil.rmtree(paths.flashchat_dir(snapshot))
        # break the blob link the way offloading originals does
        repo = paths.repo_root_dir(self.tmp.name, self.moe.hf_repo)
        shutil.rmtree(os.path.join(repo, "blobs"))
        plan = recipes.plan(self.moe, "q4", snapshot)
        self.assertTrue(plan.needs_download)


if __name__ == "__main__":
    unittest.main()
