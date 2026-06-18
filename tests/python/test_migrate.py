"""Layout v0 -> v1 migration: dedup, adoption, config/state, idempotency."""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import configfile, migrate, paths
from modelmgr.artifacts import ArtifactDir, variant_ready
from modelmgr.registry import Registry, RegistryState

from treebuilder import make_legacy_snapshot


class MigrateBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, "hub")
        self.config_dir = os.path.join(self.tmp.name, "config")
        os.makedirs(self.cache)
        os.environ["FLASHCHAT_CONFIG_DIR"] = self.config_dir
        self.registry = Registry.load()
        self.moe = self.registry.get("qwen3.6-35b-a3b")

    def tearDown(self):
        del os.environ["FLASHCHAT_CONFIG_DIR"]
        self.tmp.cleanup()


class TestDedup(MigrateBase):
    def test_duplicate_vocab_and_bf16_deduped(self):
        snapshot = make_legacy_snapshot(self.cache, self.moe, with_bf16=True)
        plan = migrate.build_plan(self.registry, self.cache)
        snap_plan = plan.snapshots[0]
        self.assertEqual(sorted(snap_plan.variants_present), ["q4", "q8"])
        rels = {d.relpath: d for d in snap_plan.dedups}
        self.assertIn("vocab.bin", rels)
        self.assertIn("bf16/", rels)
        self.assertEqual(rels["vocab.bin"].duplicates, ["q8"])
        self.assertGreater(plan.bytes_reclaimed, 0)

        migrate.execute_snapshot(snap_plan)
        shared = paths.shared_dir(snapshot)
        self.assertTrue(os.path.isfile(os.path.join(shared, "vocab.bin")))
        self.assertTrue(os.path.isfile(os.path.join(shared, "bf16", "mtp_weights.bin")))
        for v in ("q4", "q8"):
            link = os.path.join(paths.variant_dir(snapshot, v), "vocab.bin")
            self.assertTrue(os.path.islink(link), f"{v}/vocab.bin should be a link")
            self.assertTrue(os.path.isfile(link), f"{v}/vocab.bin link should resolve")
            self.assertTrue(variant_ready(self.moe, v, snapshot), v)

    def test_conflicting_vocab_left_in_place(self):
        make_legacy_snapshot(self.cache, self.moe,
                             vocab_content=lambda v: f"vocab-{v}".encode())
        plan = migrate.build_plan(self.registry, self.cache)
        snap_plan = plan.snapshots[0]
        self.assertTrue(snap_plan.conflicts)
        vocab = next(d for d in snap_plan.dedups if d.relpath == "vocab.bin")
        self.assertEqual(vocab.duplicates, [])  # nothing deleted

    def test_idempotent_rerun_plans_nothing(self):
        make_legacy_snapshot(self.cache, self.moe, with_bf16=True)
        plan = migrate.build_plan(self.registry, self.cache)
        migrate.execute_snapshot(plan.snapshots[0])
        replan = migrate.build_plan(self.registry, self.cache)
        self.assertTrue(replan.empty, replan.snapshots[0].dedups)


class TestAdoption(MigrateBase):
    def test_artifacts_adopted_with_provenance_unhashed(self):
        snapshot = make_legacy_snapshot(self.cache, self.moe)
        plan = migrate.build_plan(self.registry, self.cache)
        self.assertGreater(plan.snapshots[0].adopt_files, 0)
        migrate.execute_snapshot(plan.snapshots[0])
        adir = ArtifactDir(paths.variant_dir(snapshot, "q4"))
        entry = adir.entries["model_weights.bin"]
        self.assertEqual(entry["step"], "compile_native:non_experts")
        self.assertIsNone(entry["sha256"])  # lazily hashed
        self.assertEqual(adir.deep_check("model_weights.bin"), "unhashed")
        self.assertEqual(adir.quick_check("model_weights.bin"), "ok")

    def test_hash_now_records_baselines(self):
        snapshot = make_legacy_snapshot(self.cache, self.moe, variants=["q4"])
        plan = migrate.build_plan(self.registry, self.cache)
        migrate.execute_snapshot(plan.snapshots[0], hash_now=True)
        adir = ArtifactDir(paths.variant_dir(snapshot, "q4"))
        self.assertEqual(adir.deep_check("model_weights.bin"), "ok")

    def test_system_prompt_cache_not_adopted(self):
        snapshot = make_legacy_snapshot(self.cache, self.moe, variants=["q4"])
        cache_file = os.path.join(paths.variant_dir(snapshot, "q4"),
                                  "system_prompt_cache", "abc.fcache")
        with open(cache_file, "w") as f:
            f.write("cache")
        plan = migrate.build_plan(self.registry, self.cache)
        migrate.execute_snapshot(plan.snapshots[0])
        adir = ArtifactDir(paths.variant_dir(snapshot, "q4"))
        self.assertFalse(any("system_prompt_cache" in k for k in adir.entries))


class TestConfigAndState(MigrateBase):
    def test_legacy_model_id_maps_to_base_and_variant(self):
        configfile.update({"MODEL": "Qwen-Qwen36-35B-A3B-8bit", "SERVER_PORT": "9999"})
        changes = migrate.migrate_user_config(self.registry)
        self.assertEqual(changes["MODEL_BASE"], "qwen3.6-35b-a3b")
        self.assertEqual(changes["MODEL_VARIANT"], "q8")
        values = configfile.load()
        self.assertEqual(values["MODEL"], "Qwen-Qwen36-35B-A3B-8bit")  # untouched
        self.assertEqual(values["CONFIG_SCHEMA_VERSION"], "3")

    def test_config_migration_is_idempotent(self):
        configfile.update({"MODEL": "Qwen-Qwen36-27B"})
        self.assertTrue(migrate.migrate_user_config(self.registry))
        self.assertEqual(migrate.migrate_user_config(self.registry), {})

    def test_state_bootstrap_enables_found_models(self):
        make_legacy_snapshot(self.cache, self.moe)
        configfile.update({"MODEL": "Qwen-Qwen36-35B-A3B"})
        migrate.bootstrap_state(self.registry, self.cache, None)
        state = RegistryState.load()
        self.assertTrue(state.enabled.get("qwen3.6-35b-a3b"))
        self.assertEqual(state.default_model, "qwen3.6-35b-a3b")
        self.assertEqual(state.layout_version, 1)
        self.assertFalse(migrate.needed(Registry.load()))

    def test_full_run_end_to_end(self):
        snapshot = make_legacy_snapshot(self.cache, self.moe, with_bf16=True)
        configfile.update({"MODEL": "Qwen-Qwen36-35B-A3B"})
        plan = migrate.run(self.registry, self.cache)
        self.assertGreater(plan.bytes_reclaimed, 0)
        self.assertTrue(variant_ready(self.moe, "q4", snapshot))
        self.assertTrue(variant_ready(self.moe, "q8", snapshot))
        values = configfile.load()
        self.assertEqual(values["MODEL_BASE"], "qwen3.6-35b-a3b")
        self.assertFalse(migrate.needed(Registry.load()))
        # second run converges to no-op
        replan = migrate.run(Registry.load(), self.cache)
        self.assertTrue(replan.empty)


if __name__ == "__main__":
    unittest.main()
