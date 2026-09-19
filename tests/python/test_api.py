"""modelmgr api: the JSON contract the menubar app is built on."""

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import api, configfile, paths, settings
from modelmgr.registry import Registry

from treebuilder import make_snapshot

MODEL = "qwen3-next-80b-a3b-instruct"


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_dir = os.path.join(self.tmp.name, "config")
        self.cache = os.path.join(self.tmp.name, "hub")
        os.makedirs(self.config_dir)
        os.makedirs(self.cache)
        self.env = patch.dict(os.environ, {"FLASHCHAT_CONFIG_DIR": self.config_dir})
        self.env.start()
        for key in [k for k in os.environ if k.startswith("FLASHCHAT_")
                    and k != "FLASHCHAT_CONFIG_DIR"]:
            os.environ.pop(key)
        self.manifest = Registry.load().get(MODEL)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def write_config(self, **values):
        base = {"HUGGINGFACE_CACHE_DIR": self.cache, "OFFLOAD_DIR": "",
                "MODEL_BASE": MODEL, "MODEL_VARIANT": "q4"}
        base.update(values)
        configfile.initialize_defaults()
        configfile.update(base)

    def run_op(self, **kwargs):
        defaults = dict(model=MODEL, variant=None, source="auto", repair=False,
                        what=None, component=None, confirm="", repo=None,
                        generation_config=None)
        defaults.update(kwargs)
        out = io.StringIO()
        rc = api.run_operation(argparse.Namespace(**defaults), events=out)
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        return rc, events


class TestState(ApiTestCase):
    def test_state_reports_selected_ready_variant_and_schema(self):
        make_snapshot(self.cache, self.manifest, variants=["q4"])
        self.write_config()
        data = api.state(check_offload=False)
        self.assertEqual(data["schema"], api.API_SCHEMA)
        self.assertEqual(data["selected"]["model"], MODEL)
        self.assertTrue(data["selected"]["ready"])
        self.assertGreater(data["selected"]["memory"]["total_bytes"], 0)
        model = next(m for m in data["models"] if m["id"] == MODEL)
        self.assertTrue(model["selected"])
        self.assertEqual([v["name"] for v in model["variants"]], ["q4", "q8"])
        keys = [s["key"] for s in data["settings"]]
        self.assertEqual(keys, [s.key for s in settings.ALL])
        json.dumps(data)

    def test_state_without_config_is_first_run(self):
        data = api.state(check_offload=False)
        self.assertFalse(data["config_exists"])


class TestSettings(ApiTestCase):
    def test_bool_and_numbers_normalize(self):
        self.write_config()
        result = api.apply_settings({"SERVER_DEBUG": "true", "TOP_K": "40",
                                     "TEMPERATURE": "0.7"})
        self.assertEqual(result["values"]["SERVER_DEBUG"], "1")
        self.assertEqual(configfile.load()["TOP_K"], "40")

    def test_invalid_values_are_rejected_without_writing(self):
        self.write_config(TOP_K="20")
        for values in ({"TOP_K": "many"}, {"SERVER_PORT": "70000"},
                       {"KV_QUANT": "q2"}, {"NOT_A_KEY": "1"}, {"MAX_TOKENS": ""}):
            with self.assertRaises(api.ApiError):
                api.apply_settings(values)
        self.assertEqual(configfile.load()["TOP_K"], "20")

    def test_clear_words_store_empty(self):
        self.write_config(OFFLOAD_DIR="/somewhere", ADAPTIVE_K_MASS="0.9")
        api.apply_settings({"OFFLOAD_DIR": "-", "ADAPTIVE_K_MASS": "off", "MTP": "auto"})
        values = configfile.load()
        self.assertEqual(values["OFFLOAD_DIR"], "")
        self.assertEqual(values["ADAPTIVE_K_MASS"], "")
        self.assertEqual(values["MTP"], "")

    def test_context_window_clamps_to_model_max(self):
        self.write_config()
        too_big = str(self.manifest.max_context * 2)
        result = api.apply_settings({"CONTEXT_WINDOW": too_big})
        self.assertEqual(configfile.load()["CONTEXT_WINDOW"], str(self.manifest.max_context))
        self.assertTrue(result["warnings"])

    def test_named_profile_fills_sampling_values(self):
        self.write_config(TEMPERATURE="1.9")
        name = self.manifest.default_sampling_profile
        api.apply_settings({"SAMPLING_PROFILE": name})
        expected = str(self.manifest.sampling_profiles[name]["temperature"])
        self.assertEqual(configfile.load()["TEMPERATURE"], expected)

    def test_select_on_first_run_creates_config_with_profile(self):
        result = api.select_model(MODEL, "q8")
        values = configfile.load()
        self.assertEqual(values["MODEL_BASE"], MODEL)
        self.assertEqual(values["MODEL_VARIANT"], "q8")
        self.assertEqual(values["SAMPLING_PROFILE"], self.manifest.default_sampling_profile)
        self.assertFalse(result["ready"])
        self.assertTrue(Registry.load().is_enabled(MODEL))


class TestOperations(ApiTestCase):
    def test_delete_requires_exact_confirmation(self):
        snapshot = make_snapshot(self.cache, self.manifest, variants=["q4", "q8"])
        self.write_config()
        rc, events = self.run_op(operation="delete", component="variant:q8", confirm="nope")
        self.assertEqual(rc, 1)
        self.assertEqual(events[-1]["code"], "confirmation")
        self.assertTrue(os.path.isdir(paths.variant_dir(snapshot, "q8")))

        rc, events = self.run_op(operation="delete", component="variant:q8", confirm=MODEL)
        self.assertEqual(rc, 0, events)
        self.assertFalse(os.path.isdir(paths.variant_dir(snapshot, "q8")))
        self.assertTrue(os.path.isdir(paths.variant_dir(snapshot, "q4")))

    def test_verify_streams_artifacts_then_done(self):
        make_snapshot(self.cache, self.manifest, variants=["q4"])
        self.write_config()
        rc, events = self.run_op(operation="verify")
        self.assertEqual(rc, 0, events)
        kinds = {e["event"] for e in events}
        self.assertIn("artifact", kinds)
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["corrupt"], 0)

    def test_build_without_source_reports_needs_source(self):
        self.write_config()
        rc, events = self.run_op(operation="build", variant="q4", source="local")
        self.assertEqual(rc, 1)
        self.assertEqual(events[-1]["code"], "needs_source")

    def test_plan_build_lists_download_sources_when_missing(self):
        self.write_config()
        plan = api.plan_build(MODEL, "q4")
        self.assertFalse(plan["ready"])
        self.assertIn("download-local", [s["id"] for s in plan["sources"]])
        self.assertTrue(plan["steps"])
        pending = os.path.join(paths.repo_root_dir(self.cache, self.manifest.hf_repo))
        self.assertFalse(os.path.exists(pending))

    def test_prints_inside_operations_become_log_events(self):
        out = io.StringIO()
        em = api.Emitter(out)
        stream = api._LogStream(em)
        stream.write("\033[0;32mhello\033[0m\nworld")
        stream.close()
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([e["message"] for e in events], ["hello", "world"])


class TestWizardSharesSchema(unittest.TestCase):
    def test_advanced_keys_keep_wizard_order(self):
        self.assertEqual(
            [s.key for s in settings.ADVANCED][:4],
            ["IO_THREADS", "GPU_ROPE", "FUSED_ATTN", "PREFILL_RELEASE"])
        for s in settings.ALL:
            if s.parent:
                self.assertIn(s.parent, settings.BY_KEY)


if __name__ == "__main__":
    unittest.main()
