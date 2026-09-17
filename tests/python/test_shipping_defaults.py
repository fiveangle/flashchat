"""Shipping defaults and legacy choices use the real shell config resolver."""
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from modelmgr import configfile
from modelmgr.registry import Registry
from modelmgr.tui import config_wizard, onboarding

ROOT = Path(__file__).resolve().parents[2]


class ShippingDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = str(Path(self.tmp.name) / "config")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("FLASHCHAT_")}
        self.env.update(FLASHCHAT_CONFIG_DIR=self.tmp.name,
                        FLASHCHAT_CONFIG_FILE_OVERRIDE=self.config,
                        FLASHCHAT_MODEL_CONFIG=str(ROOT / "assets/model_configs.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def shell(self, script):
        return subprocess.check_output(
            ["bash", "-c", 'source lib/config.sh; ' + script],
            cwd=ROOT, env=self.env, text=True)

    def test_new_config_and_export_match_shipping_defaults(self):
        with patch.dict(os.environ, self.env, clear=True):
            configfile.initialize_defaults()
        values = configfile.load(self.config)
        self.assertEqual(values["KV_QUANT"], "q8")
        self.assertEqual(values["MODEL"], "Qwen-Qwen36-35B-A3B")
        self.assertEqual(values["SERVER_BIND"], "127.0.0.1")
        self.assertEqual(values["CONVERSATION_CACHE"], "1")
        self.assertEqual(values["FUSE_LINEAR"], "1")
        self.assertEqual(values["ADAPTIVE_K_MASS"], "")
        self.assertEqual(values["EXPERT_PIN_MAX_GB"], "8")
        self.assertEqual(values["SHOW_THINKING"], "1")
        self.assertEqual(values["CONFIG_SCHEMA_VERSION"], "15")
        with patch.dict(os.environ, self.env, clear=True):
            defaults = configfile.shipping_defaults()
        self.assertFalse(defaults.keys() - values.keys())
        for key in ("PREFILL_RELEASE", "GPU_ROPE", "FUSED_ATTN", "IO_THREADS"):
            self.assertEqual(values[key], defaults[key])
        self.assertEqual(self.shell('flashchat_load_config; flashchat_export_runtime_config; '
                                   'printf "%s %s %s" "$FLASHCHAT_KV_QUANT" '
                                   '"$FLASHCHAT_LM_HEAD_MLOCK" "$FLASHCHAT_EXPERT_SPLIT_IO"'),
                         "q8 1 1")

    def test_conversation_cache_migration_and_environment_override(self):
        Path(self.config).write_text('CONFIG_SCHEMA_VERSION="14"\n')
        self.assertEqual(self.shell('flashchat_load_config; flashchat_get CONVERSATION_CACHE').strip(), "1")
        configfile.update({"CONVERSATION_CACHE": "0"}, self.config)
        self.assertEqual(self.shell('flashchat_load_config; flashchat_export_runtime_config; '
                                   'printf "%s" "$FLASHCHAT_CONVERSATION_CACHE"'), "0")
        self.env["FLASHCHAT_CONVERSATION_CACHE"] = "1"
        self.assertEqual(self.shell('flashchat_load_config; flashchat_export_runtime_config; '
                                   'printf "%s" "$FLASHCHAT_CONVERSATION_CACHE"'), "1")
        self.assertEqual(configfile.load(self.config)["CONVERSATION_CACHE"], "0")

    def test_existing_cache_choices_are_preserved(self):
        for saved, expected in ((None, "off"), ("", "off"), ("off", "off"),
                                ("q8", "q8"), ("q4", "q4")):
            with self.subTest(saved=saved):
                configfile.update({"CONFIG_SCHEMA_VERSION": "12"}, self.config)
                # Replace only the isolated fixture between cases.
                Path(self.config).write_text('CONFIG_SCHEMA_VERSION="12"\n' +
                    (f'KV_QUANT="{saved}"\n' if saved is not None else ""))
                self.assertEqual(self.shell('flashchat_load_config; flashchat_get KV_QUANT').strip(), expected)
                if saved is not None:
                    self.assertEqual(configfile.load(self.config)["KV_QUANT"], saved)

    def test_wizard_distinguishes_new_default_and_explicit_off(self):
        with patch.dict(os.environ, self.env, clear=True), redirect_stdout(io.StringIO()):
            manifest = Registry.load().manifests["qwen3.6-35b-a3b"]
            with patch.object(config_wizard.common, "select_number", return_value=None) as select:
                self.assertEqual(config_wizard._kv_quant_setting(manifest, 65536), {"KV_QUANT": "q8"})
                self.assertEqual(select.call_args.kwargs["default"], 2)
            configfile.update({"KV_QUANT": ""}, self.config)
            with patch.object(config_wizard.common, "select_number", return_value=1):
                self.assertEqual(config_wizard._kv_quant_setting(manifest, 65536), {"KV_QUANT": "off"})

    def test_onboarding_uses_complete_current_defaults(self):
        with patch.dict(os.environ, self.env, clear=True):
            registry = Registry.load()
            manifest = registry.manifests["qwen3.6-35b-a3b"]
            with patch.object(registry.state, "save"), patch.object(onboarding.resolved, "write"):
                onboarding._save_selection(registry, manifest, "q4")
        values = configfile.load(self.config)
        self.assertEqual(values["CONFIG_SCHEMA_VERSION"], "15")
        self.assertEqual(values["KV_QUANT"], "q8")
        self.assertEqual(values["SAMPLING_PROFILE"], "instruct")

    def test_benchmark_clears_ambient_runtime_overrides(self):
        source = (ROOT / "tests/bench_api.sh").read_text()
        # Execute the real setup block without loading weights or starting a server.
        start = source.index('        export HOME="$bench_home"')
        end = source.index('        exec ./infer', start)
        script = ('id=Qwen-Qwen36-35B-A3B; bench_home="$1"; '
                  'REPO_ROOT="$2"; model_config="$2/assets/model_configs.json"; '
                  'MP=/fixture/model; WD=/fixture/weights; ED=/fixture/experts; '
                  'probe() {\n' + source[start:end] + '\n'
                  'printf "RESULT=%s,%s,%s,%s" "$FLASHCHAT_KV_QUANT" '
                  '"$FLASHCHAT_LM_HEAD_MLOCK" "$FLASHCHAT_EXPERT_SPLIT_IO" '
                  '"${FLASHCHAT_GPU_ROPE-unset}"; }; probe')
        env = dict(self.env, FLASHCHAT_KV_QUANT="q4", FLASHCHAT_LM_HEAD_MLOCK="0",
                   FLASHCHAT_EXPERT_SPLIT_IO="0", FLASHCHAT_GPU_ROPE="0")
        output = subprocess.check_output(["bash", "-c", script, "bench-defaults",
                                          self.tmp.name, str(ROOT)], env=env, text=True)
        self.assertIn("RESULT=q8,1,1,1", output)

    def test_production_launch_exports_saved_and_overridden_controls(self):
        configfile.update({"IO_THREADS": "4", "GPU_ROPE": "0",
                           "FUSED_ATTN": "0", "PREFILL_RELEASE": "0"}, self.config)
        self.env["FLASHCHAT_IO_THREADS"] = "2"
        source = (ROOT / "flashchat").read_text()
        start = source.index('    (\n        flashchat_export_runtime_config')
        end = source.index('    local infer_pid=$!', start)
        launch = source[start:end].replace(' > /dev/null 2>&1 &', '')
        # Replace only process execution; exercise the actual launcher subshell.
        script = ('flashchat_load_config; SCRIPT_DIR="$FLASHCHAT_REPO_ROOT"; '
                  '_config_file="$FLASHCHAT_CONFIG_FILE"; _model_path=/fixture/model; '
                  'server_port=8000; exec() { '
                  'printf "RESULT=%s,%s,%s,%s,%s\\n" "$FLASHCHAT_IO_THREADS" '
                  '"$FLASHCHAT_GPU_ROPE" "$FLASHCHAT_FUSED_ATTN" '
                  '"$FLASHCHAT_PREFILL_RELEASE" "$FLASHCHAT_SESSIONS_DIR"; '
                  'printf "ARG=%s\\n" "$@"; };\n' + launch)
        output = self.shell(script)
        self.assertIn(f"RESULT=2,0,0,0,{self.tmp.name}/sessions", output)
        self.assertIn("ARG=--config\nARG=" + self.config, output)
        self.assertEqual(configfile.load(self.config)["IO_THREADS"], "4")

    def test_wizard_uses_shipping_defaults_without_creating_config(self):
        with patch.dict(os.environ, self.env, clear=True), redirect_stdout(io.StringIO()):
            manifest = Registry.load().manifests["qwen3.6-35b-a3b"]
            with patch.object(config_wizard.common, "confirm", return_value=True), \
                 patch.object(config_wizard.common, "prompt", side_effect=lambda label, default="": default), \
                 patch.object(config_wizard.common, "prompt_clearable", side_effect=lambda label, default="", **kw: default):
                values = config_wizard._advanced_settings(manifest, "q4")
            defaults = configfile.shipping_defaults()
            for key, value in values.items():
                self.assertEqual(value, defaults[key], key)
            self.assertFalse(configfile.exists())
