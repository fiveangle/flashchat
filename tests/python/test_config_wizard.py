"""Config wizard custom profile prompts."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import configfile
from modelmgr.registry import Registry
from modelmgr.tui import config_wizard


class TestConfigWizardCustomProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.tmp.name, "config")
        with open(self.config_path, "w") as f:
            f.write('SAMPLING_PROFILE="custom"\n')
            f.write('TEMPERATURE="0.1"\n')
            f.write('TOP_P="0.8"\n')
            f.write('TOP_K="20"\n')
            f.write('MIN_P="0.0"\n')
            f.write('PRESENCE_PENALTY="1.5"\n')
            f.write('REPETITION_PENALTY="1.0"\n')
            f.write('REASONING="0"\n')
        self.env = patch.dict(os.environ, {
            "FLASHCHAT_CONFIG_FILE_OVERRIDE": self.config_path,
            "FLASHCHAT_COLOR_OUTPUT": "0",
        })
        self.env.start()
        self.registry = Registry.load()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_custom_profile_prompts_k_with_model_default(self):
        manifest = self.registry.get("qwen3.6-35b-a3b")
        replies = "\n".join(["4", "0.7", "0.8", "20", "0.0", "1.5", "1.0", "1", ""]) + "\n"

        out = io.StringIO()
        with patch("sys.stdin", io.StringIO(replies)), redirect_stdout(out):
            changes = config_wizard._select_sampling_profile(manifest)

        self.assertIn("custom — set each parameter yourself (current)", out.getvalue())
        self.assertIn("Profile [4]:", out.getvalue())
        self.assertIn("Active experts (K, default 8, max 16) [8]:", out.getvalue())
        self.assertEqual(changes["SAMPLING_PROFILE"], "custom")
        self.assertEqual(changes["ACTIVE_EXPERTS"], "")

    def test_custom_profile_enter_keeps_custom_profile(self):
        manifest = self.registry.get("qwen3.6-35b-a3b")
        replies = "\n".join(["", "0.7", "0.8", "20", "0.0", "1.5", "1.0", "1", ""]) + "\n"

        out = io.StringIO()
        with patch("sys.stdin", io.StringIO(replies)), redirect_stdout(out):
            changes = config_wizard._select_sampling_profile(manifest)

        self.assertIn("Profile [4]:", out.getvalue())
        self.assertEqual(changes["SAMPLING_PROFILE"], "custom")

    def test_custom_profile_saves_k_override(self):
        manifest = self.registry.get("qwen3.6-35b-a3b")
        replies = "\n".join(["4", "0.7", "0.8", "20", "0.0", "1.5", "1.0", "1", "6"]) + "\n"

        with patch("sys.stdin", io.StringIO(replies)), redirect_stdout(io.StringIO()):
            changes = config_wizard._select_sampling_profile(manifest)

        self.assertEqual(changes["ACTIVE_EXPERTS"], "6")

    def test_custom_profile_clamps_k_above_runtime_max(self):
        manifest = self.registry.get("qwen3.6-35b-a3b")
        replies = "\n".join(["4", "0.7", "0.8", "20", "0.0", "1.5", "1.0", "1", "32"]) + "\n"

        out = io.StringIO()
        with patch("sys.stdin", io.StringIO(replies)), redirect_stdout(out):
            changes = config_wizard._select_sampling_profile(manifest)

        self.assertIn("K=32 exceeds runtime max 16; saving 16", out.getvalue())
        self.assertEqual(changes["ACTIVE_EXPERTS"], "16")

    def test_custom_profile_replaces_stale_saved_k_above_runtime_max(self):
        with open(self.config_path, "a") as f:
            f.write('ACTIVE_EXPERTS="32"\n')
        manifest = self.registry.get("qwen3.6-35b-a3b")
        replies = "\n".join(["4", "0.7", "0.8", "20", "0.0", "1.5", "1.0", "1", ""]) + "\n"

        out = io.StringIO()
        with patch("sys.stdin", io.StringIO(replies)), redirect_stdout(out):
            changes = config_wizard._select_sampling_profile(manifest)

        self.assertIn("saved K=32 exceeds runtime max 16; using 16", out.getvalue())
        self.assertIn("Active experts (K, default 8, max 16) [16]:", out.getvalue())
        self.assertEqual(changes["ACTIVE_EXPERTS"], "16")


if __name__ == "__main__":
    unittest.main()
