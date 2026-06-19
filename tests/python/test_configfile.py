"""Config file IO: engine-compatible quoting, in-place edits, append-only."""

import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr import configfile


class TestConfigFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text):
        with open(self.path, "w") as f:
            f.write(text)

    def test_load_parses_quoted_values(self):
        self.write('MODEL="Qwen-Qwen36-35B-A3B"\nSERVER_PORT="9999"\n')
        values = configfile.load(self.path)
        self.assertEqual(values["MODEL"], "Qwen-Qwen36-35B-A3B")
        self.assertEqual(values["SERVER_PORT"], "9999")

    def test_last_occurrence_wins_like_bash_source(self):
        self.write('MODEL="a"\nMODEL="b"\n')
        self.assertEqual(configfile.load(self.path)["MODEL"], "b")

    def test_update_edits_in_place_preserving_comments_and_order(self):
        self.write('# my config\nMODEL="old"\nSERVER_PORT="9999"\n')
        configfile.update({"MODEL": "new"}, path=self.path)
        with open(self.path) as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], "# my config")
        self.assertEqual(lines[1], 'MODEL="new"')
        self.assertEqual(lines[2], 'SERVER_PORT="9999"')

    def test_update_appends_new_keys(self):
        self.write('MODEL="m"\n')
        configfile.update({"MODEL_BASE": "qwen3.6-35b-a3b", "MODEL_VARIANT": "q4"}, path=self.path)
        values = configfile.load(self.path)
        self.assertEqual(values["MODEL_BASE"], "qwen3.6-35b-a3b")
        self.assertEqual(values["MODEL_VARIANT"], "q4")
        with open(self.path) as f:
            first = f.readline().strip()
        self.assertEqual(first, 'MODEL="m"')  # existing lines untouched

    def test_values_always_double_quoted_for_engine(self):
        configfile.update({"MODEL": "x"}, path=self.path)
        with open(self.path) as f:
            self.assertIn('MODEL="x"', f.read())

    def test_update_creates_missing_file(self):
        nested = os.path.join(self.tmp.name, "sub", "config")
        configfile.update({"MODEL": "x"}, path=nested)
        self.assertEqual(configfile.load(nested)["MODEL"], "x")

    def test_env_override_in_get(self):
        self.write('MODEL="file-value"\n')
        os.environ["FLASHCHAT_MODEL"] = "env-value"
        try:
            self.assertEqual(configfile.get("MODEL", path=self.path), "env-value")
        finally:
            del os.environ["FLASHCHAT_MODEL"]
        self.assertEqual(configfile.get("MODEL", path=self.path), "file-value")


if __name__ == "__main__":
    unittest.main()
