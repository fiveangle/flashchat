"""bench_report counts regressions only for models installed on this machine."""

import csv
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "python"))

_TMP_CONFIG = tempfile.mkdtemp(prefix="flashchat-test-config.")
os.environ["FLASHCHAT_CONFIG_DIR"] = _TMP_CONFIG

from modelmgr.registry import Registry

from treebuilder import make_snapshot

REPORT = os.path.join(REPO_ROOT, "tests", "bench_report.py")
HW = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
FIELDS = ["timestamp", "commit", "hw_model", "model", "server_mode", "scenario",
          "metric_type", "metric_value", "tok_per_sec", "status"]


class TestInstalledModelsOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, "hub")
        self.offload = os.path.join(self.tmp.name, "offload")
        os.makedirs(self.cache)
        os.makedirs(self.offload)
        manifest = Registry.load().get("qwen3.6-35b-a3b")
        make_snapshot(self.cache, manifest, variants=["q4"])
        make_snapshot(self.offload, manifest, variants=["q8"])
        self.log = os.path.join(self.tmp.name, "api_perf_log.tsv")

    def tearDown(self):
        self.tmp.cleanup()

    def write_log(self, models):
        with open(self.log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
            w.writeheader()
            for model in models:
                for ts, commit, tps in (("2026-01-01T00:00:00Z", "aaaaaaa", 20.0),
                                        ("2026-01-02T00:00:00Z", "bbbbbbb", 10.0)):
                    w.writerow({"timestamp": ts, "commit": commit, "hw_model": HW,
                                "model": model, "server_mode": "bench",
                                "scenario": "chat_stream:technical",
                                "metric_type": "decode_tok_per_sec", "metric_value": tps,
                                "tok_per_sec": tps, "status": "pass"})

    def run_report(self, *args):
        env = dict(os.environ,
                   FLASHCHAT_CONFIG_DIR=_TMP_CONFIG,
                   FLASHCHAT_HUGGINGFACE_CACHE_DIR=self.cache,
                   FLASHCHAT_OFFLOAD_DIR=self.offload)
        return subprocess.run([sys.executable, REPORT, "--log", self.log, *args],
                              env=env, capture_output=True, text=True)

    def test_installed_regression_fails(self):
        self.write_log(["Qwen-Qwen36-35B-A3B"])
        result = self.run_report()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("# 1 regression(s)", result.stdout)

    def test_uninstalled_regressions_are_reported_but_not_counted(self):
        self.write_log(["Qwen-Qwen36-35B-A3B-q8", "Qwen-Qwen36-35B-A3B-8bit",
                        "Qwen-Qwen3-Next-80B-A3B-Instruct"])
        result = self.run_report()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("# 0 regression(s)", result.stdout)
        self.assertIn("Qwen-Qwen36-35B-A3B-q8   [%s]   (offloaded — not counted)" % HW, result.stdout)
        self.assertIn("(no longer in registry — not counted)", result.stdout)
        self.assertIn("(not installed — not counted)", result.stdout)

    def test_include_uninstalled_counts_everything(self):
        self.write_log(["Qwen-Qwen36-35B-A3B", "Qwen-Qwen36-35B-A3B-q8",
                        "Qwen-Qwen36-35B-A3B-8bit"])
        result = self.run_report("--include-uninstalled")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("# 3 regression(s)", result.stdout)
        self.assertNotIn("not counted", result.stdout)


if __name__ == "__main__":
    unittest.main()
