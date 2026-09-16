"""Render the real menu meter against coherent server snapshots and outages."""
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "flashchat").read_text()
METER = SOURCE[SOURCE.index("render_context_meter() {"):SOURCE.index("flashchat_set_config_value() {")]


def render(snapshot, server="Running"):
    script = '''
source lib/config.sh
flashchat_get() {
    case "$1" in
        MODEL) echo Qwen-Qwen3-Next-80B-A3B-Instruct ;;
        KV_QUANT) echo q8 ;;
        CONTEXT_WINDOW) echo "" ;;
    esac
}
FLASHCHAT_MODEL_CONFIG="$PWD/assets/model_configs.json"
FLASHCHAT_HEALTH_JSON="$1"
''' + METER + '\nrender_context_meter "$2"\n'
    return subprocess.check_output(["bash", "-c", script, "meter", snapshot, server],
                                   cwd=ROOT, text=True)


class MeterTests(unittest.TestCase):
    def snapshot(self, **fields):
        return json.dumps(dict(context_used=14293, max_context=65536,
                               kv_total_bytes=811597824, **fields))

    def test_prefill_with_cached_context(self):
        output = render(self.snapshot(phase="prefill", cached_tokens=10197,
                        prompt_tokens=10541, prefill_done=4096,
                        chunk=2, chunks=3, layer=23, layers=48))
        self.assertIn("Context Used: 14k [|||||||", output)
        self.assertIn("64k (774 MiB)", output)
        self.assertIn("Processing: Prompt 39% (4,096/10,541 tokens) | layer 23/48", output)

    def test_decode_and_idle(self):
        self.assertIn("Processing: Generating response | 85 tokens",
                      render(self.snapshot(phase="generating", generated_tokens=85)))
        output = render(self.snapshot(phase="idle", generated_tokens=85, cached_tokens=42))
        self.assertIn("Processing: Idle - ready for a request", output)
        self.assertNotIn("85", output)
        self.assertNotIn("No active request", output)

    def test_unreachable_does_not_claim_zero(self):
        output = render("")
        self.assertIn("Context Used: unavailable", output)
        self.assertIn("64k (774 MiB)", output)

    def test_stopped(self):
        output = render("", "Not running")
        self.assertIn("Context Used: 0k", output)
        self.assertIn("Processing: Server stopped", output)

    def test_fixed_fields_across_all_states(self):
        snapshots = [self.snapshot(phase=phase) for phase in
                     ("idle", "preparing", "prefill", "generating")]
        snapshots += [self.snapshot(), ""]
        expected = ["Context Used", "Processing"]
        for snapshot, server in [(s, "Running") for s in snapshots] + [("", "Not running")]:
            with self.subTest(snapshot=snapshot, server=server):
                lines = render(snapshot, server).splitlines()
                self.assertEqual([line.split(":", 1)[0] for line in lines], expected)
                self.assertTrue(all(line.split(":", 1)[1].strip() for line in lines))

    def test_small_context_does_not_round_to_zero(self):
        snapshot = json.loads(self.snapshot(phase="idle"))
        snapshot["context_used"] = 257
        self.assertIn("Context Used: 257 tokens", render(json.dumps(snapshot)))

    def test_model_missing_maximum_uses_default_window(self):
        # Unknown model still has a known configured/default window.
        script = '''source lib/config.sh
FLASHCHAT_MODEL_CONFIG="$PWD/assets/model_configs.json"
flashchat_kv_meter_data missing-model 65536 q8
flashchat_kv_meter_data Qwen-Qwen3-Next-80B-A3B-Instruct 524288 q8
'''
        output = subprocess.check_output(["bash", "-c", script], cwd=ROOT, text=True)
        self.assertEqual(output.splitlines()[0], "65536 0")
        self.assertTrue(output.splitlines()[1].startswith("262144 "))


if __name__ == "__main__":
    unittest.main()
