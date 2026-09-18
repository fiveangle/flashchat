"""Add-model manifest derivation."""

import os
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr.addmodel import AddModelError, derive_manifest, load_generation_config, sampling_profile
from modelmgr.manifest import parse_manifest
from modelmgr.registry import Registry
from modelmgr.resolved import flat_entry


class TestAddModelDerivation(unittest.TestCase):
    def test_qwen3_next_native_manifest_derives(self):
        hf_config = {
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "full_attention_interval": 4,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "partial_rotary_factor": 0.25,
            "rope_theta": 5000000,
        }

        manifest = derive_manifest("Qwen/Qwen3-Coder-Next", hf_config, Registry.load(),
                                   thinking_capable=False,
                                   generation_config={"temperature": 1.0, "top_p": 0.95, "top_k": 40})
        parsed = parse_manifest(manifest, user_defined=True)

        self.assertEqual(parsed.id, "qwen-qwen3-coder-next")
        self.assertEqual(parsed.source_format, "native_bf16")
        self.assertEqual(parsed.architecture["num_experts"], 512)
        self.assertEqual(parsed.architecture["num_experts_per_tok"], 10)
        self.assertEqual(parsed.architecture["rope_theta"], 5000000)
        self.assertEqual(set(parsed.variants), {"q4", "q8"})
        self.assertIn("packed_experts/", parsed.variants["q4"].artifacts)
        self.assertEqual(parsed.special_tokens["eos_1"], 151645)
        self.assertFalse(parsed.thinking_capable)
        self.assertFalse(flat_entry(parsed, "q4")["thinking_capable"])
        self.assertEqual(parsed.default_sampling_profile, "model-default")
        self.assertEqual(set(parsed.sampling_profiles), {"model-default"})
        profile = parsed.sampling_profiles["model-default"]
        self.assertEqual((profile["temperature"], profile["top_p"], profile["top_k"]), (1.0, 0.95, 40))
        self.assertEqual(profile["reasoning"], 0)
        self.assertEqual(profile["presence_penalty"], 0)
        self.assertEqual(profile["repetition_penalty"], 1)
        self.assertEqual(flat_entry(parsed, "q4")["sampling_profiles"], parsed.sampling_profiles)

    def test_manifest_derivation_defaults_to_thinking_capable_when_unknown(self):
        hf_config = {
            "model_type": "qwen3_next",
            "vocab_size": 151936,
            "num_experts": 512,
        }

        manifest = derive_manifest("Qwen/Qwen3-Next-Unknown", hf_config, Registry.load(),
                                   generation_config={"temperature": 0.8, "top_p": 0.9, "top_k": 30})
        parsed = parse_manifest(manifest, user_defined=True)

        self.assertTrue(parsed.thinking_capable)
        self.assertNotIn("thinking_capable", flat_entry(parsed, "q4"))
        self.assertEqual(parsed.sampling_profiles["model-default"]["reasoning"], 0)

    def test_same_vocabulary_does_not_share_sampling_settings(self):
        architecture = {"model_type": "qwen3_next", "vocab_size": 151936, "num_experts": 512}
        registry = Registry.load()
        configs = [{"temperature": 0.3, "top_p": 0.8, "top_k": 15},
                   {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.1}]
        for i, generation in enumerate(configs):
            model = derive_manifest(f"Example/Model-{i}", architecture, registry,
                                    generation_config=generation)
            for key, value in generation.items():
                self.assertEqual(model["sampling_profiles"]["model-default"][key], value)

    def test_missing_or_unsupported_settings_are_not_guessed(self):
        for settings in (None, {}, {"temperature": 1.0, "top_p": 0.95},
                         {"temperature": float("nan"), "top_p": 0.95, "top_k": 40},
                         {"temperature": 1.0, "top_p": 0.95, "top_k": "40"}):
            with self.subTest(settings=settings), self.assertRaises(AddModelError):
                sampling_profile("Example/Model", settings, False)

    def test_explicit_greedy_generation_settings(self):
        p = sampling_profile("Example/Model", {"do_sample": False}, False)
        self.assertEqual((p["temperature"], p["top_k"]), (0, 1))

    def test_generation_download_failure_is_actionable(self):
        with patch("modelmgr.steps.download.download_file", side_effect=OSError("not found")):
            with self.assertRaisesRegex(AddModelError, "--generation-config FILE"):
                load_generation_config("Example/Model", "/unused")


if __name__ == "__main__":
    unittest.main()
