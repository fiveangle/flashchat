"""Add-model manifest derivation."""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from modelmgr.addmodel import derive_manifest
from modelmgr.manifest import parse_manifest
from modelmgr.registry import Registry


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

        manifest = derive_manifest("Qwen/Qwen3-Coder-Next", hf_config, Registry.load())
        parsed = parse_manifest(manifest, user_defined=True)

        self.assertEqual(parsed.id, "qwen-qwen3-coder-next")
        self.assertEqual(parsed.source_format, "native_bf16")
        self.assertEqual(parsed.architecture["num_experts"], 512)
        self.assertEqual(parsed.architecture["num_experts_per_tok"], 10)
        self.assertEqual(parsed.architecture["rope_theta"], 5000000)
        self.assertEqual(set(parsed.variants), {"q4", "q8"})
        self.assertIn("packed_experts/", parsed.variants["q4"].artifacts)
        self.assertEqual(parsed.special_tokens["eos_1"], 151645)


if __name__ == "__main__":
    unittest.main()
