#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from flashchat_quant import (  # noqa: E402
    bf16_to_f32,
    convert_8bit_to_4bit,
    dequantize_4bit_affine_rows,
    f32_to_bf16,
    quantize_f32_to_4bit_affine_rows,
    split_qwen_gate_up_proj,
)
from compile_native_qwen import should_shift_native_norm, shift_native_norm_data  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def test_bf16_roundtrip():
    values = np.array([-3.5, -0.25, 0.0, 1.0, 17.75], dtype=np.float32)
    bf16 = f32_to_bf16(values)
    restored = bf16_to_f32(bf16)
    require(restored.shape == values.shape, "BF16 restored shape mismatch")
    require(np.allclose(restored, values, rtol=0.01, atol=0.01), "BF16 conversion drifted too far")


def test_4bit_affine_quantization():
    values = np.linspace(-2.0, 3.0, 128, dtype=np.float32).reshape(2, 64)
    weight, scales, biases = quantize_f32_to_4bit_affine_rows(values, group_size=64)
    restored = dequantize_4bit_affine_rows(weight, scales, biases, values.shape[1], group_size=64)

    require(weight.dtype == np.uint32, "packed weight dtype should be U32")
    require(weight.shape == (2, 8), "packed weight shape mismatch")
    require(scales.dtype == np.uint16, "scales should be stored as BF16 words")
    require(biases.dtype == np.uint16, "biases should be stored as BF16 words")
    require(restored.shape == values.shape, "dequantized shape mismatch")
    require(np.max(np.abs(restored - values)) < 0.45, "4-bit reconstruction error too large")


def test_8bit_to_4bit_legacy_contract():
    out_dim = 2
    in_dim = 64
    group_size = 64
    u8 = np.arange(out_dim * in_dim, dtype=np.uint8).reshape(out_dim, in_dim)
    packed_u8 = np.zeros((out_dim, in_dim // 4), dtype=np.uint32)
    for i in range(4):
        packed_u8 |= u8[:, i::4].astype(np.uint32) << (8 * i)

    scales = f32_to_bf16(np.ones((out_dim, 1), dtype=np.float32))
    biases = f32_to_bf16(np.zeros((out_dim, 1), dtype=np.float32))
    weight, new_scales, new_biases = convert_8bit_to_4bit(
        packed_u8, scales.tobytes(), biases.tobytes(), out_dim, in_dim, group_size
    )

    require(weight.shape == (out_dim, in_dim // 8), "8-bit conversion packed shape mismatch")
    require(weight.dtype == np.uint32, "8-bit conversion weight dtype mismatch")
    require(new_scales.shape == (out_dim, in_dim // group_size), "8-bit conversion scales shape mismatch")
    require(new_biases.shape == (out_dim, in_dim // group_size), "8-bit conversion biases shape mismatch")


def test_qwen_gate_up_split():
    gate = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    up = gate + 1000
    fused = np.concatenate([gate, up], axis=-2)
    got_gate, got_up = split_qwen_gate_up_proj(fused)

    require(np.array_equal(got_gate, gate), "gate half mismatch")
    require(np.array_equal(got_up, up), "up half mismatch")


def test_native_qwen_norm_shift_policy():
    meta = {"dtype": "BF16"}
    shifted = [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ]
    unchanged = [
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.mlp.gate.weight",
    ]

    for name in shifted:
        require(should_shift_native_norm(name, meta), f"{name} should use native norm shift")
    for name in unchanged:
        require(not should_shift_native_norm(name, meta), f"{name} should not use native norm shift")

    raw = f32_to_bf16(np.array([-0.25, 0.0, 1.5], dtype=np.float32)).tobytes()
    got = bf16_to_f32(shift_native_norm_data(raw))
    require(np.allclose(got, np.array([0.75, 1.0, 2.5], dtype=np.float32), atol=0.01),
            "native norm shift should add one before writing BF16")


def test_known_bf16_mtp_snapshot_metadata():
    snapshot = Path(
        "/Volumes/usr/Users/speedster/dev/models/hf/models--Qwen--Qwen3.6-35B-A3B/"
        "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
    )
    if not snapshot.exists():
        print(f"SKIP: native BF16 snapshot not found: {snapshot}")
        return

    with open(snapshot / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    required = {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.norm.weight",
    }
    missing = required - set(weight_map)
    require(not missing, f"BF16 MTP snapshot missing expected tensors: {sorted(missing)}")


def main():
    test_bf16_roundtrip()
    test_4bit_affine_quantization()
    test_8bit_to_4bit_legacy_contract()
    test_qwen_gate_up_split()
    test_native_qwen_norm_shift_policy()
    test_known_bf16_mtp_snapshot_metadata()
    print("flashchat quant helper tests passed")


if __name__ == "__main__":
    main()
