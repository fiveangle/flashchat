#!/usr/bin/env python3
"""Extract all non-expert weights from Qwen3.6-35B-A3B-4bit into a single binary file."""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
import re
import numpy as np


def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def get_default_model_path():
    model_repo = os.environ.get('FLASHCHAT_MODEL_REPO', 'mlx-community/Qwen3.6-35B-A3B-4bit')
    escaped_repo = model_repo.replace('/', '--')
    hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
    snapshot_dir = f"{hf_cache}/models--{escaped_repo}/snapshots"
    
    if os.path.isdir(snapshot_dir):
        snapshots = sorted(os.listdir(snapshot_dir))
        if snapshots:
            return f"{snapshot_dir}/{snapshots[-1]}"
    
    return os.path.expanduser(f'~/.cache/huggingface/hub/models--{escaped_repo}/snapshots/<snapshot>')


def bf16_to_f32(bf16_bytes):
    u16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(f32_arr):
    u32 = f32_arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def convert_8bit_to_4bit(weight_u32, scales_bf16, biases_bf16, out_dim, in_dim, group_size=64):
    num_groups = in_dim // group_size
    scales_f32 = bf16_to_f32(scales_bf16).reshape(out_dim, num_groups)
    biases_f32 = bf16_to_f32(biases_bf16).reshape(out_dim, num_groups)
    
    u8_vals = np.zeros((out_dim, in_dim), dtype=np.uint8)
    for i in range(4):
        u8_vals[:, i::4] = (weight_u32[:, :in_dim // 4] >> (8 * i)) & 0xFF
    
    f32_vals = np.zeros((out_dim, in_dim), dtype=np.float32)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        f32_vals[:, start:end] = (u8_vals[:, start:end].astype(np.float32) * 
                                   scales_f32[:, g:g+1] + biases_f32[:, g:g+1])
    
    new_scales_f32 = np.zeros((out_dim, num_groups), dtype=np.float32)
    new_biases_f32 = np.zeros((out_dim, num_groups), dtype=np.float32)
    u4_vals = np.zeros((out_dim, in_dim), dtype=np.uint8)
    
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group_data = f32_vals[:, start:end]
        min_vals = group_data.min(axis=1, keepdims=True)
        max_vals = group_data.max(axis=1, keepdims=True)
        
        ranges = np.where(max_vals - min_vals < 1e-8, 1e-8, max_vals - min_vals)
        new_scales_f32[:, g:g+1] = ranges / 15.0
        new_biases_f32[:, g:g+1] = min_vals
        
        u4 = np.clip(np.round((group_data - min_vals) / new_scales_f32[:, g:g+1]), 0, 15).astype(np.uint8)
        u4_vals[:, start:end] = u4
    
    new_weight_u32 = np.zeros((out_dim, in_dim // 8), dtype=np.uint32)
    for i in range(8):
        new_weight_u32 |= (u4_vals[:, i::8].astype(np.uint32) << (4 * i))
    
    new_scales_bf16 = f32_to_bf16(new_scales_f32.flatten())
    new_biases_bf16 = f32_to_bf16(new_biases_f32.flatten())
    
    return new_weight_u32, new_scales_bf16, new_biases_bf16


def load_8bit_tensor_overrides(config_path):
    overrides = {}
    if not config_path.exists():
        return overrides
    
    with open(config_path) as f:
        config = json.load(f)
    
    quant_config = config.get('quantization') or config.get('quantization_config', {})
    
    for key, qinfo in quant_config.items():
        if isinstance(qinfo, dict) and qinfo.get('bits') == 8:
            if key.startswith('language_model.'):
                key = key[len('language_model.'):]
            overrides[key] = qinfo
    
    return overrides


def main():
    parser = argparse.ArgumentParser(description='Extract non-expert weights to binary')
    parser.add_argument('--model', type=str,
                        default=os.environ.get('FLASHCHAT_MODEL_PATH') or get_default_model_path(),
                        help='Path to model directory (or set FLASHCHAT_MODEL_PATH)')
    parser.add_argument('--output', type=str,
                        default=os.environ.get('FLASHCHAT_WEIGHTS_DIR') or None,
                        help='Output directory for model_weights.bin and .json (default: MODEL_PATH/flashchat)')
    parser.add_argument('--include-experts', action='store_true',
                        help='Also extract expert weights (huge, not recommended)')
    args = parser.parse_args()

    model_path = Path(args.model)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = model_path / 'flashchat'
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}
    skipped_expert = 0
    skipped_vision = 0

    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            skipped_vision += 1
            continue
        if not args.include_experts and expert_pattern.search(name):
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}")
    print(f"Skipped expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    eight_bit_overrides = load_8bit_tensor_overrides(model_path / 'config.json')
    if eight_bit_overrides:
        print(f"Found {len(eight_bit_overrides)} 8-bit quantization overrides (will convert to 4-bit)")

    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    print("\nParsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    def sanitize_name(name):
        if name.startswith("language_model."):
            return name[len("language_model."):]
        return name

    all_tensors = []
    for name in sorted(tensors_to_extract.keys()):
        san_name = sanitize_name(name)
        all_tensors.append((san_name, name, tensors_to_extract[name]))

    bin_path = output_dir / 'model_weights.bin'
    manifest = {
        "model": str(model_path),
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "full_attention_interval": 4,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
        }
    }

    layer_types = []
    for i in range(40):
        if (i + 1) % 4 == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    print(f"\nWriting {bin_path}...")
    t0 = time.time()
    offset = 0
    total_bytes = 0
    skipped_8bit_components = set()

    ALIGN = 64

    with open(bin_path, 'wb') as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            if orig_name in skipped_8bit_components:
                continue

            filepath = model_path / filename
            header, data_start = header_cache[filename]

            if orig_name not in header:
                print(f"  WARNING: {orig_name} not found in {filename}, skipping")
                continue

            meta = header[orig_name]
            tensor_offsets = meta['data_offsets']
            byte_len = tensor_offsets[1] - tensor_offsets[0]
            shape = meta['shape']
            dtype = meta['dtype']

            is_8bit_prefix = None
            for prefix in eight_bit_overrides:
                if san_name.startswith(prefix + '.'):
                    is_8bit_prefix = prefix
                    break

            if is_8bit_prefix and orig_name.endswith('.weight'):
                base_name = orig_name[:-len('.weight')]
                scales_name = base_name + '.scales'
                biases_name = base_name + '.biases'

                scales_meta = header[scales_name]
                biases_meta = header[biases_name]

                with open(filepath, 'rb') as sf:
                    sf.seek(data_start + tensor_offsets[0])
                    weight_data = sf.read(byte_len)
                    sf.seek(data_start + scales_meta['data_offsets'][0])
                    scales_data = sf.read(scales_meta['data_offsets'][1] - scales_meta['data_offsets'][0])
                    sf.seek(data_start + biases_meta['data_offsets'][0])
                    biases_data = sf.read(biases_meta['data_offsets'][1] - biases_meta['data_offsets'][0])

                weight_u32 = np.frombuffer(weight_data, dtype=np.uint32).reshape(shape)
                out_dim = shape[0]
                in_dim = out_dim
                if 'gate' in orig_name and 'shared_expert_gate' not in orig_name:
                    in_dim = manifest['config']['hidden_size']
                else:
                    in_dim = manifest['config']['hidden_size']

                new_w_u32, new_s_bf16, new_b_bf16 = convert_8bit_to_4bit(
                    weight_u32, scales_data, biases_data, out_dim, in_dim, group_size=64
                )

                weight_data = new_w_u32.tobytes()
                scales_data = new_s_bf16
                biases_data = new_b_bf16

                if offset % ALIGN != 0:
                    pad = ALIGN - (offset % ALIGN)
                    out_f.write(b'\x00' * pad)
                    offset += pad

                out_f.write(weight_data)
                manifest["tensors"][san_name] = {
                    "offset": offset,
                    "size": len(weight_data),
                    "shape": [out_dim, in_dim // 8],
                    "dtype": dtype,
                }
                offset += len(weight_data)
                total_bytes += len(weight_data)

                if offset % ALIGN != 0:
                    pad = ALIGN - (offset % ALIGN)
                    out_f.write(b'\x00' * pad)
                    offset += pad

                san_scales = sanitize_name(scales_name)
                out_f.write(scales_data)
                manifest["tensors"][san_scales] = {
                    "offset": offset,
                    "size": len(scales_data),
                    "shape": scales_meta['shape'],
                    "dtype": scales_meta['dtype'],
                }
                offset += len(scales_data)
                total_bytes += len(scales_data)

                if offset % ALIGN != 0:
                    pad = ALIGN - (offset % ALIGN)
                    out_f.write(b'\x00' * pad)
                    offset += pad

                san_biases = sanitize_name(biases_name)
                out_f.write(biases_data)
                manifest["tensors"][san_biases] = {
                    "offset": offset,
                    "size": len(biases_data),
                    "shape": biases_meta['shape'],
                    "dtype": biases_meta['dtype'],
                }
                offset += len(biases_data)
                total_bytes += len(biases_data)

                skipped_8bit_components.add(scales_name)
                skipped_8bit_components.add(biases_name)
                continue

            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            with open(filepath, 'rb') as sf:
                sf.seek(data_start + tensor_offsets[0])
                data = sf.read(byte_len)

            out_f.write(data)

            manifest["tensors"][san_name] = {
                "offset": offset,
                "size": byte_len,
                "shape": shape,
                "dtype": dtype,
            }

            offset += byte_len
            total_bytes += byte_len

            if (i + 1) % 100 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

    elapsed = time.time() - t0
    throughput = total_bytes / elapsed / 1e9

    print(f"\nDone: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
    print(f"Binary: {bin_path} ({os.path.getsize(bin_path) / 1e9:.2f} GB)")

    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")

    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for san_name, info in manifest["tensors"].items():
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        elif "switch_mlp" in san_name:
            cat = "routed_experts"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


if __name__ == '__main__':
    main()
