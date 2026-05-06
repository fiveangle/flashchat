#!/usr/bin/env python3
"""Generate expert_index.json from safetensors files."""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from collections import defaultdict
import re


def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def get_default_model_path():
    model_repo = 'mlx-community/Qwen3.6-35B-A3B-4bit'
    escaped_repo = model_repo.replace('/', '--')
    hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
    snapshot_dir = f"{hf_cache}/models--{escaped_repo}/snapshots"
    
    if os.path.isdir(snapshot_dir):
        snapshots = sorted(os.listdir(snapshot_dir))
        if snapshots:
            return f"{snapshot_dir}/{snapshots[-1]}"
    
    return os.path.expanduser(f'~/.cache/huggingface/hub/models--{escaped_repo}/snapshots/<snapshot>')


def main():
    parser = argparse.ArgumentParser(description='Generate expert_index.json from safetensors')
    parser.add_argument('--model', type=str,
                        default=os.environ.get('FLASHCHAT_MODEL_PATH') or get_default_model_path(),
                        help='Path to model directory (or set FLASHCHAT_MODEL_PATH)')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Output directory (default: MODEL_PATH/flashchat)')
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
    
    expert_pattern = re.compile(
        r'(?:^|\.)(?:language_model\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\.'
        r'(gate_proj|up_proj|down_proj)\.'
        r'(weight|scales|biases)$'
    )
    expert_pattern_alt = re.compile(
        r'(?:^|\.)(?:language_model\.)?model\.layers\.(\d+)\.switch_mlp\.'
        r'(gate_proj|up_proj|down_proj)\.'
        r'(weight|scales|biases)$'
    )
    
    expert_tensors = []
    
    for name, filename in weight_map.items():
        m = expert_pattern.search(name)
        if not m:
            m = expert_pattern_alt.search(name)
        
        if m:
            layer_idx = int(m.group(1))
            proj_name = m.group(2)
            param_name = m.group(3)
            component_name = f"{proj_name}.{param_name}"
            expert_tensors.append((layer_idx, component_name, name, filename))
    
    print(f"Found {len(expert_tensors)} expert tensors")
    
    if not expert_tensors:
        print("ERROR: No expert tensors found. Check the model structure.", file=sys.stderr)
        sys.exit(1)
    
    by_layer = defaultdict(list)
    for layer_idx, comp_name, tensor_name, filename in expert_tensors:
        by_layer[layer_idx].append((comp_name, tensor_name, filename))
    
    num_layers = max(by_layer.keys()) + 1
    print(f"Layers: {num_layers} ({min(by_layer.keys())} to {max(by_layer.keys())})")
    
    expected_components = {'gate_proj.weight', 'gate_proj.scales', 'gate_proj.biases',
                           'up_proj.weight', 'up_proj.scales', 'up_proj.biases',
                           'down_proj.weight', 'down_proj.scales', 'down_proj.biases'}
    
    for layer_idx in sorted(by_layer.keys()):
        components = set(c for c, _, _ in by_layer[layer_idx])
        missing = expected_components - components
        if missing:
            print(f"WARNING: Layer {layer_idx} missing components: {missing}")
    
    needed_files = set(filename for _, _, _, filename in expert_tensors)
    header_cache = {}
    print(f"Parsing {len(needed_files)} safetensors headers...")
    
    for filename in sorted(needed_files):
        filepath = model_path / filename
        if not filepath.exists():
            filepath = filepath.resolve()
        header_cache[filename] = parse_safetensors_header(str(filepath))
    
    expert_reads = {}
    
    for layer_idx in sorted(by_layer.keys()):
        layer_key = str(layer_idx)
        expert_reads[layer_key] = {}
        
        for comp_name, tensor_name, filename in by_layer[layer_idx]:
            header, data_start = header_cache[filename]
            
            if tensor_name not in header:
                print(f"WARNING: {tensor_name} not found in {filename}")
                continue
            
            meta = header[tensor_name]
            shape = meta['shape']
            tensor_offsets = meta['data_offsets']
            
            total_size = tensor_offsets[1] - tensor_offsets[0]
            abs_offset = data_start + tensor_offsets[0]
            
            num_experts_in_tensor = shape[0]
            expert_stride = total_size // num_experts_in_tensor
            expert_size = expert_stride
            
            expert_reads[layer_key][comp_name] = {
                'file': filename,
                'abs_offset': abs_offset,
                'expert_stride': expert_stride,
                'expert_size': expert_size,
                'total_size': total_size,
                'shape': shape,
            }
    
    output = {
        'model_path': str(model_path.resolve()),
        'expert_reads': expert_reads,
    }
    
    output_path = output_dir / 'expert_index.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nWrote {output_path}")
    print(f"  Layers: {len(expert_reads)}")
    print(f"  Total expert tensors: {len(expert_tensors)}")
    
    first_layer = sorted(expert_reads.keys())[0]
    print(f"\nSample (layer {first_layer}):")
    for comp_name in sorted(expert_reads[first_layer].keys()):
        info = expert_reads[first_layer][comp_name]
        tensor_name_match = None
        for tn in header_cache[info['file']][0]:
            if f"layers.{first_layer}" in tn and comp_name.replace('.', '.') in tn:
                tensor_name_match = tn
                break
        dtype_str = "?"
        if tensor_name_match and tensor_name_match in header_cache[info['file']][0]:
            dtype_str = header_cache[info['file']][0][tensor_name_match]['dtype']
        print(f"  {comp_name}: file={info['file']}, offset={info['abs_offset']}, "
              f"stride={info['expert_stride']}, shape={info['shape']}, dtype={dtype_str}")


if __name__ == '__main__':
    main()
