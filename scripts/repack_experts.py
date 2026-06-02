#!/usr/bin/env python3
"""Repack expert weights from scattered safetensors into contiguous per-layer binary files.

Reads model dimensions from assets/model_configs.json and computes the expert
component layout deterministically. No model-specific hardcoding required.

Creates one binary file per layer: packed_experts/layer_XX.bin
Within each expert block, 9 components packed in fixed order:
  gate_proj.weight → scales → biases
  up_proj.weight   → scales → biases
  down_proj.weight  → scales → biases

Usage:
    python scripts/repack_experts.py --model-id mlx-community-Qwen36-35B-A3B-4bit
    python scripts/repack_experts.py --model-id mlx-community-Qwen36-35B-A3B-4bit --index /path/expert_index.json
    python scripts/repack_experts.py --model-id mlx-community-Qwen36-35B-A3B-4bit --layers 0-4
    python scripts/repack_experts.py --model-id mlx-community-Qwen36-35B-A3B-4bit --dry-run
    python scripts/repack_experts.py --model-id mlx-community-Qwen36-35B-A3B-4bit --verify-only 0
"""

import argparse
import json
import os
import time
import sys
import errno

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry():
    path = os.environ.get("FLASHCHAT_MODEL_CONFIG", os.path.join(REPO_ROOT, "assets", "model_configs.json"))
    with open(path) as f:
        return json.load(f)

def get_model_entry(model_id):
    registry = load_registry()
    entry = registry.get("models", {}).get(model_id)
    if not entry:
        print(f"ERROR: Model '{model_id}' not found in model_configs.json", file=sys.stderr)
        sys.exit(1)
    return entry

# ---------------------------------------------------------------------------
# Component layout computation
#
# The expert repack format is deterministic from model dimensions:
#
#   gate_proj / up_proj:  out_dim = moe_intermediate_size, in_dim = hidden_size
#   down_proj:            out_dim = hidden_size, in_dim = moe_intermediate_size
#
#   weight_size = out_dim * (in_dim // values_per_word) * 4
#   scale_size  = out_dim * (in_dim // G) * 2       (BF16)
#   bias_size   = out_dim * (in_dim // G) * 2       (BF16)
#
# where G = group_size (default 64).
# ---------------------------------------------------------------------------

def build_components(entry):
    hidden = entry["hidden_size"]
    moe_int = entry["moe_intermediate_size"]
    quant = entry.get("quantization", {})
    gs = quant.get("group_size", 64)
    bits = quant.get("bits", 4)
    if bits not in (4, 8):
        raise ValueError(f"unsupported quantization bits={bits}")
    values_per_word = 32 // bits

    gate_up_out = moe_int
    gate_up_in = hidden
    down_out = hidden
    down_in = moe_int

    u32_size = 4    # each U32 word is 4 bytes

    gate_up_wsize = gate_up_out * (gate_up_in // values_per_word) * u32_size
    gate_up_ssize = gate_up_out * (gate_up_in // gs) * 2
    gate_up_bsize = gate_up_out * (gate_up_in // gs) * 2

    down_wsize = down_out * (down_in // values_per_word) * u32_size
    down_ssize = down_out * (down_in // gs) * 2
    down_bsize = down_out * (down_in // gs) * 2

    offset = 0
    components = []

    for name, size, out_d, in_d, s_groups in [
        ("gate_proj.weight", gate_up_wsize, gate_up_out, gate_up_in // values_per_word, None),
        ("gate_proj.scales", gate_up_ssize, gate_up_out, gate_up_in // gs, gate_up_in // gs),
        ("gate_proj.biases", gate_up_bsize, gate_up_out, gate_up_in // gs, gate_up_in // gs),
        ("up_proj.weight",   gate_up_wsize, gate_up_out, gate_up_in // values_per_word, None),
        ("up_proj.scales",   gate_up_ssize, gate_up_out, gate_up_in // gs, gate_up_in // gs),
        ("up_proj.biases",   gate_up_bsize, gate_up_out, gate_up_in // gs, gate_up_in // gs),
        ("down_proj.weight",  down_wsize,   down_out,   down_in // values_per_word, None),
        ("down_proj.scales",  down_ssize,   down_out,   down_in // gs, down_in // gs),
        ("down_proj.biases",  down_bsize,   down_out,   down_in // gs, down_in // gs),
    ]:
        dtype = "U32" if name.endswith(".weight") else "BF16"
        shape = [out_d, in_d]
        components.append({
            "name": name, "offset": offset, "size": size,
            "dtype": dtype, "shape": shape,
        })
        offset += size

    return components

# ---------------------------------------------------------------------------
# Shared helpers (identical across models)
# ---------------------------------------------------------------------------

def parse_layers(spec, num_layers):
    if spec is None or spec == "all":
        return list(range(num_layers))
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))

def load_index(index_path):
    with open(index_path) as f:
        idx = json.load(f)
    model_path_raw = idx["model_path"]
    model_path = os.path.abspath(os.path.expanduser(model_path_raw))
    if model_path != model_path_raw:
        print(f"Resolved model path: {model_path_raw} -> {model_path}")
    return idx["expert_reads"], model_path

def verify_component_sizes(expert_reads, components):
    expected = {c["name"]: c["size"] for c in components}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                print(f"WARNING: unknown component {comp_name} in layer {layer_key}")
                continue
            if info["expert_size"] != expected[comp_name]:
                print(f"MISMATCH: layer {layer_key}, {comp_name}: "
                      f"index says {info['expert_size']}, expected {expected[comp_name]}")
                return False
    print("Component sizes verified: all match expected layout")
    return True

def open_source_files(expert_reads, model_path, layers):
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info["file"])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        if not os.path.exists(path):
            print("ERROR: source shard not found")
            print(f"  shard: {fname}")
            print(f"  attempted path: {path}")
            print(f"  model_path: {model_path}")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds

def repack_layer(layer_idx, expert_reads, fds, output_dir, components,
                 expert_size, num_experts, layer_size, dry_run=False):
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        for expert_idx in range(num_experts):
            for comp in components:
                info = layer_info[comp["name"]]
                src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
                dst_offset = expert_idx * expert_size + comp["offset"]
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {layer_size:,} bytes to {out_path}")
        return layer_size, 0.0

    t0 = time.monotonic()

    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    bytes_written = 0

    read_plan = []
    for expert_idx in range(num_experts):
        for comp in components:
            info = layer_info[comp["name"]]
            src_fd = fds[info["file"]]
            src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
            dst_offset = expert_idx * expert_size + comp["offset"]
            read_plan.append((src_fd, src_offset, dst_offset, comp["size"]))

    read_plan.sort(key=lambda x: (x[0], x[1]))

    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} "
                          f"at offset {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0
    return bytes_written, elapsed

def verify_layer(layer_idx, expert_reads, fds, output_dir, components,
                 expert_size, num_experts):
    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)

    mismatches = 0
    for expert_idx in [0, 1, num_experts - 1]:
        for comp in components:
            info = layer_info[comp["name"]]
            src_fd = fds[info["file"]]
            src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
            dst_offset = expert_idx * expert_size + comp["offset"]

            original = os.pread(src_fd, comp["size"], src_offset)
            packed = os.pread(fd_packed, comp["size"], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED (experts 0, 1, {num_experts - 1})")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0

def write_layout(output_dir, components, num_layers, num_experts, expert_size):
    layout = {
        "expert_size": expert_size,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "components": components,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")

def get_default_index_path():
    return os.environ.get("FLASHCHAT_EXPERT_INDEX")

def detect_model_path(hf_repo):
    escaped = hf_repo.replace("/", "--")
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    snap_dir = os.path.join(hf_cache, f"models--{escaped}", "snapshots")
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        if snaps:
            return os.path.join(snap_dir, snaps[-1])
    return os.path.join(hf_cache, f"models--{escaped}", "snapshots", "<snapshot>")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Model ID from model_configs.json (e.g. mlx-community-Qwen36-35B-A3B-4bit)")
    parser.add_argument("--index",
                        default=get_default_index_path(),
                        help="Path to expert_index.json (default: auto-detect from model path)")
    parser.add_argument("--output", default=os.environ.get("FLASHCHAT_EXPERTS_DIR"),
                        help="Output directory for packed expert layer files (default: MODEL_PATH/flashchat/q<bits>/packed_experts)")
    parser.add_argument("--layers", default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify offsets without writing")
    parser.add_argument("--verify-only", type=int, default=None, metavar="LAYER",
                        help="Verify a specific layer against originals")
    args = parser.parse_args()

    entry = get_model_entry(args.model_id)
    hidden = entry["hidden_size"]
    moe_int = entry["moe_intermediate_size"]
    num_layers = entry["num_hidden_layers"]
    num_experts = entry["num_experts"]
    quant = entry.get("quantization", {})
    gs = quant.get("group_size", 64)
    bits = int(quant.get("bits", 4) or 4)

    components = build_components(entry)
    expert_size = sum(c["size"] for c in components)
    layer_size = num_experts * expert_size

    print(f"Model: {entry.get('name', args.model_id)}")
    print(f"  hidden={hidden}, moe_intermediate={moe_int}, num_layers={num_layers}")
    print(f"  num_experts={num_experts}, group_size={gs}")
    print(f"  expert_size={expert_size:,} bytes")
    print(f"  layer_size={layer_size:,} bytes (~{layer_size/(1024**3):.2f} GB)")
    print()

    if args.index:
        index_path = args.index
    else:
        hf_repo = entry["hf_repo"]
        model_path_detected = detect_model_path(hf_repo)
        index_path = os.path.join(model_path_detected, "flashchat", f"q{bits}", "expert_index.json")

    if not args.index and not os.path.exists(index_path):
        print(f"ERROR: expert_index.json not found at {index_path}")
        print("Generate it first: python scripts/generate_expert_index.py --model <path> --output <path>/flashchat/q<bits>")
        sys.exit(1)

    print("Loading expert index...")
    expert_reads, model_path = load_index(index_path)
    print(f"Model path: {model_path}")
    print(f"Layers in index: {len(expert_reads)}")

    if not verify_component_sizes(expert_reads, components):
        print("ABORTING: component size mismatch")
        sys.exit(1)

    output_dir = args.output or os.path.join(model_path, "flashchat", f"q{bits}", "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers, num_layers)

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * layer_size
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes:
            per_layer_gb = layer_size / (1024**3)
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            print(f"Hint: use --layers to process a subset, e.g. --layers 0-{int(free_gb / per_layer_gb) - 1}")
            sys.exit(1)

    fds = open_source_files(expert_reads, model_path, layers)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, fds, output_dir,
                     components, expert_size, num_experts)
        for fd in fds.values():
            os.close(fd)
        return

    write_layout(output_dir, components, num_layers, num_experts, expert_size)
    print()

    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        t_layer = time.monotonic()
        bytes_written, elapsed = repack_layer(
            layer_idx, expert_reads, fds, output_dir, components,
            expert_size, num_experts, layer_size, dry_run=args.dry_run
        )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float("inf")
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers)*layer_size/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            if not verify_layer(layer_idx, expert_reads, fds, output_dir,
                                components, expert_size, num_experts):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    for fd in fds.values():
        os.close(fd)

    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == "__main__":
    main()
