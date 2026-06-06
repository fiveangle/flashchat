#!/usr/bin/env python3
"""Compile native Qwen BF16 safetensors into Flashchat runtime artifacts."""

import argparse
import json
import os
import re
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from flashchat_quant import (
    bf16_to_f32,
    f32_to_bf16,
    quantize_f32_to_4bit_affine_rows,
    quantize_f32_to_8bit_affine_rows,
    split_qwen_gate_up_proj,
)
from repack_experts import build_components, get_model_entry, parse_layers


ALIGN = 64
STACKED_EXPERT_RE = re.compile(
    r"^(?:model|mtp)\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
INDIVIDUAL_EXPERT_RE = re.compile(
    r"^(?:model|mtp)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def parse_safetensors_header(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def sanitize_name(name):
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def is_routed_expert(name):
    san = sanitize_name(name)
    return STACKED_EXPERT_RE.match(san) is not None or INDIVIDUAL_EXPERT_RE.match(san) is not None


def read_tensor_raw(model_path, filename, header, data_start, name):
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(model_path / filename, "rb") as f:
        f.seek(data_start + start)
        data = f.read(end - start)
    if len(data) != end - start:
        raise IOError(f"short read for {name}")
    return data


def read_tensor_bf16(model_path, filename, header, data_start, name):
    meta = header[name]
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name} is {meta['dtype']}, expected BF16")
    raw = read_tensor_raw(model_path, filename, header, data_start, name)
    return bf16_to_f32(raw).reshape(meta["shape"])


def read_expert_bf16(model_path, filename, header, data_start, name, expert_idx):
    meta = header[name]
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name} is {meta['dtype']}, expected BF16")
    shape = meta["shape"]
    if len(shape) != 3:
        raise ValueError(f"{name} shape {shape} is not an expert stack")
    if expert_idx < 0 or expert_idx >= shape[0]:
        raise ValueError(f"expert {expert_idx} outside {name} shape {shape}")

    start, end = meta["data_offsets"]
    stride = (end - start) // shape[0]
    with open(model_path / filename, "rb") as f:
        f.seek(data_start + start + expert_idx * stride)
        data = f.read(stride)
    if len(data) != stride:
        raise IOError(f"short read for {name} expert {expert_idx}")
    return bf16_to_f32(data).reshape(shape[1:])


def write_aligned(out_f, offset, data):
    if offset % ALIGN != 0:
        pad = ALIGN - (offset % ALIGN)
        out_f.write(b"\x00" * pad)
        offset += pad
    start = offset
    out_f.write(data)
    offset += len(data)
    return start, offset


def quantize_matrix_to_entries(name, matrix, bits, group_size):
    if bits == 4:
        weight, scales, biases = quantize_f32_to_4bit_affine_rows(matrix, group_size)
    elif bits == 8:
        weight, scales, biases = quantize_f32_to_8bit_affine_rows(matrix, group_size)
    else:
        raise ValueError(f"unsupported quantization bits={bits}")
    values_per_word = 32 // bits
    in_dim = matrix.shape[1]
    return [
        (name, weight.tobytes(), [matrix.shape[0], in_dim // values_per_word], "U32"),
        (name[:-len(".weight")] + ".scales", scales.tobytes(), [matrix.shape[0], in_dim // group_size], "BF16"),
        (name[:-len(".weight")] + ".biases", biases.tobytes(), [matrix.shape[0], in_dim // group_size], "BF16"),
    ]


def linear_attention_dims(entry):
    key_dim = int(entry["linear_num_key_heads"]) * int(entry["linear_key_head_dim"])
    value_dim = int(entry["linear_num_value_heads"]) * int(entry["linear_value_head_dim"])
    conv_dim = key_dim * 2 + value_dim
    gate_dim = int(entry["linear_num_value_heads"])
    return conv_dim, value_dim, gate_dim


def expand_native_projection_matrix(san_name, matrix, entry):
    # Qwen3-Next ships the GatedDeltaNet input projections stacked AND interleaved
    # per key-head (see Qwen3NextGatedDeltaNet.fix_query_key_value_ordering in HF).
    # in_proj_qkvz rows are grouped as num_k_heads blocks of
    #   [q: head_k_dim, k: head_k_dim, v: (num_v_heads/num_k_heads)*head_v_dim,
    #    z: (num_v_heads/num_k_heads)*head_v_dim]
    # which HF reshapes to head-contiguous query/key/value/z, then forms
    #   mixed_qkv = cat(query, key, value)  -> [q_all, k_all, v_all]
    # The engine copies conv1d.weight verbatim and consumes in_proj_qkv in exactly
    # that head-contiguous [q_all, k_all, v_all] order, so we must de-interleave
    # here. A flat slice (matrix[:conv_dim]) mixes heads together and silently
    # corrupts linear attention in every GatedDeltaNet layer.
    nkh = int(entry["linear_num_key_heads"])
    nvh = int(entry["linear_num_value_heads"])
    hkd = int(entry["linear_key_head_dim"])
    hvd = int(entry["linear_value_head_dim"])
    vh_per_kh = nvh // nkh  # value heads grouped under each key head

    if san_name.endswith(".linear_attn.in_proj_qkvz.weight"):
        conv_dim, value_dim, _ = linear_attention_dims(entry)
        expected = conv_dim + value_dim
        if matrix.shape[0] != expected:
            raise ValueError(f"{san_name} row count {matrix.shape[0]} != expected {expected}")
        in_features = matrix.shape[1]
        v_blk = vh_per_kh * hvd  # v (and z) rows per key-head block
        blk = 2 * hkd + 2 * v_blk
        per_head = matrix.reshape(nkh, blk, in_features)
        q = per_head[:, 0:hkd, :].reshape(nkh * hkd, in_features)
        k = per_head[:, hkd:2 * hkd, :].reshape(nkh * hkd, in_features)
        v = per_head[:, 2 * hkd:2 * hkd + v_blk, :].reshape(nkh * v_blk, in_features)
        z = per_head[:, 2 * hkd + v_blk:, :].reshape(nkh * v_blk, in_features)
        qkv = np.concatenate([q, k, v], axis=0)
        prefix = san_name[:-len("in_proj_qkvz.weight")]
        return [
            (prefix + "in_proj_qkv.weight", np.ascontiguousarray(qkv)),
            (prefix + "in_proj_z.weight", np.ascontiguousarray(z)),
        ]
    if san_name.endswith(".linear_attn.in_proj_ba.weight"):
        _, _, gate_dim = linear_attention_dims(entry)
        expected = gate_dim * 2
        if matrix.shape[0] != expected:
            raise ValueError(f"{san_name} row count {matrix.shape[0]} != expected {expected}")
        in_features = matrix.shape[1]
        # ba rows are grouped as num_k_heads blocks of [b: vh_per_kh, a: vh_per_kh],
        # reshaped by HF to head-contiguous b/a of length num_v_heads.
        per_head = matrix.reshape(nkh, 2 * vh_per_kh, in_features)
        b = per_head[:, 0:vh_per_kh, :].reshape(nkh * vh_per_kh, in_features)
        a = per_head[:, vh_per_kh:, :].reshape(nkh * vh_per_kh, in_features)
        prefix = san_name[:-len("in_proj_ba.weight")]
        return [
            (prefix + "in_proj_b.weight", np.ascontiguousarray(b)),
            (prefix + "in_proj_a.weight", np.ascontiguousarray(a)),
        ]
    return [(san_name, matrix)]


def should_quantize_tensor(san_name, meta, group_size):
    if meta["dtype"] != "BF16":
        return False
    shape = meta["shape"]
    if len(shape) != 2:
        return False
    if not san_name.endswith(".weight"):
        return False
    if shape[1] % group_size != 0:
        return False
    return True


def should_shift_native_norm(san_name, meta):
    if meta["dtype"] != "BF16":
        return False
    if not san_name.endswith(".weight"):
        return False
    if "linear_attn.norm.weight" in san_name:
        return False
    return (
        san_name.endswith(".input_layernorm.weight") or
        san_name.endswith(".post_attention_layernorm.weight") or
        san_name.endswith(".self_attn.q_norm.weight") or
        san_name.endswith(".self_attn.k_norm.weight") or
        san_name in {
            "model.norm.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        }
    )


def shift_native_norm_data(data):
    shifted = bf16_to_f32(data) + 1.0
    return f32_to_bf16(shifted).tobytes()


def native_text_config(model_path):
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = json.load(f)
    return config.get("text_config", config)


def build_manifest_config(entry, native_config=None):
    native_config = native_config or {}
    quant = entry.get("quantization", {})
    cfg = {
        "hidden_size": entry["hidden_size"],
        "num_hidden_layers": entry["num_hidden_layers"],
        "num_attention_heads": entry["num_attention_heads"],
        "num_key_value_heads": entry["num_key_value_heads"],
        "head_dim": entry["head_dim"],
        "vocab_size": entry["vocab_size"],
        "rms_norm_eps": entry["rms_norm_eps"],
        "num_experts": entry["num_experts"],
        "num_experts_per_tok": entry["num_experts_per_tok"],
        "moe_intermediate_size": entry["moe_intermediate_size"],
        "shared_expert_intermediate_size": entry["shared_expert_intermediate_size"],
        "intermediate_size": entry.get("intermediate_size", 0),  # dense FFN (num_experts==0 => dense)
        "full_attention_interval": entry["full_attention_interval"],
        "linear_num_value_heads": entry["linear_num_value_heads"],
        "linear_num_key_heads": entry["linear_num_key_heads"],
        "linear_key_head_dim": entry["linear_key_head_dim"],
        "linear_value_head_dim": entry["linear_value_head_dim"],
        "linear_conv_kernel_dim": entry["linear_conv_kernel_dim"],
        "partial_rotary_factor": entry["partial_rotary_factor"],
        "rope_theta": entry["rope_theta"],
        "quantization": {
            "bits": quant.get("bits", 4),
            "group_size": quant.get("group_size", 64),
        },
    }
    cfg["layer_types"] = [
        "full_attention" if (i + 1) % cfg["full_attention_interval"] == 0 else "linear_attention"
        for i in range(cfg["num_hidden_layers"])
    ]
    cfg["mtp_num_hidden_layers"] = native_config.get(
        "mtp_num_hidden_layers",
        entry.get("mtp_num_hidden_layers", 0),
    )
    return cfg


def planned_non_expert_tensors(weight_map, headers, group_size, include_mtp):
    tensors = []
    skipped_experts = 0
    skipped_mtp = 0
    for name, filename in weight_map.items():
        san = sanitize_name(name)
        if san.startswith(("vision_tower", "model.visual")):
            continue
        if san.startswith("mtp.") and not include_mtp:
            skipped_mtp += 1
            continue
        if is_routed_expert(name):
            skipped_experts += 1
            continue
        meta = headers[filename][0][name]
        will_quantize = should_quantize_tensor(san, meta, group_size)
        tensors.append((san, name, filename, will_quantize))
    return sorted(tensors), skipped_experts, skipped_mtp


def compile_non_experts(model_path, output_dir, weight_map, headers, entry, native_config,
                        include_mtp, dry_run, limit, name_regex):
    quant = entry.get("quantization", {})
    bits = quant.get("bits", 4)
    group_size = quant.get("group_size", 64)
    tensors, skipped_experts, skipped_mtp = planned_non_expert_tensors(weight_map, headers, group_size, include_mtp)
    if name_regex:
        pattern = re.compile(name_regex)
        tensors = [t for t in tensors if pattern.search(t[0])]
    if limit is not None:
        tensors = tensors[:limit]

    print(f"Non-expert tensors planned: {len(tensors)}")
    print(f"Skipped routed expert tensors: {skipped_experts}")
    if skipped_mtp:
        print(f"Skipped MTP tensors: {skipped_mtp}")

    if dry_run:
        quantized = sum(1 for _, _, _, q in tensors if q)
        copied = len(tensors) - quantized
        print(f"Dry run: would quantize {quantized} BF16 matrices and copy {copied} tensors")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / "model_weights.bin"
    manifest = {
        "model": str(model_path),
        "num_tensors": 0,
        "tensors": {},
        "config": build_manifest_config(entry, native_config),
    }

    offset = 0
    total_bytes = 0
    started = time.time()
    with open(bin_path, "wb") as out_f:
        for idx, (san, orig, filename, will_quantize) in enumerate(tensors, 1):
            header, data_start = headers[filename]
            meta = header[orig]
            entries = []
            if will_quantize:
                matrix = read_tensor_bf16(model_path, filename, header, data_start, orig)
                for out_name, out_matrix in expand_native_projection_matrix(san, matrix, entry):
                    entries.extend(quantize_matrix_to_entries(out_name, out_matrix, bits, group_size))
            else:
                raw = read_tensor_raw(model_path, filename, header, data_start, orig)
                if should_shift_native_norm(san, meta):
                    raw = shift_native_norm_data(raw)
                entries = [(san, raw, meta["shape"], meta["dtype"])]

            for out_name, data, shape, dtype in entries:
                start, offset = write_aligned(out_f, offset, data)
                manifest["tensors"][out_name] = {
                    "offset": start,
                    "size": len(data),
                    "shape": shape,
                    "dtype": dtype,
                }
                total_bytes += len(data)

            if idx % 25 == 0 or idx == len(tensors):
                print(f"  [{idx}/{len(tensors)}] {total_bytes / 1e9:.2f} GB written")

    manifest["num_tensors"] = len(manifest["tensors"])
    with open(output_dir / "model_weights.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - started
    print(f"Wrote {bin_path} ({total_bytes / 1e9:.2f} GB in {elapsed:.1f}s)")
    print(f"Wrote {output_dir / 'model_weights.json'}")


def compile_bf16_mtp(model_path, output_dir, weight_map, headers, entry, native_config, dry_run):
    """Extract MTP head weights in native BF16 to a separate flashchat/bf16/ directory.

    This keeps the BF16 predictor path alive alongside the quantized 8-bit path,
    so researchers can A/B them without re-downloading safetensors.
    """
    bf16_dir = output_dir / "bf16"
    bf16_dir.mkdir(parents=True, exist_ok=True)
    bin_path = bf16_dir / "mtp_weights.bin"
    manifest = {
        "model": str(model_path),
        "num_tensors": 0,
        "tensors": {},
        "config": build_manifest_config(entry, native_config),
    }

    # Collect only MTP tensors
    mtp_tensors = []
    for name, filename in weight_map.items():
        san = sanitize_name(name)
        if san.startswith("mtp.") and not is_routed_expert(name):
            mtp_tensors.append((san, name, filename))
    mtp_tensors.sort()

    print(f"BF16 MTP tensors planned: {len(mtp_tensors)}")
    if dry_run:
        print(f"Dry run: would write {len(mtp_tensors)} BF16 MTP tensors")
        return

    offset = 0
    total_bytes = 0
    started = time.time()
    with open(bin_path, "wb") as out_f:
        for idx, (san, orig, filename) in enumerate(mtp_tensors, 1):
            header, data_start = headers[filename]
            meta = header[orig]
            raw = read_tensor_raw(model_path, filename, header, data_start, orig)
            if should_shift_native_norm(san, meta):
                raw = shift_native_norm_data(raw)
            start, offset = write_aligned(out_f, offset, raw)
            manifest["tensors"][san] = {
                "offset": start,
                "size": len(raw),
                "shape": meta["shape"],
                "dtype": meta["dtype"],
            }
            total_bytes += len(raw)
            if idx % 25 == 0 or idx == len(mtp_tensors):
                print(f"  [{idx}/{len(mtp_tensors)}] {total_bytes / 1e6:.1f} MB written")

    manifest["num_tensors"] = len(manifest["tensors"])
    with open(bf16_dir / "mtp_weights.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - started
    print(f"Wrote {bin_path} ({total_bytes / 1e6:.1f} MB in {elapsed:.1f}s)")
    print(f"Wrote {bf16_dir / 'mtp_weights.json'}")


def build_expert_sources(weight_map, prefix):
    sources = defaultdict(lambda: {"stacked": {}, "individual": defaultdict(dict)})
    for name, filename in weight_map.items():
        san = sanitize_name(name)
        if not san.startswith(prefix):
            continue
        match = STACKED_EXPERT_RE.match(san)
        if match:
            layer = int(match.group(1))
            key = "gate_up" if match.group(2) == "gate_up_proj" else "down"
            sources[layer]["stacked"][key] = (name, filename)
            continue
        match = INDIVIDUAL_EXPERT_RE.match(san)
        if match:
            layer = int(match.group(1))
            expert = int(match.group(2))
            sources[layer]["individual"][expert][match.group(3)] = (name, filename)
    return sources


def expert_source_layout(layer_sources, experts_to_write):
    if {"gate_up", "down"} <= set(layer_sources.get("stacked", {})):
        return "stacked"
    individual = layer_sources.get("individual", {})
    required = {"gate_proj", "up_proj", "down_proj"}
    for expert in range(experts_to_write):
        if required - set(individual.get(expert, {})):
            return None
    return "individual"


def compile_routed_experts(model_path, output_dir, weight_map, headers, entry, layers,
                           dry_run, max_experts, prefix, packed_name, label, layout_layers):
    quant = entry.get("quantization", {})
    bits = quant.get("bits", 4)
    group_size = quant.get("group_size", 64)
    components = build_components(entry)
    comp_by_name = {c["name"]: c for c in components}
    expert_size = sum(c["size"] for c in components)
    num_experts = entry["num_experts"]
    experts_to_write = min(num_experts, max_experts) if max_experts else num_experts
    layer_size = expert_size * num_experts
    sources = build_expert_sources(weight_map, prefix)

    print(f"{label} routed expert layers planned: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Expert size: {expert_size:,} bytes")
    print(f"Layer size: {layer_size / (1024 ** 3):.2f} GB")
    if experts_to_write != num_experts:
        print(f"Smoke mode: writing first {experts_to_write} of {num_experts} experts")
    if dry_run:
        missing = [layer for layer in layers if not expert_source_layout(sources.get(layer, {}), experts_to_write)]
        if missing:
            raise RuntimeError(f"missing routed expert tensors for layers: {missing[:8]}")
        planned = len(layers) * experts_to_write * expert_size
        print(f"Dry run: would write {planned / (1024 ** 3):.3f} GB of packed experts")
        return

    packed_dir = output_dir / packed_name
    packed_dir.mkdir(parents=True, exist_ok=True)
    with open(packed_dir / "layout.json", "w") as f:
        json.dump({
            "expert_size": expert_size,
            "num_layers": layout_layers,
            "num_experts": num_experts,
            "artifact": packed_name,
            "components": components,
        }, f, indent=2)

    for layer in layers:
        layer_sources = sources.get(layer, {})
        layout = expert_source_layout(layer_sources, experts_to_write)
        if not layout:
            raise RuntimeError(f"missing routed expert tensors for layer {layer}")
        out_path = packed_dir / f"layer_{layer:02d}.bin"
        fd = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, layer_size)
        started = time.time()

        if layout == "stacked":
            gate_up_name, gate_up_file = layer_sources["stacked"]["gate_up"]
            down_name, down_file = layer_sources["stacked"]["down"]
            gate_up_header, gate_up_start = headers[gate_up_file]
            down_header, down_start = headers[down_file]

        for expert in range(experts_to_write):
            if layout == "stacked":
                gate_up = read_expert_bf16(model_path, gate_up_file, gate_up_header, gate_up_start, gate_up_name, expert)
                gate, up = split_qwen_gate_up_proj(gate_up)
                down = read_expert_bf16(model_path, down_file, down_header, down_start, down_name, expert)
                matrices = {
                    "gate_proj": gate,
                    "up_proj": up,
                    "down_proj": down,
                }
            else:
                matrices = {}
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensor_name, tensor_file = layer_sources["individual"][expert][proj]
                    tensor_header, tensor_start = headers[tensor_file]
                    matrices[proj] = read_tensor_bf16(
                        model_path, tensor_file, tensor_header, tensor_start, tensor_name
                    )

            for proj, matrix in matrices.items():
                for name, data, _, _ in quantize_matrix_to_entries(f"{proj}.weight", matrix, bits, group_size):
                    comp_name = name
                    comp = comp_by_name[comp_name]
                    dst = expert * expert_size + comp["offset"]
                    os.pwrite(fd, data, dst)

        os.close(fd)
        elapsed = time.time() - started
        print(f"  Layer {layer:2d}: wrote {out_path} in {elapsed:.1f}s")


def load_headers(model_path, weight_map):
    by_file = defaultdict(list)
    for name, filename in weight_map.items():
        by_file[filename].append(name)
    headers = {}
    print(f"Parsing {len(by_file)} safetensors headers...")
    for filename in sorted(by_file):
        headers[filename] = parse_safetensors_header(model_path / filename)
    return headers


def main():
    parser = argparse.ArgumentParser(description="Compile native Qwen BF16 checkpoints for Flashchat")
    parser.add_argument("--model-id", required=True, help="Registry model id to use for dimensions")
    parser.add_argument("--model", required=True, help="Path to native BF16 model snapshot")
    parser.add_argument("--output", default=None, help="Output directory (default: MODEL/flashchat/q<bits>)")
    parser.add_argument("--layers", default=None, help='Expert layers to compile, e.g. "0", "0-3", "all"')
    parser.add_argument("--non-experts", action="store_true", help="Compile non-expert model_weights artifacts")
    parser.add_argument("--experts", action="store_true", help="Compile routed expert packed layer artifacts")
    parser.add_argument("--mtp-experts", action="store_true", help="Compile MTP routed expert packed layer artifacts")
    parser.add_argument("--include-mtp", action="store_true", help="Include MTP tensors in model_weights artifacts")
    parser.add_argument("--bf16-mtp", action="store_true", help="Extract MTP head weights in native BF16 to a separate flashchat/bf16/ directory (kept alongside quantized weights for research/fallback)")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan without writing artifacts")
    parser.add_argument("--limit-tensors", type=int, default=None, help="Only compile the first N non-expert tensors")
    parser.add_argument("--name-regex", default=None, help="Only compile non-expert tensors whose sanitized name matches")
    parser.add_argument("--max-experts", type=int, default=None, help="Only compile the first N routed experts in each selected layer")
    args = parser.parse_args()

    if not args.non_experts and not args.experts and not args.mtp_experts and not args.bf16_mtp:
        args.non_experts = True
        args.experts = True

    entry = get_model_entry(args.model_id)
    model_path = Path(args.model)
    native_config = native_text_config(model_path)
    _bits = int(entry.get("quantization", {}).get("bits", 4) or 4)
    _default_out = model_path / "flashchat" / f"q{_bits}"
    output_dir = Path(args.output) if args.output else _default_out
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Registry model id: {args.model_id}")
    print(f"Total indexed tensors: {len(weight_map)}")

    headers = load_headers(model_path, weight_map)
    if args.non_experts:
        compile_non_experts(model_path, output_dir, weight_map, headers, entry, native_config,
                            args.include_mtp, args.dry_run, args.limit_tensors, args.name_regex)
    if args.bf16_mtp:
        compile_bf16_mtp(model_path, output_dir, weight_map, headers, entry, native_config, args.dry_run)
    if args.experts:
        layers = parse_layers(args.layers, entry["num_hidden_layers"])
        compile_routed_experts(model_path, output_dir, weight_map, headers, entry,
                               layers, args.dry_run, args.max_experts,
                               "model.layers.", "packed_experts", "Backbone",
                               entry["num_hidden_layers"])
    if args.mtp_experts:
        mtp_layers = native_config.get("mtp_num_hidden_layers", entry.get("mtp_num_hidden_layers", 0))
        if not mtp_layers:
            print("ERROR: native config does not declare mtp_num_hidden_layers", file=sys.stderr)
            sys.exit(1)
        layers = parse_layers(args.layers, mtp_layers)
        compile_routed_experts(model_path, output_dir, weight_map, headers, entry,
                               layers, args.dry_run, args.max_experts,
                               "mtp.layers.", "packed_mtp_experts", "MTP",
                               mtp_layers)


if __name__ == "__main__":
    main()
