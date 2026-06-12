"""compile_native steps: native BF16 Qwen checkpoints -> runtime artifacts.

Four sub-steps share the safetensors loading and quantization machinery:
  compile_native:non_experts  model_weights.bin/.json (quantized, +MTP tensors)
  compile_native:experts      packed_experts/ per-layer packs
  compile_native:mtp_experts  packed_mtp_experts/ per-layer packs
  compile_native:bf16_mtp     shared/bf16/mtp_weights.bin/.json (unquantized)

The sequential writers stream through hash-on-write sinks; the expert
packers write via pwrite (expert-major destination layout) and hash each
finished layer while page-cache warm.
"""

import json
import os
import re
import struct
import time
from collections import defaultdict

import numpy as np

from . import StepContext, step_version
from ..artifacts import ArtifactDir
from ..quant import (
    bf16_to_f32,
    f32_to_bf16,
    quantize_f32_to_4bit_affine_rows,
    quantize_f32_to_8bit_affine_rows,
    split_qwen_gate_up_proj,
)
from .repack_experts import build_components, parse_layers

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
    with open(os.path.join(model_path, filename), "rb") as f:
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
    with open(os.path.join(model_path, filename), "rb") as f:
        f.seek(data_start + start + expert_idx * stride)
        data = f.read(stride)
    if len(data) != stride:
        raise IOError(f"short read for {name} expert {expert_idx}")
    return bf16_to_f32(data).reshape(shape[1:])


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
        (name[:-len(".weight")] + ".scales", scales.tobytes(),
         [matrix.shape[0], in_dim // group_size], "BF16"),
        (name[:-len(".weight")] + ".biases", biases.tobytes(),
         [matrix.shape[0], in_dim // group_size], "BF16"),
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
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
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


class _Aligner:
    def __init__(self, sink):
        self.sink = sink
        self.offset = 0

    def write_tensor(self, data) -> int:
        if self.offset % ALIGN != 0:
            pad = ALIGN - (self.offset % ALIGN)
            self.sink.write(b"\x00" * pad)
            self.offset += pad
        start = self.offset
        self.sink.write(data)
        self.offset += len(data)
        return start


def compile_non_experts(model_path, weight_map, headers, entry, native_config,
                        include_mtp, sink, dry_run=False, limit=None,
                        name_regex=None, progress=None):
    """Quantize/copy all non-expert tensors into `sink`; returns the manifest."""
    quant = entry.get("quantization", {})
    bits = quant.get("bits", 4)
    group_size = quant.get("group_size", 64)
    tensors, _skipped_experts, _skipped_mtp = planned_non_expert_tensors(
        weight_map, headers, group_size, include_mtp)
    if name_regex:
        pattern = re.compile(name_regex)
        tensors = [t for t in tensors if pattern.search(t[0])]
    if limit is not None:
        tensors = tensors[:limit]

    manifest = {
        "model": str(model_path),
        "num_tensors": 0,
        "tensors": {},
        "config": build_manifest_config(entry, native_config),
    }
    if dry_run:
        return manifest

    out = _Aligner(sink)
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
            start = out.write_tensor(data)
            manifest["tensors"][out_name] = {
                "offset": start, "size": len(data), "shape": shape, "dtype": dtype,
            }
        if progress and (idx % 25 == 0 or idx == len(tensors)):
            progress("compile_native:non_experts", idx, len(tensors),
                     f"{out.offset / 1e9:.2f} GB written")

    manifest["num_tensors"] = len(manifest["tensors"])
    return manifest


def compile_bf16_mtp(model_path, weight_map, headers, entry, native_config,
                     sink, dry_run=False, progress=None):
    """Extract MTP head weights in native BF16; returns the manifest.

    Keeps the BF16 predictor path alive alongside the quantized path so the
    two can be A/B'd without re-downloading safetensors. Variant-independent
    -> lives in shared/bf16/.
    """
    manifest = {
        "model": str(model_path),
        "num_tensors": 0,
        "tensors": {},
        "config": build_manifest_config(entry, native_config),
    }
    mtp_tensors = []
    for name, filename in weight_map.items():
        san = sanitize_name(name)
        if san.startswith("mtp.") and not is_routed_expert(name):
            mtp_tensors.append((san, name, filename))
    mtp_tensors.sort()
    if dry_run:
        return manifest

    out = _Aligner(sink)
    for idx, (san, orig, filename) in enumerate(mtp_tensors, 1):
        header, data_start = headers[filename]
        meta = header[orig]
        raw = read_tensor_raw(model_path, filename, header, data_start, orig)
        if should_shift_native_norm(san, meta):
            raw = shift_native_norm_data(raw)
        start = out.write_tensor(raw)
        manifest["tensors"][san] = {
            "offset": start, "size": len(raw), "shape": meta["shape"], "dtype": meta["dtype"],
        }
        if progress and (idx % 25 == 0 or idx == len(mtp_tensors)):
            progress("compile_native:bf16_mtp", idx, len(mtp_tensors),
                     f"{out.offset / 1e6:.1f} MB written")

    manifest["num_tensors"] = len(manifest["tensors"])
    return manifest


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


def compile_routed_experts(model_path, packed_dir, weight_map, headers, entry, layers,
                           prefix, packed_name, layout_layers, dry_run=False,
                           max_experts=None, progress=None, on_layer_complete=None):
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

    if dry_run:
        missing = [layer for layer in layers
                   if not expert_source_layout(sources.get(layer, {}), experts_to_write)]
        if missing:
            raise RuntimeError(f"missing routed expert tensors for layers: {missing[:8]}")
        return

    os.makedirs(packed_dir, exist_ok=True)
    with open(os.path.join(packed_dir, "layout.json"), "w") as f:
        json.dump({
            "expert_size": expert_size,
            "num_layers": layout_layers,
            "num_experts": num_experts,
            "artifact": packed_name,
            "components": components,
        }, f, indent=2)

    for i, layer in enumerate(layers):
        layer_sources = sources.get(layer, {})
        layout = expert_source_layout(layer_sources, experts_to_write)
        if not layout:
            raise RuntimeError(f"missing routed expert tensors for layer {layer}")
        out_path = os.path.join(packed_dir, f"layer_{layer:02d}.bin")
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
                gate_up = read_expert_bf16(model_path, gate_up_file, gate_up_header,
                                           gate_up_start, gate_up_name, expert)
                gate, up = split_qwen_gate_up_proj(gate_up)
                down = read_expert_bf16(model_path, down_file, down_header,
                                        down_start, down_name, expert)
                matrices = {"gate_proj": gate, "up_proj": up, "down_proj": down}
            else:
                matrices = {}
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    tensor_name, tensor_file = layer_sources["individual"][expert][proj]
                    tensor_header, tensor_start = headers[tensor_file]
                    matrices[proj] = read_tensor_bf16(
                        model_path, tensor_file, tensor_header, tensor_start, tensor_name)

            for proj, matrix in matrices.items():
                for name, data, _, _ in quantize_matrix_to_entries(
                        f"{proj}.weight", matrix, bits, group_size):
                    comp = comp_by_name[name]
                    os.pwrite(fd, data, expert * expert_size + comp["offset"])

        os.close(fd)
        if on_layer_complete:
            on_layer_complete(f"layer_{layer:02d}.bin")
        if progress:
            progress(f"compile_native:{packed_name}", i + 1, len(layers),
                     f"layer {layer} ({time.time() - started:.1f}s)")


def load_headers(model_path, weight_map):
    headers = {}
    for filename in sorted(set(weight_map.values())):
        headers[filename] = parse_safetensors_header(os.path.join(model_path, filename))
    return headers


def load_weight_map(model_path):
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        return json.load(f)["weight_map"]


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _prepare(ctx: StepContext):
    from ..resolved import flat_entry

    entry = flat_entry(ctx.manifest, ctx.variant_name or next(iter(ctx.manifest.variants)))
    weight_map = load_weight_map(ctx.snapshot)
    headers = load_headers(ctx.snapshot, weight_map)
    native_config = native_text_config(ctx.snapshot)
    return entry, weight_map, headers, native_config


def _run_non_experts(ctx: StepContext, planned=None) -> None:
    entry, weight_map, headers, native_config = _prepare(ctx)
    adir = ArtifactDir(ctx.variant_dir, ctx.manifest.id, ctx.variant_name)
    include_mtp = ctx.manifest.mtp_artifacts_required
    version = step_version("compile_native:non_experts")
    if ctx.dry_run:
        compile_non_experts(ctx.snapshot, weight_map, headers, entry, native_config,
                            include_mtp, sink=None, dry_run=True)
        return
    with adir.open("model_weights.bin", step="compile_native:non_experts",
                   step_version=version) as sink:
        manifest = compile_non_experts(ctx.snapshot, weight_map, headers, entry,
                                       native_config, include_mtp, sink,
                                       progress=ctx.progress)
    with adir.open("model_weights.json", step="compile_native:non_experts",
                   step_version=version) as sink:
        sink.write(json.dumps(manifest, indent=2).encode())
    adir.commit()


def _run_bf16_mtp(ctx: StepContext, planned=None) -> None:
    entry, weight_map, headers, native_config = _prepare(ctx)
    adir = ArtifactDir(ctx.shared_dir, ctx.manifest.id, "shared")
    version = step_version("compile_native:bf16_mtp")
    if ctx.dry_run:
        return
    with adir.open("bf16/mtp_weights.bin", step="compile_native:bf16_mtp",
                   step_version=version) as sink:
        manifest = compile_bf16_mtp(ctx.snapshot, weight_map, headers, entry,
                                    native_config, sink, progress=ctx.progress)
    with adir.open("bf16/mtp_weights.json", step="compile_native:bf16_mtp",
                   step_version=version) as sink:
        sink.write(json.dumps(manifest, indent=2).encode())
    adir.commit()


def _run_experts(ctx: StepContext, packed_name: str, prefix: str,
                 layout_layers: int, step_name: str) -> None:
    entry, weight_map, headers, _native_config = _prepare(ctx)
    adir = ArtifactDir(ctx.variant_dir, ctx.manifest.id, ctx.variant_name)
    packed_dir = os.path.join(ctx.variant_dir, packed_name)
    version = step_version(step_name)
    layers = list(range(layout_layers))

    def on_layer_complete(filename):
        # Just written + page-cache warm: hashing re-reads RAM, not SSD.
        adir.backfill(f"{packed_name}/{filename}", step=step_name, step_version=version)
        adir.commit()

    if not ctx.dry_run:
        adir.forget(packed_name + "/")
    compile_routed_experts(ctx.snapshot, packed_dir, weight_map, headers, entry,
                           layers, prefix, packed_name, layout_layers,
                           dry_run=ctx.dry_run, progress=ctx.progress,
                           on_layer_complete=None if ctx.dry_run else on_layer_complete)
    if not ctx.dry_run:
        adir.backfill(f"{packed_name}/layout.json", step=step_name, step_version=version)
        adir.commit()


def _run_backbone_experts(ctx: StepContext, planned=None) -> None:
    _run_experts(ctx, "packed_experts", "model.layers.",
                 int(ctx.manifest.architecture["num_hidden_layers"]),
                 "compile_native:experts")


def _run_mtp_experts(ctx: StepContext, planned=None) -> None:
    mtp_layers = int((ctx.manifest.mtp or {}).get("num_hidden_layers", 0))
    if not mtp_layers:
        raise RuntimeError(f"model {ctx.manifest.id} declares no MTP layers")
    _run_experts(ctx, "packed_mtp_experts", "mtp.layers.", mtp_layers,
                 "compile_native:mtp_experts")


_RUNNERS = {
    "compile_native:non_experts": _run_non_experts,
    "compile_native:experts": _run_backbone_experts,
    "compile_native:mtp_experts": _run_mtp_experts,
    "compile_native:bf16_mtp": _run_bf16_mtp,
}


def get_runner(name: str):
    return _RUNNERS[name]
