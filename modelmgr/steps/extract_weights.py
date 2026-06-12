"""extract_weights step (mlx-quantized models): non-expert weights -> blob.

Reads the quantized safetensors shards and writes model_weights.bin (one
64-byte-aligned tensor after another) plus the model_weights.json manifest
the engine mmaps against. 8-bit per-tensor overrides in the source
config.json are converted to the target bits when extracting a 4-bit
variant.
"""

import json
import os
import re
import struct
from collections import defaultdict

from . import StepContext, step_version
from ..artifacts import ArtifactDir

ALIGN = 64

EXPERT_PATTERN = re.compile(
    r"\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")
VISION_PATTERN = re.compile(r"^(vision_tower|model\.visual)")


def parse_safetensors_header(filepath):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def sanitize_name(name):
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def load_8bit_tensor_overrides(config_path):
    overrides = {}
    if not os.path.exists(config_path):
        return overrides
    with open(config_path) as f:
        config = json.load(f)
    quant_config = config.get("quantization") or config.get("quantization_config", {})
    for key, qinfo in quant_config.items():
        if isinstance(qinfo, dict) and qinfo.get("bits") == 8:
            if key.startswith("language_model."):
                key = key[len("language_model."):]
            overrides[key] = qinfo
    return overrides


def build_manifest_config(entry):
    interval = entry["full_attention_interval"]
    num_layers = entry["num_hidden_layers"]
    quant = entry.get("quantization", {})
    config = {
        "hidden_size": entry["hidden_size"],
        "num_hidden_layers": num_layers,
        "num_attention_heads": entry["num_attention_heads"],
        "num_key_value_heads": entry["num_key_value_heads"],
        "head_dim": entry["head_dim"],
        "vocab_size": entry["vocab_size"],
        "rms_norm_eps": entry["rms_norm_eps"],
        "num_experts": entry["num_experts"],
        "num_experts_per_tok": entry["num_experts_per_tok"],
        "moe_intermediate_size": entry["moe_intermediate_size"],
        "shared_expert_intermediate_size": entry["shared_expert_intermediate_size"],
        "full_attention_interval": interval,
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
    config["layer_types"] = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]
    return config


class _Aligner:
    """Tracks the output offset and pads to ALIGN before each tensor."""

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


def extract_non_expert_weights(entry: dict, model_path: str, sink, progress=None,
                               include_experts: bool = False) -> dict:
    """Stream all non-expert tensors into `sink`; returns the engine manifest."""
    import numpy as np

    from ..quant import convert_8bit_to_4bit

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    tensors_to_extract = {}
    for name, filename in weight_map.items():
        if VISION_PATTERN.match(name):
            continue
        if not include_experts and EXPERT_PATTERN.search(name):
            continue
        tensors_to_extract[name] = filename

    header_cache = {}
    for filename in sorted(set(tensors_to_extract.values())):
        header_cache[filename] = parse_safetensors_header(os.path.join(model_path, filename))

    all_tensors = [(sanitize_name(name), name, tensors_to_extract[name])
                   for name in sorted(tensors_to_extract)]

    quant = entry.get("quantization", {})
    target_bits = quant.get("bits", 4)
    group_size = quant.get("group_size", 64)
    if target_bits not in (4, 8):
        raise ValueError(f"unsupported quantization bits={target_bits}")

    eight_bit_overrides = load_8bit_tensor_overrides(os.path.join(model_path, "config.json"))

    manifest = {
        "model": str(model_path),
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": build_manifest_config(entry),
    }

    out = _Aligner(sink)
    skipped_8bit_components = set()

    def record(san_name, data, shape, dtype):
        start = out.write_tensor(data)
        manifest["tensors"][san_name] = {
            "offset": start, "size": len(data), "shape": shape, "dtype": dtype,
        }

    for i, (san_name, orig_name, filename) in enumerate(all_tensors):
        if orig_name in skipped_8bit_components:
            continue
        filepath = os.path.join(model_path, filename)
        header, data_start = header_cache[filename]
        if orig_name not in header:
            continue
        meta = header[orig_name]
        tensor_offsets = meta["data_offsets"]
        byte_len = tensor_offsets[1] - tensor_offsets[0]
        shape = meta["shape"]
        dtype = meta["dtype"]

        is_8bit_prefix = None
        for prefix in eight_bit_overrides:
            if san_name.startswith(prefix + "."):
                is_8bit_prefix = prefix
                break

        if is_8bit_prefix and orig_name.endswith(".weight"):
            base_name = orig_name[: -len(".weight")]
            scales_name = base_name + ".scales"
            biases_name = base_name + ".biases"
            scales_meta = header[scales_name]
            biases_meta = header[biases_name]

            with open(filepath, "rb") as sf:
                sf.seek(data_start + tensor_offsets[0])
                weight_data = sf.read(byte_len)
                sf.seek(data_start + scales_meta["data_offsets"][0])
                scales_data = sf.read(
                    scales_meta["data_offsets"][1] - scales_meta["data_offsets"][0])
                sf.seek(data_start + biases_meta["data_offsets"][0])
                biases_data = sf.read(
                    biases_meta["data_offsets"][1] - biases_meta["data_offsets"][0])

            out_dim = shape[0]
            in_dim = shape[1] * 4
            packed_cols = shape[1]
            if target_bits == 4:
                weight_u32 = np.frombuffer(weight_data, dtype=np.uint32).reshape(shape)
                new_w, new_s, new_b = convert_8bit_to_4bit(
                    weight_u32, scales_data, biases_data, out_dim, in_dim,
                    group_size=group_size)
                weight_data = new_w.tobytes()
                scales_data = new_s.tobytes()
                biases_data = new_b.tobytes()
                packed_cols = in_dim // 8

            record(san_name, weight_data, [out_dim, packed_cols], dtype)
            record(sanitize_name(scales_name), scales_data,
                   scales_meta["shape"], scales_meta["dtype"])
            record(sanitize_name(biases_name), biases_data,
                   biases_meta["shape"], biases_meta["dtype"])
            skipped_8bit_components.add(scales_name)
            skipped_8bit_components.add(biases_name)
            continue

        with open(filepath, "rb") as sf:
            sf.seek(data_start + tensor_offsets[0])
            data = sf.read(byte_len)
        record(san_name, data, shape, dtype)

        if progress and ((i + 1) % 100 == 0 or i == len(all_tensors) - 1):
            progress("extract_weights", i + 1, len(all_tensors),
                     f"{out.offset / 1e9:.2f} GB written")

    return manifest


def run(ctx: StepContext, planned=None) -> None:
    from ..resolved import flat_entry

    entry = flat_entry(ctx.manifest, ctx.variant_name)
    adir = ArtifactDir(ctx.variant_dir, ctx.manifest.id, ctx.variant_name)
    if ctx.dry_run:
        ctx.report("extract_weights", 0, 1, "dry run")
        return
    version = step_version("extract_weights")
    with adir.open("model_weights.bin", step="extract_weights", step_version=version) as sink:
        manifest = extract_non_expert_weights(entry, ctx.snapshot, sink, progress=ctx.progress)
    with adir.open("model_weights.json", step="extract_weights", step_version=version) as sink:
        sink.write(json.dumps(manifest, indent=2).encode())
    adir.commit()


def get_runner(name: str):
    return run
