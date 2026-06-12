"""generate_expert_index step (mlx-quantized models only).

Scans model.safetensors.index.json plus the shard headers to locate every
routed-expert tensor and records absolute file offsets/strides so
repack_experts can read experts straight out of the shards. The output
embeds the absolute snapshot path, so it is variant-local and must be
regenerated when the snapshot moves (offload restore checks this).
"""

import json
import os
import re
import struct
from collections import defaultdict

from . import StepContext, step_version
from ..artifacts import ArtifactDir

EXPECTED_COMPONENTS = {
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight", "up_proj.scales", "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
}

_PATTERNS = (
    re.compile(
        r"(?:^|\.)(?:language_model\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\."
        r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
    ),
    re.compile(
        r"(?:^|\.)(?:language_model\.)?model\.layers\.(\d+)\.switch_mlp\."
        r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
    ),
)


def parse_safetensors_header(filepath: str):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def build_expert_index(model_path: str) -> dict:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    expert_tensors = []
    for name, filename in weight_map.items():
        for pattern in _PATTERNS:
            m = pattern.search(name)
            if m:
                expert_tensors.append(
                    (int(m.group(1)), f"{m.group(2)}.{m.group(3)}", name, filename))
                break
    if not expert_tensors:
        raise RuntimeError(f"no expert tensors found in {index_path}")

    by_layer = defaultdict(list)
    for layer_idx, comp_name, tensor_name, filename in expert_tensors:
        by_layer[layer_idx].append((comp_name, tensor_name, filename))
    for layer_idx in sorted(by_layer):
        missing = EXPECTED_COMPONENTS - {c for c, _, _ in by_layer[layer_idx]}
        if missing:
            raise RuntimeError(f"layer {layer_idx} missing expert components: {missing}")

    header_cache = {}
    for filename in sorted({f for _, _, _, f in expert_tensors}):
        header_cache[filename] = parse_safetensors_header(os.path.join(model_path, filename))

    expert_reads = {}
    for layer_idx in sorted(by_layer):
        reads = expert_reads[str(layer_idx)] = {}
        for comp_name, tensor_name, filename in by_layer[layer_idx]:
            header, data_start = header_cache[filename]
            meta = header[tensor_name]
            offsets = meta["data_offsets"]
            total_size = offsets[1] - offsets[0]
            stride = total_size // meta["shape"][0]
            reads[comp_name] = {
                "file": filename,
                "abs_offset": data_start + offsets[0],
                "expert_stride": stride,
                "expert_size": stride,
                "total_size": total_size,
                "shape": meta["shape"],
            }

    return {"model_path": os.path.realpath(model_path), "expert_reads": expert_reads}


def run(ctx: StepContext, planned=None) -> None:
    adir = ArtifactDir(ctx.variant_dir, ctx.manifest.id, ctx.variant_name)
    ctx.report("generate_expert_index", 0, 1, "scanning safetensors headers")
    if ctx.dry_run:
        return
    index = build_expert_index(ctx.snapshot)
    with adir.open("expert_index.json", step="generate_expert_index",
                   step_version=step_version("generate_expert_index")) as sink:
        sink.write(json.dumps(index, indent=2).encode())
    adir.commit()
    ctx.report("generate_expert_index", 1, 1,
               f"{len(index['expert_reads'])} layers indexed")


def get_runner(name: str):
    return run
