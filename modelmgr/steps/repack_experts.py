"""repack_experts step (mlx-quantized models): shards -> per-layer packs.

Reads expert tensors straight out of the safetensors shards using the
offsets recorded in expert_index.json and writes one contiguous binary per
layer (expert-major, 9 components in fixed order). Reads are sorted by
source offset for sequential SSD throughput, so writes land scattered via
pwrite; each finished layer is hashed immediately afterwards while still
page-cache warm.
"""

import errno
import json
import os
import time

from . import StepContext, step_version
from ..artifacts import ArtifactDir


def build_components(entry):
    """Deterministic 9-component expert layout from model dimensions."""
    hidden = entry["hidden_size"]
    moe_int = entry["moe_intermediate_size"]
    quant = entry.get("quantization", {})
    gs = quant.get("group_size", 64)
    bits = quant.get("bits", 4)
    if bits not in (4, 8):
        raise ValueError(f"unsupported quantization bits={bits}")
    values_per_word = 32 // bits

    gate_up_wsize = moe_int * (hidden // values_per_word) * 4
    gate_up_ssize = moe_int * (hidden // gs) * 2
    down_wsize = hidden * (moe_int // values_per_word) * 4
    down_ssize = hidden * (moe_int // gs) * 2

    offset = 0
    components = []
    for name, size, out_d, in_d in [
        ("gate_proj.weight", gate_up_wsize, moe_int, hidden // values_per_word),
        ("gate_proj.scales", gate_up_ssize, moe_int, hidden // gs),
        ("gate_proj.biases", gate_up_ssize, moe_int, hidden // gs),
        ("up_proj.weight",   gate_up_wsize, moe_int, hidden // values_per_word),
        ("up_proj.scales",   gate_up_ssize, moe_int, hidden // gs),
        ("up_proj.biases",   gate_up_ssize, moe_int, hidden // gs),
        ("down_proj.weight", down_wsize,   hidden,  moe_int // values_per_word),
        ("down_proj.scales", down_ssize,   hidden,  moe_int // gs),
        ("down_proj.biases", down_ssize,   hidden,  moe_int // gs),
    ]:
        components.append({
            "name": name, "offset": offset, "size": size,
            "dtype": "U32" if name.endswith(".weight") else "BF16",
            "shape": [out_d, in_d],
        })
        offset += size
    return components


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
    model_path = os.path.abspath(os.path.expanduser(idx["model_path"]))
    return idx["expert_reads"], model_path


def verify_component_sizes(expert_reads, components):
    expected = {c["name"]: c["size"] for c in components}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                continue
            if info["expert_size"] != expected[comp_name]:
                return False
    return True


def open_source_files(expert_reads, model_path, layers):
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info["file"])
    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        fds[fname] = os.open(path, os.O_RDONLY)
    return fds


def repack_layer(layer_idx, expert_reads, fds, output_dir, components,
                 expert_size, num_experts, layer_size, dry_run=False):
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        return 0, 0.0
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        return layer_size, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    read_plan = []
    for expert_idx in range(num_experts):
        for comp in components:
            info = layer_info[comp["name"]]
            read_plan.append((
                fds[info["file"]],
                info["abs_offset"] + expert_idx * info["expert_stride"],
                expert_idx * expert_size + comp["offset"],
                comp["size"],
            ))
    read_plan.sort(key=lambda x: (x[0], x[1]))

    bytes_written = 0
    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"short read: expected {size}, got {len(data)} at {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    return bytes_written, time.monotonic() - t0


def verify_layer(layer_idx, expert_reads, fds, output_dir, components,
                 expert_size, num_experts):
    """Spot-check experts 0, 1 and last against the source shards."""
    layer_info = expert_reads[str(layer_idx)]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    if not os.path.exists(out_path):
        return False
    fd_packed = os.open(out_path, os.O_RDONLY)
    mismatches = 0
    for expert_idx in [0, 1, num_experts - 1]:
        for comp in components:
            info = layer_info[comp["name"]]
            src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
            dst_offset = expert_idx * expert_size + comp["offset"]
            original = os.pread(fds[info["file"]], comp["size"], src_offset)
            packed = os.pread(fd_packed, comp["size"], dst_offset)
            if original != packed:
                mismatches += 1
    os.close(fd_packed)
    return mismatches == 0


def repack(entry: dict, index_path: str, packed_dir: str, layers=None,
           dry_run: bool = False, progress=None, on_layer_complete=None) -> None:
    components = build_components(entry)
    expert_size = sum(c["size"] for c in components)
    num_experts = entry["num_experts"]
    num_layers = entry["num_hidden_layers"]
    layer_size = num_experts * expert_size
    layers = layers if layers is not None else list(range(num_layers))

    expert_reads, model_path = load_index(index_path)
    if not verify_component_sizes(expert_reads, components):
        raise RuntimeError("expert index component sizes do not match expected layout")

    if not dry_run:
        os.makedirs(packed_dir, exist_ok=True)
        total = len(layers) * layer_size
        stat = os.statvfs(packed_dir)
        if stat.f_bavail * stat.f_frsize < total:
            raise RuntimeError(
                f"not enough free space: need {total / 1024**3:.1f} GiB "
                f"in {packed_dir}")
        with open(os.path.join(packed_dir, "layout.json"), "w") as f:
            json.dump({
                "expert_size": expert_size,
                "num_layers": num_layers,
                "num_experts": num_experts,
                "components": components,
            }, f, indent=2)

    fds = open_source_files(expert_reads, model_path, layers)
    try:
        for i, layer_idx in enumerate(layers):
            bytes_written, elapsed = repack_layer(
                layer_idx, expert_reads, fds, packed_dir, components,
                expert_size, num_experts, layer_size, dry_run=dry_run)
            if not dry_run and bytes_written > 0:
                if not verify_layer(layer_idx, expert_reads, fds, packed_dir,
                                    components, expert_size, num_experts):
                    raise RuntimeError(f"verification failed for layer {layer_idx}")
                if on_layer_complete:
                    on_layer_complete(f"layer_{layer_idx:02d}.bin")
            if progress:
                progress("repack_experts", i + 1, len(layers),
                         f"layer {layer_idx} ({bytes_written / 1024**3:.2f} GiB, {elapsed:.1f}s)")
    finally:
        for fd in fds.values():
            os.close(fd)


def run(ctx: StepContext, planned=None) -> None:
    from ..resolved import flat_entry

    entry = flat_entry(ctx.manifest, ctx.variant_name)
    adir = ArtifactDir(ctx.variant_dir, ctx.manifest.id, ctx.variant_name)
    index_path = os.path.join(ctx.variant_dir, "expert_index.json")
    packed_dir = os.path.join(ctx.variant_dir, "packed_experts")
    version = step_version("repack_experts")

    def on_layer_complete(filename):
        # The layer was just written and spot-verified; its pages are warm,
        # so the hashing re-read costs RAM bandwidth, not SSD time.
        adir.backfill(f"packed_experts/{filename}", step="repack_experts",
                      step_version=version)
        adir.commit()

    adir.forget("packed_experts/")
    repack(entry, index_path, packed_dir, dry_run=ctx.dry_run,
           progress=ctx.progress, on_layer_complete=on_layer_complete)
    if not ctx.dry_run:
        adir.backfill("packed_experts/layout.json", step="repack_experts",
                      step_version=version)
        adir.commit()


def get_runner(name: str):
    return run
