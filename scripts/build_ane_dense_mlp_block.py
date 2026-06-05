#!/usr/bin/env python3
"""Build an experimental private-ANE dense MLP block artifact."""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_ane_dense_projection import GROUP_SIZE, compile_model, import_coremltools, read_matrix
from compile_native_qwen import sanitize_name
from repack_experts import get_model_entry

EXPANSION_CHUNK = 8704


def artifact_name_for(layer, tokens, hidden_dim, intermediate_dim):
    return f"model_layers_{layer}_mlp_block_n{tokens}_h{hidden_dim}_i{intermediate_dim}_g64_uint4"


def default_output_dir(model_path, model_id, artifact_name):
    bits = 4
    if model_id:
        entry = get_model_entry(model_id)
        bits = int(entry.get("quantization", {}).get("bits", 4) or 4)
        if bits != 4:
            raise SystemExit("ERROR: the ANE dense MLP block probe currently builds q4 artifacts only")
    return model_path / "flashchat" / f"q{bits}" / "ane_dense" / artifact_name


def grouped_linear(mb, x, matrix16, tokens, group_size, prefix):
    out_dim, in_dim = matrix16.shape
    groups = in_dim // group_size
    acc = None
    for group in range(groups):
        col_start = group * group_size
        col_end = col_start + group_size
        xg = mb.slice_by_index(
            x=x,
            begin=[0, col_start],
            end=[tokens, col_end],
            name=f"{prefix}_x_g{group}",
        )
        yg = mb.linear(
            x=xg,
            weight=matrix16[:, col_start:col_end],
            name=f"{prefix}_linear_g{group}",
        )
        acc = yg if acc is None else mb.add(x=acc, y=yg, name=f"{prefix}_add_g{group}")
    return acc


def grouped_linear_rows(mb, x, matrix16, tokens, row_start, row_end, group_size, prefix):
    return grouped_linear(mb, x, matrix16[row_start:row_end, :], tokens, group_size, prefix)


def grouped_down_from_chunks(mb, act_chunks, down16, tokens, group_size):
    out_dim, in_dim = down16.shape
    groups = in_dim // group_size
    acc = None
    for group in range(groups):
        col_start = group * group_size
        col_end = col_start + group_size
        source = None
        local_start = 0
        for chunk_start, chunk_end, chunk_value in act_chunks:
            if col_start >= chunk_start and col_end <= chunk_end:
                source = chunk_value
                local_start = col_start - chunk_start
                break
        if source is None:
            raise ValueError(f"down group [{col_start},{col_end}) crosses an expansion chunk")
        xg = mb.slice_by_index(
            x=source,
            begin=[0, local_start],
            end=[tokens, local_start + group_size],
            name=f"down_x_g{group}",
        )
        yg = mb.linear(
            x=xg,
            weight=down16[:, col_start:col_end],
            name=f"down_linear_g{group}",
        )
        acc = yg if acc is None else mb.add(x=acc, y=yg, name=f"down_add_g{group}")
    return acc


def build_program(ct, mb, types, gate, up, down, tokens):
    hidden_dim = gate.shape[1]
    gate16 = np.asarray(gate, dtype=np.float16)
    up16 = np.asarray(up, dtype=np.float16)
    down16 = np.asarray(down, dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(tokens, hidden_dim), dtype=types.fp32)],
                opset_version=ct.target.iOS18)
    def program(x):
        x16 = mb.cast(x=x, dtype="fp16", name="x_fp16")
        act_chunks = []
        for row_start in range(0, gate16.shape[0], EXPANSION_CHUNK):
            row_end = min(row_start + EXPANSION_CHUNK, gate16.shape[0])
            tag = f"{row_start}_{row_end}"
            gate_out = grouped_linear_rows(mb, x16, gate16, tokens, row_start, row_end, GROUP_SIZE, f"gate_{tag}")
            up_out = grouped_linear_rows(mb, x16, up16, tokens, row_start, row_end, GROUP_SIZE, f"up_{tag}")
            sig = mb.sigmoid(x=gate_out, name=f"gate_sigmoid_{tag}")
            silu = mb.mul(x=gate_out, y=sig, name=f"gate_silu_{tag}")
            act = mb.mul(x=silu, y=up_out, name=f"swiglu_{tag}")
            act_chunks.append((row_start, row_end, act))
        out = grouped_down_from_chunks(mb, act_chunks, down16, tokens, GROUP_SIZE)
        return mb.cast(x=out, dtype="fp32", name="y")

    return program


def write_reference(qmodel, output_name, output_dir, tokens, hidden_dim, seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((tokens, hidden_dim), dtype=np.float32) * 0.25).astype(np.float32)
    pred = qmodel.predict({"x": x})
    y = np.asarray(pred[output_name], dtype=np.float32)
    x.tofile(output_dir / "input_f32.bin")
    y.tofile(output_dir / "reference_f32.bin")
    return [{
        "name": output_name,
        "start": 0,
        "end": int(hidden_dim),
        "dim": int(hidden_dim),
    }]


def main():
    parser = argparse.ArgumentParser(description="Build a q4 group-64 CoreML/ANE dense MLP block probe")
    parser.add_argument("--model", required=True, help="Path to a native BF16 model snapshot")
    parser.add_argument("--model-id", default=None, help="Registry model id, used for default runtime placement")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to compile")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--tokens", type=int, default=8, help="Static prompt chunk size")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic probe input seed")
    parser.add_argument("--keep-package", action="store_true", help="Keep the intermediate .mlpackage")
    args = parser.parse_args()

    ct, mb, types, OptimizationConfig, OpLinearQuantizerConfig = import_coremltools()

    model_path = Path(args.model)
    prefix = f"model.language_model.layers.{args.layer}.mlp"
    gate_name, gate = read_matrix(model_path, f"{prefix}.gate_proj.weight")
    up_name, up = read_matrix(model_path, f"{prefix}.up_proj.weight")
    down_name, down = read_matrix(model_path, f"{prefix}.down_proj.weight")

    if gate.shape != up.shape:
        raise SystemExit(f"ERROR: gate shape {gate.shape} != up shape {up.shape}")
    if down.shape[1] != gate.shape[0] or down.shape[0] != gate.shape[1]:
        raise SystemExit(f"ERROR: incompatible down shape {down.shape} for gate/up {gate.shape}")
    if down.shape[1] % GROUP_SIZE != 0:
        raise SystemExit(f"ERROR: intermediate dim {down.shape[1]} is not divisible by {GROUP_SIZE}")

    intermediate_dim, hidden_dim = gate.shape
    artifact_name = artifact_name_for(args.layer, args.tokens, hidden_dim, intermediate_dim)
    output_dir = Path(args.output) if args.output else default_output_dir(model_path, args.model_id, artifact_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Layer: {args.layer}")
    print(f"Hidden: {hidden_dim}")
    print(f"Intermediate: {intermediate_dim}")
    print(f"Output: {output_dir}")

    started = time.time()
    program = build_program(ct, mb, types, gate, up, down, args.tokens)
    mlmodel = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear",
            dtype=types.uint4,
            granularity="per_channel",
            weight_threshold=0,
        )
    )
    qmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)

    package_dir = output_dir / f"{artifact_name}.mlpackage"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    qmodel.save(str(package_dir))

    output_name = qmodel._spec.description.output[0].name
    outputs = write_reference(qmodel, output_name, output_dir, args.tokens, hidden_dim, args.seed)
    modelc_dir = compile_model(package_dir, output_dir, artifact_name)
    if not args.keep_package:
        shutil.rmtree(package_dir, ignore_errors=True)

    manifest = {
        "format_version": 1,
        "kind": "private_ane_dense_mlp_block_probe",
        "model_id": args.model_id or "",
        "model_path": str(model_path),
        "source_tensor": f"{prefix}.block",
        "sanitized_tensor": f"model.layers.{args.layer}.mlp.block",
        "source_tensors": [gate_name, up_name, down_name],
        "tokens": args.tokens,
        "input_dim": int(hidden_dim),
        "intermediate_dim": int(intermediate_dim),
        "output_dim": int(hidden_dim),
        "group_size": GROUP_SIZE,
        "quantization": {
            "bits": 4,
            "style": "coreml_linear_zero_point_group64_uint4",
            "note": "Experimental fused ANE MLP block, not bit-exact to Flashchat MLX affine q4.",
        },
        "mlmodelc": modelc_dir.name,
        "input_file": "input_f32.bin",
        "reference_file": "reference_f32.bin",
        "outputs": outputs,
        "coremltools_version": ct.__version__,
        "build_seconds": round(time.time() - started, 3),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
