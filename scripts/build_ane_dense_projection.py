#!/usr/bin/env python3
"""Build an experimental private-ANE dense projection probe artifact."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compile_native_qwen import parse_safetensors_header, read_tensor_bf16, sanitize_name
from repack_experts import get_model_entry

GROUP_SIZE = 64


def import_coremltools():
    try:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig
    except Exception as exc:
        raise SystemExit(
            "ERROR: coremltools is required for ANE dense projection artifacts. "
            "Run this with a Python environment that has coremltools installed."
        ) from exc
    return ct, mb, types, OptimizationConfig, OpLinearQuantizerConfig


def candidate_tensor_names(name):
    names = [name]
    if name.startswith("model.") and not name.startswith("model.language_model."):
        names.append("model.language_model." + name[len("model."):])
    if name.startswith("language_model."):
        names.append("model." + name)
    if not name.startswith("model.") and not name.startswith("language_model."):
        names.append("model.language_model." + name)
        names.append("language_model." + name)
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def read_matrix(model_path, tensor_name):
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"ERROR: {index_path} not found")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    actual_name = None
    for name in candidate_tensor_names(tensor_name):
        if name in weight_map:
            actual_name = name
            break
    if actual_name is None:
        raise SystemExit(f"ERROR: tensor '{tensor_name}' was not found in {index_path}")

    filename = weight_map[actual_name]
    header, data_start = parse_safetensors_header(model_path / filename)
    matrix = read_tensor_bf16(model_path, filename, header, data_start, actual_name)
    if matrix.ndim != 2:
        raise SystemExit(f"ERROR: {actual_name} shape {matrix.shape} is not a 2D projection")
    if matrix.shape[1] % GROUP_SIZE != 0:
        raise SystemExit(f"ERROR: {actual_name} input dim {matrix.shape[1]} is not divisible by {GROUP_SIZE}")
    return actual_name, np.asarray(matrix, dtype=np.float32)


def safe_artifact_name(tensor_name, tokens, in_dim, out_dim):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sanitize_name(tensor_name)).replace(".", "_")
    return f"{safe}_n{tokens}_i{in_dim}_o{out_dim}_g64_uint4"


def default_output_dir(model_path, model_id, artifact_name):
    bits = 4
    if model_id:
        entry = get_model_entry(model_id)
        bits = int(entry.get("quantization", {}).get("bits", 4) or 4)
        if bits != 4:
            raise SystemExit("ERROR: the ANE dense projection probe currently builds q4 artifacts only")
    return model_path / "flashchat" / f"q{bits}" / "ane_dense" / artifact_name


def output_slices(out_dim, split_size):
    if split_size <= 0:
        return [(0, out_dim)]
    if out_dim <= 12288:
        return [(0, out_dim)]
    parts = (out_dim + split_size - 1) // split_size
    chunk = (out_dim + parts - 1) // parts
    return [(start, min(start + chunk, out_dim)) for start in range(0, out_dim, chunk)]


def build_program(ct, mb, types, matrix, tokens, slices):
    in_dim = matrix.shape[1]
    groups = in_dim // GROUP_SIZE
    matrix16 = np.asarray(matrix, dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(tokens, in_dim), dtype=types.fp32)],
                opset_version=ct.target.iOS18)
    def program(x):
        x16 = mb.cast(x=x, dtype="fp16", name="x_fp16")
        outs = []
        for split_idx, (row_start, row_end) in enumerate(slices):
            acc = None
            for group in range(groups):
                col_start = group * GROUP_SIZE
                col_end = col_start + GROUP_SIZE
                xg = mb.slice_by_index(
                    x=x16,
                    begin=[0, col_start],
                    end=[tokens, col_end],
                    name=f"x_s{split_idx}_g{group}",
                )
                weight = matrix16[row_start:row_end, col_start:col_end]
                yg = mb.linear(x=xg, weight=weight, name=f"linear_s{split_idx}_g{group}")
                if acc is None:
                    acc = yg
                else:
                    acc = mb.add(x=acc, y=yg, name=f"add_s{split_idx}_g{group}")
            outs.append(mb.cast(x=acc, dtype="fp32", name=f"y{split_idx}"))
        return outs[0] if len(outs) == 1 else tuple(outs)

    return program


def compile_model(package_dir, output_dir, artifact_name):
    coremlc = shutil.which("xcrun")
    if coremlc is None:
        raise SystemExit("ERROR: xcrun not found; Xcode Command Line Tools are required")
    tmp_parent = output_dir / f".{artifact_name}.compile"
    if tmp_parent.exists():
        shutil.rmtree(tmp_parent)
    tmp_parent.mkdir(parents=True)
    try:
        subprocess.run(
            ["xcrun", "coremlc", "compile", str(package_dir), str(tmp_parent)],
            check=True,
        )
        produced = sorted(tmp_parent.glob("*.mlmodelc"))
        if not produced:
            raise SystemExit("ERROR: coremlc did not produce an .mlmodelc directory")
        dest = output_dir / f"{artifact_name}.mlmodelc"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(produced[0]), dest)
        return dest
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


def write_reference(qmodel, output_names, slices, output_dir, tokens, in_dim, seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((tokens, in_dim), dtype=np.float32) * 0.25).astype(np.float32)
    pred = qmodel.predict({"x": x})
    chunks = [np.asarray(pred[name], dtype=np.float32) for name in output_names]
    y = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=1)
    x.tofile(output_dir / "input_f32.bin")
    y.tofile(output_dir / "reference_f32.bin")
    return [
        {
            "name": output_names[i],
            "start": int(start),
            "end": int(end),
            "dim": int(end - start),
        }
        for i, (start, end) in enumerate(slices)
    ]


def main():
    parser = argparse.ArgumentParser(description="Build a q4 group-64 CoreML/ANE dense projection probe")
    parser.add_argument("--model", required=True, help="Path to a native BF16 model snapshot")
    parser.add_argument("--model-id", default=None, help="Registry model id, used for default runtime placement")
    parser.add_argument("--tensor", required=True, help="BF16 projection tensor name to compile")
    parser.add_argument("--output", default=None, help="Output directory (default: MODEL/flashchat/q4/ane_dense/ARTIFACT)")
    parser.add_argument("--tokens", type=int, default=8, help="Static prompt chunk size")
    parser.add_argument("--input-limit", type=int, default=0, help="Only use the first N input columns")
    parser.add_argument("--output-limit", type=int, default=0, help="Only use the first N output rows")
    parser.add_argument("--output-split-size", type=int, default=8704, help="Split outputs larger than 12288 rows into chunks of about this size; 0 disables splitting")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic probe input seed")
    parser.add_argument("--keep-package", action="store_true", help="Keep the intermediate .mlpackage")
    args = parser.parse_args()

    ct, mb, types, OptimizationConfig, OpLinearQuantizerConfig = import_coremltools()

    model_path = Path(args.model)
    actual_name, matrix = read_matrix(model_path, args.tensor)
    if args.input_limit:
        if args.input_limit % GROUP_SIZE != 0:
            raise SystemExit(f"ERROR: --input-limit must be divisible by {GROUP_SIZE}")
        matrix = matrix[:, :args.input_limit]
    if args.output_limit:
        matrix = matrix[:args.output_limit, :]
    out_dim, in_dim = matrix.shape
    if out_dim <= 0 or in_dim <= 0:
        raise SystemExit("ERROR: projection slice is empty")

    artifact_name = safe_artifact_name(actual_name, args.tokens, in_dim, out_dim)
    output_dir = Path(args.output) if args.output else default_output_dir(model_path, args.model_id, artifact_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    slices = output_slices(out_dim, args.output_split_size)
    print(f"Tensor: {actual_name}")
    print(f"Shape: [{out_dim}, {in_dim}]")
    print(f"Output: {output_dir}")
    print(f"Output splits: {[end - start for start, end in slices]}")

    started = time.time()
    program = build_program(ct, mb, types, matrix, args.tokens, slices)
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

    output_names = [spec.name for spec in qmodel._spec.description.output]
    outputs = write_reference(qmodel, output_names, slices, output_dir, args.tokens, in_dim, args.seed)
    modelc_dir = compile_model(package_dir, output_dir, artifact_name)
    if not args.keep_package:
        shutil.rmtree(package_dir, ignore_errors=True)

    manifest = {
        "format_version": 1,
        "kind": "private_ane_dense_projection_probe",
        "model_id": args.model_id or "",
        "model_path": str(model_path),
        "source_tensor": actual_name,
        "sanitized_tensor": sanitize_name(actual_name),
        "tokens": args.tokens,
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "group_size": GROUP_SIZE,
        "quantization": {
            "bits": 4,
            "style": "coreml_linear_zero_point_group64_uint4",
            "note": "Experimental ANE q4 is group-64 but not bit-exact to Flashchat MLX affine q4.",
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
