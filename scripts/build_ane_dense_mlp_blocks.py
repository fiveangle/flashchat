#!/usr/bin/env python3
"""Build experimental private-ANE dense MLP block artifacts for layer ranges."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_layers(spec):
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise SystemExit(f"ERROR: invalid descending layer range '{part}'")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    seen = set()
    return [layer for layer in layers if not (layer in seen or seen.add(layer))]


def main():
    parser = argparse.ArgumentParser(description="Build fused ANE dense MLP block artifacts for one or more layers")
    parser.add_argument("--model", required=True, help="Path to a native BF16 model snapshot")
    parser.add_argument("--model-id", default=None, help="Registry model id, used for default runtime placement")
    parser.add_argument("--layers", required=True, help="Layer list/ranges, for example 0,3,7-10")
    parser.add_argument("--output-root", default=None, help="Parent output directory; layer artifacts go under layer_XX")
    parser.add_argument("--tokens", type=int, default=8, help="Static prompt chunk size")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic probe input seed")
    parser.add_argument("--keep-package", action="store_true", help="Keep intermediate .mlpackage directories")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    if not layers:
        raise SystemExit("ERROR: no layers requested")

    builder = REPO_ROOT / "scripts" / "build_ane_dense_mlp_block.py"
    for layer in layers:
        cmd = [
            sys.executable,
            str(builder),
            "--model", args.model,
            "--layer", str(layer),
            "--tokens", str(args.tokens),
            "--seed", str(args.seed),
        ]
        if args.model_id:
            cmd += ["--model-id", args.model_id]
        if args.output_root:
            output = Path(args.output_root) / f"layer_{layer:02d}"
            cmd += ["--output", str(output)]
        if args.keep_package:
            cmd.append("--keep-package")
        print(f"=== building ANE dense MLP block layer {layer} ===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
