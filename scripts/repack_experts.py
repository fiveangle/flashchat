#!/usr/bin/env python3
"""
repack_experts.py — Dispatcher for model-specific repack_experts scripts.

Reads assets/model_configs.json to determine which model-specific script to run.
Supports --model-id flag or FLASHCHAT_MODEL environment variable.
All other arguments are passed through to the model-specific script.

Usage:
    python repack_experts.py --model-id qwen3.5-397B-A17B [other args...]
"""

import argparse
import os
import sys

from flashchat_registry import load_registry, model_ids, model_script_path, registry_path, repo_root

def main():
    parser = argparse.ArgumentParser(description='Repack experts dispatcher')
    parser.add_argument('--model-id', type=str,
                        default=os.environ.get('FLASHCHAT_MODEL'),
                        help='Model ID from assets/model_configs.json (or set FLASHCHAT_MODEL)')
    parser.add_argument('--model', type=str,
                        help='Path to model directory (passed through to model-specific script)')
    args, remaining = parser.parse_known_args()

    if not args.model_id:
        print("ERROR: No model-id specified. Use --model-id or set FLASHCHAT_MODEL", file=sys.stderr)
        sys.exit(1)

    configs = load_registry()
    config_path = registry_path()

    if args.model_id not in configs['models']:
        print(f"ERROR: Model '{args.model_id}' not found in {config_path}", file=sys.stderr)
        print(f"Available models: {model_ids(configs)}", file=sys.stderr)
        sys.exit(1)

    script_path = model_script_path(args.model_id, 'repack_experts', configs)

    if not script_path or not os.path.exists(script_path):
        print(f"ERROR: Script {script_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Running {script_path.relative_to(repo_root())} for model {args.model_id}")
    script_args = [str(script_path)]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model-id':
            i += 2
            continue
        script_args.append(sys.argv[i])
        i += 1
    os.execv(sys.executable, [sys.executable] + script_args)

if __name__ == '__main__':
    main()
