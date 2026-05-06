#!/usr/bin/env python3
"""
extract_weights.py — Dispatcher for model-specific extract_weights scripts.

Reads assets/model_configs.json to determine which model-specific script to run.
Supports --model-id flag or FLASHCHAT_MODEL environment variable.
All other arguments (including --model PATH) are passed through to the model-specific script.

Usage:
    python extract_weights.py --model-id qwen3.5-397B-A17B --model /path/to/model --output /path/to/output
"""

import argparse
import json
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Extract weights dispatcher')
    parser.add_argument('--model-id', type=str,
                        default=os.environ.get('FLASHCHAT_MODEL'),
                        help='Model ID from assets/model_configs.json (or set FLASHCHAT_MODEL)')
    parser.add_argument('--model', type=str,
                        help='Path to model directory (passed through to model-specific script)')
    args, remaining = parser.parse_known_args()

    if not args.model_id:
        print("ERROR: No model-id specified. Use --model-id or set FLASHCHAT_MODEL", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(os.environ.get('FLASHCHAT_MODEL_CONFIG', repo_root / 'assets' / 'model_configs.json'))
    with open(config_path) as f:
        configs = json.load(f)

    if args.model_id not in configs['models']:
        print(f"ERROR: Model '{args.model_id}' not found in {config_path}", file=sys.stderr)
        print(f"Available models: {list(configs['models'].keys())}", file=sys.stderr)
        sys.exit(1)

    script_name = configs['models'][args.model_id]['scripts']['extract_weights']
    script_path = repo_root / script_name

    if not os.path.exists(script_path):
        print(f"ERROR: Script {script_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Running {script_name} for model {args.model_id}")
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
