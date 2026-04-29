#!/usr/bin/env python3
"""Export HuggingFace tokenizer.json to a compact binary format for C.

Usage: python export_tokenizer.py [tokenizer.json] [output.bin]

Binary format:
  Header:
    magic: "BPET" (4 bytes)
    version: uint32
    vocab_size: uint32
    num_merges: uint32
    num_added: uint32
  Vocab section (sorted by token_id):
    For each entry: uint32 token_id, uint16 str_len, char[str_len] (UTF-8 bytes of the BPE string)
  Merges section (ordered by priority, index 0 = highest priority):
    For each entry: uint16 len_a, char[len_a], uint16 len_b, char[len_b]
  Added tokens section:
    For each entry: uint32 token_id, uint16 str_len, char[str_len]
"""
import json
import struct
import sys
import os


def get_default_model_path():
    """Get default model path from environment or compute from MODEL_REPO."""
    model_repo = os.environ.get('FLASHCHAT_MODEL_REPO', 'mlx-community/Qwen3.5-397B-A17B-4bit')
    escaped_repo = model_repo.replace('/', '--')
    hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
    snapshot_dir = f"{hf_cache}/models--{escaped_repo}/snapshots"
    
    if os.path.isdir(snapshot_dir):
        snapshots = sorted(os.listdir(snapshot_dir))
        if snapshots:
            return f"{snapshot_dir}/{snapshots[-1]}"
    
    return os.path.expanduser(f'~/.cache/huggingface/hub/models--{escaped_repo}/snapshots/<snapshot>')


def main():
    # Allow env vars to override: FLASHCHAT_MODEL_PATH, FLASHCHAT_WEIGHTS_DIR
    default_model_path = os.environ.get('FLASHCHAT_MODEL_PATH') or get_default_model_path()
    default_weights_dir = os.environ.get('FLASHCHAT_WEIGHTS_DIR') or f"{default_model_path}/flashchat"
    
    tok_path = sys.argv[1] if len(sys.argv) > 1 else f"{default_model_path}/tokenizer.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{default_weights_dir}/vocab.bin"

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    model = t['model']
    vocab = model['vocab']       # str -> int
    merges = model['merges']     # list of [str, str]
    added = t['added_tokens']    # list of {id, content, special, ...}

    # Sort vocab by token_id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(out_path, 'wb') as f:
        # Header
        f.write(b'BPET')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(sorted_vocab)))
        f.write(struct.pack('<I', len(merges)))
        f.write(struct.pack('<I', len(added)))

        # Vocab
        for token_str, token_id in sorted_vocab:
            b = token_str.encode('utf-8')
            f.write(struct.pack('<I', token_id))
            f.write(struct.pack('<H', len(b)))
            f.write(b)

        # Merges
        for pair in merges:
            a, b = pair[0], pair[1]
            ab = a.encode('utf-8')
            bb = b.encode('utf-8')
            f.write(struct.pack('<H', len(ab)))
            f.write(ab)
            f.write(struct.pack('<H', len(bb)))
            f.write(bb)

        # Added tokens
        for tok in added:
            b = tok['content'].encode('utf-8')
            f.write(struct.pack('<I', tok['id']))
            f.write(struct.pack('<H', len(b)))
            f.write(b)

    print(f"Exported to {out_path}:")
    print(f"  Vocab: {len(sorted_vocab)} entries")
    print(f"  Merges: {len(merges)} rules")
    print(f"  Added tokens: {len(added)} entries")

    sz = os.path.getsize(out_path)
    print(f"  File size: {sz / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    main()
