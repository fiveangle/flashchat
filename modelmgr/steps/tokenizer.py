"""export_tokenizer step: HF tokenizer.json -> shared/vocab.bin.

Binary format (consumed by metal_infer/tokenizer.h):
  Header: magic "BPET", uint32 version, vocab_size, num_merges, num_added
  Vocab (sorted by token_id): uint32 id, uint16 len, utf-8 bytes
  Merges (priority order):    uint16 len_a, bytes, uint16 len_b, bytes
  Added tokens:               uint32 id, uint16 len, utf-8 bytes
"""

import json
import os
import struct

from . import StepContext, step_version
from ..artifacts import ArtifactDir


def parse_merge_pair(pair):
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return pair[0], pair[1]
    if isinstance(pair, str):
        parts = pair.split(" ")
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"Unsupported merge pair format: {pair!r}")


def export_vocab(tokenizer_json_path: str, sink) -> dict:
    """Write the BPET blob to a file-like sink; returns counts for reporting."""
    with open(tokenizer_json_path, encoding="utf-8") as f:
        t = json.load(f)

    model = t["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    added = t["added_tokens"]
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    sink.write(b"BPET")
    sink.write(struct.pack("<I", 1))
    sink.write(struct.pack("<I", len(sorted_vocab)))
    sink.write(struct.pack("<I", len(merges)))
    sink.write(struct.pack("<I", len(added)))

    for token_str, token_id in sorted_vocab:
        b = token_str.encode("utf-8")
        sink.write(struct.pack("<I", token_id))
        sink.write(struct.pack("<H", len(b)))
        sink.write(b)

    for pair in merges:
        a, b = parse_merge_pair(pair)
        ab, bb = a.encode("utf-8"), b.encode("utf-8")
        sink.write(struct.pack("<H", len(ab)))
        sink.write(ab)
        sink.write(struct.pack("<H", len(bb)))
        sink.write(bb)

    for tok in added:
        b = tok["content"].encode("utf-8")
        sink.write(struct.pack("<I", tok["id"]))
        sink.write(struct.pack("<H", len(b)))
        sink.write(b)

    return {"vocab": len(sorted_vocab), "merges": len(merges), "added": len(added)}


def run(ctx: StepContext, planned=None) -> None:
    tokenizer_json = os.path.join(ctx.snapshot, "tokenizer.json")
    adir = ArtifactDir(ctx.shared_dir, ctx.manifest.id, "shared")
    ctx.report("export_tokenizer", 0, 1, "exporting vocab.bin")
    if ctx.dry_run:
        return
    with adir.open("vocab.bin", step="export_tokenizer",
                   step_version=step_version("export_tokenizer")) as sink:
        counts = export_vocab(tokenizer_json, sink)
    adir.commit()
    ctx.report("export_tokenizer", 1, 1,
               f"vocab {counts['vocab']}, merges {counts['merges']}, added {counts['added']}")


def get_runner(name: str):
    return run
