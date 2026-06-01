#!/usr/bin/env python3
import importlib.util
from pathlib import Path


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_tokenizer", root / "scripts" / "export_tokenizer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.parse_merge_pair(["a", "b"]) == ("a", "b")
assert module.parse_merge_pair("a b") == ("a", "b")
assert module.parse_merge_pair("Ġ t") == ("Ġ", "t")

try:
    module.parse_merge_pair("abc")
except ValueError:
    pass
else:
    raise AssertionError("expected invalid merge format to raise ValueError")

print("tokenizer export tests passed")
