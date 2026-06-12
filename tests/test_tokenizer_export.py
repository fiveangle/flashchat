#!/usr/bin/env python3
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from modelmgr.steps import tokenizer as module

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
