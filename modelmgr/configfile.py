"""Read/write ~/.config/flashchat/config.

The file is consumed by three readers with different parsers, so writes
must stay maximally conservative:
- the C engine prefix-matches `KEY="value"` lines and ignores unknown keys
  (metal_infer/infer.m), values must be double-quoted;
- bash `source`s it;
- this module.

Updates edit a key's line in place when present and append otherwise —
never reorder, never rewrite untouched lines (preserves user comments and
the append-only migration contract).
"""
from __future__ import annotations

import os
import re

from . import paths

_LINE_RE = re.compile(r'^(\s*)([A-Z][A-Z0-9_]*)=(".*"|\S*)\s*$')


def _parse_line(line: str):
    m = _LINE_RE.match(line)
    if not m:
        return None
    key, raw = m.group(2), m.group(3)
    value = raw[1:-1] if raw.startswith('"') and raw.endswith('"') else raw
    return key, value


def load(path: str | None = None) -> dict:
    """Last occurrence wins, matching bash `source` semantics."""
    path = path or paths.config_file_path()
    values: dict = {}
    if not os.path.isfile(path):
        return values
    with open(path) as f:
        for line in f:
            parsed = _parse_line(line)
            if parsed:
                values[parsed[0]] = parsed[1]
    return values


def get(key: str, default: str = "", path: str | None = None) -> str:
    env = os.environ.get(f"FLASHCHAT_{key}")
    if env is not None:
        return env
    return load(path).get(key, default)


def update(changes: dict, path: str | None = None) -> None:
    """Set keys, editing existing lines in place and appending new ones."""
    path = path or paths.config_file_path()
    lines: list = []
    if os.path.isfile(path):
        with open(path) as f:
            lines = f.read().splitlines()

    remaining = dict(changes)
    for i, line in enumerate(lines):
        parsed = _parse_line(line)
        if parsed and parsed[0] in remaining:
            lines[i] = f'{parsed[0]}="{remaining.pop(parsed[0])}"'
    for key, value in remaining.items():
        lines.append(f'{key}="{value}"')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_text = "\n".join(lines) + "\n"
    if os.path.isfile(path):
        with open(path) as f:
            if f.read() == new_text:
                return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(new_text)
    os.replace(tmp, path)


def exists(path: str | None = None) -> bool:
    return os.path.isfile(path or paths.config_file_path())
