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
import subprocess
from functools import lru_cache
from pathlib import Path

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


@lru_cache(maxsize=8)
def _shipping_defaults(home: str, config_dir: str) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, HOME=home, FLASHCHAT_CONFIG_DIR=config_dir)
    output = subprocess.check_output([
        "bash", "-c", 'source "$1/lib/config.sh"; flashchat_dump_defaults',
        "defaults", str(root),
    ], env=env)
    fields = output.decode().split("\0")[:-1]
    return dict(zip(fields[::2], fields[1::2]))


def shipping_defaults() -> dict[str, str]:
    """Read launcher defaults without creating or migrating user configuration."""
    return _shipping_defaults(os.path.expanduser("~"), paths.config_dir()).copy()


def get(key: str, default: str | None = None, path: str | None = None) -> str:
    env = os.environ.get(f"FLASHCHAT_{key}")
    if env is not None:
        return env
    values = load(path)
    if key in values:
        return values[key]
    if default is None:
        default = shipping_defaults().get(key, "")
    return default


def mtp_enabled(path: str | None = None) -> bool:
    value = get("MTP", "", path).strip().lower()
    return value not in ("", "0", "off", "no", "false", "default", "registry")


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


def initialize_defaults(path: str | None = None) -> None:
    """Initialize or migrate through the same defaults used by the launcher."""
    target = path or paths.config_file_path()
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        "bash", "-c",
        'source "$1/lib/config.sh"; '
        'FLASHCHAT_CONFIG_FILE_OVERRIDE="$2"; '
        'if [ ! -f "$2" ]; then FLASHCHAT_CONFIG_FILE="$2"; '
        'flashchat_create_default_config; fi; flashchat_load_config',
        "defaults", str(root), target,
    ], check=True)
