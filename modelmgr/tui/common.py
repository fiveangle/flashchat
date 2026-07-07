"""Prompt/print helpers shared by the TUI flows.

Plain stdin/stdout (works under the existing piped-input test harness),
ANSI color gated on COLOR_OUTPUT + isatty.
"""
from __future__ import annotations

import sys

from .. import configfile


def color_enabled() -> bool:
    return sys.stdout.isatty() and configfile.get("COLOR_OUTPUT", "1") != "0"


def _c(code: str, text: str) -> str:
    if not color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t):  return _c("0;32", t)   # noqa: E704
def red(t):    return _c("0;31", t)   # noqa: E704
def yellow(t): return _c("0;33", t)   # noqa: E704
def dim(t):    return _c("2", t)      # noqa: E704
def bold(t):   return _c("1", t)      # noqa: E704


def heading(text: str) -> None:
    print(f"\n=== {text} ===\n")


def prompt(message: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        reply = input(f"{message}{suffix}: ").strip()
    except EOFError:
        return default
    return reply or default


def confirm(message: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        reply = input(f"{message} {hint}: ").strip().lower()
    except EOFError:
        # No interactive user. Never treat exhausted input as consent —
        # confirms gate downloads, builds, and deletes.
        return False
    if not reply:
        return default
    return reply in ("y", "yes")


def confirm_exact(expected: str, what: str) -> bool:
    """Destructive-op confirmation: the user must type the exact id."""
    print(f"Type the model id exactly to confirm {what}: {bold(expected)}")
    try:
        return input("> ").strip() == expected
    except EOFError:
        return False


def select_number(count: int, message: str = "Select", default: int | None = None,
                  extras: dict | None = None):
    """Pick 1..count, or one of `extras` keys (single letters). Returns an
    int index (1-based) or the extra key; None on cancel/empty/EOF."""
    default_str = str(default) if default else ""
    suffix = f" [{default_str}]" if default_str else ""
    while True:
        # Read input directly (not via prompt()): prompt() maps EOF to the
        # default, and if the default is itself out of range this retry loop
        # would spin forever on closed/exhausted stdin — printing the error
        # line at CPU speed (2026-07-07: a redirected-stdin test hit exactly
        # this and allocated 42GB of buffered output before being killed).
        try:
            reply = input(f"{message}{suffix}: ").strip() or default_str
        except EOFError:
            return None  # closed input is a cancel, never a retry
        if not reply or reply.lower() == "q":
            return None
        low = reply.lower()
        if extras and low in extras:
            return low
        if reply.isdigit() and 1 <= int(reply) <= count:
            return int(reply)
        print(f"  Enter a number 1-{count}"
              + (f" or one of: {', '.join(extras)}" if extras else ""))


class ProgressLine:
    """Single-line progress renderer fed by step progress callbacks."""

    def __init__(self):
        self._last_len = 0

    def __call__(self, phase: str, current: int = 0, total: int = 0, msg: str = "") -> None:
        if total:
            line = f"  [{phase}] [{current}/{total}] {msg}"
        else:
            line = f"  [{phase}] {msg}"
        if sys.stdout.isatty():
            pad = max(0, self._last_len - len(line))
            sys.stdout.write("\r" + line + " " * pad)
            if current >= total and total:
                sys.stdout.write("\n")
                self._last_len = 0
            else:
                self._last_len = len(line)
            sys.stdout.flush()
        else:
            print(line)

    def finish(self) -> None:
        if sys.stdout.isatty() and self._last_len:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_len = 0
