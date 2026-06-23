#!/bin/bash
# python.sh — Flashchat Python Interpreter Resolver
#
# Locates a Python interpreter that satisfies the project's minimum version
# (currently 3.10 — modelmgr uses PEP 604 union syntax unconditionally).
#
# Resolution order:
#   1. $FLASHCHAT_PYTHON            — explicit override
#   2. /opt/homebrew/bin/python3    — Apple Silicon Homebrew (most common)
#   3. /usr/local/bin/python3       — Intel Homebrew (legacy prefix)
#   4. python3 from PATH            — last resort
#
# Each candidate must report sys.version_info >= (3, 10) to be accepted.
# The function never modifies the user's PATH or default `python3`; it only
# picks an existing interpreter and prints its path to stdout.
#
# Usage:
#   source lib/python.sh
#   PY=$(flashchat_python_bin)
#   "$PY" -m venv "$FLASHCHAT_VENV"
#
# Notes:
#   - The Homebrew prefix differs by Mac architecture (Apple Silicon vs Intel),
#     so both must be probed. CommandLineTools' /usr/bin/python3 (always 3.9
#     on macOS 26) is intentionally NOT listed — it would fail the version check.
#   - This helper is intentionally non-interactive. If no candidate satisfies
#     the floor, the function prints an actionable error to stderr and returns
#     non-zero so the caller can surface it.

set -e

FLASHCHAT_MIN_PYTHON_MAJOR=3
FLASHCHAT_MIN_PYTHON_MINOR=10

_flashchat_python_meets_floor() {
    local py="$1"
    "$py" -c "import sys; sys.exit(0 if sys.version_info >= (${FLASHCHAT_MIN_PYTHON_MAJOR}, ${FLASHCHAT_MIN_PYTHON_MINOR}) else 1)" 2>/dev/null
}

_flashchat_python_version() {
    local py="$1"
    "$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null
}

flashchat_python_bin() {
    local candidate version

    if [ -n "${FLASHCHAT_PYTHON:-}" ]; then
        if [ ! -x "${FLASHCHAT_PYTHON}" ]; then
            echo "FLASHCHAT_PYTHON is set to '${FLASHCHAT_PYTHON}' but that path is not executable." >&2
            return 1
        fi
        if _flashchat_python_meets_floor "${FLASHCHAT_PYTHON}"; then
            echo "${FLASHCHAT_PYTHON}"
            return 0
        fi
        version=$(_flashchat_python_version "${FLASHCHAT_PYTHON}")
        echo "FLASHCHAT_PYTHON ('${FLASHCHAT_PYTHON}', Python ${version:-unknown}) is older than the required ${FLASHCHAT_MIN_PYTHON_MAJOR}.${FLASHCHAT_MIN_PYTHON_MINOR}." >&2
        return 1
    fi

    for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3; do
        if [ -x "$candidate" ] && _flashchat_python_meets_floor "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    if command -v python3 >/dev/null 2>&1; then
        candidate=$(command -v python3)
        if _flashchat_python_meets_floor "$candidate"; then
            echo "$candidate"
            return 0
        fi
        version=$(_flashchat_python_version "$candidate")
        echo "" >&2
        echo "Flashchat requires Python ${FLASHCHAT_MIN_PYTHON_MAJOR}.${FLASHCHAT_MIN_PYTHON_MINOR}+, but '$candidate' reports ${version:-an unknown version}." >&2
        echo "" >&2
        echo "Your system's default python3 is too old (this is normal on macOS 26, where /usr/bin/python3 is 3.9 from CommandLineTools)." >&2
        echo "Install a newer Python via Homebrew and re-run flashchat:" >&2
        echo "    brew install python" >&2
        echo "Or pin an interpreter explicitly:" >&2
        echo "    export FLASHCHAT_PYTHON=/path/to/python3.11" >&2
        return 1
    fi

    echo "No python3 interpreter found on PATH." >&2
    echo "Install one via 'brew install python' or set FLASHCHAT_PYTHON." >&2
    return 1
}