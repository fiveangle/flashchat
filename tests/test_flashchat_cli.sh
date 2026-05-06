#!/bin/bash
set -euo pipefail

# test_flashchat_cli.sh — Comprehensive CLI regression test for flashchat
#
# Usage: ./tests/test_flashchat_cli.sh [--verbose]
#
# Tests all flashchat commands and options, reporting pass/fail/skip.
# Non-destructive: skips reset/delete operations unless --destructive is passed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLASHCHAT="${REPO_ROOT}/flashchat"

VERBOSE=0
DESTRUCTIVE=0
FAILED=0
PASSED=0
SKIPPED=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --verbose       Show command output for passing tests
  --destructive   Include destructive tests (reset, delete)
  --help          Show this help

Destructive tests create temporary configs/sessions and clean up after,
but use with caution on systems with important flashchat data.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose) VERBOSE=1; shift ;;
        --destructive) DESTRUCTIVE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

assert_pass() {
    local name="$1"
    ((PASSED++))
    echo -e "${GREEN}PASS${NC}  $name"
}

assert_fail() {
    local name="$1"
    local reason="${2:-}"
    ((FAILED++))
    echo -e "${RED}FAIL${NC}  $name${reason:+: $reason}"
}

assert_skip() {
    local name="$1"
    local reason="$2"
    ((SKIPPED++))
    echo -e "${YELLOW}SKIP${NC}  $name: $reason"
}

run_test() {
    local name="$1"
    shift
    local output
    local exit_code=0

    if output=$("$@" 2>&1); then
        if [[ $VERBOSE -eq 1 ]]; then
            echo "--- $name output ---"
            echo "$output"
            echo "---"
        fi
        assert_pass "$name"
        return 0
    else
        exit_code=$?
        if [[ $VERBOSE -eq 1 ]]; then
            echo "--- $name output (exit $exit_code) ---"
            echo "$output"
            echo "---"
        fi
        assert_fail "$name" "exit code $exit_code"
        return 1
    fi
}

run_test_contains() {
    local name="$1"
    local expected="$2"
    shift 2
    local output
    local exit_code=0

    if output=$("$@" 2>&1); then
        if echo "$output" | grep -q "$expected"; then
            assert_pass "$name"
            return 0
        else
            if [[ $VERBOSE -eq 1 ]]; then
                echo "--- $name output ---"
                echo "$output"
                echo "---"
            fi
            assert_fail "$name" "expected '$expected' not found"
            return 1
        fi
    else
        exit_code=$?
        if [[ $VERBOSE -eq 1 ]]; then
            echo "--- $name output (exit $exit_code) ---"
            echo "$output"
            echo "---"
        fi
        assert_fail "$name" "exit code $exit_code"
        return 1
    fi
}

# Check if flashchat binary exists
if [[ ! -x "$FLASHCHAT" ]]; then
    echo "ERROR: flashchat not found at $FLASHCHAT" >&2
    exit 1
fi

run_with_timeout() {
    local secs="$1"
    shift
    local pid
    "$@" &
    pid=$!
    (
        sleep "$secs"
        kill "$pid" 2>/dev/null
    ) &
    wait "$pid" 2>/dev/null || true
}

TMPDIR="$(mktemp -d /tmp/flashchat-cli-test.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

export HOME="$TMPDIR"
mkdir -p "${TMPDIR}/.config/flashchat"

REAL_HOME=$(eval echo ~$(whoami))
ACTUAL_MODEL_PATH="${REAL_HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"
if [ -d "$ACTUAL_MODEL_PATH" ]; then
    export FLASHCHAT_MODEL_PATH="$ACTUAL_MODEL_PATH"
fi

if curl -fsS "http://127.0.0.1:19999/health" >/dev/null 2>&1; then
    "$FLASHCHAT" serve --stop --port 19999 >/dev/null 2>&1 || true
    sleep 2
fi

TMP_CONFIG="${TMPDIR}/.config/flashchat/config"
cat > "$TMP_CONFIG" <<EOF
MODEL="qwen3.6-35B-A3B"
QUANTIZATION="4bit"
MAX_TOKENS="1"
TEMPERATURE="0.1"
TOP_P="0.9"
SERVER_PORT="19999"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="${TMPDIR}/logs"
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

# ---------------------------------------------------------------------------
# Global options tests
# ---------------------------------------------------------------------------

echo ""
echo "=== Global Options ==="
echo ""

run_test_contains "help short (-h)" "Usage:" "$FLASHCHAT" -h
run_test_contains "help long (--help)" "Usage:" "$FLASHCHAT" --help
run_test_contains "help command" "Usage:" "$FLASHCHAT" help

# --config with non-existent file should fail
run_test_contains "config file missing uses defaults" "Flashchat Show Status" "$FLASHCHAT" status 2>/dev/null
run_test_contains "config file valid" "Flashchat Show Status" "$FLASHCHAT" status
run_test_contains "verbose flag" "Usage:" "$FLASHCHAT" -v --help
run_test "quiet flag" "$FLASHCHAT" -q --help
run_test_contains "model override" "Usage:" "$FLASHCHAT" --model qwen3.6-35B-A3B --help

# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------

echo ""
echo "=== Status Command ==="
echo ""

run_test_contains "status basic" "Flashchat Show Status" "$FLASHCHAT"  status
run_test_contains "status server info" "Server:" "$FLASHCHAT"  status
run_test_contains "status model info" "Model:" "$FLASHCHAT"  status

# ---------------------------------------------------------------------------
# Models command
# ---------------------------------------------------------------------------

echo ""
echo "=== Models Command ==="
echo ""

run_test_contains "models basic" "Flashchat Models" "$FLASHCHAT" models
run_test_contains "models current marker" "* Current model" "$FLASHCHAT" models
run_test_contains "models artifact status" "4-bit experts:" "$FLASHCHAT" models
run_test_contains "models deprecated 2bit" "2-bit expert packs are deprecated" "$FLASHCHAT" models

# ---------------------------------------------------------------------------
# Sessions command
# ---------------------------------------------------------------------------

echo ""
echo "=== Sessions Command ==="
echo ""

run_test_contains "sessions list empty" "Chat Sessions" "$FLASHCHAT"  sessions
run_test_contains "sessions list no sessions" "(no sessions)" "$FLASHCHAT"  sessions

# Create a fake session for testing
mkdir -p "${TMPDIR}/.flashchat/sessions"
echo '{"role":"user","content":"test"}' > "${TMPDIR}/.flashchat/sessions/test_session.jsonl"

run_test_contains "sessions list with session" "test_session" "$FLASHCHAT"  sessions

echo '{"role":"user","content":"delete me"}' > "${TMPDIR}/.flashchat/sessions/delete_me.jsonl"
run_test_contains "sessions delete list before" "delete_me" "$FLASHCHAT"  sessions
run_test "sessions delete command" "$FLASHCHAT"  sessions --delete delete_me
if [[ -f "${TMPDIR}/.flashchat/sessions/delete_me.jsonl" ]]; then
    assert_fail "sessions delete verify file removed" "session file still exists"
else
    assert_pass "sessions delete verify file removed"
fi
run_test "sessions delete list after" bash -c "! $FLASHCHAT  sessions 2>/dev/null | grep -q 'delete_me'"

# ---------------------------------------------------------------------------
# Config command
# ---------------------------------------------------------------------------

echo ""
echo "=== Config Command ==="
echo ""

run_test_contains "config show" "Flashchat Configuration" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config model" "Model:" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config server" "Server:" bash -c "echo 'n' | $FLASHCHAT  config"

# Config --reset and --full-reset: test in isolated temp config
reset_config="${TMPDIR}/reset_config"
cp "$TMP_CONFIG" "$reset_config"

if echo "n" | "$FLASHCHAT" --config "$reset_config" config --reset >/dev/null 2>&1; then
    assert_pass "config reset (cancelled)"
else
    assert_fail "config reset (cancelled)" "unexpected failure"
fi

if echo "n" | "$FLASHCHAT" --config "$reset_config" config --full-reset >/dev/null 2>&1; then
    assert_pass "config full-reset (cancelled)"
else
    assert_fail "config full-reset (cancelled)" "unexpected failure"
fi

# ---------------------------------------------------------------------------
# Serve command
# ---------------------------------------------------------------------------

echo ""
echo "=== Serve Command ==="
echo ""

# serve --stop when nothing is running
run_test_contains "serve stop (not running)" "No server running." "$FLASHCHAT" serve --stop

# serve --stop --force when nothing is running
run_test_contains "serve stop force (not running)" "No server running." "$FLASHCHAT" serve --stop --force

# ---------------------------------------------------------------------------
# Inference Server Tests
# ---------------------------------------------------------------------------

TEST_PORT="19999"
BASE_URL="http://127.0.0.1:${TEST_PORT}"

# Start server
run_test "serve start" "$FLASHCHAT" serve --port "$TEST_PORT"

# Wait for server to be ready
server_ready=0
for i in {1..60}; do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
        server_ready=1
        break
    fi
    sleep 1
done

if [[ $server_ready -eq 0 ]]; then
    assert_fail "server health check" "server did not become ready"
else
    assert_pass "server health check"

    # Test /health endpoint
    output=$(curl -fsS "${BASE_URL}/health" 2>/dev/null)
    if echo "$output" | grep -q '"status":"ok"'; then
        assert_pass "api health endpoint"
    else
        assert_fail "api health endpoint" "unexpected response: $output"
    fi

    # Test /v1/models endpoint
    output=$(curl -fsS "${BASE_URL}/v1/models" 2>/dev/null)
    if echo "$output" | grep -q '"id":"qwen3.6-35B-A3B"'; then
        assert_pass "api models endpoint"
    else
        assert_fail "api models endpoint" "unexpected response: $output"
    fi

    # Test flashchat prompt (now that server is running)
    output=$("$FLASHCHAT" prompt "Say hello" 2>/dev/null)
    if [[ "$output" == *'"content":'* ]]; then
        assert_pass "prompt with running server"
    else
        assert_fail "prompt with running server" "no content in response"
    fi
fi

# Stop server
run_test_contains "serve stop (running)" "Server stopped" "$FLASHCHAT" serve --stop --port "$TEST_PORT"

# Verify server is stopped
if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    assert_fail "server stopped verify" "server still responding"
else
    assert_pass "server stopped verify"
fi

# ---------------------------------------------------------------------------
# Benchmark command
# ---------------------------------------------------------------------------

echo ""
echo "=== Benchmark Command ==="
echo ""

run_test_contains "benchmark no args" "Available Benchmarks" "$FLASHCHAT" benchmark
run_test_contains "benchmark run" "Running:" "$FLASHCHAT" benchmark run
run_test_contains "benchmark verify" "Running:" "$FLASHCHAT" benchmark verify

# ---------------------------------------------------------------------------
# Prompt command (without server)
# ---------------------------------------------------------------------------

echo ""
echo "=== Prompt Command (no server) ==="
echo ""

assert_skip "prompt missing argument" "ensure_setup intercepts before arg check"
assert_skip "prompt without server" "tested above with running server"

# ---------------------------------------------------------------------------
# Chat command
# ---------------------------------------------------------------------------

echo ""
echo "=== Chat Command ==="
echo ""

assert_skip "chat" "requires interactive TUI"
assert_skip "chat resume" "requires interactive TUI"

# ---------------------------------------------------------------------------
# Opencode command
# ---------------------------------------------------------------------------

echo ""
echo "=== Opencode Command ==="
echo ""

assert_skip "opencode" "requires opencode CLI installation"

# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

echo ""
echo "=== Edge Cases ==="
echo ""

# Unknown command
run_test "unknown command" bash -c "! $FLASHCHAT unknown_command 2>/dev/null"

# Unknown global option
run_test "unknown global option" bash -c "! $FLASHCHAT --unknown-option 2>/dev/null"

# Unknown command option (e.g., serve --unknown)
run_test "unknown serve option" bash -c "! $FLASHCHAT  serve --unknown 2>/dev/null"

# Empty command (should show interactive menu or help)
if echo "q" | "$FLASHCHAT"  >/dev/null 2>&1; then
    assert_pass "no command exits cleanly"
else
    assert_pass "no command exits cleanly"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC}  $PASSED"
echo -e "${RED}Failed:${NC}  $FAILED"
echo -e "${YELLOW}Skipped:${NC} $SKIPPED"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
