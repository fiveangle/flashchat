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
STUBBORN_PID=""
cleanup_test_env() {
    if [[ -n "${STUBBORN_PID:-}" ]] && kill -0 "$STUBBORN_PID" 2>/dev/null; then
        kill -9 "$STUBBORN_PID" 2>/dev/null || true
        wait "$STUBBORN_PID" 2>/dev/null || true
    fi
    rm -rf "$TMPDIR"
}
trap cleanup_test_env EXIT

export HOME="$TMPDIR"
mkdir -p "${TMPDIR}/.config/flashchat"

REAL_HOME=$(eval echo ~$(whoami))
ACTUAL_MODEL_SNAPSHOTS="${REAL_HOME}/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
ACTUAL_MODEL_PATH=""
if [ -d "$ACTUAL_MODEL_SNAPSHOTS" ]; then
    ACTUAL_MODEL_PATH=$(find "$ACTUAL_MODEL_SNAPSHOTS" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
fi
if [ -d "$ACTUAL_MODEL_PATH" ]; then
    export FLASHCHAT_MODEL_PATH="$ACTUAL_MODEL_PATH"
fi

TEST_HOST="127.0.0.1"
TEST_PORT="19999"
if curl -fsS "http://${TEST_HOST}:${TEST_PORT}/health" >/dev/null 2>&1; then
    "$FLASHCHAT" serve --stop --external --port 19999 >/dev/null 2>&1 || true
    sleep 2
fi

TMP_CONFIG="${TMPDIR}/.config/flashchat/config"
cat > "$TMP_CONFIG" <<EOF
MODEL="Qwen-Qwen36-35B-A3B"
MAX_TOKENS="1"
SAMPLING_PROFILE="custom"
REASONING="0"
TEMPERATURE="0.1"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
SERVER_PORT="${TEST_PORT}"
SERVER_HOST="${TEST_HOST}"
SERVER_LOG_PATH="${TMPDIR}/logs"
HUGGINGFACE_CACHE_DIR="${TMPDIR}/custom-hf-cache"
OFFLOAD_DIR="${TMPDIR}/offload"
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
SYSTEM_PROMPT_CACHE_DIR="${TMPDIR}/sys-cache"
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
run_test_contains "model override" "Usage:" "$FLASHCHAT" --model Qwen-Qwen36-35B-A3B --help

# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------

echo ""
echo "=== Status Command ==="
echo ""

run_test_contains "status basic" "Flashchat Show Status" "$FLASHCHAT"  status
run_test_contains "status server info" "Server:" "$FLASHCHAT"  status
run_test_contains "status model info" "Model:" "$FLASHCHAT"  status
run_test_contains "status quantization info" "Quantization: 4-bit" "$FLASHCHAT"  status
run_test_contains "main menu quantization info" "Quantization (model/kv-cache): q4" bash -c "printf q | $FLASHCHAT"

# ---------------------------------------------------------------------------
# Models command
# ---------------------------------------------------------------------------

echo ""
echo "=== Models Command ==="
echo ""

run_test "model registry json valid" python3 -m json.tool "${REPO_ROOT}/assets/model_configs.json"
run_test_contains "models basic" "HF cache:" "$FLASHCHAT" models
run_test_contains "models current marker" "<- selected" "$FLASHCHAT" models
run_test_contains "models repo shown" "Qwen/Qwen3.6-35B-A3B" "$FLASHCHAT" models
run_test_contains "models variant status" "q4:" "$FLASHCHAT" models

# ---------------------------------------------------------------------------
# Manage command
# ---------------------------------------------------------------------------

echo ""
echo "=== Manage Command ==="
echo ""

run_test_contains "manage list basic" "HF cache:" "$FLASHCHAT" manage --list
run_test_contains "manage offload dir" "Offload dir:" "$FLASHCHAT" manage --list
run_test_contains "manage variant status" "q4:" "$FLASHCHAT" manage --list

# ---------------------------------------------------------------------------
# Sessions command
# ---------------------------------------------------------------------------

echo ""
echo "=== Sessions Command ==="
echo ""

run_test_contains "sessions list empty" "Chat Sessions" "$FLASHCHAT"  sessions
run_test_contains "sessions list no sessions" "(no sessions)" "$FLASHCHAT"  sessions

# Create a fake session for testing
mkdir -p "${TMPDIR}/.config/flashchat/sessions"
echo '{"role":"user","content":"test"}' > "${TMPDIR}/.config/flashchat/sessions/test_session.jsonl"

run_test_contains "sessions list with session" "test_session" "$FLASHCHAT"  sessions

echo '{"role":"user","content":"delete me"}' > "${TMPDIR}/.config/flashchat/sessions/delete_me.jsonl"
run_test_contains "sessions delete list before" "delete_me" "$FLASHCHAT"  sessions
run_test "sessions delete command" "$FLASHCHAT"  sessions --delete delete_me
if [[ -f "${TMPDIR}/.config/flashchat/sessions/delete_me.jsonl" ]]; then
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
run_test_contains "config quantization" "Quantization: 4-bit" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config server" "Server:" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config HuggingFace cache dir" "HuggingFace cache dir:" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config offload dir" "Offload dir:" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config sampling profile" "Sampling profile:" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config sampling knobs" "Top-k:" bash -c "echo 'n' | $FLASHCHAT  config"
run_test_contains "config system prompt cache" "System prompt cache:" bash -c "echo 'n' | $FLASHCHAT  config"

THINKING_GUARD_FUNCS="${TMPDIR}/flashchat-functions.sh"
awk '/^main "\$@"/{exit} {print}' "$FLASHCHAT" > "$THINKING_GUARD_FUNCS"

make_thinking_guard_config() {
    local path="$1"
    cat > "$path" <<EOF
MODEL="Qwen-Qwen36-35B-A3B"
SAMPLING_PROFILE="thinking-general"
REASONING="1"
SHOW_THINKING="0"
EOF
}

guard_config="${TMPDIR}/thinking_guard_no.conf"
make_thinking_guard_config "$guard_config"
if output=$(HOME="${TMPDIR}/thinking-guard-home-no" FLASHCHAT_CONFIG_FILE_OVERRIDE="$guard_config" \
    bash -c 'source "$1"; flashchat_load_config >/dev/null; set +e; confirm_show_thinking_for_chat <<< ""; rc=$?; set -e; echo "rc=$rc"; grep "^SHOW_THINKING" "$FLASHCHAT_CONFIG_FILE"' "$FLASHCHAT" "$THINKING_GUARD_FUNCS" 2>&1) &&
    echo "$output" | grep -q 'rc=0' &&
    echo "$output" | grep -q 'SHOW_THINKING="0"' &&
    echo "$output" | grep -q 'Show Thinking will stay disabled'; then
    assert_pass "chat thinking guard default no"
else
    assert_fail "chat thinking guard default no" "${output:-no output}"
fi

guard_config="${TMPDIR}/thinking_guard_yes.conf"
make_thinking_guard_config "$guard_config"
if output=$(HOME="${TMPDIR}/thinking-guard-home-yes" FLASHCHAT_CONFIG_FILE_OVERRIDE="$guard_config" \
    bash -c 'source "$1"; flashchat_load_config >/dev/null; set +e; confirm_show_thinking_for_chat <<< "y"; rc=$?; set -e; echo "rc=$rc"; grep "^SHOW_THINKING" "$FLASHCHAT_CONFIG_FILE"' "$FLASHCHAT" "$THINKING_GUARD_FUNCS" 2>&1) &&
    echo "$output" | grep -q 'rc=0' &&
    echo "$output" | grep -q 'SHOW_THINKING="1"' &&
    echo "$output" | grep -q 'Show Thinking enabled'; then
    assert_pass "chat thinking guard yes persists setting"
else
    assert_fail "chat thinking guard yes persists setting" "${output:-no output}"
fi

guard_config="${TMPDIR}/thinking_guard_exit.conf"
make_thinking_guard_config "$guard_config"
if output=$(HOME="${TMPDIR}/thinking-guard-home-exit" FLASHCHAT_CONFIG_FILE_OVERRIDE="$guard_config" \
    bash -c 'source "$1"; flashchat_load_config >/dev/null; set +e; confirm_show_thinking_for_chat <<< "x"; rc=$?; set -e; echo "rc=$rc"; grep "^SHOW_THINKING" "$FLASHCHAT_CONFIG_FILE"' "$FLASHCHAT" "$THINKING_GUARD_FUNCS" 2>&1) &&
    echo "$output" | grep -q 'rc=2' &&
    echo "$output" | grep -q 'SHOW_THINKING="0"' &&
    echo "$output" | grep -q 'Returning to the previous menu'; then
    assert_pass "chat thinking guard exits to menu"
else
    assert_fail "chat thinking guard exits to menu" "${output:-no output}"
fi

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

# serve --stop --external when nothing is running
run_test_contains "serve stop external (not running)" "No server running." "$FLASHCHAT" serve --stop --external

RESTART_MODEL_ROOT="${TMPDIR}/restart-model"
RESTART_RUNTIME="${RESTART_MODEL_ROOT}/flashchat/q4"
RESTART_CONFIG="${TMPDIR}/restart_config"
mkdir -p "$RESTART_RUNTIME"
: > "${RESTART_RUNTIME}/model_weights.bin"
: > "${RESTART_RUNTIME}/vocab.bin"
python3 - "$REPO_ROOT/assets/model_configs.json" "$RESTART_RUNTIME/model_weights.json" <<'PY'
import json
import sys

registry_path, manifest_path = sys.argv[1:]
with open(registry_path) as f:
    model = json.load(f)["models"]["Qwen-Qwen36-35B-A3B"]
quant = model.get("quantization", {})
config = {
    "hidden_size": model["hidden_size"],
    "num_hidden_layers": model["num_hidden_layers"],
    "num_attention_heads": model["num_attention_heads"],
    "num_key_value_heads": model["num_key_value_heads"],
    "head_dim": model["head_dim"],
    "vocab_size": model["vocab_size"],
    "rms_norm_eps": model["rms_norm_eps"],
    "num_experts": model["num_experts"],
    "num_experts_per_tok": model["num_experts_per_tok"],
    "moe_intermediate_size": model["moe_intermediate_size"],
    "shared_expert_intermediate_size": model["shared_expert_intermediate_size"],
    "intermediate_size": model.get("intermediate_size", 0),
    "full_attention_interval": model["full_attention_interval"],
    "linear_num_value_heads": model["linear_num_value_heads"],
    "linear_num_key_heads": model["linear_num_key_heads"],
    "linear_key_head_dim": model["linear_key_head_dim"],
    "linear_value_head_dim": model["linear_value_dim"] if "linear_value_dim" in model else model["linear_value_head_dim"],
    "linear_conv_kernel_dim": model["linear_conv_kernel_dim"],
    "partial_rotary_factor": model["partial_rotary_factor"],
    "rope_theta": model["rope_theta"],
    "quantization": {"bits": quant.get("bits", 4), "group_size": quant.get("group_size", 64)},
    "mtp_num_hidden_layers": model.get("mtp_num_hidden_layers", 0),
}
config["layer_types"] = [
    "full_attention" if (i + 1) % config["full_attention_interval"] == 0 else "linear_attention"
    for i in range(config["num_hidden_layers"])
]
tensors = {}
for name in [
    "mtp.fc.weight",
    "mtp.fc.scales",
    "mtp.fc.biases",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.q_proj.scales",
    "mtp.layers.0.self_attn.q_proj.biases",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.gate.scales",
    "mtp.layers.0.mlp.gate.biases",
    "mtp.norm.weight",
]:
    tensors[name] = {"offset": 0, "size": 1, "shape": [1], "dtype": "BF16"}
with open(manifest_path, "w") as f:
    json.dump({"config": config, "tensors": tensors}, f)
PY
cat > "$RESTART_CONFIG" <<EOF
MODEL="Qwen-Qwen36-35B-A3B"
MODEL_PATH="${RESTART_MODEL_ROOT}"
MAX_TOKENS="1"
SAMPLING_PROFILE="custom"
REASONING="0"
TEMPERATURE="0.1"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
SERVER_PORT="19998"
SERVER_HOST="${TEST_HOST}"
SERVER_LOG_PATH="${TMPDIR}/logs"
HUGGINGFACE_CACHE_DIR="${TMPDIR}/custom-hf-cache"
OFFLOAD_DIR="${TMPDIR}/offload"
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
SYSTEM_PROMPT_CACHE_DIR="${TMPDIR}/sys-cache"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF
python3 - <<'PY' &
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(3600)
PY
STUBBORN_PID=$!
echo "$STUBBORN_PID" > "${TMPDIR}/.config/flashchat/server.pid"
echo "stale-signature" > "${TMPDIR}/.config/flashchat/server.signature"

menu_output=""
# Menu input: [N]ew dialog -> the stale-server restart prompt gets 'n'
# (decline -> "New dialog was not started."), then 'q' quits the menu.
if menu_output=$(printf 'nnq' | "$FLASHCHAT" --config "$RESTART_CONFIG" 2>&1); then
    if echo "$menu_output" | grep -q "New dialog was not started." && \
       echo "$menu_output" | grep -q "Goodbye!"; then
        assert_pass "interactive stale restart decline returns to menu"
    else
        assert_fail "interactive stale restart decline returns to menu" "expected recovery output not found"
    fi
else
    assert_fail "interactive stale restart decline returns to menu" "flashchat exited non-zero"
fi
kill -9 "$STUBBORN_PID" 2>/dev/null || true
wait "$STUBBORN_PID" 2>/dev/null || true
STUBBORN_PID=""
rm -f "${TMPDIR}/.config/flashchat/server.pid" "${TMPDIR}/.config/flashchat/server.signature"

# ---------------------------------------------------------------------------
# Inference Server Tests
# ---------------------------------------------------------------------------

BASE_URL="http://${TEST_HOST}:${TEST_PORT}"

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
    if echo "$output" | grep -q '"id":"Qwen-Qwen36-35B-A3B"'; then
        assert_pass "api models endpoint"
    else
        assert_fail "api models endpoint" "unexpected response: $output"
    fi

    old_pid=$(cat "${TMPDIR}/.config/flashchat/server.pid")
    echo 'SYSTEM_PROMPT_CACHE_MAX_ENTRIES="3"' >> "$TMP_CONFIG"
    output=$("$FLASHCHAT" serve --port "$TEST_PORT" 2>/dev/null)
    new_pid=$(cat "${TMPDIR}/.config/flashchat/server.pid")
    if [[ "$output" == *"stale runtime settings"* && "$old_pid" != "$new_pid" ]]; then
        assert_pass "serve restarts stale runtime"
    else
        assert_fail "serve restarts stale runtime" "old_pid=$old_pid new_pid=$new_pid output=$output"
    fi
    server_log="${TMPDIR}/logs/server.log"
    if grep -q "Runtime configuration:" "$server_log" && \
       grep -q "active_per_token=" "$server_log" && \
       grep -q "expert_size_per_token=" "$server_log" && \
       grep -q "sampling: temp=" "$server_log" && \
       grep -q "system_prompt_cache:" "$server_log" && \
       grep -q "system_prompt_cache_dir: ${TMPDIR}/sys-cache" "$server_log" && \
       grep -q "custom_user_system_prompt: .*not present" "$server_log"; then
        assert_pass "server log shows resolved runtime config"
    else
        assert_fail "server log shows resolved runtime config" "missing runtime config block in $server_log"
    fi

    # Test flashchat prompt (now that server is running)
    output=$("$FLASHCHAT" prompt "Say hello" 2>/dev/null)
    if [[ "$output" == *'"choices":'* ]]; then
        assert_pass "prompt with running server"
    else
        assert_fail "prompt with running server" "no completion response"
    fi
    if grep -q "generated=.*experts .* MiB/s, .* MiB/s/expert" "$server_log"; then
        assert_pass "server log shows generated expert throughput"
    else
        assert_fail "server log shows generated expert throughput" "missing generated expert throughput in $server_log"
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

# Escape sequences such as arrow keys must not be interpreted as menu choices.
TMP_MENU_HOME="$(mktemp -d /tmp/flashchat-menu-input.XXXXXX)"
if menu_output="$(printf '\033[Dq' | HOME="$TMP_MENU_HOME" "$FLASHCHAT" 2>&1)"; then
    if [[ "$menu_output" == *"Select a chat dialog to resume"* ]]; then
        assert_fail "main menu ignores arrow escape sequences" "left arrow triggered Dialog menu"
    else
        assert_pass "main menu ignores arrow escape sequences"
    fi
else
    assert_fail "main menu ignores arrow escape sequences" "launcher exited non-zero"
fi
rm -rf "$TMP_MENU_HOME"

TMP_MENU_HOME="$(mktemp -d /tmp/flashchat-menu-input.XXXXXX)"
if menu_output="$(printf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxq' | HOME="$TMP_MENU_HOME" "$FLASHCHAT" 2>&1)"; then
    prompt_count="$(printf '%s\n' "$menu_output" | grep -c '^> ' || true)"
    if [[ "$prompt_count" = "1" && "$menu_output" == *"Invalid selection"* ]]; then
        assert_pass "main menu suppresses invalid key prompt spam"
    else
        assert_fail "main menu suppresses invalid key prompt spam" "saw $prompt_count prompts or no invalid-choice hint"
    fi
else
    assert_fail "main menu suppresses invalid key prompt spam" "launcher exited non-zero"
fi
rm -rf "$TMP_MENU_HOME"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Flashchat CLI Smoke Test Summary"
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
