#!/bin/bash
set -euo pipefail

HOST="127.0.0.1"
PORT="9999"
START_SERVER=1
STARTED_SERVER=0
SERVER_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${HOME}/.config/flashchat/config"

MODEL_ID=""
MODEL_PATH=""
WEIGHTS_DIR=""
EXPERTS_DIR=""
MODEL_CONFIG=""
PERF_LOG_ENABLED=1
PERF_LOG_PATH="${REPO_ROOT}/assets/api_perf_log.tsv"
SERVER_MODE="reused"
LAST_DURATION_MS=""
HOSTNAME_VALUE=""
HW_MODEL_VALUE=""
RAM_GIB_VALUE=""
CPU_SUMMARY_VALUE=""
PASSED=0
FAILED=0
SKIPPED=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [--port N] [--host HOST] [--no-start] [--no-perf-log] [--perf-log FILE]

Smoke-tests the local Flashchat HTTP API:
  - GET /health
  - GET /v1
  - GET /v1/models
  - POST /v1/chat/completions (stream=false)
  - POST /v1/chat/completions (stream=true)
  - POST /v1/chat/completions tool-call round trip
  - POST /v1/responses (stream=false)
  - POST /v1/responses (stream=true)
  - POST /v1/responses tool-call round trip

By default, starts metal_infer/infer --serve if nothing is already listening.
When perf logging is enabled, appends timing rows to a local TSV.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --no-start)
            START_SERVER=0
            shift
            ;;
        --no-perf-log)
            PERF_LOG_ENABLED=0
            shift
            ;;
        --perf-log)
            PERF_LOG_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

BASE_URL="http://${HOST}:${PORT}"
TMPDIR="$(mktemp -d /tmp/flashchat-api-smoke.XXXXXX)"

cleanup() {
    if [[ $STARTED_SERVER -eq 1 && -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    rm -rf "${TMPDIR}"
}
trap cleanup EXIT

load_flashchat_config() {
    if [[ -n "${MODEL_ID}" ]]; then
        export FLASHCHAT_MODEL="${MODEL_ID}"
    fi

    set +u
    source "${REPO_ROOT}/lib/config.sh"
    flashchat_load_config
    set -u

    MODEL_ID="$(flashchat_get MODEL)"
    MODEL_PATH="$(flashchat_get MODEL_PATH)"
    WEIGHTS_DIR="$(flashchat_get WEIGHTS_DIR)"
    EXPERTS_DIR="$(flashchat_get EXPERTS_DIR)"
    MODEL_CONFIG="$(flashchat_get MODEL_CONFIG)"

    export FLASHCHAT_MODEL="${MODEL_ID}"
    export FLASHCHAT_MODEL_PATH="${MODEL_PATH}"
    export FLASHCHAT_WEIGHTS_DIR="${WEIGHTS_DIR}"
    export FLASHCHAT_EXPERTS_DIR="${EXPERTS_DIR}"
    export FLASHCHAT_MODEL_CONFIG="${MODEL_CONFIG}"
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC}  ${label} preflight" >&2
        echo "ERROR: ${label} is not available for ${MODEL_ID}." >&2
        echo "Expected: ${path}" >&2
        echo "Run ./flashchat setup first, or select a configured model with completed runtime artifacts." >&2
        print_summary
        exit 1
    fi
}

preflight_model_artifacts() {
    if [[ -z "${MODEL_PATH}" || "${MODEL_PATH}" == *"<snapshot>"* || ! -d "${MODEL_PATH}" ]]; then
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC}  model snapshot preflight" >&2
        echo "ERROR: Model is not downloaded for ${MODEL_ID}." >&2
        echo "Expected model snapshot: ${MODEL_PATH}" >&2
        echo "Run ./flashchat setup first, or select a configured model with downloaded weights." >&2
        print_summary
        exit 1
    fi

    require_file "${WEIGHTS_DIR}/model_weights.bin" "Extracted model weights"
    require_file "${WEIGHTS_DIR}/model_weights.json" "Model weights manifest"
    require_file "${WEIGHTS_DIR}/vocab.bin" "Tokenizer vocabulary"
    require_file "${EXPERTS_DIR}/layer_00.bin" "Packed experts"
}

now_ms() {
    python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
}

git_branch() {
    git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"
}

git_commit() {
    git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

timestamp_iso() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

safe_sysctl() {
    local key="$1"
    sysctl -n "$key" 2>/dev/null || true
}

detect_hostname() {
    hostname 2>/dev/null || uname -n 2>/dev/null || echo "unknown"
}

detect_hw_model() {
    local value
    value="$(safe_sysctl hw.model)"
    if [[ -n "$value" ]]; then
        echo "$value"
    else
        echo "unknown"
    fi
}

detect_ram_gib() {
    local bytes
    bytes="$(safe_sysctl hw.memsize)"
    if [[ -z "$bytes" ]]; then
        echo ""
        return 0
    fi
    awk -v bytes="$bytes" 'BEGIN { printf "%.1f", bytes / (1024*1024*1024) }'
}

detect_gpu_cores() {
    local value
    for key in hw.gpucores hw.gpu.core_count hw.optional.gpu_core_count; do
        value="$(safe_sysctl "$key")"
        if [[ -n "$value" ]]; then
            echo "$value"
            return 0
        fi
    done

    if command -v system_profiler >/dev/null 2>&1; then
        value="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/ { print $2; exit }')"
        if [[ -n "$value" ]]; then
            echo "$value" | sed -E 's/[^0-9].*$//'
            return 0
        fi
    fi
    echo ""
}

detect_cpu_summary() {
    local p e g total
    p="$(safe_sysctl hw.perflevel0.physicalcpu)"
    e="$(safe_sysctl hw.perflevel1.physicalcpu)"
    g="$(detect_gpu_cores)"
    total="$(safe_sysctl hw.physicalcpu_max)"

    if [[ -z "$p" && -n "$total" ]]; then
        p="$total"
    fi
    [[ -z "$p" ]] && p="?"
    [[ -z "$e" ]] && e="?"
    [[ -z "$g" ]] && g="?"
    printf "%sp %se %sg" "$p" "$e" "$g"
}

populate_machine_metadata() {
    HOSTNAME_VALUE="$(detect_hostname)"
    HW_MODEL_VALUE="$(detect_hw_model)"
    RAM_GIB_VALUE="$(detect_ram_gib)"
    CPU_SUMMARY_VALUE="$(detect_cpu_summary)"
}

ensure_perf_log_header() {
    if [[ $PERF_LOG_ENABLED -ne 1 ]]; then
        return 0
    fi
    if [[ ! -f "${PERF_LOG_PATH}" ]]; then
        mkdir -p "$(dirname "${PERF_LOG_PATH}")"
        printf "timestamp\tbranch\tcommit\thostname\thw_model\tram_gib\tcpu_summary\tmodel\tserver_mode\tscenario\tendpoint\tstream\ttool_mode\treasoning\ttemperature\ttop_p\tduration_ms\tmetric_type\tmetric_value\ttok_per_sec\tstatus\tnotes\n" > "${PERF_LOG_PATH}"
    fi
}

log_perf_row() {
    local scenario="$1"
    local endpoint="$2"
    local stream="$3"
    local tool_mode="$4"
    local reasoning="$5"
    local temperature="$6"
    local top_p="$7"
    local duration_ms="$8"
    local metric_type="$9"
    local metric_value="${10}"
    local status="${11}"
    local notes="${12}"
    local tok_per_sec=""

    if [[ $PERF_LOG_ENABLED -ne 1 ]]; then
        return 0
    fi

    if [[ -n "${metric_value}" && "${metric_value}" != "0" && -n "${duration_ms}" && "${duration_ms}" != "0" && "${metric_type}" == "stream_deltas" ]]; then
        tok_per_sec="$(awk -v tokens="${metric_value}" -v dur="${duration_ms}" 'BEGIN { printf "%.3f", (tokens * 1000.0) / dur }')"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(timestamp_iso)" \
        "$(git_branch)" \
        "$(git_commit)" \
        "${HOSTNAME_VALUE}" \
        "${HW_MODEL_VALUE}" \
        "${RAM_GIB_VALUE}" \
        "${CPU_SUMMARY_VALUE}" \
        "${MODEL_ID}" \
        "${SERVER_MODE}" \
        "${scenario}" \
        "${endpoint}" \
        "${stream}" \
        "${tool_mode}" \
        "${reasoning}" \
        "${temperature}" \
        "${top_p}" \
        "${duration_ms}" \
        "${metric_type}" \
        "${metric_value}" \
        "${tok_per_sec}" \
        "${status}" \
        "${notes}" >> "${PERF_LOG_PATH}"
}

measure_request_to_file() {
    local outfile="$1"
    shift
    local start_ms end_ms
    start_ms="$(now_ms)"
    "$@" > "${outfile}"
    end_ms="$(now_ms)"
    LAST_DURATION_MS="$((end_ms - start_ms))"
}

count_chat_stream_deltas() {
    local file="$1"
    grep -c '"content":"' "${file}" || true
}

count_responses_stream_deltas() {
    local file="$1"
    grep -c 'response.output_text.delta' "${file}" || true
}

extract_json_text_length() {
    local file="$1"
    python3 - "$file" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print("")
    raise SystemExit(0)

text = ""
if data.get("object") == "chat.completion":
    choices = data.get("choices") or []
    if choices:
        text = ((choices[0].get("message") or {}).get("content")) or ""
elif data.get("object") == "response":
    text = data.get("output_text") or ""
print(len(text))
PY
}

wait_for_server() {
    local tries=0
    while [[ $tries -lt 60 ]]; do
        if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [[ $STARTED_SERVER -eq 1 && -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            return 1
        fi
        sleep 1
        tries=$((tries + 1))
    done
    return 1
}

assert_contains() {
    local file="$1"
    local pattern="$2"
    local label="$3"
    if ! grep -q "$pattern" "$file"; then
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC}  ${label}" >&2
        echo "--- ${file} ---" >&2
        cat "$file" >&2
        print_summary
        exit 1
    fi
    PASSED=$((PASSED + 1))
    echo -e "${GREEN}PASS${NC}  ${label}"
}

assert_not_contains() {
    local file="$1"
    local pattern="$2"
    local label="$3"
    if grep -q "$pattern" "$file"; then
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC}  ${label}" >&2
        echo "--- ${file} ---" >&2
        cat "$file" >&2
        print_summary
        exit 1
    fi
    PASSED=$((PASSED + 1))
    echo -e "${GREEN}PASS${NC}  ${label}"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "Flashchat API Smoke Test Summary"
    echo "========================================"
    echo -e "${GREEN}Passed:${NC}  $PASSED"
    echo -e "${RED}Failed:${NC}  $FAILED"
    echo -e "${YELLOW}Skipped:${NC} $SKIPPED"
    echo ""
    if [[ $FAILED -gt 0 ]]; then
        echo -e "${RED}Some tests failed.${NC}"
    else
        echo -e "${GREEN}All tests passed!${NC}"
    fi
}

echo "=== Flashchat API Smoke Test ==="
echo "Base URL: ${BASE_URL}"
load_flashchat_config
echo "Model: ${MODEL_ID}"
populate_machine_metadata
ensure_perf_log_header

if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    if [[ $START_SERVER -ne 1 ]]; then
        echo "ERROR: server is not reachable at ${BASE_URL} and --no-start was used" >&2
        exit 1
    fi
    echo ""
    echo "--- Starting local server ---"
    preflight_model_artifacts
    (
        cd "${REPO_ROOT}/metal_infer"
        ./infer --serve "${PORT}" --model-id "${MODEL_ID}" --model "${MODEL_PATH}" >"${TMPDIR}/server.log" 2>&1
    ) &
    SERVER_PID="$!"
    STARTED_SERVER=1
    SERVER_MODE="fresh"
    if ! wait_for_server; then
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC}  server became ready" >&2
        echo "ERROR: server did not become ready" >&2
        [[ -f "${TMPDIR}/server.log" ]] && cat "${TMPDIR}/server.log" >&2
        print_summary
        exit 1
    fi
else
    echo ""
    echo "--- Reusing existing server ---"
fi

echo ""
echo "--- GET /health ---"
measure_request_to_file "${TMPDIR}/health.json" curl -fsS "${BASE_URL}/health"
cat "${TMPDIR}/health.json"
echo ""
assert_contains "${TMPDIR}/health.json" '"status":"ok"' "health returned ok"
log_perf_row "health" "/health" "false" "none" "" "" "" "${LAST_DURATION_MS}" "none" "" "pass" ""

echo ""
echo "--- GET /v1 ---"
measure_request_to_file "${TMPDIR}/v1_root.json" curl -fsS "${BASE_URL}/v1"
cat "${TMPDIR}/v1_root.json"
echo ""
assert_contains "${TMPDIR}/v1_root.json" '"object":"service"' "v1 root returned service object"
log_perf_row "v1_root" "/v1" "false" "none" "" "" "" "${LAST_DURATION_MS}" "none" "" "pass" ""

echo ""
echo "--- GET /v1/models ---"
measure_request_to_file "${TMPDIR}/models.json" curl -fsS "${BASE_URL}/v1/models"
cat "${TMPDIR}/models.json"
echo ""
assert_contains "${TMPDIR}/models.json" "\"${MODEL_ID}\"" "models lists expected id"
log_perf_row "models" "/v1/models" "false" "none" "" "" "" "${LAST_DURATION_MS}" "none" "" "pass" ""

echo ""
echo "--- POST /v1/chat/completions (stream=false) ---"
measure_request_to_file "${TMPDIR}/chat.json" curl -fsS -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "messages": [{"role":"user","content":"Reply with exactly: smoke ok"}],
      "stream": false,
      "max_tokens": 32,
      "temperature": 0
    }'
cat "${TMPDIR}/chat.json"
echo ""
assert_contains "${TMPDIR}/chat.json" '"object":"chat.completion"' "chat completion object present"
assert_contains "${TMPDIR}/chat.json" '"content":"' "chat completion includes assistant content"
log_perf_row "chat_text_nonstream" "/v1/chat/completions" "false" "none" "default" "0" "" "${LAST_DURATION_MS}" "text_chars" "$(extract_json_text_length "${TMPDIR}/chat.json")" "pass" ""

echo ""
echo "--- POST /v1/chat/completions (stream=true) ---"
measure_request_to_file "${TMPDIR}/chat_stream.txt" curl -fsS -N -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "messages": [{"role":"user","content":"Repeat the word benchmark 24 times separated by spaces and nothing else."}],
      "stream": true,
      "max_tokens": 96,
      "temperature": 0
    }'
head -n 20 "${TMPDIR}/chat_stream.txt"
assert_contains "${TMPDIR}/chat_stream.txt" 'chat.completion.chunk' "chat stream emitted chunks"
assert_contains "${TMPDIR}/chat_stream.txt" '\[DONE\]' "chat stream completed"
log_perf_row "chat_text_stream" "/v1/chat/completions" "true" "none" "default" "0" "" "${LAST_DURATION_MS}" "stream_deltas" "$(count_chat_stream_deltas "${TMPDIR}/chat_stream.txt")" "pass" "repeat benchmark x24"

echo ""
echo "--- POST /v1/chat/completions tool-call round trip ---"
measure_request_to_file "${TMPDIR}/chat_tool_call.json" curl -fsS -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "messages": [{"role":"user","content":"Use the provided tool before answering."}],
      "stream": false,
      "max_tokens": 96,
      "temperature": 0,
      "tools": [{
        "type": "function",
        "function": {
          "name": "record_result",
          "description": "Record a provided result string for the assistant to use in its final answer.",
          "parameters": {
            "type": "object",
            "properties": {
              "result": {"type": "string"}
            },
            "required": ["result"]
          }
        }
      }],
      "tool_choice": {
        "type": "function",
        "function": {"name": "record_result"}
      }
    }'
cat "${TMPDIR}/chat_tool_call.json"
echo ""
assert_contains "${TMPDIR}/chat_tool_call.json" '"tool_calls"' "chat tool call emitted"
assert_contains "${TMPDIR}/chat_tool_call.json" '"finish_reason":"tool_calls"' "chat tool call finish reason"
assert_contains "${TMPDIR}/chat_tool_call.json" '"name":"record_result"' "chat tool call used forced function"
log_perf_row "chat_tool_call" "/v1/chat/completions" "false" "forced" "default" "0" "" "${LAST_DURATION_MS}" "tool_calls" "1" "pass" "forced record_result"

measure_request_to_file "${TMPDIR}/chat_tool_followup.json" curl -fsS -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "messages": [
        {"role":"user","content":"Use the provided tool before answering."},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_test","type":"function","function":{"name":"record_result","arguments":"{\"result\":\"tool smoke ok\"}"}}]},
        {"role":"tool","tool_call_id":"call_test","name":"record_result","content":"tool smoke ok"}
      ],
      "stream": false,
      "max_tokens": 64,
      "temperature": 0,
      "tools": [{
        "type": "function",
        "function": {
          "name": "record_result",
          "description": "Record a provided result string for the assistant to use in its final answer.",
          "parameters": {
            "type": "object",
            "properties": {
              "result": {"type": "string"}
            },
            "required": ["result"]
          }
        }
      }],
      "tool_choice": "none"
    }'
cat "${TMPDIR}/chat_tool_followup.json"
echo ""
assert_contains "${TMPDIR}/chat_tool_followup.json" '"content":"' "chat follow-up returned content"
assert_not_contains "${TMPDIR}/chat_tool_followup.json" '"tool_calls"' "chat follow-up did not emit another tool call"
log_perf_row "chat_tool_followup" "/v1/chat/completions" "false" "followup" "default" "0" "" "${LAST_DURATION_MS}" "text_chars" "$(extract_json_text_length "${TMPDIR}/chat_tool_followup.json")" "pass" "tool_choice none"

echo ""
echo "--- POST /v1/responses (stream=false) ---"
measure_request_to_file "${TMPDIR}/responses.json" curl -fsS -X POST "${BASE_URL}/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "input": "Reply with exactly: responses ok",
      "stream": false,
      "max_output_tokens": 32,
      "temperature": 0
    }'
cat "${TMPDIR}/responses.json"
echo ""
assert_contains "${TMPDIR}/responses.json" '"object":"response"' "responses object present"
assert_contains "${TMPDIR}/responses.json" '"output"' "responses output present"
log_perf_row "responses_text_nonstream" "/v1/responses" "false" "none" "default" "0" "" "${LAST_DURATION_MS}" "text_chars" "$(extract_json_text_length "${TMPDIR}/responses.json")" "pass" ""

echo ""
echo "--- POST /v1/responses (stream=true) ---"
measure_request_to_file "${TMPDIR}/responses_stream.txt" curl -fsS -N -X POST "${BASE_URL}/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "input": "Repeat the word benchmark 24 times separated by spaces and nothing else.",
      "stream": true,
      "max_output_tokens": 96,
      "temperature": 0
    }'
head -n 20 "${TMPDIR}/responses_stream.txt"
assert_contains "${TMPDIR}/responses_stream.txt" 'response.completed' "responses stream completed event present"
assert_contains "${TMPDIR}/responses_stream.txt" '\[DONE\]' "responses stream completed"
log_perf_row "responses_text_stream" "/v1/responses" "true" "none" "default" "0" "" "${LAST_DURATION_MS}" "stream_deltas" "$(count_responses_stream_deltas "${TMPDIR}/responses_stream.txt")" "pass" "repeat benchmark x24"

echo ""
echo "--- POST /v1/responses tool-call round trip ---"
measure_request_to_file "${TMPDIR}/responses_tool_call.json" curl -fsS -X POST "${BASE_URL}/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "input": "Use the provided tool before answering.",
      "stream": false,
      "max_output_tokens": 96,
      "temperature": 0,
      "tools": [{
        "type": "function",
        "function": {
          "name": "record_result",
          "description": "Record a provided result string for the assistant to use in its final answer.",
          "parameters": {
            "type": "object",
            "properties": {
              "result": {"type": "string"}
            },
            "required": ["result"]
          }
        }
      }],
      "tool_choice": {
        "type": "function",
        "function": {"name": "record_result"}
      }
    }'
cat "${TMPDIR}/responses_tool_call.json"
echo ""
assert_contains "${TMPDIR}/responses_tool_call.json" '"type":"function_call"' "responses function_call emitted"
assert_contains "${TMPDIR}/responses_tool_call.json" '"name":"record_result"' "responses tool call used forced function"
log_perf_row "responses_tool_call" "/v1/responses" "false" "forced" "default" "0" "" "${LAST_DURATION_MS}" "tool_calls" "1" "pass" "forced record_result"

measure_request_to_file "${TMPDIR}/responses_tool_followup.json" curl -fsS -X POST "${BASE_URL}/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "${MODEL_ID}",
      "input": [
        {"type":"message","role":"user","content":"Use the provided tool before answering."},
        {"type":"function_call","call_id":"call_resp","name":"record_result","arguments":"{\"result\":\"responses tool smoke ok\"}"},
        {"type":"function_call_output","call_id":"call_resp","name":"record_result","output":"responses tool smoke ok"}
      ],
      "stream": false,
      "max_output_tokens": 64,
      "temperature": 0,
      "tool_choice": "none",
      "tools": [{
        "type": "function",
        "function": {
          "name": "record_result",
          "description": "Record a provided result string for the assistant to use in its final answer.",
          "parameters": {
            "type": "object",
            "properties": {
              "result": {"type": "string"}
            },
            "required": ["result"]
          }
        }
      }]
    }'
cat "${TMPDIR}/responses_tool_followup.json"
echo ""
assert_contains "${TMPDIR}/responses_tool_followup.json" '"output_text":"' "responses follow-up returned output text"
assert_not_contains "${TMPDIR}/responses_tool_followup.json" '"type":"function_call"' "responses follow-up did not emit another function call"
log_perf_row "responses_tool_followup" "/v1/responses" "false" "followup" "default" "0" "" "${LAST_DURATION_MS}" "text_chars" "$(extract_json_text_length "${TMPDIR}/responses_tool_followup.json")" "pass" "tool_choice none"

echo ""
if [[ $PERF_LOG_ENABLED -eq 1 ]]; then
    echo "Perf log: ${PERF_LOG_PATH}"
fi
print_summary
