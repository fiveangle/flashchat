#!/bin/bash
# Performance regression benchmark for the Flashchat HTTP API.
#
# Design: the matrix is DERIVED FROM THE MODEL REGISTRY (the same source of truth the
# TUI/`/v1/models` use), not a hand-maintained list — so coverage can't drift from what
# the product supports. For every installed registry model that isn't opted out
# (`"benchmark": false`), start the server in its AS-SHIPPED DEFAULT config and run one
# UNIFORM spec (fixed prompts, temp 0, warmup + N timed repeats), recording prefill (TTFT)
# and decode (tok/s) separately into assets/api_perf_log.tsv.
#
# Adding a model to the registry (required for the TUI to select it) automatically adds it
# here. The only thing not auto-covered is feature axes that aren't a model — see AGENTS.md.
#
# Usage: tests/bench_api.sh [--port N] [--repeats N] [--max-tokens N]
#                           [--model-id ID] [--no-perf-log] [--perf-log FILE]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT="9999"
HOST="127.0.0.1"
WARMUP="${BENCH_WARMUP:-1}"
REPEATS="${BENCH_REPEATS:-2}"
# NOTE: lib/config.sh defines globals named MODEL, MAX_TOKENS, TEMPERATURE, WEIGHTS_DIR, etc.
# Use bench-prefixed names here so sourcing it / flashchat_load_config can't clobber ours.
BENCH_MAX_TOK="${BENCH_MAX_TOKENS:-128}"
ONLY_MODEL=""
PERF_LOG_ENABLED=1
PERF_LOG_PATH="${REPO_ROOT}/assets/api_perf_log.tsv"
SERVER_MODE="bench"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --max-tokens) BENCH_MAX_TOK="$2"; shift 2 ;;
        --model-id) ONLY_MODEL="$2"; shift 2 ;;
        --no-perf-log) PERF_LOG_ENABLED=0; shift ;;
        --perf-log) PERF_LOG_PATH="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

BASE_URL="http://${HOST}:${PORT}"
TMPDIR="$(mktemp -d)"
SERVER_PID=""
HOSTNAME_VALUE=""; HW_MODEL_VALUE=""; RAM_GIB_VALUE=""; CPU_SUMMARY_VALUE=""
CASES_RUN=0; CASES_SKIPPED=0

cleanup() { stop_server; rm -rf "${TMPDIR}"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Machine metadata + perf-log row (identical schema to tests/test_api_smoke.sh)
# ---------------------------------------------------------------------------
now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
timestamp_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
git_branch() { git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown; }
git_commit() { git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown; }
safe_sysctl() { sysctl -n "$1" 2>/dev/null || true; }
detect_hostname() { hostname 2>/dev/null || uname -n 2>/dev/null || echo unknown; }
detect_hw_model() { local v; v="$(safe_sysctl hw.model)"; echo "${v:-unknown}"; }
detect_ram_gib() { local b; b="$(safe_sysctl hw.memsize)"; [[ -z "$b" ]] && { echo ""; return; }; awk -v b="$b" 'BEGIN{printf "%.1f",b/(1024*1024*1024)}'; }
detect_gpu_cores() {
    local v
    for k in hw.gpucores hw.gpu.core_count hw.optional.gpu_core_count; do
        v="$(safe_sysctl "$k")"; [[ -n "$v" ]] && { echo "$v"; return; }
    done
    if command -v system_profiler >/dev/null 2>&1; then
        v="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2;exit}')"
        [[ -n "$v" ]] && { echo "$v" | sed -E 's/[^0-9].*$//'; return; }
    fi
    echo ""
}
detect_cpu_summary() {
    local p e g; p="$(safe_sysctl hw.perflevel0.physicalcpu)"; e="$(safe_sysctl hw.perflevel1.physicalcpu)"; g="$(detect_gpu_cores)"
    [[ -z "$p" ]] && p="$(safe_sysctl hw.physicalcpu_max)"; [[ -z "$p" ]] && p="?"; [[ -z "$e" ]] && e="?"; [[ -z "$g" ]] && g="?"
    printf "%sp %se %sg" "$p" "$e" "$g"
}
populate_machine_metadata() {
    HOSTNAME_VALUE="$(detect_hostname)"; HW_MODEL_VALUE="$(detect_hw_model)"
    RAM_GIB_VALUE="$(detect_ram_gib)"; CPU_SUMMARY_VALUE="$(detect_cpu_summary)"
}
PERF_HEADER="timestamp	branch	commit	hostname	hw_model	ram_gib	cpu_summary	model	server_mode	scenario	endpoint	stream	tool_mode	reasoning	temperature	top_p	top_k	min_p	presence_penalty	repetition_penalty	duration_ms	metric_type	metric_value	tok_per_sec	status	notes"
ensure_perf_log_header() {
    [[ $PERF_LOG_ENABLED -ne 1 ]] && return 0
    if [[ ! -f "${PERF_LOG_PATH}" ]]; then
        mkdir -p "$(dirname "${PERF_LOG_PATH}")"
        printf "%s\n" "${PERF_HEADER}" > "${PERF_LOG_PATH}"
    fi
}
# log_perf_row <model> <scenario> <endpoint> <duration_ms> <metric_type> <metric_value> <tok_per_sec> <status> <notes>
log_perf_row() {
    [[ $PERF_LOG_ENABLED -ne 1 ]] && return 0
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(timestamp_iso)" "$(git_branch)" "$(git_commit)" "${HOSTNAME_VALUE}" "${HW_MODEL_VALUE}" \
        "${RAM_GIB_VALUE}" "${CPU_SUMMARY_VALUE}" "$1" "${SERVER_MODE}" "$2" "$3" "true" "none" "0" "0" \
        "1" "0" "0" "0" "1.0" "$4" "$5" "$6" "$7" "$8" "$9" >> "${PERF_LOG_PATH}"
}

# ---------------------------------------------------------------------------
# Registry-driven model resolution
# ---------------------------------------------------------------------------
set +u
export FLASHCHAT_MODEL_CONFIG="${FLASHCHAT_MODEL_CONFIG:-${REPO_ROOT}/assets/model_configs.json}"
source "${REPO_ROOT}/lib/config.sh"
set -u

# Resolve weights/experts dirs for a model id; probe flashchat/q{bits} fallbacks.
# Sets globals MP WD ED BITS NE. Returns 0 if the model's weights are present locally.
MP=""; WD=""; ED=""; BITS=""; NE=""
resolve_model() {
    local id="$1"
    set +u
    FLASHCHAT_MODEL="$id" FLASHCHAT_MODEL_PATH="" FLASHCHAT_WEIGHTS_DIR="" FLASHCHAT_EXPERTS_DIR="" \
        flashchat_load_config >/dev/null 2>&1
    MP="$(flashchat_get MODEL_PATH)"; WD="$(flashchat_get WEIGHTS_DIR)"; ED="$(flashchat_get EXPERTS_DIR)"
    BITS="$(flashchat_model_quant_bits "$id" 2>/dev/null)"; NE="$(flashchat_model_num_experts "$id" 2>/dev/null)"
    set -u
    [[ -z "$MP" || ! -d "$MP" ]] && return 1
    # Weights-dir fallback probe (handles the flashchat/q{bits} layout).
    if [[ -z "$WD" || ! -f "${WD}/model_weights.bin" ]]; then
        for cand in "${WD}" "${MP}/flashchat/q${BITS}" "${MP}/flashchat"; do
            [[ -n "$cand" && -f "${cand}/model_weights.bin" ]] && { WD="$cand"; break; }
        done
    fi
    [[ -f "${WD}/model_weights.bin" ]] || return 1
    # Experts dir. The engine uses FLASHCHAT_EXPERTS_DIR *directly* as the dir
    # holding layer_XX.bin (it does NOT append packed_experts) — so it must point AT
    # packed_experts, not its parent. Earlier this set ED to the parent, so layer files were
    # never found, experts didn't load, and generation was garbage (which is what made MoE
    # "fail" tool calls and produced bogus benchmark numbers). Resolve against WD/q{bits}.
    if [[ "${NE:-0}" -gt 0 ]]; then
        ED=""
        for cand in "${WD}/packed_experts" "${MP}/flashchat/q${BITS}/packed_experts" "${MP}/flashchat/packed_experts"; do
            [[ -d "$cand" && ( -f "$cand/layer_00.bin" || -n "$(ls "$cand" 2>/dev/null | head -1)" ) ]] && { ED="$cand"; break; }
        done
        [[ -n "$ED" ]] || return 1
    fi
    return 0
}

start_server() {
    local id="$1"
    ( cd "${REPO_ROOT}/metal_infer"
      FLASHCHAT_MODEL_CONFIG="${FLASHCHAT_MODEL_CONFIG}" FLASHCHAT_MODEL="${id}" \
      FLASHCHAT_MODEL_PATH="${MP}" FLASHCHAT_WEIGHTS_DIR="${WD}" FLASHCHAT_EXPERTS_DIR="${ED}" \
        ./infer --serve "${PORT}" --model-id "${id}" --model "${MP}" >"${TMPDIR}/server.log" 2>&1 ) &
    SERVER_PID="$!"
    local tries=0
    while [[ $tries -lt 120 ]]; do
        curl -fsS "${BASE_URL}/health" >/dev/null 2>&1 && return 0
        kill -0 "${SERVER_PID}" 2>/dev/null || return 1
        sleep 1; tries=$((tries+1))
    done
    return 1
}
stop_server() {
    [[ -z "${SERVER_PID}" ]] && return 0
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    # belt-and-suspenders: free the port for the next model
    lsof -ti tcp:"${PORT}" 2>/dev/null | xargs -r kill 2>/dev/null || true
    sleep 1
}

# ---------------------------------------------------------------------------
# Uniform benchmark spec (one global block — identical for every model)
# ---------------------------------------------------------------------------
PROMPT_NAMES=("technical" "creative" "worstcase")
PROMPT_TEXTS=(
"Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list. Respond with only the code."
"Write a short story of about 120 words about a lighthouse keeper who befriends a migrating whale."
"First write a Python function to compute the nth Fibonacci number iteratively, then explain its time and space complexity in clear prose, then end with a haiku about recursion."
)

# json_body <endpoint> <prompt_text> -> writes JSON payload to stdout
json_body() {
    python3 - "$1" "$2" "$BENCH_MAX_TOK" <<'PY'
import json, sys
endpoint, prompt, mx = sys.argv[1], sys.argv[2], int(sys.argv[3])
if endpoint == "chat":
    body = {"model":"bench","messages":[{"role":"user","content":prompt}],
            "stream":True,"max_tokens":mx,"temperature":0}
else:
    body = {"model":"bench","input":prompt,"stream":True,
            "max_output_tokens":mx,"temperature":0}
print(json.dumps(body))
PY
}

# run_stream <endpoint> <prompt_text> -> echoes "ttft_ms decode_tps total_ms deltas"
# Timestamps each SSE line as it arrives (perl, resident) and keys off REAL token deltas:
#  - the server flushes the role/preamble chunk instantly, so curl's first-byte time is NOT
#    the first token -> TTFT must be the time of the first token delta.
#  - the model runs in reasoning mode by default, so token deltas are "reasoning_content"
#    then "content" (chat) / "response.output_text.delta" (responses); count both.
# TTFT = first_delta - start; decode tok/s = (n-1) / (last_delta - first_delta).
run_stream() {
    local endpoint="$1" prompt="$2"
    local url payload stamped grep_pat start
    if [[ "$endpoint" == "chat" ]]; then
        url="${BASE_URL}/v1/chat/completions"; grep_pat='"(reasoning_)?content":"'
    else
        # Match the JSON "type" field, NOT the bare event name: the responses endpoint frames
        # each token as an `event:` line AND a `data:` line, so the bare string would count 2x.
        url="${BASE_URL}/v1/responses"; grep_pat='"type":"response\.output_text\.delta"'
    fi
    payload="${TMPDIR}/payload.json"; stamped="${TMPDIR}/stamped.txt"
    json_body "$endpoint" "$prompt" > "$payload"
    start="$(now_ms)"
    curl -sS -N -X POST "$url" -H 'Content-Type: application/json' --data-binary @"$payload" 2>/dev/null \
        | perl -MTime::HiRes=time -ne 'BEGIN{$|=1} printf "%d\t%s", time()*1000, $_' > "$stamped"
    awk -F'\t' -v start="$start" -v pat="$grep_pat" '
        $2 ~ pat { if (first == "") first=$1; last=$1; n++ }
        END {
            ttft  = (first=="") ? 0 : first-start;
            dt    = (n>1) ? (last-first)/1000.0 : 0;
            tps   = (n>1 && dt>0) ? (n-1)/dt : 0;
            total = (last=="") ? 0 : last-start;
            printf "%.1f %.3f %.0f %d", ttft, tps, total, n;
        }' "$stamped"
}

median() { printf "%s\n" "$@" | sort -n | awk '{a[NR]=$1} END{ if(NR==0){print 0} else if(NR%2){print a[(NR+1)/2]} else {printf "%.3f",(a[NR/2]+a[NR/2+1])/2} }'; }
minval()  { printf "%s\n" "$@" | sort -n | head -1; }

# bench_case <model> <endpoint> <prompt_name> <prompt_text>
bench_case() {
    local model="$1" endpoint="$2" pname="$3" ptext="$4"
    local ep_path scenario; scenario="${endpoint}_stream:${pname}"
    [[ "$endpoint" == "chat" ]] && ep_path="/v1/chat/completions" || ep_path="/v1/responses"
    # warmup (discarded)
    local w; for ((w=0; w<WARMUP; w++)); do run_stream "$endpoint" "$ptext" >/dev/null; done
    local tps_list=() ttft_list=() total_list=() r out
    for ((r=0; r<REPEATS; r++)); do
        out="$(run_stream "$endpoint" "$ptext")"
        ttft_list+=("$(echo "$out" | awk '{print $1}')")
        tps_list+=("$(echo "$out" | awk '{print $2}')")
        total_list+=("$(echo "$out" | awk '{print $3}')")
    done
    local m_tps m_ttft min_tps m_total status
    m_tps="$(median "${tps_list[@]}")"; min_tps="$(minval "${tps_list[@]}")"
    m_ttft="$(median "${ttft_list[@]}")"; m_total="$(median "${total_list[@]}")"
    status="pass"; awk -v t="$m_tps" 'BEGIN{exit !(t>0)}' || status="warn"
    printf "    %-22s decode=%6.2f tok/s (min %.2f)  prefill=%.0f ms\n" "$scenario" "$m_tps" "$min_tps" "$m_ttft"
    log_perf_row "$model" "$scenario" "$ep_path" "$m_total" "decode_tok_per_sec" "$m_tps" "$m_tps" "$status" "min_tps=${min_tps} n=${REPEATS}"
    log_perf_row "$model" "$scenario" "$ep_path" "$m_ttft" "prefill_ms"         "$m_ttft" ""       "$status" "n=${REPEATS}"
}

# bench_tool <model> -> one forced tool-call latency row (fixed prompt, coverage)
bench_tool() {
    local model="$1" body start end dur status
    body="${TMPDIR}/tool.json"
    start="$(now_ms)"
    curl -sS -X POST "${BASE_URL}/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @- > "$body" <<'JSON'
{"model":"bench","messages":[{"role":"user","content":"Record the result 'ok'."}],
 "tools":[{"type":"function","function":{"name":"record_result","description":"Record a result",
   "parameters":{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}}}],
 "tool_choice":{"type":"function","function":{"name":"record_result"}},"max_tokens":256,"temperature":0}
JSON
    end="$(now_ms)"; dur=$((end-start))
    status="pass"; grep -q '"name":"record_result"' "$body" || status="warn"
    printf "    %-22s %d ms (%s)\n" "tool_call" "$dur" "$status"
    log_perf_row "$model" "chat_tool_call" "/v1/chat/completions" "$dur" "tool_call_ms" "$dur" "" "$status" "forced record_result"
}

# ---------------------------------------------------------------------------
# Main: iterate the registry
# ---------------------------------------------------------------------------
echo "=== Flashchat API performance benchmark ==="
populate_machine_metadata
ensure_perf_log_header
echo "machine: ${HOSTNAME_VALUE} ${HW_MODEL_VALUE} ${RAM_GIB_VALUE}GB (${CPU_SUMMARY_VALUE})"
echo "commit:  $(git_branch)@$(git_commit)  | repeats=${REPEATS} warmup=${WARMUP} max_tokens=${BENCH_MAX_TOK}"
echo "log:     ${PERF_LOG_PATH}"
echo

MODEL_IDS="$(flashchat_list_models 2>/dev/null | awk -F'\t' '{print $1}')"
[[ -z "$MODEL_IDS" ]] && { echo "ERROR: no models in registry (${FLASHCHAT_MODEL_CONFIG})" >&2; exit 1; }

for id in $MODEL_IDS; do
    [[ -n "$ONLY_MODEL" && "$id" != "$ONLY_MODEL" ]] && continue
    # Lowercase: flashchat_model_field renders the JSON boolean false as Python's "False".
    bench_flag="$(set +u; flashchat_model_field "$id" benchmark 2>/dev/null; set -u)"
    bench_flag="$(printf '%s' "$bench_flag" | tr '[:upper:]' '[:lower:]')"
    if [[ "$bench_flag" == "false" || "$bench_flag" == "0" || "$bench_flag" == "no" ]]; then
        echo -e "${YELLOW}skip${NC}  ${id}  (benchmark=false)"; CASES_SKIPPED=$((CASES_SKIPPED+1)); continue
    fi
    if ! resolve_model "$id"; then
        echo -e "${YELLOW}skip${NC}  ${id}  (weights not installed locally)"; CASES_SKIPPED=$((CASES_SKIPPED+1)); continue
    fi
    echo -e "${GREEN}bench${NC} ${id}  experts=${NE} bits=${BITS}"
    echo "      weights=${WD}"
    if ! start_server "$id"; then
        echo -e "${RED}FAIL${NC}  ${id} server did not become ready" >&2
        [[ -f "${TMPDIR}/server.log" ]] && tail -5 "${TMPDIR}/server.log" >&2
        stop_server; CASES_SKIPPED=$((CASES_SKIPPED+1)); continue
    fi
    grep -m1 '^\[perf\]' "${TMPDIR}/server.log" 2>/dev/null | sed 's/^/      /' || true
    for i in "${!PROMPT_NAMES[@]}"; do
        for ep in chat responses; do
            bench_case "$id" "$ep" "${PROMPT_NAMES[$i]}" "${PROMPT_TEXTS[$i]}"
        done
    done
    bench_tool "$id"
    stop_server
    CASES_RUN=$((CASES_RUN+1))
    echo
done

echo "=== done: ${CASES_RUN} model(s) benchmarked, ${CASES_SKIPPED} skipped ==="
[[ $PERF_LOG_ENABLED -eq 1 ]] && echo "rows appended to ${PERF_LOG_PATH} — run 'make bench-report' to compare."
exit 0
