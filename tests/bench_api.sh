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
#                           [--allow-busy-system]
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
ALLOW_BUSY_SYSTEM=0

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --max-tokens) BENCH_MAX_TOK="$2"; shift 2 ;;
        --model-id) ONLY_MODEL="$2"; shift 2 ;;
        --no-perf-log) PERF_LOG_ENABLED=0; shift ;;
        --perf-log) PERF_LOG_PATH="$2"; shift 2 ;;
        --allow-busy-system) ALLOW_BUSY_SYSTEM=1; shift ;;
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
SYSTEM_NOTES=""
SYSTEM_CONFOUNDED=0
SYSTEM_REASONS=()

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
sample_system_health() {
    local top_output tm_output therm_output infer_pids
    local cpu_idle="unknown" gpu_util="unknown" load1="unknown"
    local mem_free="unknown" swap_mb="unknown" tm_running="unknown" thermal="unknown"

    top_output="$(top -l 2 -s 1 -n 0 2>/dev/null || true)"
    cpu_idle="$(printf '%s\n' "$top_output" | awk '/CPU usage/{gsub("%", "", $7); idle=$7} END{print idle}')"
    [[ -z "$cpu_idle" ]] && cpu_idle="unknown"
    gpu_util="$(ioreg -r -d 1 -w 0 -c AGXAccelerator 2>/dev/null \
        | sed -n 's/.*"Device Utilization %"=\([0-9][0-9]*\).*/\1/p' | head -1)"
    [[ -z "$gpu_util" ]] && gpu_util="unknown"
    load1="$(safe_sysctl vm.loadavg | awk '{print $2}')"
    [[ -z "$load1" ]] && load1="unknown"
    mem_free="$(memory_pressure -Q 2>/dev/null \
        | awk '/System-wide memory free percentage:/{gsub("%", "", $5); print $5}')"
    [[ -z "$mem_free" ]] && mem_free="unknown"
    swap_mb="$(safe_sysctl vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p')"
    [[ -z "$swap_mb" ]] && swap_mb="unknown"

    tm_output="$(tmutil status 2>/dev/null || true)"
    tm_running="$(printf '%s\n' "$tm_output" \
        | sed -n 's/.*Running = \([01]\).*/\1/p' | head -1)"
    [[ -z "$tm_running" ]] && tm_running="unknown"
    therm_output="$(pmset -g therm 2>/dev/null || true)"
    if printf '%s\n' "$therm_output" | grep -q 'No thermal warning level' \
        && printf '%s\n' "$therm_output" | grep -q 'No performance warning level'; then
        thermal="ok"
    elif [[ -n "$therm_output" ]]; then
        thermal="warning"
    fi

    infer_pids="$(pgrep -f '[/]metal_infer/infer|[/]infer --serve' 2>/dev/null || true)"
    [[ "$tm_running" == "1" ]] && SYSTEM_REASONS+=("Time Machine backup is active")
    [[ "$thermal" == "warning" ]] && SYSTEM_REASONS+=("macOS reports thermal or performance pressure")
    [[ -n "$infer_pids" ]] && SYSTEM_REASONS+=("another inference process is running: ${infer_pids//$'\n'/,}")
    if [[ "$cpu_idle" != "unknown" ]] && awk -v idle="$cpu_idle" 'BEGIN{exit !(idle < 50.0)}'; then
        SYSTEM_REASONS+=("CPU idle is only ${cpu_idle}%")
    fi
    if [[ "$gpu_util" != "unknown" ]] && awk -v util="$gpu_util" 'BEGIN{exit !(util >= 70.0)}'; then
        SYSTEM_REASONS+=("GPU utilization is already ${gpu_util}%")
    fi

    SYSTEM_NOTES="env_cpu_idle=${cpu_idle}%;env_gpu_util=${gpu_util}%;env_load1=${load1};env_mem_free=${mem_free}%;env_swap_mb=${swap_mb};env_time_machine=${tm_running};env_thermal=${thermal}"
    echo "system:  cpu_idle=${cpu_idle}% gpu_util=${gpu_util}% load1=${load1} mem_free=${mem_free}% swap=${swap_mb}MB time_machine=${tm_running} thermal=${thermal}"

    if [[ ${#SYSTEM_REASONS[@]} -eq 0 ]]; then
        return 0
    fi
    echo -e "${RED}Benchmark preflight found invalidating system contention:${NC}" >&2
    local reason
    for reason in "${SYSTEM_REASONS[@]}"; do
        echo "  - $reason" >&2
    done
    if [[ $ALLOW_BUSY_SYSTEM -eq 0 ]]; then
        echo "Wait for the machine to become idle, then rerun." >&2
        echo "For diagnostic reproduction only, pass --allow-busy-system; its rows are marked confounded." >&2
        return 1
    fi
    SYSTEM_CONFOUNDED=1
    echo -e "${YELLOW}Continuing only because --allow-busy-system was supplied; rows will be marked confounded.${NC}" >&2
    return 0
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
    local status="$8" notes="$9"
    if [[ $SYSTEM_CONFOUNDED -eq 1 ]]; then
        status="confounded"
    fi
    [[ -n "$notes" ]] && notes="${notes};"
    notes="${notes}${SYSTEM_NOTES}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(timestamp_iso)" "$(git_branch)" "$(git_commit)" "${HOSTNAME_VALUE}" "${HW_MODEL_VALUE}" \
        "${RAM_GIB_VALUE}" "${CPU_SUMMARY_VALUE}" "$1" "${SERVER_MODE}" "$2" "$3" "true" "none" "0" "0" \
        "1" "0" "0" "0" "1.0" "$4" "$5" "$6" "$7" "$status" "$notes" >> "${PERF_LOG_PATH}"
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
    local id="$1" bench_home="${TMPDIR}/home-${id}" model_config="$FLASHCHAT_MODEL_CONFIG"
    mkdir -p "$bench_home"
    (
        export HOME="$bench_home"
        # A benchmark measures product defaults, not the invoking shell's overrides.
        # Model artifact paths are restored explicitly after clearing persistent keys.
        local key
        for key in \
            MODEL MODEL_PATH WEIGHTS_DIR EXPERTS_DIR MAX_TOKENS SAMPLING_PROFILE \
            REASONING TEMPERATURE TOP_P TOP_K MIN_P PRESENCE_PENALTY REPETITION_PENALTY \
            SERVER_PORT SERVER_HOST SERVER_LOG SERVER_DEBUG SERVER_HTTP_LOG \
            FUSE_LINEAR ADAPTIVE_K_MASS BATCH_PREFILL PREFILL_CHUNK PREFILL_DEBUG PREFILL_RELEASE \
            ANE_PREFILL ANE_MIN_CHUNK PREAD_PROFILE PREAD_PROFILE_CAP \
            EXPERT_PIN_MAX_GB EXPERT_PIN_MAX_EXPERTS EXPERT_PIN_AUTO_FRAC EXPERT_PIN_MLOCK \
            SYSTEM_PROMPT_CACHE SYSTEM_PROMPT_CACHE_MAX_ENTRIES SYSTEM_PROMPT_CACHE_DIR \
            MTP MTP_BF16 ACTIVE_EXPERTS KV_QUANT CONTEXT_WINDOW; do
            unset "FLASHCHAT_${key}"
        done
        unset FLASHCHAT_CONFIG_FILE_OVERRIDE CONFIG_FILE
        export FLASHCHAT_MODEL_CONFIG="$model_config"
        export FLASHCHAT_MODEL="$id"
        export FLASHCHAT_MODEL_PATH="$MP"
        export FLASHCHAT_WEIGHTS_DIR="$WD"
        export FLASHCHAT_EXPERTS_DIR="$ED"

        # Re-source after changing HOME so config paths and defaults are isolated.
        set +u
        source "${REPO_ROOT}/lib/config.sh"
        MODEL="$id"
        flashchat_create_default_config >/dev/null
        flashchat_load_config >/dev/null
        set -u

        local config_file
        config_file="$(flashchat_get CONFIG_FILE)"
        printf '[bench-config] model=%s fuse=%s adaptive_k=%s pin_gb=%s pin_experts=%s pin_auto=%s pin_mlock=%s batch_prefill=%s ane_prefill=%s kv_quant=%s config=%s\n' \
            "$id" "$(flashchat_get FUSE_LINEAR)" "$(flashchat_get ADAPTIVE_K_MASS)" \
            "$(flashchat_get EXPERT_PIN_MAX_GB)" "$(flashchat_get EXPERT_PIN_MAX_EXPERTS)" \
            "$(flashchat_get EXPERT_PIN_AUTO_FRAC)" "$(flashchat_get EXPERT_PIN_MLOCK)" \
            "$(flashchat_get BATCH_PREFILL)" "$(flashchat_get ANE_PREFILL)" \
            "$(flashchat_get KV_QUANT)" "$config_file"

        cd "${REPO_ROOT}/metal_infer"
        FLASHCHAT_FUSE_LINEAR="$(flashchat_get FUSE_LINEAR)" \
        FLASHCHAT_ADAPTIVE_K_MASS="$(flashchat_get ADAPTIVE_K_MASS)" \
        FLASHCHAT_BATCH_PREFILL="$(flashchat_get BATCH_PREFILL)" \
        FLASHCHAT_PREFILL_CHUNK="$(flashchat_get PREFILL_CHUNK)" \
        FLASHCHAT_PREFILL_DEBUG="$(flashchat_get PREFILL_DEBUG)" \
        FLASHCHAT_ANE_PREFILL="$(flashchat_get ANE_PREFILL)" \
        FLASHCHAT_ANE_MIN_CHUNK="$(flashchat_get ANE_MIN_CHUNK)" \
        FLASHCHAT_PREAD_PROFILE="$(flashchat_get PREAD_PROFILE)" \
        FLASHCHAT_PREAD_PROFILE_CAP="$(flashchat_get PREAD_PROFILE_CAP)" \
        FLASHCHAT_EXPERT_PIN_MAX_GB="$(flashchat_get EXPERT_PIN_MAX_GB)" \
        FLASHCHAT_EXPERT_PIN_MAX_EXPERTS="$(flashchat_get EXPERT_PIN_MAX_EXPERTS)" \
        FLASHCHAT_EXPERT_PIN_AUTO_FRAC="$(flashchat_get EXPERT_PIN_AUTO_FRAC)" \
        FLASHCHAT_EXPERT_PIN_MLOCK="$(flashchat_get EXPERT_PIN_MLOCK)" \
        FLASHCHAT_SESSIONS_DIR="$(flashchat_get_sessions_dir)" \
        FLASHCHAT_SYSTEM_PROMPT="$(flashchat_get_system_prompt_file)" \
        FLASHCHAT_ACTIVE_EXPERTS="$(flashchat_get ACTIVE_EXPERTS)" \
        FLASHCHAT_CONTEXT_WINDOW="$(flashchat_get CONTEXT_WINDOW)" \
        FLASHCHAT_KV_QUANT="$(flashchat_get KV_QUANT)" \
            exec ./infer --serve "$PORT" --config "$config_file" \
                --model-id "$id" --model "$MP"
    ) >"${TMPDIR}/server.log" 2>&1 &
    SERVER_PID="$!"
    local tries=0
    while [[ $tries -lt 120 ]]; do
        curl -fsS "${BASE_URL}/health" >/dev/null 2>&1 && return 0
        kill -0 "${SERVER_PID}" 2>/dev/null || return 1
        sleep 1; tries=$((tries+1))
    done
    cat "${TMPDIR}/server.log" >&2
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
echo "machine: ${HOSTNAME_VALUE} ${HW_MODEL_VALUE} ${RAM_GIB_VALUE}GB (${CPU_SUMMARY_VALUE})"
sample_system_health || exit 3
ensure_perf_log_header
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
    grep -E -m3 '^\[(bench-config|perf)\]|^\[serve\]   expert_pin_cache:' \
        "${TMPDIR}/server.log" 2>/dev/null | sed 's/^/      /' || true
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
