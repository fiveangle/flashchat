#!/bin/bash
set -euo pipefail

HOST="127.0.0.1"
PORT="9999"
REQUEST_JSON=""
LOG_PATH=""
MAX_TOKENS="1"
REPEATS="1"
STREAM_MODE="preserve"

usage() {
    cat <<EOF
Usage: $0 --request-json FILE [--host HOST] [--port N] [--log FILE]
          [--max-tokens N] [--repeats N] [--stream preserve|true|false]

Replays a captured /v1/chat/completions request against an already-running
Flashchat server, caps generation, and reports the server-side prefill timing
from the persistent server log.

The input request file is never modified.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --request-json) REQUEST_JSON="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --log) LOG_PATH="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --stream) STREAM_MODE="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$REQUEST_JSON" ]]; then
    echo "Missing --request-json FILE" >&2
    usage >&2
    exit 2
fi
if [[ ! -f "$REQUEST_JSON" ]]; then
    echo "Request JSON not found: $REQUEST_JSON" >&2
    exit 1
fi
if [[ "$STREAM_MODE" != "preserve" && "$STREAM_MODE" != "true" && "$STREAM_MODE" != "false" ]]; then
    echo "--stream must be preserve, true, or false" >&2
    exit 2
fi
if ! [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] || [[ "$MAX_TOKENS" -lt 1 ]]; then
    echo "--max-tokens must be a positive integer" >&2
    exit 2
fi
if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
    echo "--repeats must be a positive integer" >&2
    exit 2
fi

CONFIG_FILE="$HOME/.config/flashchat/config"
TMPDIR="$(mktemp -d /tmp/flashchat-prefill-replay.XXXXXX)"

cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [[ -z "$LOG_PATH" ]]; then
    if [[ -f "$CONFIG_FILE" ]]; then
        LOG_PATH="$(python3 - "$CONFIG_FILE" <<'PY'
import pathlib
import shlex
import sys

config = pathlib.Path(sys.argv[1])
values = {}
for line in config.read_text(errors="replace").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, raw = line.split("=", 1)
    try:
        values[key] = shlex.split(raw)[0]
    except Exception:
        values[key] = raw.strip().strip('"')

log_path = values.get("SERVER_LOG_PATH", "")
if log_path:
    p = pathlib.Path(log_path).expanduser()
    if p.suffix:
        print(p)
    else:
        print(p / "server.log")
else:
    print(pathlib.Path.home() / ".config" / "flashchat" / "logs" / "server.log")
PY
)"
    else
        LOG_PATH="$HOME/.config/flashchat/logs/server.log"
    fi
fi

if [[ ! -f "$LOG_PATH" ]]; then
    echo "Server log not found: $LOG_PATH" >&2
    echo "Pass --log FILE if the server writes logs somewhere else." >&2
    exit 1
fi

BASE_URL="http://$HOST:$PORT"
if ! curl -fsS "$BASE_URL/health" >/dev/null; then
    echo "Flashchat server is not reachable at $BASE_URL" >&2
    echo "Start it first, then rerun this script." >&2
    exit 1
fi

make_replay_request() {
    local out="$1"
    python3 - "$REQUEST_JSON" "$out" "$MAX_TOKENS" "$STREAM_MODE" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
max_tokens = int(sys.argv[3])
stream_mode = sys.argv[4]

data = json.loads(src.read_text())
data["max_tokens"] = max_tokens
if stream_mode == "true":
    data["stream"] = True
elif stream_mode == "false":
    data["stream"] = False

dst.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
}

summarize_request() {
    python3 - "$REQUEST_JSON" "$MAX_TOKENS" "$STREAM_MODE" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
messages = data.get("messages") or []
tools = data.get("tools") or []
text = json.dumps(data, ensure_ascii=False)
chars = 0
for msg in messages:
    content = msg.get("content", "")
    if isinstance(content, str):
        chars += len(content)
    else:
        chars += len(json.dumps(content, ensure_ascii=False))

print(f"request={path}")
print(f"request_bytes={path.stat().st_size}")
print(f"messages={len(messages)} tools={len(tools)} content_chars={chars} opencode={str('opencode' in text.lower()).lower()}")
print(f"source_stream={data.get('stream', '<absent>')} replay_stream={sys.argv[3]} source_max_tokens={data.get('max_tokens', '<absent>')} replay_max_tokens={sys.argv[2]}")
PY
}

parse_log_window() {
    local start_offset="$1"
    python3 - "$LOG_PATH" "$start_offset" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
offset = int(sys.argv[2])
with path.open("rb") as f:
    f.seek(offset)
    text = f.read().decode("utf-8", "replace")

ids = re.findall(r"\[serve\] ((?:chatcmpl|resp)-\d+) endpoint=", text)
if not ids:
    print("status=missing_log_window")
    print("No request id found in server log after replay.")
    sys.exit(1)

request_id = ids[0]
lines = [line for line in text.splitlines() if f"[serve] {request_id} " in line or f"[mtp] {request_id} " in line]

fields = {"request_id": request_id}
patterns = [
    ("endpoint", r"endpoint=([^ ]+)"),
    ("prompt_chars", r"prompt_chars=(\d+)"),
    ("tools", r"tools=(\d+)"),
    ("active_experts", r"active_experts=([^ ]+)"),
    ("reasoning", r"reasoning=(\d+)"),
    ("snapshot", r"snapshot=(\d+)"),
    ("cache_hit_tokens", r"sys_prompt_cache (?:disk )?hit hash=\d+ tokens=(\d+)"),
    ("cache_miss_tokens", r"sys_prompt_cache miss .*tokens=(\d+)"),
    ("tokenized_prompt_tokens", r"tokenized prompt tokens=(\d+)"),
    ("tokenized_continuation_tokens", r"tokenized snapshot continuation tokens=(\d+)"),
    ("prefill_tokens", r"prefill=(\d+) tokens in ([0-9.]+)ms"),
    ("first_token", r"first_token=([^ ]+)"),
    ("generated_tokens", r"generated=(\d+) tokens in ([0-9.]+)ms"),
]

for line in lines:
    for key, pattern in patterns:
        m = re.search(pattern, line)
        if not m:
            continue
        if key == "prefill_tokens":
            fields["prefill_tokens"] = m.group(1)
            fields["prefill_ms"] = m.group(2)
        elif key == "generated_tokens":
            fields["generated_tokens"] = m.group(1)
            fields["generated_ms"] = m.group(2)
        else:
            fields[key] = m.group(1)

print("status=ok")
for key in [
    "request_id",
    "endpoint",
    "prompt_chars",
    "tools",
    "active_experts",
    "reasoning",
    "snapshot",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "tokenized_prompt_tokens",
    "tokenized_continuation_tokens",
    "prefill_tokens",
    "prefill_ms",
    "first_token",
    "generated_tokens",
    "generated_ms",
]:
    if key in fields:
        print(f"{key}={fields[key]}")

if "prefill_tokens" in fields and "prefill_ms" in fields:
    tokens = float(fields["prefill_tokens"])
    ms = float(fields["prefill_ms"])
    if ms > 0:
        print(f"prefill_tok_per_sec={tokens / (ms / 1000.0):.3f}")
        print(f"prefill_ms_per_token={ms / tokens:.3f}")
PY
}

summarize_request
echo "base_url=$BASE_URL"
echo "server_log=$LOG_PATH"

for ((i = 1; i <= REPEATS; i++)); do
    replay_json="$TMPDIR/replay-$i.json"
    response_file="$TMPDIR/response-$i.txt"
    make_replay_request "$replay_json"
    start_offset="$(stat -f %z "$LOG_PATH")"
    start_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"
    curl -fsS -N -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data-binary @"$replay_json" >"$response_file"
    end_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"
    echo "--- repeat $i/$REPEATS ---"
    echo "client_duration_ms=$((end_ms - start_ms))"
    parse_log_window "$start_offset"
done
