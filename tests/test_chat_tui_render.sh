#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHAT="${REPO_ROOT}/metal_infer/chat"

TMPDIR="$(mktemp -d /tmp/flashchat-chat-render.XXXXXX)"
SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

mkdir -p "${TMPDIR}/home"
port_file="${TMPDIR}/port"
body_file="${TMPDIR}/request-body.txt"

python3 - "$port_file" "$body_file" <<'PY' &
import json
import socket
import sys
import time

port_file = sys.argv[1]
body_file = sys.argv[2]
chunks = [
    ("reasoning_content", "SECRET_THINK_STEP\n"),
    "Here is code:\n\n",
    "```html <!DOCTYPE html> <html><body><canvas id=\"game\"></canvas></body></html>```\n",
    "Unicode: \u201cquote\u201d \u2014 done.\n",
]

with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    with open(port_file, "w") as f:
        f.write(str(server.getsockname()[1]))

    for request_index, request_path in enumerate([body_file, body_file + ".show", body_file + ".plain"]):
        conn, _ = server.accept()
        conn.close()

        conn, _ = server.accept()
        with conn:
            request = conn.recv(65536)
            with open(request_path, "wb") as f:
                f.write(request)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            response_chunks = chunks if request_index < 2 else [chunk for chunk in chunks if not isinstance(chunk, tuple)]
            for chunk in response_chunks:
                if isinstance(chunk, tuple):
                    key, text = chunk
                else:
                    key, text = "content", chunk
                payload = {"choices": [{"delta": {key: text}, "finish_reason": None}]}
                line = "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"
                conn.sendall(line.encode("utf-8"))
                time.sleep(0.01)
            if request_index < 2:
                usage_stats = {
                    "completion_tokens": 2559,
                    "thinking_tokens": 1249,
                    "response_tokens": 1120,
                    "ttft_ms": 2200,
                    "generation_ms": 159938,
                    "thinking_ms": 78063,
                    "response_ms": 70000,
                    "experts_mib_per_sec": 432.1,
                    "experts_mib_per_sec_per_expert": 54.0,
                }
            else:
                usage_stats = {
                    "completion_tokens": 1126,
                    "thinking_tokens": 0,
                    "response_tokens": 1126,
                    "ttft_ms": 2000,
                    "generation_ms": 93058,
                    "thinking_ms": 0,
                    "response_ms": 93058,
                    "experts_mib_per_sec": 199.2,
                    "experts_mib_per_sec_per_expert": 19.9,
                }
            usage = {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage_stats,
            }
            line = "data: " + json.dumps(usage, separators=(",", ":")) + "\n\n"
            conn.sendall(line.encode("utf-8"))
            conn.sendall(b"data: [DONE]\n\n")
PY
SERVER_PID=$!

for _ in $(seq 1 50); do
    [[ -f "$port_file" ]] && break
    sleep 0.1
done
if [[ ! -f "$port_file" ]]; then
    echo "FAIL: mock server did not start" >&2
    exit 1
fi

port="$(cat "$port_file")"
raw_output="$(
    printf 'hello\n/quit\n' |
        HOME="${TMPDIR}/home" FLASHCHAT_MODEL=Mock FLASHCHAT_REASONING=1 FLASHCHAT_SHOW_THINKING=0 \
            "$CHAT" --host 127.0.0.1 --port "$port" --max-tokens 200 2>&1
)"
clean_output="$(printf '%s\n' "$raw_output" | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g')"

if [[ "$clean_output" != *'<canvas id="game"></canvas>'* ]]; then
    echo "FAIL: one-line fenced HTML code was not rendered" >&2
    printf '%s\n' "$clean_output" >&2
    exit 1
fi
if [[ "$clean_output" == *'SECRET_THINK_STEP'* ]]; then
    echo "FAIL: hidden reasoning_content was rendered with show thinking disabled" >&2
    printf '%s\n' "$clean_output" >&2
    exit 1
fi
if [[ "$clean_output" != *'2559 tokens, 16.0 tok/s, TTFT 2.2s (1249@16.0tok/s think, 1120@16.0tok/s response), experts 432.1 MiB/s, 54.0 MiB/s/expert'* ]]; then
    echo "FAIL: chat footer did not render server timing breakdown" >&2
    printf '%s\n' "$clean_output" >&2
    exit 1
fi

if ! grep -q '"reasoning":true' "$body_file"; then
    echo "FAIL: chat request did not preserve configured reasoning mode" >&2
    cat "$body_file" >&2
    exit 1
fi

raw_show_output="$(
    printf 'hello again\n/quit\n' |
        HOME="${TMPDIR}/home-show" FLASHCHAT_MODEL=Mock FLASHCHAT_REASONING=1 FLASHCHAT_SHOW_THINKING=1 \
            "$CHAT" --host 127.0.0.1 --port "$port" --max-tokens 200 2>&1
)"
clean_show_output="$(printf '%s\n' "$raw_show_output" | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g')"
if [[ "$clean_show_output" != *'SECRET_THINK_STEP'* ]]; then
    echo "FAIL: reasoning_content was not rendered with show thinking enabled" >&2
    printf '%s\n' "$clean_show_output" >&2
    exit 1
fi
if ! grep -q '"reasoning":true' "${body_file}.show"; then
    echo "FAIL: show-thinking chat request did not preserve configured reasoning mode" >&2
    cat "${body_file}.show" >&2
    exit 1
fi

raw_plain_output="$(
    printf 'hello without thinking\n/quit\n' |
        HOME="${TMPDIR}/home-plain" FLASHCHAT_MODEL=Mock FLASHCHAT_REASONING=0 FLASHCHAT_SHOW_THINKING=0 \
            "$CHAT" --host 127.0.0.1 --port "$port" --max-tokens 200 2>&1
)"
clean_plain_output="$(printf '%s\n' "$raw_plain_output" | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g')"
if [[ "$clean_plain_output" != *'1126 tokens, 12.1 tok/s, TTFT 2.0s, experts 199.2 MiB/s, 19.9 MiB/s/expert'* ]]; then
    echo "FAIL: non-thinking chat footer did not render aggregate timing" >&2
    printf '%s\n' "$clean_plain_output" >&2
    exit 1
fi
if [[ "$clean_plain_output" == *'1126@12.1tok/s response'* ]]; then
    echo "FAIL: non-thinking chat footer rendered redundant response breakdown" >&2
    printf '%s\n' "$clean_plain_output" >&2
    exit 1
fi
if ! grep -q '"reasoning":false' "${body_file}.plain"; then
    echo "FAIL: non-thinking chat request did not preserve disabled reasoning mode" >&2
    cat "${body_file}.plain" >&2
    exit 1
fi

python3 - "${TMPDIR}/home/.config/flashchat/sessions" <<'PY'
import json
import pathlib
import sys

sessions = sorted(pathlib.Path(sys.argv[1]).glob("*.jsonl"))
if not sessions:
    raise SystemExit("FAIL: no session file was written")

assistant = []
for line in sessions[-1].read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    if obj.get("role") == "assistant":
        assistant.append(obj.get("content", ""))

content = "\n".join(assistant)
if '<canvas id="game"></canvas>' not in content:
    raise SystemExit("FAIL: one-line fenced HTML code was not saved")
if "Unicode: \u201cquote\u201d \u2014 done." not in content:
    raise SystemExit("FAIL: Unicode punctuation was not preserved")
if "SECRET_THINK_STEP" in content:
    raise SystemExit("FAIL: reasoning_content was saved as assistant content")
PY

echo "PASS: chat TUI renders one-line code fences and preserves Unicode"
