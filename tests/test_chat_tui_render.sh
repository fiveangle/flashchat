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

python3 - "$port_file" <<'PY' &
import json
import socket
import sys
import time

port_file = sys.argv[1]
chunks = [
    "Here is code:\n\n",
    "```html <!DOCTYPE html> <html><body><canvas id=\"game\"></canvas></body></html>```\n",
    "Unicode: \u201cquote\u201d \u2014 done.\n",
]

with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    with open(port_file, "w") as f:
        f.write(str(server.getsockname()[1]))

    conn, _ = server.accept()
    conn.close()

    conn, _ = server.accept()
    with conn:
        conn.recv(65536)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Connection: close\r\n"
            b"\r\n"
        )
        for chunk in chunks:
            payload = {"choices": [{"delta": {"content": chunk}, "finish_reason": None}]}
            line = "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"
            conn.sendall(line.encode("utf-8"))
            time.sleep(0.01)
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
        HOME="${TMPDIR}/home" FLASHCHAT_MODEL=Mock "$CHAT" --host 127.0.0.1 --port "$port" --max-tokens 200 2>&1
)"
clean_output="$(printf '%s\n' "$raw_output" | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g')"

if [[ "$clean_output" != *'<canvas id="game"></canvas>'* ]]; then
    echo "FAIL: one-line fenced HTML code was not rendered" >&2
    printf '%s\n' "$clean_output" >&2
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
PY

echo "PASS: chat TUI renders one-line code fences and preserves Unicode"
