#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER="${REPO_ROOT}/metal_infer/infer"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
PASSED=0
FAILED=0

assert_pass() {
    PASSED=$((PASSED + 1))
    echo -e "${GREEN}PASS${NC}  $1"
}

assert_fail() {
    FAILED=$((FAILED + 1))
    echo -e "${RED}FAIL${NC}  $1${2:+: $2}"
}

assert_contains() {
    local name="$1"
    local needle="$2"
    local path="$3"
    if grep -q "$needle" "$path"; then
        assert_pass "$name"
    else
        assert_fail "$name" "expected '$needle' in $path"
    fi
}

TMPDIR="$(mktemp -d /tmp/flashchat-tool-template.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

export HOME="${TMPDIR}/home"
mkdir -p "$HOME"

REQUEST_JSON="${TMPDIR}/request.json"
RENDER_DIR="${TMPDIR}/rendered"
TOOL_CALL_TXT="${TMPDIR}/tool_call.txt"

python3 - "$REQUEST_JSON" <<'PY'
import json
import sys

tail = "x" * 6000 + " SCHEMA_TAIL_MARKER"
tools = [{
    "type": "function",
    "function": {
        "name": "record_result",
        "description": "Record a result.",
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": tail
                },
                "count": {"type": "integer"},
                "ok": {"type": "boolean"},
                "items": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["result", "count", "ok", "items"]
        }
    }
}]
for i in range(1, 30):
    tools.append({
        "type": "function",
        "function": {
            "name": f"dummy_tool_{i:02d}",
            "description": f"Dummy tool {i}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": []
            }
        }
    })
request = {
    "model": "qwen3.6-35B-A3B",
    "stream": False,
    "messages": [
        {"role": "system", "content": "You are validating Flashchat's native tool renderer."},
        {"role": "user", "content": "Use the tool if needed."}
    ],
    "tools": tools,
    "tool_choice": "auto"
}
with open(sys.argv[1], "w") as f:
    json.dump(request, f)
PY

echo ""
echo "=== Flashchat Tool Template Render Smoke ==="
echo ""

render_output=$("$INFER" --model-id qwen3.6-35B-A3B --render-request "$REQUEST_JSON" --render-output "$RENDER_DIR" 2>&1)
if printf "%s" "$render_output" | grep -q "tools=30"; then
    assert_pass "render request reports thirty tools"
else
    assert_fail "render request reports thirty tools" "$render_output"
fi

assert_contains "system prompt has native tools header" "# Tools" "${RENDER_DIR}/system_prompt.txt"
assert_contains "system prompt has tools block" "<tools>" "${RENDER_DIR}/system_prompt.txt"
assert_contains "system prompt keeps late tool definitions" "dummy_tool_29" "${RENDER_DIR}/system_prompt.txt"
assert_contains "system prompt keeps required fields" '"required":\["result","count","ok","items"\]' "${RENDER_DIR}/system_prompt.txt"
assert_contains "system prompt preserves large schema tail" "SCHEMA_TAIL_MARKER" "${RENDER_DIR}/system_prompt.txt"
assert_contains "system prompt has native important block" "<IMPORTANT>" "${RENDER_DIR}/system_prompt.txt"
assert_contains "conversation uses empty think for reasoning off" "<think>" "${RENDER_DIR}/conversation.txt"
assert_contains "assembled prompt opens assistant turn" "<|im_start|>assistant" "${RENDER_DIR}/assembled_prompt.txt"

cat > "$TOOL_CALL_TXT" <<'EOF'
some leading text
<tool_call>
<function=record_result>
<parameter=result>
line one
line two
</parameter>
<parameter=count>
3
</parameter>
<parameter=ok>
true
</parameter>
<parameter=items>
["a","b"]
</parameter>
</function>
</tool_call>
EOF

parsed=$("$INFER" --model-id qwen3.6-35B-A3B --parse-tool-call "$TOOL_CALL_TXT" | tail -1)
python3 - "$parsed" <<'PY'
import json
import sys

outer = json.loads(sys.argv[1])
args = json.loads(outer["arguments"])
assert outer["name"] == "record_result"
assert args["result"] == "line one\nline two"
assert args["count"] == 3
assert args["ok"] is True
assert args["items"] == ["a", "b"]
PY
if [ $? -eq 0 ]; then
    assert_pass "native XML parser preserves multiline and typed params"
else
    assert_fail "native XML parser preserves multiline and typed params"
fi

echo ""
echo "========================================"
echo "Flashchat Tool Template Render Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC}  $PASSED"
echo -e "${RED}Failed:${NC}  $FAILED"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi

echo -e "${GREEN}All tests passed!${NC}"
