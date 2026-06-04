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
    "model": "mlx-community-Qwen36-35B-A3B-4bit",
    "stream": False,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
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

render_output=$("$INFER" --model-id mlx-community-Qwen36-35B-A3B-4bit --render-request "$REQUEST_JSON" --render-output "$RENDER_DIR" 2>&1)
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

# Native chat_template.jinja places the tool block FIRST, then a `\n\n`
# separator, then the user-supplied system content. The 35B-A3B model was
# post-trained with this order; reversing it shifts the tool grammar far
# out of distribution and degrades tool-calling reliability vs lmstudio.
sys_prompt_path="${RENDER_DIR}/system_prompt.txt"
tools_offset=$(grep -b -m1 '^# Tools' "$sys_prompt_path" | cut -d: -f1)
user_offset=$(grep -b -m1 'You are validating' "$sys_prompt_path" | cut -d: -f1)
if [[ -n "$tools_offset" && -n "$user_offset" && "$tools_offset" -lt "$user_offset" ]]; then
    assert_pass "tool block precedes user system content (native order)"
else
    assert_fail "tool block precedes user system content (native order)" \
        "tools_offset=$tools_offset user_offset=$user_offset"
fi

assert_contains "conversation uses empty think for reasoning off" "<think>" "${RENDER_DIR}/conversation.txt"
assert_contains "assembled prompt opens assistant turn" "<|im_start|>assistant" "${RENDER_DIR}/assembled_prompt.txt"
assert_contains "summary includes top_k" '"top_k": 20' "${RENDER_DIR}/summary.json"
assert_contains "summary includes presence penalty" '"presence_penalty": 0.000' "${RENDER_DIR}/summary.json"

# ----- froggeric v19 chat-template fixes -----
# Build a second request that exercises: developer role, <|think_off|> toggle,
# preserved <think> on old assistant turn, empty-reasoning new assistant turn,
# and a Tier-2 tool-error escalation (2 consecutive errors).

V19_REQUEST_JSON="${TMPDIR}/v19_request.json"
V19_RENDER_DIR="${TMPDIR}/v19_rendered"

python3 - "$V19_REQUEST_JSON" <<'PY'
import json, sys
request = {
    "model": "mlx-community-Qwen36-35B-A3B-4bit",
    "stream": False,
    "messages": [
        {"role": "developer", "content": "Developer-supplied instructions: V19_DEV_MARKER"},
        {"role": "system", "content": "You are validating the v19 fixes."},
        {"role": "user", "content": "First user question. <|think_off|>"},
        {"role": "assistant", "content": "<think>\nOLD_REASONING_MARKER\n</think>\n\nFirst answer."},
        {"role": "user", "content": "Try the tool please."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "record_result",
             "arguments": {"result": "x", "count": 1, "ok": True, "items": []}}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"error\": \"missing parameter description\"}"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "function": {"name": "record_result",
             "arguments": {"result": "x", "count": 1, "ok": True, "items": []}}}
        ]},
        {"role": "tool", "tool_call_id": "c2", "content": "Traceback (most recent call last):\n  File X line 1\nValueError: again"}
    ],
    "tools": [{"type": "function", "function": {
        "name": "record_result",
        "description": "Record a result.",
        "parameters": {"type": "object", "properties": {
            "result": {"type": "string"}, "count": {"type": "integer"},
            "ok": {"type": "boolean"}, "items": {"type": "array", "items": {"type": "string"}}
        }, "required": ["result", "count", "ok", "items"]}
    }}],
    "tool_choice": "auto"
}
with open(sys.argv[1], "w") as f:
    json.dump(request, f)
PY

"$INFER" --model-id mlx-community-Qwen36-35B-A3B-4bit --render-request "$V19_REQUEST_JSON" --render-output "$V19_RENDER_DIR" >/dev/null 2>&1

V19_SYS="${V19_RENDER_DIR}/system_prompt.txt"
V19_CONV="${V19_RENDER_DIR}/conversation.txt"
V19_FULL="${V19_RENDER_DIR}/assembled_prompt.txt"

assert_contains "developer role folded into system prompt" "V19_DEV_MARKER" "$V19_SYS"

# think_off toggle was present in a user message: reasoning should be OFF,
# so the generation prompt MUST contain the empty think block (generation
# prefix only — past turns must not).
if grep -q "<|im_start|>assistant" "$V19_FULL" && \
   tail -c 60 "$V19_FULL" | grep -q "<think>"; then
    assert_pass "think_off toggle flips generation prompt to reasoning-off form"
else
    assert_fail "think_off toggle flips generation prompt to reasoning-off form" \
        "expected empty-think prefix at end of assembled_prompt.txt"
fi

# v19 abolishes empty think on past assistant turns. The two tool-calling
# assistant turns have empty reasoning — they MUST NOT render
# `<think>\n\n</think>` followed by past-turn content. The ONLY allowed
# occurrence of an empty `<think></think>` is the reasoning-off generation
# prefix, which lives at the end of the file. We strip the trailing
# generation prefix in Python (grep is line-oriented and cannot match the
# multi-line pattern).
if python3 - "$V19_CONV" <<'PY'
import sys
src = open(sys.argv[1], "r").read()
idx = src.rfind("<|im_start|>assistant")
past = src[:idx] if idx >= 0 else src
sys.exit(0 if "<think>\n\n</think>" in past else 1)
PY
then
    assert_fail "no empty-<think> wrapping on past assistant turns" \
        "found '<think>\\n\\n</think>' in past-turn-only slice"
else
    assert_pass "no empty-<think> wrapping on past assistant turns"
fi

assert_contains "preserve_thinking keeps old assistant reasoning" "OLD_REASONING_MARKER" "$V19_CONV"

# Tier-2 escalation: 2 consecutive tool errors -> <IMPORTANT> directive in the
# user turn carrying the LAST tool_response (the Traceback). Should not appear
# attached to the first error.
if grep -q "twice in a row" "$V19_CONV"; then
    assert_pass "Tier-2 IMPORTANT directive injected after second tool error"
else
    assert_fail "Tier-2 IMPORTANT directive injected after second tool error" \
        "expected 'twice in a row' marker in conversation.txt"
fi
# The directive must be inside the LAST tool_response user turn, not the first.
tier2_offset=$(grep -b -m1 'twice in a row' "$V19_CONV" | cut -d: -f1)
last_resp_offset=$(grep -b 'Traceback' "$V19_CONV" | tail -1 | cut -d: -f1)
first_resp_offset=$(grep -b 'missing parameter description' "$V19_CONV" | head -1 | cut -d: -f1)
if [[ -n "$tier2_offset" && -n "$last_resp_offset" && -n "$first_resp_offset" \
      && "$tier2_offset" -gt "$last_resp_offset" \
      && "$tier2_offset" -gt "$first_resp_offset" ]]; then
    assert_pass "Tier-2 directive placed after the most recent tool_response"
else
    assert_fail "Tier-2 directive placed after the most recent tool_response" \
        "tier2=$tier2_offset last=$last_resp_offset first=$first_resp_offset"
fi

# ----- Tier-1 escalation (single error) -----
V19_TIER1_JSON="${TMPDIR}/v19_tier1_request.json"
V19_TIER1_DIR="${TMPDIR}/v19_tier1_rendered"

python3 - "$V19_TIER1_JSON" <<'PY'
import json, sys
request = {
    "model": "mlx-community-Qwen36-35B-A3B-4bit",
    "stream": False,
    "messages": [
        {"role": "user", "content": "Try the tool."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "record_result",
             "arguments": {"result": "x", "count": 1, "ok": True, "items": []}}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"error\": \"bad param\"}"}
    ],
    "tools": [{"type": "function", "function": {
        "name": "record_result", "description": "Record a result.",
        "parameters": {"type": "object", "properties": {
            "result": {"type": "string"}, "count": {"type": "integer"},
            "ok": {"type": "boolean"}, "items": {"type": "array", "items": {"type": "string"}}
        }, "required": ["result", "count", "ok", "items"]}
    }}],
    "tool_choice": "auto",
    "reasoning": True
}
with open(sys.argv[1], "w") as f:
    json.dump(request, f)
PY

"$INFER" --model-id mlx-community-Qwen36-35B-A3B-4bit --render-request "$V19_TIER1_JSON" --render-output "$V19_TIER1_DIR" >/dev/null 2>&1

V19_TIER1_FULL="${V19_TIER1_DIR}/assembled_prompt.txt"
if grep -q "reconsider" "$V19_TIER1_FULL"; then
    assert_pass "Tier-1 correction seed injected into assistant think block"
else
    assert_fail "Tier-1 correction seed injected into assistant think block" \
        "expected 'reconsider' marker after generation <think> opener"
fi
# Tier-1 should NOT trigger the Tier-2 user-turn directive.
if grep -q "twice in a row" "$V19_TIER1_FULL"; then
    assert_fail "Tier-1 alone does not trigger Tier-2 IMPORTANT directive" \
        "Tier-2 directive leaked into single-error case"
else
    assert_pass "Tier-1 alone does not trigger Tier-2 IMPORTANT directive"
fi

# ----- Smart error detection: false positives must NOT escalate -----
V19_OK_JSON="${TMPDIR}/v19_ok_request.json"
V19_OK_DIR="${TMPDIR}/v19_ok_rendered"

python3 - "$V19_OK_JSON" <<'PY'
import json, sys
ok_payload = json.dumps({"status": "ok", "error_rate": 0.0, "items": ["a", "b"]})
request = {
    "model": "mlx-community-Qwen36-35B-A3B-4bit",
    "stream": False,
    "messages": [
        {"role": "user", "content": "Try the tool."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "record_result",
             "arguments": {"result": "x", "count": 1, "ok": True, "items": []}}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": ok_payload}
    ],
    "tools": [{"type": "function", "function": {
        "name": "record_result", "description": "Record a result.",
        "parameters": {"type": "object", "properties": {
            "result": {"type": "string"}, "count": {"type": "integer"},
            "ok": {"type": "boolean"}, "items": {"type": "array", "items": {"type": "string"}}
        }, "required": ["result", "count", "ok", "items"]}
    }}],
    "tool_choice": "auto",
    "reasoning": True
}
with open(sys.argv[1], "w") as f:
    json.dump(request, f)
PY

"$INFER" --model-id mlx-community-Qwen36-35B-A3B-4bit --render-request "$V19_OK_JSON" --render-output "$V19_OK_DIR" >/dev/null 2>&1
V19_OK_FULL="${V19_OK_DIR}/assembled_prompt.txt"
if grep -q "reconsider\|twice in a row" "$V19_OK_FULL"; then
    assert_fail "successful response with 'error_rate' does not trigger escalation" \
        "smart detection should ignore 'error_rate' key"
else
    assert_pass "successful response with 'error_rate' does not trigger escalation"
fi

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

parsed=$("$INFER" --model-id mlx-community-Qwen36-35B-A3B-4bit --parse-tool-call "$TOOL_CALL_TXT" | tail -1)
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

# qwen3_coder dialect: a single <parameters>{json}</parameters> block, closed with
# </function> and no </tool_call> (as some Qwen3.6 quants emit under greedy). The parser
# must extract it just like the xml form.
CODER_TOOL_CALL_TXT="${TMPDIR}/coder_tool_call.txt"
cat > "$CODER_TOOL_CALL_TXT" <<'EOF'
<tool_call>
<function=record_result>
<parameters>{"result": "done", "count": 3, "ok": true}</parameters>
</function>
EOF
coder_parsed=$("$INFER" --model-id mlx-community-Qwen36-35B-A3B-4bit --parse-tool-call "$CODER_TOOL_CALL_TXT" | tail -1)
python3 - "$coder_parsed" <<'PY'
import json, sys
outer = json.loads(sys.argv[1])
args = json.loads(outer["arguments"])
assert outer["name"] == "record_result", outer
assert args["result"] == "done", args
assert args["count"] == 3, args
assert args["ok"] is True, args
PY
if [ $? -eq 0 ]; then
    assert_pass "native coder parser extracts <parameters>{json}</parameters> (no </tool_call>)"
else
    assert_fail "native coder parser extracts <parameters>{json}</parameters> (no </tool_call>)"
fi

# coder nested-XML variant: <parameters><KEY>value</KEY>...</parameters> (seen from Qwen3.6
# when a cached system-prompt snapshot nudges the format choice). Typed values like the xml path.
NESTED_TOOL_CALL_TXT="${TMPDIR}/nested_tool_call.txt"
cat > "$NESTED_TOOL_CALL_TXT" <<'EOF'
<tool_call>
<function=record_result>
<parameters>
<result>ok</result>
<count>3</count>
<ok>true</ok>
</parameters>
</function>
</tool_call>
EOF
nested_parsed=$("$INFER" --model-id mlx-community-Qwen36-35B-A3B-4bit --parse-tool-call "$NESTED_TOOL_CALL_TXT" | tail -1)
python3 - "$nested_parsed" <<'PY'
import json, sys
outer = json.loads(sys.argv[1])
args = json.loads(outer["arguments"])
assert outer["name"] == "record_result", outer
assert args["result"] == "ok", args
assert args["count"] == 3, args
assert args["ok"] is True, args
PY
if [ $? -eq 0 ]; then
    assert_pass "native coder parser extracts <parameters><KEY>v</KEY></parameters> (nested-XML)"
else
    assert_fail "native coder parser extracts <parameters><KEY>v</KEY></parameters> (nested-XML)"
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
