# Stage 0 — Native chat template vs. flashchat current rendering

## Summary

Flashchat hand-rolls prompt assembly. The model (Qwen3.6-35B-A3B-4bit) ships
a Jinja2 chat_template that defines the format it was post-trained on.
Honoring that template — instead of approximating it — eliminates the entire
class of symptoms we've been patching.

**Size comparison** (same opencode request, same 11 tools):

| Render | chars | Notes |
|---|---|---|
| Flashchat verbatim (legacy) | ~48,000 | Original `args=...` prose, prefill hits opencode timeout |
| Flashchat compact mode | 17,815 | Strips JSONSchema info, breaks accuracy |
| **Native template** | **34,392** | Correct format the model was trained on |

Native is ~2× compact mode in chars but is the format the model is *home* in.
Speed work belongs after this.

---

## Critical structural differences

### 1. Tool call output format is XML, not JSON

**Native (what the model emits):**
```
<tool_call>
<function=bash>
<parameter=command>
ls -la
</parameter>
<parameter=description>
List files
</parameter>
</function>
</tool_call>
```

**What flashchat tells the model to emit (system prompt prose):**
```
<tool_call>
{"name":"tool_name","arguments":{...}}
</tool_call>
```

**What flashchat's parser expects:**
JSON inside `<tool_call>...</tool_call>` tags.

**Consequences observed:**
- Model fell back to markdown ` ```json{...}``` ` because we gave it
  conflicting instructions vs. training prior — neither format matched
  what it knew, so it defaulted to "code in markdown."
- Even when it produced JSON, the parser only matched the *wrapper* but
  lost native-format calls entirely.
- Our markdown-fence parser branch (commit `666f71e` on the experimental
  branch) was a workaround for this, not a fix.

**Implication:** parser must be rewritten for the XML form. The current
JSON-based parser would not match a correctly-trained model output.

### 2. System block ordering is inverted

**Native:**
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{tool1_json}
{tool2_json}
...
</tools>

If you choose to call a function ONLY reply in the following format...

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format...
- Required parameters MUST be specified
- You may provide optional reasoning...
</IMPORTANT>

{user's actual system prompt content}<|im_end|>
```

**Flashchat current:**
```
<|im_start|>system
{user's system prompt}

Tools:
If a tool is needed, reply with only:
<tool_call>...
Functions:
- bash: ...
  params: ...
<|im_end|>
```

**Implication:** native puts tool definitions *first*, user system content
*last*, with explicit `<IMPORTANT>` reminder language about required
parameters. The "missing required `description`" bug is a direct
consequence of stripping these reminders.

### 3. Tool definitions use full JSON, one per line

**Native:** each tool's complete OpenAI-spec JSON object on its own line:
```
<tools>
{"function":{"description":"...","name":"bash","parameters":{"properties":{...},"required":["command","description"],"type":"object"}},"type":"function"}
{"function":{"description":"...","name":"read_file","parameters":{...}},"type":"function"}
...
</tools>
```

**Flashchat compact:** lossy custom rendering that strips JSONSchema
metadata including `required`, nested type info, etc.

**Flashchat verbatim:** keeps schemas but in custom prose format the model
wasn't trained on.

**Implication:** native preserves all schema fidelity (required fields,
types, descriptions, nested structures) in the form the model expects.
The compaction work was the wrong direction — it stripped the very fields
that drive schema-compliant tool calls.

### 4. `<think>` tag handling

**Native:** assistant prompt always opens with `<think>\n` (reasoning on)
or injects `<think>\n\n</think>\n\n` if `enable_thinking=false`. The
template guarantees this — there is no "reasoning off" without explicitly
emitting empty think tags.

**Flashchat:** appears to handle think tokens via runtime token-id
detection (suppressing them from output) without rendering the actual
`<think>` framing in the prompt.

**Implication:** model emits `first_token=248068` (`<think>`) every time
because its training expects to start with one. The early "thinking
bleeding into chat" symptoms came from us not framing the assistant turn
correctly.

### 5. Tool result wrapping

**Native:**
```
<|im_start|>user
<tool_response>
{result}
</tool_response><|im_end|>
```

**Flashchat:** plain user message with the result as content.

**Implication:** subsequent turns of a tool-using conversation render
differently. Currently moot because we're only validating round-1 calls,
but matters as soon as multi-step agent flows are tested.

### 6. Tool call instructions phrasing

The native template includes specific post-trained phrasing:

> If you choose to call a function ONLY reply in the following format with NO suffix
>
> ...
>
> `<IMPORTANT>`
> Reminder:
> - Function calls MUST follow the specified format: an inner `<function=...></function>` block must be nested within `<tool_call></tool_call>` XML tags
> - Required parameters MUST be specified
> - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
> - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
> `</IMPORTANT>`

These exact strings are what the model was trained against. Approximations
work less reliably.

---

## Mapping symptoms → root cause

| Symptom we patched | Root cause |
|---|---|
| Markdown ` ```json ``` ` instead of `<tool_call>` | Model defaulted to markdown because we asked for JSON-in-`<tool_call>` (neither training nor canonical) |
| Required arg `description` omitted | Custom rendering stripped JSONSchema `required: [...]` info |
| Wrong field name `path` vs `workdir` | Same — schema flattening hid the actual property names from the model |
| `<think>` bleeding into chat | Assistant turn never framed with proper `<think>` opening |
| Title-gen runaway loop | Non-native chat template + sampling drove model out of trained EOS distribution |
| Verbatim tool prompt too large | Tool definitions duplicated from `tools[]` field as prose, redundant |

Every symptom traces back to: **we are not using the model's chat template.**

---

## What this implies for next stages

### Stage 1 (ship native rendering for one model)

**Light path** (recommended first):
- Port Qwen3.6's specific chat template logic to C/Objective-C
- Replace `build_system_prompt_for_request` and the tool-instruction
  builder with a single template renderer
- Rewrite `parse_tool_call_from_buffer` for the XML format:
  `<function=name>\n<parameter=key>\nvalue\n</parameter>...</function>`
- Remove the markdown-fence parser branch (becomes dead code)
- Likely also revert title-gen short-circuit (EOS behavior should
  correct itself once we render the template properly — verify before
  removing)

**Estimated effort:** 3–5 days, single model, validates the architectural
premise.

### Stage 2 (generalize)

Either:
- Hardcode a second model family's template (pragmatic, exposes pattern)
- Build a Jinja2-subset evaluator that interprets `chat_template.jinja`
  generically. HF templates use a small subset (`{% for %}`, `{% if %}`,
  `{{ var }}`, `tojson`, `trim`, `string` checks, namespaces, macros).
  ~500–1000 lines of C, pays for itself starting at model 2.

**Estimated effort:** 3–5 days hardcoded second model OR 1–2 weeks for
generic evaluator. Decision deferred until we have one working.

### Stage 3 (speed)

Once accuracy is locked:
- KV snapshot caching of the rendered system+tools block (already works,
  remains valid)
- Prefill batching
- Larger batch sizes during prefill
- Other model-orthogonal optimizations

---

## Recommended path forward

1. **Greenlight Stage 1 light path.**
2. Implement Qwen3.6 native template renderer in C, replacing the current
   prompt assembly path.
3. Rewrite tool-call parser for XML format.
4. Validate with opencode and nanocode.
5. Revert experimental branch's compact-tools and markdown-fence commits
   if validation passes.
6. Decide Stage 2 path (hardcoded #2 vs. generic Jinja) based on what
   Stage 1 reveals about template variability.

The size hit (~17K → ~34K chars) is real but acceptable: prefill caching
makes turn 2+ free, and accuracy is non-negotiable per stated goals.
