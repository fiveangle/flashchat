# Tool Calling Investigation: Learnings & Debugging Walkthrough

> **Date**: 2026-05-06 / 2026-05-07
> **Branch**: `implimen-native-qwen-chat-template`
> **Symptom**: nanocoder (running against flashchat) produced garbled or missing tool calls; the same model running in lmstudio / Inferencer.app produced clean tool calls with default settings.
> **Root cause**: flashchat hardcoded `K=4` active experts per token regardless of model. The Qwen3.6-35B-A3B model expects `K=8` per its config; running it under-experted produced systematically perturbed logits, making the model fragile under any non-trivial sampling and outright wrong under greedy decoding.

This document captures the full investigation arc, including the wrong turns. The point isn't to celebrate the fix — it's to capture the **debugging methodology** so future drift bugs (which will look superficially similar) can be diagnosed faster.

---

## Initial symptom

Nanocoder run against the flashchat server, simple "what's in this directory?" prompt, 30 tools available including `list_directory`. Instead of a proper native Qwen tool call:

```
<tool_call>
<function=list_directory>
<parameter=path>
.
</parameter>
</function>
</tool_call>
```

…flashchat produced things like:

```
list_
Path: /Users/
<function_re
<tool_use_desc>
</tool
</function
<function
</function_
```

or

```
list_directory(path=".", recursive=true, maxDepth=2)   # Python-style
```

or

```
<tool_call>
<function=list_directory>
<path>                <- bare key, missing parameter= prefix
.
</parameter></function>     <- missing </tool_call>
```

Variable, but consistently broken. Sometimes the model would just return prose ("List of directory contents:") and emit no tool call at all.

User report: "the same model running in lmstudio has no issue performing tool calling … i just want tool calling to be on parity with established inference servers."

---

## Investigation arc

### Hypothesis 1 — Chat template ordering (CORRECT but partial)

**Reasoning**: the Qwen3.6 model ships with `chat_template.jinja` (153 lines of Jinja2). When tools are present, the template emits the tool grammar **first** as the system block primer, then user system content after a `\n\n` separator:

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{tool1 json}
{tool2 json}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:
... example ...
<IMPORTANT>... reminder ...</IMPORTANT>

{user_system_content}<|im_end|>
```

Flashchat's `build_system_prompt_for_request` had it **reversed**: user system content first, then tool block. Comment in code even said "Keeping the concrete tool grammar last gives it precedence over client-provided system text" — but that contradicts the trained format.

With a 11kB nanocoder system prompt, the format primer was buried 220 lines deep, far out of distribution from how the model was trained.

**Verification**: rendered the same request via Python jinja2 against the actual `chat_template.jinja` — diff showed only cosmetic JSON key ordering after the reorder fix.

**Result**: 5/5 success on chatcmpl-2 replay (nanocoder request) with explicit sampling params (`temp=0.6, top_p=0.95, presence_penalty=0`). Looked like the bug was solved.

[metal_infer/infer.m:6466](metal_infer/infer.m:6466) — fix landed.

### Hypothesis 2 — Persistent disk cache corrupting state (WRONG, but valuable detour)

After my reorder, the user did `make clean`, restarted the server, and reported nanocoder produced **garbage** output: `<function=execute_3Bash>` with an injected `3` mid-token, fragments, no `<tool_call>` envelope. The output looked like genuine state corruption — not just a bad sample.

The persistent disk cache feature (commit `ca4efaa`) saves KV state to a `.fcache` file after the first prefill, so subsequent server starts skip the cold prefill. After `make clean`, the binary was rebuilt; the `.fcache` was left intact and reloaded.

**Hypothesis**: bug in the disk save/load path corrupted state. In-memory cache hits worked, disk-loaded cache hits produced garbage.

#### What we built (and kept)

Three-layer test infrastructure to localize where state went wrong:

1. **Synthetic unit test** — `infer --cache-roundtrip-test`. Synthesizes deterministic snapshot data sized like real KV/conv/SSM tensors, runs through `save_system_prompt_disk_cache` + `load_system_prompt_disk_cache` into a temp dir, memcmps loaded vs original. Catches: serializer bugs, header rewrite at line 8135, LZFSE roundtrip, FNV1a checksum, chunk ordering, struct layout drift.

   - File: [tests/test_disk_cache_roundtrip.sh](tests/test_disk_cache_roundtrip.sh)
   - Wired into `make cache-roundtrip-smoke` and `make test`.
   - Result: **PASS** — serializer is byte-correct.

2. **Runtime validator** — `FLASHCHAT_CACHE_VALIDATE=1`. After `save_system_prompt_disk_cache` writes the file, immediately reload it into shadow buffers and `memcmp` against the in-memory snapshot we just persisted. Catches: actual save/load mismatches with real live data.

   - Result: **PASS** — disk file bytes match in-memory snapshot bytes for all 140 chunks.

3. **Live-state fingerprint** — `FLASHCHAT_CACHE_FINGERPRINT=1`. Hashes (FNV1a 64-bit) every persistent runtime buffer immediately after cold prefill (post-capture) and again immediately after restore-from-snapshot (in-memory or disk). Per-buffer diff. Catches: snapshot doesn't faithfully represent live runtime state.

   - Result: **PASS** — every buffer (KV K/V CPU+GPU, linear conv/ssm CPU, GPU buf_delta_state, GPU buf_conv_state) byte-identical between post-cold-prefill and post-restore.

#### Conclusion at this point: cache is innocent

All three layers passed. The cache infrastructure correctly:
- Serializes state to disk
- Reloads bytes identical to what was saved
- Restores into a runtime that exactly matches the live post-prefill runtime

Yet output still varied wildly between runs.

### Hypothesis 3 — Sampling defaults (PARTIAL fix, also wrong root cause)

After finding the cache wasn't the bug, I ran 8 cache-hit requests against the same snapshot (so state was identical) and got 3/8 success. With temp=0.6, top_p=0.95 it improved to 4/5. Greedy (temp=0): 0/5.

I proposed: when tools are active, override defaults to thinking-mode sampling (temp=0.6, top_p=0.95), matching Qwen3 model card guidance.

**Why this didn't smell right**: greedy decoding (temp=0) is deterministic — same input must produce same output if math is identical. Yet greedy on flashchat consistently picked prose ("List of directory contents:") instead of `<tool_call>`, while Inferencer with greedy on the same model produced reliable tool calls. Two engines, same weights, greedy decoding, different output → math must differ.

This is the moment the user said: *"I'm not so confident it's the sampling specifically … makes me wonder more if our experts extraction offsets are 100% correct (as that is the fundamental difference between our 'streaming' model and these native ram-resident std style engines)."*

**The user was right.** The sampling tweaks were papering over the real bug.

### Root cause — K=4 hardcoded, model expects K=8

Inferencer's settings (screenshots from user):
- Temperature: 0.00 (greedy)
- Number of Experts: **8**
- Thinking: off
- Repetition Penalty: 1.10

Flashchat's `assets/model_configs.json` for Qwen3.6-35B-A3B:
```json
"num_experts": 256,
"num_experts_per_tok": 8,
```

But flashchat's main loop hardcoded `int K = 4;` and never read `num_experts_per_tok` from config. The CLAUDE.md note said "K=4 for speed" — which was a deliberate tradeoff for the **397B** model (where 10 active experts × 6.75MB = 67MB of SSD I/O per token vs K=4's 27MB), but the same hardcode silently applied to all models including the 35B-A3B that fits in RAM.

For Qwen3.5-397B-A17B, `num_experts_per_tok=10` — K=4 means **40%** of trained-active experts.
For Qwen3.6-35B-A3B, `num_experts_per_tok=8` — K=4 means **50%** of trained-active experts.

Half the experts means each MoE layer's output is a (different) weighted combination of half the experts. The model still produces vaguely-correct output most of the time, but logits are systematically perturbed. Under greedy this manifests as different argmax → different tokens → different sequences. Under sampling, the perturbation happens to push the right tokens out of the top-p mass sometimes; high-entropy sampling occasionally recovers.

#### Verification

After patching K to read from config (defaulting to `num_experts_per_tok` when `--k` not specified), reran the **same** nanocoder request with the **same** Inferencer-matching settings (temp=0.01, top_p=1.0, repetition_penalty=1.10, reasoning=off):

```
run 1 t=652.0s  OK  list_directory({"path":"."})  <- cold prefill
run 2 t=  5.4s  OK  list_directory({"path":"."})  <- cache hit
run 3 t=  4.2s  OK  list_directory({"path":"."})
run 4 t=  6.1s  OK  list_directory({"path":"."})
run 5 t=  8.9s  OK  list_directory({"path":"."})
SUMMARY  TOOL_OK=5/5  WEIRD=0/5  NO_TOOL=0/5
DETERMINISTIC: all runs identical
```

5/5, byte-identical output across all runs. This is what greedy decoding *should* look like. With K=4 the same test produced 0/5.

[metal_infer/infer.m:9487](metal_infer/infer.m:9487) and [metal_infer/infer.m:9707](metal_infer/infer.m:9707) — fix landed.

---

## Why this was hard to find

Several things conspired to hide K=4 as the root cause:

1. **The hardcode was old and load-bearing** for the 397B showcase model. Nobody added a TODO when 35B was added, and the CLAUDE.md note implied K=4 was a universal "for speed" choice rather than a per-model tradeoff.

2. **The under-experting was graceful**, not catastrophic. The model produced *almost* the right tokens, *almost* always — just enough to look like sampling instability rather than systematic numerical drift. This is the worst kind of bug: the failure mode looks like normal model fragility.

3. **The disk-cache rabbit hole was plausible**. The persistent cache was added recently (`ca4efaa`), the failure was correlated with a `make clean` + restart, and "garbage output after restoring from disk" is exactly what a cache corruption would produce. Spent significant effort building three layers of validators before concluding the cache was innocent.

4. **The chat-template fix was real and produced an immediate 5/5 win** in early testing, which reinforced the impression that the surface-level format issues were the root cause. They weren't — the early 5/5 used `top_p=0.95`, which provided enough sampling diversity to recover from the K=4 perturbation.

5. **Greedy parity was the test we should have run earlier.** Same model, same weights, same prompt, deterministic decoding → output **must** match across engines if math is correct. We had the means to run this test all along; we just didn't until the user nudged us toward the experts-extraction hypothesis.

---

## Methodology lessons (for future drift bugs)

### Always run the greedy-parity test early when comparing inference engines

If two engines purport to run the same model, they *must* produce identical token sequences under greedy decoding (modulo deliberate non-determinism like atomic GPU reductions, which are rare and bounded). When they don't, the math differs. This collapses an open-ended debugging space ("why is output bad?") into a localized one ("which step's output differs?").

In this case, greedy-flashchat (K=4) on first token = `<` to start prose, greedy-Inferencer (K=8) on first token = `<tool_call>` opener. Two different argmax → math differs → fix the math.

### Build instrumentation that isolates by elimination, not by guessing

The three-layer cache validator was built to localize a hypothesized bug to a specific tensor. The PASS results, while ruling out the hypothesis, were valuable: they collapsed the suspect surface, freeing attention for other suspects. *Negative results from instrumentation are progress.*

The instrumentation is permanent (gated behind env vars / make targets). Future regressions in the same code paths will trip these tests automatically.

### Distrust hardcoded constants that smell like global tuning

`#define MAX_K 8`, `int K = 4;`, `NUM_IO_THREADS 4` — three constants that all encoded the same assumption (K=4 active experts per token). Each of these is a per-model property that should derive from the model's config. The `MAX_K=8` cap also means the codebase **silently caps K at 8** even for models with K>8 (e.g., the 397B's K=10) — that's a known under-experting case still present in this branch and called out with a loud warning at startup.

### Don't trust your own "the early fix worked" signal

The chat-template reorder produced 5/5 in my first replay test. That looked like proof. It wasn't — the test request had explicit `top_p=0.95`, which masked the K=4 perturbation. The user's nanocoder traffic uses *no sampling overrides*, so flashchat's defaults (top_p=0.8) kicked in, the perturbation was no longer masked, and the bug reappeared. Always test in the *actual user configuration*, not in the contrived test that reproduces "the bug."

---

## Summary of fixes landed in this session

| # | Fix | File | Mechanism |
|---|---|---|---|
| 1 | Native chat-template ordering: `# Tools` block first, user system content after `\n\n` separator | [metal_infer/infer.m:6466](metal_infer/infer.m:6466) | Match `chat_template.jinja` order |
| 2 | Reasoning-on-by-default when tools active and client doesn't pin it | [metal_infer/infer.m:6605](metal_infer/infer.m:6605), [:6701](metal_infer/infer.m:6701) | Mirror chat_template.jinja's `enable_thinking` semantics |
| 3 | **K reads from config's `num_experts_per_tok`** (`--k` still overrides) | [metal_infer/infer.m:9487](metal_infer/infer.m:9487), [:9707](metal_infer/infer.m:9707) | Respect the model's trained expert count |
| 4 | Per-request log line shows `active_experts=K/config_K` for visibility | [metal_infer/infer.m:8980](metal_infer/infer.m:8980) | Surface the value that would otherwise be invisible |
| 5 | Disk-cache roundtrip unit test | [tests/test_disk_cache_roundtrip.sh](tests/test_disk_cache_roundtrip.sh), Makefile | `make cache-roundtrip-smoke` |
| 6 | Runtime cache validator | [metal_infer/infer.m](metal_infer/infer.m) | `FLASHCHAT_CACHE_VALIDATE=1` |
| 7 | Live-state fingerprint diff | [metal_infer/infer.m](metal_infer/infer.m) | `FLASHCHAT_CACHE_FINGERPRINT=1` |
| 8 | Tool-template render regression test (asserts ordering) | [tests/test_tool_template_render.sh](tests/test_tool_template_render.sh) | Catches future template-order regressions |

Fixes 1–3 are correctness; 4 is observability; 5–8 are durable test infrastructure.

---

## Known follow-ups

- **Qwen3.5-397B-A17B is still under-experted** (`MAX_K=8` caps it from running at its trained `num_experts_per_tok=10`). The startup log emits a loud warning. Fixing requires expanding `MAX_K` and the multi-expert buffer slots in `metal_setup`. This is a real performance-correctness tradeoff: K=10 doubles the SSD I/O per token vs K=4, but the model was trained with K=10. Owner's call.
- **Persistent disk cache should be re-enabled by default** in user config now that we've proven it's correct. Was disabled mid-investigation as a workaround.
- **`flashchat config` should expose K** — the active-experts setting is now per-model and can be tuned (e.g., for the 397B model, a user might want K=4 for streaming speed). Should be a first-class field in the config UI alongside temperature/top_p.
