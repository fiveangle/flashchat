# 26 — Exact conversation-state reuse

Date: 2026-09-16. Branch: `exp/26-conversation-state-reuse`.
Base: `d193340`; measurements include the uncommitted experiment changes.
Status: **Closed; validated and approved for commit.**
Verdict: **Keep** the single-conversation cache, enabled by default.

## Hypothesis and recovered history

Preserving the active conversation avoids repeatedly prefilling earlier turns.
Use exact rendered-token comparisons, not a session ID as evidence of matching
state. Retain persistent system/tool caching as the fallback.

`d965514` introduced a single active session; `d5e5e71` fixed destructive JSON
parsing and forwarded the final EOS before continuation. `bbaa549` removed active
session reuse during the API rewrite. Today's rendering normalizes assistant
reasoning and tool arguments; internal schema-repair turns are not sent back by
clients. Directly restoring the old session-ID shortcut would be incorrect.

## Implementation and boundaries

- One active token ledger, with a checkpoint before the newest assistant header.
  Full input is tokenized before touching live state; exact matching chooses live
  reuse, checkpoint rollback, or the existing system/tool-cache fallback.
- The live ledger contains only tokens actually forwarded, including thinking
  markers. An emitted-but-unforwarded token or EOS is not falsely counted as cached;
  the next request processes it if its rendered sequence includes it.
- Checkpoints copy CPU and GPU recurrent state and retain attention lengths. They
  share the append-only attention buffers; rollback truncates lengths without
  copying the growing attention history. They are valid only while those buffers
  remain intact. This is one bounded checkpoint, not a multi-session prefix tree.
- Client normalization can prevent live reuse. The checkpoint still preserves all
  earlier turns and reprocesses the latest assistant turn plus new input. The live
  Qwen fixtures exercised this fallback. This does not promise to skip every token
  of the latest answer or tool call.
- Speculative decoding and internal tool repair disable live-ledger reuse but keep
  the pre-assistant checkpoint. Cancellation invalidates both. Changed or shortened
  histories and changed session IDs require a matching checkpoint or a fresh start.
- Sampler penalties still account for the entire conversation, including reused
  input. Progress reports newly processed tokens separately from cached positions.
  Oversized prompts fail cleanly; generation reserves enough context for its next
  forward or speculative batch.
- `CONVERSATION_CACHE=1` ships through config, menu, runtime export/signature and
  engine startup. Set `0` for a controlled comparison or rollback. It is independent
  of `SYSTEM_PROMPT_CACHE`. No conversation disk cache was introduced.

The checkpoint's calculated recurrent-copy allocation for this model is about
**187 MiB** (30 linear layers, CPU and GPU copies), plus a bounded token ledger.
This is an allocation estimate from dimensions, not a measured peak-RAM result.
It can compete with expert residency on tighter-memory machines; real 16 GiB
hardware was not tested.

## Functional validation

- `make infer`: passed, no compiler warnings.
- `make conversation-cache-smoke`: passed exact-prefix selection, input changes,
  shortening, session changes, CPU/GPU recurrent rollback, attention-length
  restoration, cache invalidation, and absolute-position progress accounting.
  This uses synthetic state and no model inference. Metal access was required;
  the sandboxed first invocation could not obtain a device, then the permitted
  invocation passed.
- `make tool-template-smoke`: all render/parser checks passed.
- `make py-tests`: 132 passed, including config migration/default/export tests.
  Old schema-version assertions were updated from 14 to 15 for the new setting.
- `tests/test_conversation_cache_live.py`: 11 cache-off/cache-on output comparisons
  passed on Qwen3.6-35B-A3B q4 with q8 context, 4096 positions, pinning/MTP off.
  Covers three chat turns, streaming, edited/shortened history, identical retry,
  changed system prompt, forced tool call, tool result, and two Responses turns.
  Output comparison preserves content/reasoning, compares parsed tool arguments,
  and excludes generated tool-call IDs. Expected hits/misses were verified in logs.
- `--edge-only`: passed conversation reuse with system caching disabled, session
  change, oversized-input rejection, disconnect during generation, and a subsequent
  cache miss. Each isolated test server was stopped and verified exited.

Evidence: [requests](functional-requests.json),
[off outputs](functional-responses-off.json), [on outputs](functional-responses-on.json),
[off server log](functional-cache-off.log), [on server log](functional-cache-on.log),
[edge-case log](functional-edge.log).

Limitations: live comparisons covered the shipping hybrid model, not every model
or context-cache precision. Synthetic tests exercise both CPU and GPU state storage.
The bounded benchmark used shipping MTP defaults; the detailed off/on correctness
fixtures used MTP disabled. More demanding task quality is not inferred from an OK
response or from faster prefill.

## Approved performance scope and method

Dave approved one standard Qwen3.6 q4/q8 Make suite (one warmup and two measured
repeats per standard case) plus one three-turn cache-off/cache-on comparison.
No further performance runs were made. Hardware: Mac17,2, 32 GiB RAM. Model:
`Qwen-Qwen36-35B-A3B`, 8 active experts, shipping q4 weights/q8 context, 65,536
positions, 8 GiB pin cap with 0.5 free-memory fraction and locking, fusion and
batched prefill enabled. A cap is not proof the arena reached 8 GiB.

The initial standard suite ran with application windows hidden and Finder foreground.
Dave subsequently reported that Codex/the desktop had been restored after that run,
so both three-turn runs included visible Codex UI and its residual GPU load. The
automated visibility checks did not establish the required desktop state throughout
measurement. Each harness preflight passed: CPU idle 90.94% / 94.47% / 92.97%;
GPU 0% / 1% / 2%; Time Machine idle; no thermal warning. Those point-in-time samples
do not rule out UI contention during the runs. The six conversation measurements
are therefore classified as **confounded**; the effect on timing is unquantified.
No reruns were made, as Dave requested. Final desktop restoration was verified.

Commands (each prefixed by `make bench-api`):

```
BENCH_ARGS='--model-id Qwen-Qwen36-35B-A3B --repeats 2'
BENCH_ARGS='--model-id Qwen-Qwen36-35B-A3B --conversation-only --conversation-cache 0'
BENCH_ARGS='--model-id Qwen-Qwen36-35B-A3B --conversation-only --conversation-cache 1'
```

The three-turn probe appends ~305 tokens per turn and carries actual assistant
replies into the next request. Temperature 0, reasoning off, maximum 8 generated
tokens; all six replies were exactly OK after trimming. Timing starts before the
HTTP request and ends at the first content token, excluding initial role/keepalive
frames. Both runs start new servers and retain system caching; their disk caches
are isolated. Mode 0 ran before mode 1, one observation per turn: this is initial
bounded evidence, not an interleaved confidence interval or universal speed claim.

## Measured three-turn result (confounded by visible Codex UI)

| Turn | Cache off first-token wait | Cache on first-token wait | Change in wait | Newly prefilled, off → on | Cached positions, off → on |
|---|---:|---:|---:|---:|---:|
| 1 (cold) | 8.574 s | 8.784 s | +2.5%, noise-scale | 341 → 341 | 0 → 0 |
| 2 | 13.430 s | 7.965 s | **−40.7%** | 611 → 312 | 35 → 334 |
| 3 | 18.104 s | 8.344 s | **−53.9%** | 916 → 312 | 35 → 639 |

Across all three requests, summed first-token wait fell **40.108→25.093 s
(−37.4%)**. Summed HTTP completion time fell **40.269→25.336 s (−37.1%)**.
The key mechanism is visible in token counts: growing-history work becomes a
roughly constant-sized continuation. The reduced prefill counts demonstrate state reuse. The observed latency reductions
are diagnostic evidence, not clean baseline gains or a decode-kernel throughput claim;
we cannot separate the timing effect of visible Codex UI from this comparison. Sources:
[off run](bench-conversation-off.log), [on run](bench-conversation-on.log), and
[`assets/api_perf_log.tsv`](../../../assets/api_perf_log.tsv) (six new conversation rows).

## Standard suite and report

| Case | Decode tokens/sec | First-token wait |
|---|---:|---:|
| Chat technical | 30.325 | 273.5 ms |
| Responses technical | 30.709 | 242.0 ms |
| Chat creative | 27.678 | 316.0 ms |
| Responses creative | 29.645 | 268.5 ms |
| Chat worst-case | 25.886 | 268.5 ms |
| Responses worst-case | 26.816 | 262.5 ms |

Forced tool call: **2360 ms, passed**. The standard repeated prompts now benefit
from the pre-assistant checkpoint, so their warm first-token times describe that
cache state. The historical report shows 75.9–86.0% less first-token wait, but it
compares another commit/session and is not isolated attribution. Decode changes
span −5.2% to +2.6%; no material decode gain or regression is established.

`make bench-report` returned nonzero for four flags: three pre-existing rows for
other models and this model's Responses technical decode at −5.2% versus the
previous session. That small historical difference did not justify reruns under
our noise policy. Sources: [standard suite](bench-api.log), [report](bench-report.txt).
The saved report predates the subsequent reclassification of the six conversation
rows as confounded and is retained as the original report artifact.

**Harness incident:** all six standard cases and the tool probe had completed and
appended their 13 rows when an agent edit of the running shell script caused a
post-case `fi` parsing error. The edit isolated disk-cache paths for future runs;
it did not change the running engine. Preserve the output and limitation: do not
claim a clean standard-script exit. Current script syntax and config tests pass;
the two subsequent conversation invocations completed normally. No case was rerun
to conceal the incident. All 19 new watchdog measurements are retained. Following Dave's desktop-state
correction on 2026-09-16, only the status and notes of the six conversation rows
were updated to record the confound; their numbers and original health samples
remain unchanged.

The harness now supports `BENCH_ARGS`, explicit conversation-only off/on scenarios,
and isolated config/disk-cache paths, avoiding mutation of the user's saved config
while resolving installed model artifacts. This is part of making 26 reproducible.

## Environment effects

All test/benchmark servers were stopped; no production server was started. Saved
user configuration and model weight files were not edited. The original standard
harness refreshed/pruned the model's generated prompt-cache directory before disk
cache isolation was added. The resulting generated files outside the project are:

- `/Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/flashchat/q4/system_prompt_cache/40151882ca0c04f9-v2.fcache`
- `/Users/speedster/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/flashchat/q4/system_prompt_cache/6517233bbcdcf8c3-v2.fcache`

These contain regenerated system-prompt state, not weights or conversation data.
The prior cache directory listing was not captured, so older pruned entries cannot
be enumerated from this run. Subsequent off/on measurements used isolated caches.

## Decision and next steps

Keep exact single-conversation reuse with the checkpoint fallback. Implementation
and the approved validation scope are complete. Dave approved committing the feature
after the desktop-state correction; no additional measurements were requested.
Revisit for a reproducible mismatch, memory-pressure cost, or a workload where
normalization repeatedly invalidates older history. Multi-conversation storage
and partial-prefix branching remain experiment 27; prepared-prompt/token caching
remains 28. No additional benchmark campaign is pending.
