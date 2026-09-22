# 28 — Bounded prepared-prompt cache

Date: 2026-09-21. Branch: `exp/28-prepared-prompt-cache`.
Base: `9c62991`.
Status (2026-09-22): implemented; correctness checks, standard watchdog, and focused
preparation comparison completed. **About 3.9 ms saved per hit on the tested 26 KiB
tool prefix; no demonstrated end-to-end speedup.** Retained as a modest CPU
preparation optimization; the standard watchdog remains inconclusive.

## Hypothesis

Repeated system/tool inputs need not be rendered and tokenized again before the
existing inference-state cache lookup. Tool validation can reuse the parsed schema
objects instead of reparsing their JSON for every call or repair attempt.

## Implementation and bounds

The inference owner retains one prepared entry for the lifetime of `serve_loop`.
It holds the exact rendered system body, wrapped prefix, compact token IDs/count,
rendering metadata, and immutable parsed schema objects. A matching request skips
system/tool rendering and prefix tokenization. The conversation suffix is always
rendered by the existing API parser and tokenized from its current bytes.

Retention limits are explicit: a 256 KiB length-framed input key, at most 512 KiB
each for the system body and wrapped prefix allocation, and 2 MiB of token IDs.
Schema objects are limited to the parse of at most 64 KiB of serialized schemas
in total, across at most 64 tools. Foundation object overhead is additional but
bounded by this input limit; this is not a measured resident-memory figure.
Oversized inputs use request-local preparation and do not evict the retained
entry. Tokenization/allocation failures do not replace it. A private autorelease
pool drains temporary rendering/key/schema objects on every preparation.

Request schema references have their own retains and survive entry eviction.
Request cleanup and server shutdown release them. Cancellation does not invalidate
immutable prepared data; the existing inference-state cancellation rules remain.
There is no new disk cache, inference snapshot, server, or background worker.

## Exact invalidation and behavior

The key compares full length-framed bytes, not a hash or schema equivalence:

- Template/preparation revision and loaded tokenizer generation.
- Runtime model and weights paths, model identifier, quantization, context window,
  and the request's model identifier/API format.
- Freshly read default/custom system contents and the request's system contents,
  including null versus empty; reasoning mode.
- Tool-choice mode, forced name, ordered tool definitions, descriptions,
  parameter-presence flags, and exact serialized parameter bytes.

The tokenizer/model/template are immutable within a server lifetime. Every
successful tokenizer load advances its generation. The new preparation header is
included in Make dependencies, launcher build freshness, and the runtime restart
signature. The custom system file is reread before each lookup; same-length edits,
replacement, deletion, and fallback changes cannot reuse stale preparation.

Sampling, session IDs, streaming, output limits, and conversation messages do not
affect the cached representation. They retain their existing request behavior.
Message/tool-result edits are re-encoded and still go through the existing exact
conversation-token matching before any model state is reused.

Prefix/suffix token concatenation is allowed only when the suffix starts with a
tokenizer added token and no added token can span the join. Otherwise the whole
prompt is tokenized. This preserves ordinary byte-pair merges and unusual added
tokens; arbitrary text boundaries are never assumed safe. Returned token arrays
are request-owned copies. Debug prompt dumps retain the full assembled bytes.

Schema preparation uses the same Foundation JSON parser and fragment option as
the existing validator. Invalid schemas retain their unsupported-schema behavior;
validation errors, paths, unsupported-keyword reporting, and bounded repair remain
unchanged. Tool order and first-match lookup for duplicate names are preserved.

Serialization normalization is deliberately deferred: equivalent schemas with
different serialized bytes miss, because changing their rendering would violate
this exploration's exact-behavior constraint. The current request parser still
parses JSON and serializes tool parameters to establish these exact inputs.

## Correctness evidence

- `make infer`: passed without compiler warnings.
- `make prepared-prompt-smoke`: passed. Uses the production request parsers,
  renderer, byte-pair tokenizer, and validator with a synthetic byte vocabulary,
  a real newline merge, and added tokens. No model weights or server are used.
  Compares rendered bytes, counts, and complete token arrays with uncached
  preparation; compares validation results including error/unsupported paths.
  Covers both API formats, Unicode, embedded special tokens, changed messages,
  tool choice/order/duplicates, schema edits, invalid/false schemas, reasoning,
  model/quantization/context/tokenizer changes, system file edits/deletion,
  oversized bypass, replacement, ownership, and unsafe joins.
- `make tool-template-smoke`: existing render/parser/schema validation suite passed.
- `make request-sampling-smoke`: passed; tool/forced-call sampling remains unchanged.

The fixture establishes preparation equivalence, not live model output or speed.
At the initial implementation checkpoint on 2026-09-21, no performance tests had
run; the standard watchdog awaited Dave's approval. Subsequent results follow.

## 2026-09-22 — Approved standard watchdog

Ran `make bench-api BENCH_ARGS='--model-id Qwen-Qwen36-35B-A3B --repeats 2'`
followed by `make bench-report`, on the finalized engine source/binary. Qwen3.6
35B-A3B q4, q8 context, shipping settings, 128-token limit, one warmup and two
measured repeats per standard streaming case. Mac17,2, M5, 32 GiB.

Finder was activated **before** hiding other applications. Hiding Codex before
making Finder visible failed; those attempts stopped before measurement. All
other app windows were verified hidden and Finder foreground before and after
the successful suite. Preflight: CPU idle 92.51%, GPU 2%, free memory 73%, swap
280 MiB, no Time Machine backup, no thermal warning, no other inference process.
The previous visibility/foreground state was restored and verified afterward.

| Case | First-token wait, ms | Decode, tokens/s |
| --- | ---: | ---: |
| Chat technical | 352.5 | 30.065 |
| Responses technical | 240.5 | 30.958 |
| Chat creative | 271.5 | 27.573 |
| Responses creative | 264.5 | 29.863 |
| Chat mixed code/prose | 301.0 | 23.831 |
| Responses mixed code/prose | 300.5 | 25.082 |

The tool case took 9745 ms versus 9741 ms in the previous watchdog: effectively
unchanged. The suite completed successfully and appended **13 rows**, all kept
in `assets/api_perf_log.tsv`. These rows label the uncommitted experiment with its
base commit, `9c62991`, and branch `exp/28-prepared-prompt-cache`.

The report exited nonzero: seven current-q4 metrics crossed its 5% regression
threshold, plus five historical flags for other models. Compared with `f52b993`,
mixed code/prose decode was **14.6% slower for Chat and 12.3% slower for Responses**;
their first-token waits increased 15.1% and 17.6%. Chat technical first-token wait
increased 31.3%. Smaller differences are not persuasive on their own. The large
flags remain unresolved; do not call this a clean performance pass or dismiss
them as proven noise. No rerun was performed.

The captured preparation log confirms 12 hits across the 18 streaming requests;
all measured streaming repeats hit. But these use only a **123-byte system body**
and no tools. The single tool case was a miss. This suite does not directly
measure the intended benefit of reusing a large system/tool prefix, and comparing
with a prior date does not isolate the cause of the observed decode differences.

Evidence: [suite output](bench-api.log), [full report](bench-report.txt), and
[preparation-log excerpt](bench-preparation-excerpt.log). Engine source, cache
header, binary and user config hashes matched before/after. The benchmark's
temporary `/Users/speedster/.config/flashchat/menubar-quiet` marker was created and
removed; no persistent user configuration changes remain. No benchmark server
was left running.

## 2026-09-22 — Focused test crafted at Dave's request

[Test specification and commands](TEST.md) describe the new
`tests/prepared_prompt_probe.m` executable. It compares the actual pre-exp28
preparation sequence with current cache hits and misses using the installed Qwen
tokenizer, without inference or model weights. Twenty alternating paired timings
per case are available behind the explicit `--measure` option.

The executable builds without warnings. Its **untimed `--check` mode passed** for
small-prefix hits, 24-tool hits and 24-tool misses, comparing exact bytes, every
token ID, system counts/hashes and schema-validation results. Request
parsing/serialization and model execution are excluded from its timing scope.

## 2026-09-22 — Focused preparation results

Continued the authorized effectiveness testing with one fixed `--measure` run:
three cases, one untimed warmup and 20 alternating before/after pairs per case.
All 60 pairs matched exact system bytes, hashes/counts, every token ID, and schema
validation results. Expected hits were 20/20 in each hit case and 0/20 in the miss
case. All **120 measured path rows** are retained.

| Case | Before median | Exp28 median | Median paired saving | Paired saving p10–p90 |
| --- | ---: | ---: | ---: | ---: |
| Small prefix, hit (123 bytes, 31 tokens) | 0.034 ms | 0.009 ms | 0.026 ms | 0.024–0.027 ms |
| 24 tools, hit (26,248 bytes, 5,465 prefix tokens) | 3.912 ms | 0.035 ms | 3.878 ms | 3.754–4.209 ms |
| 24 tools, miss (26,263 bytes, 5,473 prefix tokens) | 4.129 ms | 1.996 ms | 2.002 ms | 1.811–2.173 ms |

The table includes preparation plus validation of one tool call. Independent path
medians and the median of paired differences need not subtract exactly. Tool-hit
CPU preparation fell 99.1%; the absolute saving is about **3.9 ms per eligible
request** for this synthetic workload. This is useful avoided work, but small
relative to the hundreds of milliseconds before a first token or seconds spent
generating. Do not present the 99.1% figure as a whole-request speedup.

Misses also improved in this sample because exp28 renders once and tokenizes the
prefix once, whereas the original path renders twice and tokenizes the prefix
both for counting and inside the assembled prompt. Preparing all schemas on
admission did not outweigh that saving here. Other schema sizes, hit rates, or
platforms remain untested; no reruns or extrapolated throughput gains are claimed.

The canonical benchmark's system-health function was reused for preflight after
Finder-first hiding: CPU idle 94.42%, GPU 1%, free memory 72%, swap 280 MiB, no
backup, no thermal warning, no inference workload. Desktop visibility was verified
before/after measurement, and its original state restored and verified. The first
wrapper attempt failed to load that function and stopped **before any timing**;
the corrected wrapper performed the single measured run. User config is unchanged;
the focused probe creates no quiet marker or server.

Evidence: [raw paired results](preparation-results.tsv),
[preflight](preparation-health.log), [source/binary/tokenizer hashes and build flags](preparation-build.txt),
and [untimed correctness check](preparation-check.txt). The native tokenizer has
248,044 vocabulary entries, 247,587 merges and 26 added tokens. Same M5/32 GiB host,
production aggressive compiler flags; no model weights or Metal execution.

## Conclusion

Prepared-prefix and schema reuse is implemented with bounded retention and exact
input matching. Correctness checks pass, and the focused paired test demonstrates
roughly 3.9 ms saved per hit on a 26 KiB tool prefix, with lower miss cost in this
sample. This supports keeping the cache as a modest CPU preparation optimization.
The standard watchdog still demonstrates no end-to-end speedup and retains its
unresolved regression flags; the preparation result neither explains nor clears
the older-run decode comparison.
