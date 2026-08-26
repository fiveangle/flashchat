# 23 — OpenCode long-context replay and expert pin-cache sizing

- Date: 2026-08-25
- Status: **MEASURED — retain 8 GiB for this 32 GiB / 262k-q8 OpenCode workload**
- Code/config changes: **none**

## Why this experiment exists

An OpenCode request through the OpenAI-compatible endpoint returned a tool
schema error (`task.description` missing). HTTP request logging preserved the
exchange, which also exposed a repeatable performance disparity:

- short prose generation: **26.18–27.11 tok/s** near positions 49–680;
- the captured OpenCode exchange: **8.64 tok/s** near position 15,972.

The work had two phases:

1. replay the real OpenCode request while varying context-window and KV-cache
   representation;
2. test whether the 8 GiB explicit expert pin cache was starving the macOS
   filesystem cache, then test upward after the smaller cache thrashed.

This is a production-workload diagnostic, not `bench-api` sign-off and not a
new shipped default. Attempt 12 remains the short-prompt/default reference.

## Reproducibility

- Machine: 32 GiB Mac17,2, Apple M5.
- Model: Qwen3.6-35B-A3B q4, 8 active experts/token.
- Git HEAD: `520fb0f057f41b63dbf5ad4186ddb8002957e1e6`.
- Binary SHA-256: `b374a0e4664b7ac961701ebf42baa8261fea70173df4ac6bacd180f8f03d45d7`.
- Binary mtime: `2026-08-25T15:26:57-0700` (working tree included attempt 21a).

Health gate before each configuration:

- no competing `metal_infer/infer` process;
- Time Machine idle;
- no thermal or performance warning;
- 83–84% system memory free before launch;
- GPU utilization below the project 70% cutoff before starting a run.

Common runtime settings:

```text
MODEL=Qwen-Qwen36-35B-A3B
MTP=0
ACTIVE_EXPERTS=8
FUSE_LINEAR=1
BATCH_PREFILL=1
PREFILL_CHUNK=4096
ANE_PREFILL=1
ANE_MIN_CHUNK=512
LM_HEAD_MLOCK=1
EXPERT_SPLIT_IO=1
SYSTEM_PROMPT_CACHE=1
SERVER_DEBUG=0
SERVER_HTTP_LOG=0
PREAD_PROFILE=disabled
```

The real 8 GiB baseline is from the original server and therefore had
`SERVER_DEBUG=1`, `SERVER_HTTP_LOG=1`, and the 2,097,152-event pread profiler
enabled. That handicaps rather than favors the 8 GiB result.

The replay changed only request `max_tokens` from 32,000 to 128 as a safety
cap. Messages, tools, streaming, and sampling defaults were preserved.

## Captured request corpus

The requests were not copied into git because they contain the full OpenCode
conversation/tool context. Their durable identifiers are:

| request | bytes | messages | roles | tools | SHA-256 |
|---------|------:|---------:|-------|------:|---------|
| `chatcmpl-6.request.json` | 70,753 | 2 | system,user | 11 | `bff95bd9a44aa62c1b60f12658cf895d910e9c2522e0bfd5775648ff5d4b8b2e` |
| `chatcmpl-11.request.json` | 72,895 | 4 | system,user,assistant,tool | 11 | `61e6c897d2ac659a49c6a53035b9a37dd46874bf136633b6ce5ea0d6f561dba4` |
| `chatcmpl-12.request.json` | 73,246 | 6 | system,user,assistant,tool,assistant,tool | 11 | `3fce1ce37f6a7128f8fc2a827566a982c8d914789ec768865904586d71f3a09e` |

Source at measurement time:
`/Volumes/usr/Users/speedster/dev/flashchat_logs/`.

All three share system/tool prefix hash `6b79852c62f406ce`:

- cached prefix: 14,579 tokens;
- request 6 continuation: 405 tokens;
- request 11 continuation: 1,393 tokens;
- request 12 continuation: 1,476 tokens.

The pin-sizing protocol was intentionally an evolving sequence rather than
repeating one request: request 6 warm-up → request 11 → request 12.

## Phase 1 — context window and KV representation

### Original evidence: short prose versus OpenCode

With the same 262k/q8 server configuration:

| workload | starting context | generated | decode |
|----------|-----------------:|----------:|-------:|
| short prose 1 | ~49 tokens | 632 | **27.11 tok/s** |
| short prose 2 | ~60 tokens | 633 | **26.18 tok/s** |
| OpenCode request 11 | 15,972 tokens | 55 | **8.64 tok/s** |

The system-prompt cache avoids recomputing the 14,579-token prefix, but decode
still occurs at position ~16k. It does not make long-context attention behave
like a fresh prose prompt.

### Controlled replay results

| KV/window | cache state | prefill | decode | tool outcome |
|-----------|-------------|--------:|-------:|--------------|
| q8 / 64k | first server request | 128.931 s | 5.60 tok/s | valid structured call |
| q8 / 64k | repeated, pin-warm | 146.523 s | **9.82 tok/s** | valid structured call |
| q8 / 262k | first full replay | 114.464 s | 8.40 tok/s | valid structured call |
| q8 / 262k | repeated, pin-warm | 114.143 s | **9.90 tok/s** | valid structured call |
| fp32 / 64k | cold full-prefix rebuild | 690.622 s | 7.14 tok/s | not parsed by 128-token cap |
| fp32 / 64k | prefix/pin warm | 114.467 s | **7.47 tok/s** | not parsed by 128-token cap |

Conclusions:

1. q8 64k versus q8 262k was neutral once warm: 9.82 versus 9.90 tok/s.
   Reducing allocated window saves 1.875 GiB but did not speed decode at the
   same ~16k position.
2. fp32 was slower and consumed 2.5 GiB at 64k, the same KV allocation as
   q8 at 262k. It also rebuilt the full saved prefix once.
3. The combined original 8.64 → clean repeated-request 9.90 difference is
   about +15%. That is an upper bound on profiler/debug contribution because
   it also includes repeated-request pin warmth; it cannot explain the roughly
   3x short-versus-long context disparity.
4. The original missing-`description` schema error did not reproduce. q8 chose
   `bash` and emitted valid arguments, so this experiment did not exercise the
   invalid-tool repair path.

### Prompt-cache observation

The fp32 server rejected the existing 187 MiB cache entry, rebuilt the full
prefix, and wrote a 623,109,188-byte entry. A later q8 server accepted that
entry and produced valid structured calls. The asymmetric reject/accept
behavior deserves a separate cache-format correctness review; no failure was
observed after returning to q8.

## Phase 2 — pin-cache sizing on real evolving requests

Hypothesis: an 8 GiB anonymous, mlocked expert arena may duplicate experts
also retained in the macOS filesystem cache. A 4 GiB arena might retain the
hottest experts while leaving 4 GiB more reclaimable for broader file-backed
coverage.

Relevant implementation facts verified in `infer.m`:

- the entire arena is allocated and mlocked at first expert use, including
  slots not yet populated;
- `FLASHCHAT_EXPERT_NOCACHE` was off, so miss reads still traverse and may
  populate the filesystem cache;
- pin hits bind persistent no-copy Metal buffers; filesystem-cache hits still
  require `pread` into a runtime expert buffer;
- eviction is LFU using cumulative expert frequency.

### Existing 8 GiB baseline (real server, diagnostics enabled)

| request | continuation | prefill | generated | decode | outcome |
|---------|-------------:|--------:|----------:|-------:|---------|
| 6 | 14,984 total (cold prefix) | 581.386 s | 62 | 8.50 tok/s | parsed tool call |
| 11 | 1,393 | 118.085 s | 55 | **8.64 tok/s** | parsed tool call |
| 12 | 1,476 | 124.329 s | 56 | **8.63 tok/s** | parsed tool call |

The original server console was unavailable at shutdown, so aggregate pin-hit
and eviction counts are not available for this exact evolving sequence.

### 4 GiB test

Resolved allocation: 4,095.6 MiB, 2,427 slots, free RAM at init 12.81 GiB.

| request | prefill | generated | decode | outcome |
|---------|--------:|----------:|-------:|---------|
| 6 warm-up | 44.162 s | 59 | 5.47 tok/s | parsed tool call |
| 11 | 156.022 s | 36 | **5.52 tok/s** | parsed tool call |
| 12 | 168.530 s | 128 | **5.58 tok/s** | tool call did not close by cap |

Shutdown statistics:

```text
hits 52569 / 72320 (72.7%), pinned 2427/2427, evictions 6920
```

Against the existing 8 GiB requests 11/12:

- decode: **−36% / −35%**;
- continuation prefill: **+32% / +36%**.

The arena was full and eviction-thrashing. Approximately 8.8 GiB remained
reclaimable immediately after the 4 GiB lock, so the filesystem cache already
had substantial headroom and did not compensate. Per the adaptive test plan,
1 GiB and zero were not run.

### Upward test: 12 GiB maximum

`MAX_GB=12`, `AUTO_FRAC=0.75`, `MLOCK=1`. The safety fraction resolved to
9,348.8 MiB (9.13 GiB), 5,540 slots, because only 12.17 GiB was reclaimable at
first expert use. This is **not** mislabeled as a true 12 GiB measurement.

| request | prefill | generated | decode | outcome |
|---------|--------:|----------:|-------:|---------|
| 6 warm-up | 36.994 s | 59 | 7.57 tok/s | parsed tool call |
| 11 | 130.107 s | 36 | **8.14 tok/s** | parsed tool call |
| 12 | 137.205 s | 128 | **8.37 tok/s** | tool call did not close by cap |

Shutdown statistics:

```text
hits 64940 / 72320 (89.8%), pinned 5540/5540, evictions 1840
```

The extra 686 slots above an 8 GiB/4,854-slot allocation improved hit rate
and reduced evictions versus 4 GiB, but did not beat the real 8 GiB server:

- request 11: 8.14 versus 8.64 tok/s; prefill 130.1 versus 118.1 s;
- request 12: 8.37 versus 8.63 tok/s; prefill 137.2 versus 124.3 s.

A true locked 12 GiB arena was not safely available: 12.17 GiB reclaimable is
less than the arena plus the engine's 512 MiB minimum headroom. Forcing it
would either skip mlock or leave essentially no filesystem/OS headroom.

## Caveats

- Requests sample at server temperature 0.7; exact tool content and completion
  length vary. Prefill token counts are fixed, and two consecutive evolving
  requests were used for each sizing conclusion.
- Page cache was not forcibly purged. Purging would disrupt the user's system
  and would not represent a warm, long-running OpenCode server.
- The 8 GiB baseline retained profiler/debug overhead, while 4 and 9.13 GiB
  ran clean. Its win is therefore conservative, but aggregate pin statistics
  for that exact original process are unavailable.
- A separate controlled 8 GiB repeated-request run reached 9.90 tok/s with
  86.1% hits and no evictions, but repeating an identical request favors LFU;
  it is not substituted for the evolving 8 GiB baseline above.
- The 128-token cap caused some sampled tool calls not to close. Those runs
  still provide decode timing but are not tool-correctness successes.

## Verdict

**Retain the current 8 GiB pin cap for this machine and workload.**

- 4 GiB is decisively capacity-constrained at ~16k OpenCode context.
- A safely resolved 9.13 GiB improves cache statistics but not end-to-end
  performance; 8 GiB is the observed knee.
- Do not infer a global shipped-default change: short prompts and lower-memory
  machines have different tradeoffs, documented in attempts 02 and 12.
- Keep q8. Choose 64k versus 262k for capacity/memory headroom, not expected
  decode speed at the same actual sequence position.
- Disable the large pread profiler during normal OpenCode operation; retain
  HTTP request logging only while schema troubleshooting requires replay data.

## Evidence files

- `RESULTS.tsv` — machine-readable timing/config table.
- `LOG_EXCERPTS.txt` — exact server result and pin-stat lines used above.
