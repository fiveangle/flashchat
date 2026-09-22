# Focused exp28 effectiveness test

Question: **How many milliseconds does exp28 save when preparing the same
system/tools for another request, and what does a cache miss cost?**

The standalone `tests/prepared_prompt_probe.m` compares the pre-exp28 preparation
sequence from `9c62991` with the current implementation in one executable, using
the installed Qwen tokenizer. It does not load model weights, initialize Metal,
start a server, or generate an answer. This isolates the work exp28 changes.

## Three fixed cases

| Case | Input | Expected exp28 behavior |
| --- | --- | --- |
| `small_prefix_hit` | Default system prompt, no tools | Reuse the prefix; reveal overhead on a tiny input |
| `24_tools_hit` | About 4 KiB of system instructions and 24 realistic tool schemas | Reuse rendered bytes, prefix tokens and parsed schemas |
| `24_tools_miss` | Same tool set, but change the system revision every request | Miss every time; expose preparation and schema-admission cost |

The conversation filename changes on every iteration so the suffix must still be
tokenized. Every comparison uses exactly the same already-parsed request and exact
serialized tool bytes for both paths. Request JSON parsing and tool JSON
serialization are outside the measurement; this tests preparation after parsing,
not endpoint latency or client serialization stability.

The baseline performs both original system renders, hashes/counts the prefix,
tokenizes the whole assembled prompt, and parses a tool schema when validating a
call. Exp28 uses its real bounded cache, then tokenizes the current suffix and
validates the same call. A miss includes admission of all 24 parsed schemas; it
cannot hide that cost by reporting validation alone.

## Correctness gate and timing design

- Compare exact rendered system bytes, system hash/count, every prompt token ID,
  and schema-validation results after every pair; fail on any difference.
- Require the expected hit or miss for every exp28 invocation.
- Use one untimed warmup pair, then **20 paired comparisons per case**; alternate
  before/after execution order. No automatic reruns.
- Use a monotonic clock and suppress tokenizer diagnostic output in both paths.
- Emit all 120 measured path rows plus per-case median time, paired milliseconds
  saved, paired-savings 10th–90th percentile range, relative change, and hit count.
- Report preparation and validation separately, plus their total. Keep cold-miss
  results alongside warm-hit results.

The meaningful result is absolute milliseconds saved per eligible request. A
consistent warm tool-prefix saving supports exp28's CPU benefit; an interval that
spans zero is inconclusive. Small percentage differences alone do not establish a
win. The miss case reveals a tradeoff, not a reason to omit unfavorable samples.
These results cannot establish faster model decoding or whole-request latency.

## Build and validate without measuring

From the project root:

```sh
make metal_infer/prepared_prompt_probe
./metal_infer/prepared_prompt_probe --check /absolute/path/to/flashchat/q4/vocab.bin
```

`--check` runs the real tokenizer and both preparation paths, but never reads a
clock. It uses one warmup and two alternating check pairs per case, including
cache misses. No performance approval is needed for this correctness check.

## Approved measurement command

After explicit approval, record/hide application windows, activate Finder first,
verify the desktop and system-health preflight, then run **once**:

```sh
./metal_infer/prepared_prompt_probe --measure /absolute/path/to/flashchat/q4/vocab.bin \
  > docs/explorations/28-prepared-prompt-cache/preparation-results.tsv
```

Restore prior visibility and foreground app on completion, failure, or
cancellation. Use the same system-health checks as the standard benchmark. Do not
run concurrently with inference or another performance workload. Retain the raw
output and record the tokenizer hash, source hashes, build flags, host, and system
health in the exploration notes. No production configuration changes are needed.

The single approved timing run completed on 2026-09-22. See
[results and interpretation](NOTES.md#2026-09-22--focused-preparation-results) and
[all raw pairs](preparation-results.tsv). The 24-tool hit case saved a median
3.878 ms per pair; correctness comparisons passed for every pair.
