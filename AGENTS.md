# AGENTS.md — Flashchat Development Guide

This is a pure C/Metal inference engine for running 397B parameter MoE models on Apple Silicon.

## Working Rules: Benchmarks and Git

- **Check the branch before making changes.** If on `main`, `develop`, or another
  shared integration/release branch, create and switch to a dedicated task branch
  before editing. Do not use `develop` as a task branch. Branch creation is
  authorized by this rule; no separate permission is needed. Name branches by
  work type: `fix/<task>` for fixes, `feat/<task>` for features,
  `refactor/<task>` for refactoring, `doc/<task>` for documentation, and
  `exp/<count>-<experiment-name>` for experiments/explorations. Use a concise,
  lowercase, hyphen-separated task name. These prefixes replace the default
  `codex/` prefix. Reuse an existing task branch only for that same task.
- **Run the standard watchdog; do not duplicate it.** Any change to `infer.m`
  requires `make bench-api` followed by `make bench-report`, even when the change
  appears unrelated to the hot path. The same applies to decode/prefill, kernel,
  attention, and speculative-decoding changes. One approved standard run covers
  the finalized batch; do not add redundant baselines, before/after probes, or
  reruns. An existing run counts only if it covers the actual code being committed.
  Do not assume an automatic watchdog ran: verify its results and include changed
  `assets/api_perf_log.tsv` rows in the commit. Additional measurements beyond the
  watchdog need a specific unanswered question or concrete anomaly and Dave's
  explicit approval. Continue to run relevant correctness tests.
- **Ask Dave before any benchmarking.** Get explicit approval for the proposed
  scope and configuration before running performance measurements, including
  ad-hoc timing probes and reruns. Approval covers the agreed bounded run or suite,
  not an open-ended benchmark campaign.
- **Benchmark with application windows hidden and Finder foreground.** After
  approval, record the current application visibility and foreground app, hide all
  running GUI applications except Finder (including Codex), and activate Finder.
  Do not quit apps or stop their background work. Run the system-health preflight
  after hiding apps; investigate remaining contention rather than merely retrying
  or bypassing the check. Keep this desktop state throughout measurement, then
  restore the prior visibility and foreground app on completion, failure, or
  cancellation. If hiding or restoration is unavailable, explain and ask before
  proceeding. Visible application rendering can materially consume the GPU even
  when no inference is running; do not assume app updates preserve idle behavior.
- **Expect noise, not single-digit precision.** SSD-streamed inference depends heavily
  on prompt type (prose versus math/code, narrow versus broad topics), available RAM,
  cache state, and competing activity. Most single-digit percentage differences are
  inconclusive, not automatic regressions or wins. A 3% change alone does not justify
  repeated baseline runs. Keep validation bounded and investigate material,
  reproducible changes; do not create more noise by launching extra memory consumers.
- **Protect performance measurements already in progress.** Before starting another
  inference server, establish the purpose and whether the existing server is being
  used for performance measurement, including by Dave. A second server can be fine
  for functional testing only when neither workload needs performance measurements.
  If either does, do not start a competing server or disturb the running workload.
  Ask when unclear; never stop Dave's server without permission.
- **Target shipping defaults, not Dave's current settings.** Use the Make-based
  benchmarks and the out-of-box Qwen3.6-35B-A3B-q4 model with q8 context cache unless
  another target is explicitly agreed. Dave's running configuration is experimental
  and malleable. An unexpected setting is not a new baseline or a reason to launch
  time-consuming benchmark campaigns: ask before changing the target.
- **Keep the performance log, do not curate it.** Always include
  `assets/api_perf_log.tsv` when committing if it has changes. It is a malleable
  watchdog, not a source of absolute truth. Preserve noisy or confounded results;
  do not delete, rewrite, or rerun them merely to make the log or report look clean.
  Explain limitations when interpreting results instead.
- **Commit headline prefixes:** `feat:` for features, `fix:` for fixes,
  `refactor:` for refactoring, and `doc:` for non-experiment documentation.
  Optimization, speed, and other material performance experiments use `exp/<count>:`
  with a permanent integer experiment ID from
  `docs/END_TO_END_SPEED_OPPORTUNITIES.md`, the cumulative exploration register.
  Reuse an existing proposal's assigned ID when starting it. Before assigning a
  new ID, check the register (including reserved proposals without directories),
  `docs/explorations/`, and Git history/branches; allocate above the highest ID
  across all three. Never renumber or reuse IDs after assignment, including
  rejected or superseded explorations. Update the register's status and evidence
  links as work advances; preserve dated findings in the exploration notes.
  Document and commit every experiment,
  including negative or inconclusive results, in
  `docs/explorations/<count>-<experiment-name>/NOTES.md`. Record the hypothesis,
  configuration, method, findings, and conclusion so explored approaches are not
  repeated without a deliberate reason. Follow the commit-approval checkpoint below.
- **One branch per exploration.** Before starting a new exploration, create and
  switch to a dedicated `exp/<count>-<experiment-name>` branch matching its
  exploration directory. Keep unrelated work off that branch; never reuse it for
  another exploration.

## Project Structure

```
metal_infer/
  infer.m           # Main inference engine
  shaders.metal     # Metal compute kernels
  chat.m            # Interactive chat TUI
  main.m            # MoE-only benchmark
  tokenizer.h       # Single-header C BPE tokenizer
  linenoise.c/h     # Line editing library

modelmgr/           # Model management core (registry, manifests, recipes,
                    # artifact hashing/verify, offload, migration, TUIs)
                    # — see docs/MODEL_FRAMEWORK.md
  steps/            # Shared step library (download, tokenizer, extract,
                    # repack, compile_native, materialize)
  tui/              # Onboarding, config wizard, manage flows

assets/models/      # Per-model manifests; assets/model_configs.json is
                    # GENERATED from them (`make registry`)

tests/
  test_flashchat_cli.sh     # CLI regression test
  test_modelmgr_cli.sh      # Model management integration test
  test_api_smoke.sh         # HTTP API smoke test
  python/                   # modelmgr unit tests (make py-tests)

root/
  Makefile          # Project build/test surface
  assets/api_perf_log.tsv   # Durable API perf trend log (make bench-api)
```

## Build Commands

```bash
# From the project root

# Build all targets
make

# List available Make targets
make help

# Build specific binaries
make metal_infer    # Main benchmark
make infer          # Full inference engine
make chat           # Interactive chat TUI
make api-smoke      # HTTP API smoke test

# Run benchmarks
make run            # Single expert forward pass
make verify         # Metal vs CPU reference verify
make bench          # Benchmark single expert (10 iterations)
make moe            # Full MoE forward (K experts, single layer)
make moebench       # Benchmark MoE (10 iterations)
make full           # Full model forward (K=4)
make fullbench      # Full forward benchmark (3 iterations)

# Inference engine (full model)
./metal_infer/infer --prompt "Hello" --tokens 100

# With timing breakdown
./metal_infer/infer --prompt "Hello" --tokens 20 --timing

# Framework regression tests
make cli-smoke      # tests/test_flashchat_cli.sh
make manage-smoke   # tests/test_modelmgr_cli.sh
make py-tests       # tests/python/ (modelmgr unit tests)
make registry-check # assets/model_configs.json matches assets/models/*.json
make tool-template-smoke
make api-smoke
make test           # everything

# Performance regression benchmark (server-level, per model) + trend report
make bench-api
make bench-report
```

## Performance Regression Testing (read before touching the hot path)

`make bench-api` is the canonical perf-regression command. It **derives its matrix from
the model registry** (`assets/model_configs.json`) — every installed model not opted out
with `"benchmark": false` is benchmarked in its as-shipped default config, against the
real HTTP server, with one uniform spec (fixed prompts, temp 0, warmup + repeats),
recording **prefill (TTFT)** and **decode (tok/s)** separately to `assets/api_perf_log.tsv`.
`make bench-report` compares the latest rows to prior commits (keyed on `hw_model`) and
flags regressions.

Preserve and commit the performance log under the working rules above, including
rows from functional smoke tests. Report flags are investigation hints, not a demand
to repair history or repeatedly benchmark until everything is green.

This design exists because the perf log silently went stale for the entire MTP + dense
arc: the suite followed a single configured model, so new models/quants were never
measured. Coverage is now structural, not manual — **but two things still require a human:**

1. **Adding a model/quant variant:** put it in the registry (you must anyway, for the TUI)
   and it is benchmarked automatically. Only set `"benchmark": false` for variants that
   physically cannot run here (e.g. the 397B). Do **not** add a parallel benchmark list —
   a second source of truth is exactly what rotted last time.
2. **Adding a perf-affecting *axis* that is not a model** (a new kernel mode, an MTP
   depth, a sampling/runtime toggle that ships on by default): it is exercised
   automatically *only* if it's part of the model's default config. If it's a new opt-in
   path, add it to the spec in `tests/bench_api.sh` so every model covers it.

**Changes to `infer.m` require the standard watchdog regardless of their apparent
hot-path impact.** Run `make bench-api` and review `make bench-report` against the
finalized code before committing; decode/prefill, kernel, attention, and
speculative-decoding changes have the same requirement. Seek the bounded approval
required above unless Dave has already authorized the run. If it cannot run safely,
report the performance-validation gap explicitly rather than treating functional
tests as watchdog coverage. Preserve and commit its log, including noisy rows and
report warnings. Do not duplicate this run with ad-hoc baselines or rerun it merely
to clear small or historical flags. Ad-hoc `--mtp-generate-*` numbers are for
approved experiments, not routine regression sign-off.
Canonical benchmarks must pass the system-health preflight. Active Time Machine,
thermal/performance warnings, another inference process, or sustained CPU/GPU pressure
make latency and throughput results invalid. The harness records the sampled state on
every row and refuses to run under known contention by default. `--allow-busy-system`
exists only for diagnostic reproduction: every row it emits must be marked `confounded`
and excluded from regression reports, but retained in the log. Material unexpected
slowdowns warrant checking prompt, memory pressure, runtime settings, and system
activity first, not automatically running a new baseline campaign. Historical absolute
throughput numbers are not universal thresholds across models, prompts, and machines.


## Debug Features Must Be Settable From the Config Menu

**Rule:** every debug/profiling feature added to the engine MUST be exposed as a
setting in the interactive config menu — not env-var-only. A flag a user can't
reach without exporting `FLASHCHAT_*` by hand is effectively undiscoverable.

The menu's advanced section is the home for these toggles:

- **Menu UI:** `modelmgr/tui/config_wizard.py` → `_advanced_settings()`
  ("Configure advanced options (debug, MTP, cache)?"). Add a
  `(KEY, "label", "default")` tuple here.

A new debug key must be wired through the whole config chain, or it won't reach
the engine. Use an existing debug flag (e.g. `SERVER_HTTP_LOG`) as the template —
grep it to find every site:

- **`lib/config.sh`** — eight sites: `FLASHCHAT_DEFAULT_<KEY>` constant, the
  reset/var declaration (top-level + inside `flashchat_load_config`), the
  `_flashchat_migrate_config` backfill list, the `FLASHCHAT_<KEY>` env-override
  apply, the finalize-default line, and the `flashchat_get` getter `case`. Bump
  `FLASHCHAT_CONFIG_SCHEMA_VERSION` so existing user configs backfill the key.
- **`flashchat`** — the `cmd_serve` launch block bridges config → environment
  (`FLASHCHAT_<KEY>="$(flashchat_get <KEY>)" \` on the `infer` exec line);
  runtime settings are read from the config file/env, not shell args. Also add
  the key to the `config --show` dump.
- **`metal_infer/infer.m`** — the engine reads the flag via
  `getenv("FLASHCHAT_<KEY>")`. Treat empty string as "unset/default" (an unset
  config key bridges through as `""`), not as a literal value.

Reference implementations — both added across exactly these sites, use either as
the worked example when adding the next debug feature:
- **pread timing profiler** — `FLASHCHAT_PREAD_PROFILE` / `FLASHCHAT_PREAD_PROFILE_CAP`
  (`prof_pread` wrapper in `infer.m`; analyze with `tools/pread_profile_analyze.py`).
- **expert pin-cache** — `FLASHCHAT_EXPERT_PIN_MAX_GB` (cap GiB, 0=off) /
  `FLASHCHAT_EXPERT_PIN_AUTO_FRAC` (free-RAM fraction) / `FLASHCHAT_EXPERT_PIN_MLOCK`.
  Keeps the hottest whole experts RAM-resident (LFU, `expert_pin_*` in `infer.m`)
  so low-RAM machines don't thrash the page cache re-streaming experts per token.
  Budget = `min(AUTO_FRAC × free RAM at first use, MAX_GB)`, fail-soft to pure pread.
  A cache slot returned as a hit remains live until the current batch has finished
  consuming it. Admission may evict only slots that are neither loading nor referenced
  by any active batch item; otherwise a later miss can overwrite bytes still queued for
  GPU upload. Exercise forced-eviction tests with a cache much smaller than one batch's
  working set, and compare generated output byte-for-byte with the cache disabled.

## Code Style

### General Principles
- Write self-documenting code with clear naming
- No comments unless required for the complexity (comments are never for historical context)
- Prefer early returns for error conditions
- Keep functions focused and single-purpose
- **Configuration policy:** shipped model metadata belongs in `assets/model_configs.json`; user settings belong in `~/.config/flashchat/config` and should select models by `MODEL` ID; generated per-model runtime artifacts belong in `<model>/flashchat/` so entire model snapshots can be moved or restored as a unit. Model offload storage is one global `OFFLOAD_DIR`, never a per-model config path.
- **User/app state policy:** Flashchat-owned user state belongs under `~/.config/flashchat/`, including sessions, prompt history, default server logs, pid files, and optional `system.md`.
- **Do not add implicit config fallbacks.** Use `--config FILE` only as an explicit override, otherwise use `~/.config/flashchat/config`, environment overrides, and defaults derived from the bundled model registry.
- **Do not rely on current working directory for model registry lookup.** Shell, Python, and C/Objective-C callers should resolve the registry via `FLASHCHAT_MODEL_CONFIG` or the repo-root `assets/model_configs.json` path.
- **Only the registry-backed production expert format is supported in active code.** Old low-precision experiments may remain in historical results/paper artifacts, but do not reintroduce alternate runtime paths, setup scripts, config settings, or user-facing UX.
- **Model storage management must be registry-aware and confirmation-gated.** Whole-model offload/reload operates on HuggingFace cache repo directories, runtime-only restore copies only `<model>/flashchat/`, and destructive model actions require typing the exact model ID.
- **Always ask the user before modifying their system** (e.g., installing packages, changing config files, running system commands)
- **Setup/dependency installs must start with one explicit consent screen (`Y/x`) before any download/install work begins.** `X` must cancel cleanly with a clear "User cancelled" style message and instructions to re-run `flashchat` later.
- **When the user gives specific instructions for moving forward, ask if you should update the AGENTS.md file to ensure the instruction is adhered to.**
- **When testing interactive scripts (especially with piped input), if output is confusing or unreadable, improve the output formatting for clarity before continuing**
- **Fresh-install/setup testing must use isolated timestamped `HOME` directories under the project, such as `debug/fresh-envs/YYYYMMDD-HHMMSS/home`. Do not test setup flows by toggling, renaming, or moving the user's real `~/.config/flashchat` or model cache.**
- **Never assume the user has knowledge of the system, commands, or syntax. Always provide instruction that is atomic and self-explanatory.**
- **Use user-friendly terminology, not technical precision.** For example, use "context window" instead of "max tokens" since that's what users expect in modern LLM interfaces, even though the technical term is different.
- **When making user-facing changes (especially output/UX), look for similar patterns elsewhere in the project.** For example, if you consolidate section headers in one command, check other commands for the same issue and apply the same fix.
- **Don't ask the user to perform manual steps that can be automated.** If something needs testing, write a script to do it automatically rather than asking the user to do it manually.
- **When changing the HTTP/API surface, add or update an automated smoke test whenever feasible.** Prefer a lightweight script that exercises the live endpoints the same way frontend harnesses will.
- **Tool-calling changes should validate rendering before live agent testing.** Use `metal_infer/infer --render-request ... --render-output ...` and `make tool-template-smoke` to verify native Qwen tool-template rendering and XML parser behavior before comparing nanocode/opencode live logs.
- **Validate generated tool arguments against the client-provided schema before emitting a call.** Validation must be tool- and client-agnostic, report recognized violations with instance/schema paths, and log unsupported schema keywords without falsely rejecting the call. Never invent missing semantic values. Intercept recognized violations, give the model one bounded correction attempt in the existing context, then stop cleanly if the corrected call is still invalid.
- **Shell helpers should preserve the caller's working directory unless changing directories is the explicit purpose of the helper.** Save and restore `pwd` inside build/setup helpers that `cd` internally.
- **Always verify the environment state before and after making changes.** Check that config files, model caches, and other state are preserved or properly restored. Test that changes don't inadvertently delete or corrupt user data.
- **Server-side debug visibility matters.** Changes to `infer --serve` should preserve persistent logging so background server runs remain debuggable without requiring an interactive launch.
- **Persistent system prompt caches belong with the model runtime artifacts.** Store them under `<model>/flashchat/system_prompt_cache/` unless set to an alternate path via the user config. validate model dimensions and prompt hash before reuse, write atomically, and keep cache counts bounded to avoid unnecessary SSD churn.
- **Running server state must be invalidated when runtime inputs change.** If model selection, model registry contents, server-affecting config, or infer source/binary state changes, Flashchat-owned servers should be treated as stale and restarted before reuse.
- **When runtime behavior does not match expectations, anchor debugging in captured logs, request dumps, build stamps, and reproducible evidence.** Do not default to assuming the user launched an old binary or made a procedural mistake unless the evidence specifically points there.
- **Server control commands must verify reality, not assume it.** Start/stop helpers should confirm health or actual process exit before reporting success or removing pid/state files.
- **Persistent server/runtime toggles should be first-class config options.** If a setting is useful beyond one-off debugging, surface it through the config file and `flashchat` configuration wizard instead of leaving it env-only.
- **Sampling/runtime generation knobs must stay consistent across config, API request parsing, server runtime signatures, docs, and smoke-test perf logging.** Do not add hidden sampler defaults that cannot be inspected or overridden from Flashchat's normal user surfaces.
- **Model-specific sampling profiles belong in `assets/model_configs.json`.** The Flashchat config should select a `SAMPLING_PROFILE`, with `custom` reserved for manually managed sampler/reasoning values.
- **Multi-token prediction policy:** MTP is a model/profile/server-default capability, not just a debugging flag. Treat user config `MTP` as a numeric draft budget: empty means use the registry fallback chain, `0` means force disabled, `1` means automatic, and `2+` means that predictor batch size. Model and profile metadata use `mtp_default_predictions`; profile values override server defaults, server defaults override model defaults, and model defaults should be `1` unless there is a specific reason otherwise. MTP-capable models can require extracted MTP artifacts independently of the selected runtime budget.
- **Python interpreter resolution must go through `lib/python.sh`.** The launcher never calls bare `python3` for venv creation or package install — always `flashchat_python_bin` (or its resolved path). On macOS 26, `/usr/bin/python3` is CLT 3.9 and will shadow any Homebrew Python on PATH; `python3 -m venv` against it produces a 3.9 venv that cannot import `modelmgr/` because the codebase uses PEP 604 union syntax unconditionally. Minimum supported interpreter is 3.10; the resolver enforces this and refuses to fall back silently.
- **Stale-venv recovery is silent.** If an existing `metal_infer/.venv/` interpreter falls below the supported Python floor, `ensure_venv` recreates it with a one-line log message — no user prompt, no consent screen, because nothing on the user's system is being modified (Homebrew Python lives under `/opt/homebrew/`, not `/usr/`). The venv is regenerated state, not user data.
- **When a verified coherent unit of work is complete, ask whether it should be committed before ending the turn.** Don’t interrupt active debugging for every small change, but after implementation plus relevant verification, make the commit checkpoint explicit instead of leaving it implicit.

### C/Objective-C

**Indentation**: 4 spaces (no tabs)

**Naming Conventions**:
- Global variables: `g_<name>` (e.g., `g_timing`, `g_cache_telemetry_enabled`)
- Constants: `UPPER_SNAKE` or `kCamelCase` (e.g., `MAX_K`, `kMetalDevice`)
- Functions: `snake_case` (e.g., `load_expert_weights`)
- Types: `snake_case_t` or `CamelCase` (e.g., `bpe_tokenizer`, `LayerTimingAccum`)

**Imports**:
```objc
// Objective-C (Foundation, Metal)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// C standard library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
```

**Error Handling**:
- Return error codes (0 for success, -1 for failure)
- Check return values immediately with early returns
- Log errors to stderr with descriptive messages
- Use `NSError **` for Objective-C methods

**File Structure**:
```objc
// ============================================================================
// Section divider comment
// ============================================================================

static helper_function() { ... }  // file-scope functions are static

// Public function
int public_function(int arg) {
    if (arg < 0) {
        return -1;
    }
    // ... implementation
    return 0;
}
```

### Metal Shaders (shaders.metal)

**Indentation**: 4 spaces

**Header Format**:
```metal
/*
 * shaders.metal — Brief description
 *
 * Detailed description of operations.
 *
 * Quantization format: ...
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Kernel: kernel_name
// ============================================================================

kernel void kernel_name(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      size   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // ... implementation
}
```

**Naming**: `snake_case` for all identifiers (functions, variables, parameters)

**Inline Functions**:
```metal
inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}
```

### Python Scripts

**Style**: PEP 8 compliant, snake_case for functions/variables

**Docstrings**: Triple quotes, one-line for simple functions
```python
def parse_layers(spec):
    """Parse layer specification like '0-4' or '0,5,10'."""
    if spec is None or spec == 'all':
        return list(range(60))
    # ...
```

**Imports**: Standard library first, then third-party
```python
import argparse
import json
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
```

### Python Interpreter Resolution

The launcher (`flashchat`) and any helper script that needs to create a venv or run `pip install` **must** source `lib/python.sh` and call `flashchat_python_bin`. Never call `python3` directly in shell. Reasons:

- **macOS 26 ships `/usr/bin/python3 = 3.9`** from CommandLineTools. That `python3` shadows any Homebrew Python the user has installed (`/opt/homebrew/bin/python3` is later on PATH for shells that have run `brew shellenv`, but a fresh terminal or a non-interactive script can still hit CLT first).
- **`python3 -m venv` inherits the parent interpreter.** A venv is a thin wrapper around an existing Python binary — there is no "use a different Python" knob. If you `python3 -m venv` against CLT 3.9, your venv is 3.9, full stop.
- **`modelmgr/` uses PEP 604 union syntax unconditionally** (`str | None`, `dict | None`, ~50+ sites across `artifacts.py`, `manifest.py`, `migrate.py`, `configfile.py`, `status.py`, etc.). 3.9 cannot parse these at import time, so the first `modelmgr` import raises `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` and the user gets a "Setup incomplete" screen for what looks like a setup bug.

`flashchat_python_bin` resolves in this priority:

1. `$FLASHCHAT_PYTHON` — explicit override (path is verified executable and meets the version floor).
2. `/opt/homebrew/bin/python3` — Apple Silicon Homebrew. Probed, must report ≥ 3.10.
3. `/usr/local/bin/python3` — Intel Homebrew legacy prefix. Probed, must report ≥ 3.10.
4. `python3` from PATH — last resort, must report ≥ 3.10.

If no candidate satisfies the floor, the helper exits non-zero with a clear actionable message ("`brew install python`" or "set `FLASHCHAT_PYTHON`"). It **never** modifies the user's `PATH`, default `python3`, or any system-level state.

**Minimum supported Python: 3.10.** PEP 604 is the floor. Do not introduce `match` statements or PEP 695 type aliases without also bumping `FLASHCHAT_MIN_PYTHON_MINOR` in `lib/python.sh`.

**Stale-venv handling.** `ensure_venv` in the launcher checks whether the existing `metal_infer/.venv/bin/python` meets the floor; if it doesn't (e.g., the venv was created with CLT 3.9 from a previous flashchat run), the venv is recreated silently using the resolver's chosen interpreter. A single `Recreating project venv (existing interpreter below Python 3.10).` line is logged. This is intentional: regenerating `metal_infer/.venv/` is regenerating derived state, not user data — no consent screen needed.

**Testing the resolver in isolation:**

```bash
# Default (PATH has whatever it has)
source lib/python.sh && flashchat_python_bin

# Simulate a CLT-only PATH — should still find Homebrew Python
PATH="/usr/bin:/bin:/usr/sbin:/sbin" bash -c 'source lib/python.sh && flashchat_python_bin'

# Force-fail: pin to CLT 3.9, expect non-zero exit and an actionable error
FLASHCHAT_PYTHON=/usr/bin/python3 bash -c 'source lib/python.sh && flashchat_python_bin'
```

## Architecture Notes

### Metal Compute Pipeline
1. CMD1: Attention projections + delta-net (GPU)
2. CPU: Flush results, softmax + topK routing
3. SSD: Parallel pread K=4 experts (2.4ms)
4. CMD2: o_proj + norm + routing (GPU)
5. CMD3: Expert forward + combine + norm (GPU, deferred)

### 4-bit Quantization Format
- MLX affine 4-bit, group_size=64
- Weights: uint32 (8 x 4-bit values packed)
- Per-group scale and bias in bfloat16
- Dequant: `value = uint4_val * scale + bias`

### Memory Management
- Non-expert weights: mmap'd (5.5GB, read-only)
- Expert weights: Streamed from SSD on demand
- Metal scratch: ~200MB
- Trust OS page cache for expert data (no custom LRU)

### Performance-Critical Patterns
- Deferred GPU command submission for pipeline overlap
- FMA-optimized dequant kernels
- BLAS (Accelerate) for linear attention
- Parallel `pread()` for expert I/O
