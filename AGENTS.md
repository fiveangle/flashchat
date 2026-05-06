# README — What is this, and how is this possible?

## History

Back when Apple began its quest to free itself from reliance on CPU makers Motorola (68k), IBM (PowerPC), and Intel (x86) for the beating heart of its microcomputer offerings, it bet hard on the fledgling ARM ecosystem and its flexible licensing model — and the bet paid off. Beyond the obvious gains in raw compute and efficiency, the move quietly opened the door for Apple to enter the then-anemic handset market. With that came the freedom to pull functions historically off-loaded to 3rd-party vendors back in-house, including the dedicated SSD processors it had been buying from the likes of Samsung and Toshiba — now folded onto the die alongside the rest of the "A" series, and later the "M" series. A huge undertaking, but Apple was no stranger to ripping out the heart of its computers and replacing it: it had already done so 3 times in the previous 2 decades.

A major benefit of this consolidation was bringing ancillary subsystems under a processor Apple now owned end-to-end. Driven primarily by cost reduction, pulling NAND die control directly onto the main CPU was one of those moves. It came at the price of easy storage upgrades, but offered something else besides not paying a 3rd party for their SSD management processor: S P E E D

What Apple has coined "Apple Fabric" — a fancy name for the CPU performing all the NAND management traditionally reserved for a specialized controller like those from SandForce (prior to its purchase by Seagate), Samsung, Marvell, and others — is essentially raw NAND wired as close as possible to the PCI bus so the CPU can drive it directly. The result is near-wire-speed storage.

## How It Changes Everything

Without an intermediary SSD controller in the path, the Apple M-series CPU in today's "Apple Silicon" systems pulls data from NAND faster than was ever possible with a separate processor in the way. It can never quite match RAM, but — much like Intel's all-but-defunct Optane and the more contemporary CXL "memory" — it gets close enough to be genuinely useful as a tier just below it.

## What Is This Project

This project, envisioned by the original author Dan Woods, is the answer to the question, "What if, instead of loading an entire Large Language Model into RAM the traditional way, we streamed the individual LLM layers directly to the GPU from storage?" Before Apple Fabric became ubiquitous across Apple's lineup — including the base-model MacBook Pro M5 I am typing this on — the idea was unheard of outside of esoteric non-volatile storage systems.

Using roughly 6GB of memory-resident space for the LLM metadata that needs low-latency access, the 60 "expert" layers are streamed on demand from Apple Fabric storage straight to the GPU as each layer is activated. At over 200GB on disk, the Qwen3.5-397B-A17B-4bit LLM has historically been reserved for big-iron systems. It works by holding 60 experts at the ready, each roughly 3.5GB, and for every token activates only the subset the prompt actually needs — streaming just that expert's data from the SSD to the GPU on the fly.

The result? Over 3.5 tok/sec on a base-model MacBook Pro M5 — a feat unheard of until now.

## Getting Started

Below is Dan Woods' original readme and paper, which I highly recommend you read. They offer a fascinating peek into the ingenuity of the human mind, as well as a wonderful demonstration of the capability multiplier machine learning brings to seeing that ingenuity through to fruition.

Dan's original project was polished just enough to publish his whitepaper, but it wasn't ready for the public to actually run, with fundamental rough edges around expert extraction and the tokenizer. This project closes that gap so the average Joe can set up and run flashchat on their own laptop without much fanfare. Using Opencode and the stealth coding-specific machine learning model Big Pickle, within a few hours I was able to turn this academic effort into a (hopefully) turn-key project you can run yourself.

You can find more details in RUN.md (written entirely by Big Pickle under my guidance), but the meat of it should be to simply:
<!-- git clone https://github.com/fiveangle/flash-moe -->
cd flash-moe
./flashchat

And follow the prompts to set up and run.

## Requirements
Apple Silicon computer with at least 16GB of RAM
At least 415GB of free space on the Apple Fabric SSD built into your Mac
Xcode Command Line Tools
# AGENTS.md — Flashchat Development Guide

This is a pure C/Metal inference engine for running 397B parameter MoE models on Apple Silicon.

## Project Structure

```
metal_infer/
  infer.m           # Main inference engine (~7000 lines)
  shaders.metal     # Metal compute kernels (~1300 lines)
  chat.m            # Interactive chat TUI (~760 lines)
  main.m            # MoE-only benchmark
  tokenizer.h       # Single-header C BPE tokenizer
  linenoise.c/h     # Line editing library
  discarded/        # Historical optimization experiments not in active testing

scripts/
  extract_weights.py        # Dispatch to model-specific non-expert extraction
  repack_experts.py         # Dispatch to model-specific expert repacking
  generate_expert_index.py  # Generate per-model expert_index.json
  export_tokenizer.py       # Export tokenizer vocabulary
  models/                   # Model-specific extraction/repack implementations

tests/
  test_flashchat_cli.sh     # CLI regression test
  test_api_smoke.sh         # HTTP API smoke test

root/
  Makefile          # Project build/test surface
  progress.py       # Results visualization
  assets/*.tsv      # Durable experiment/API metric logs
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
./tests/test_flashchat_cli.sh
make cli-smoke
make api-smoke
make test
```

## Code Style

### General Principles
- **No comments unless explicitly requested** (per project convention)
- Write self-documenting code with clear naming
- Prefer early returns for error conditions
- Keep functions focused and single-purpose
- **Configuration policy:** shipped model metadata belongs in `assets/model_configs.json`; user settings belong in `~/.config/flashchat/config` and should select models by `MODEL` ID; generated per-model runtime artifacts belong in `<model>/flashchat/` so entire model snapshots can be moved or restored as a unit.
- **User/app state policy:** Flashchat-owned user state belongs under `~/.config/flashchat/`, including sessions, prompt history, server logs, pid files, and optional `system.md`.
- **Do not add implicit config fallbacks.** Use `--config FILE` only as an explicit override, otherwise use `~/.config/flashchat/config`, environment overrides, and defaults derived from the bundled model registry.
- **Do not rely on current working directory for model registry lookup.** Shell, Python, and C/Objective-C callers should resolve the registry via `FLASHCHAT_MODEL_CONFIG` or the repo-root `assets/model_configs.json` path.
- **Only the registry-backed production expert format is supported in active code.** Old low-precision experiments may remain in historical results/paper artifacts, but do not reintroduce alternate runtime paths, setup scripts, config settings, or user-facing UX.
- **Always ask the user before modifying their system** (e.g., installing packages, changing config files, running system commands)
- **Setup/dependency installs must start with one explicit consent screen (`Y/x`) before any download/install work begins.** `X` must cancel cleanly with a clear "User cancelled" style message and instructions to re-run `flashchat` later.
- **When the user gives specific instructions for moving forward, ask if you should update the AGENTS.md file to ensure the instruction is adhered to.**
- **When testing interactive scripts (especially with piped input), if output is confusing or unreadable, improve the output formatting for clarity before continuing**
- **Fresh-install/setup testing must use isolated timestamped `HOME` directories under the project, such as `debug/fresh-envs/YYYYMMDD-HHMMSS/home`. Do not test setup flows by toggling, renaming, or moving the user's real `~/.config/flashchat` or model cache.**
- **Never assume the user has knowledge of the system, commands, or syntax. Always provide instruction that is atomic and self-explanatory.**
- **Use user-friendly terminology, not technical precision.** For example, use "context window" instead of "max tokens" since that's what users expect in modern LLM interfaces, even though the technical term is different.
- **When making user-facing changes (especially output/UX), look for similar patterns elsewhere in the project.** For example, if you consolidate section headers in one command, check other commands for the same issue and apply the same fix.
- **Never ask the user to perform manual steps that can be automated.** If something needs testing, write a script to do it automatically rather than asking the user to do it manually.
- **When changing the HTTP/API surface, add or update an automated smoke test whenever feasible.** Prefer a lightweight script that exercises the live endpoints the same way frontend harnesses will.
- **Shell helpers should preserve the caller's working directory unless changing directories is the explicit purpose of the helper.** Save and restore `pwd` inside build/setup helpers that `cd` internally.
- **Always verify the environment state before and after making changes.** Check that config files, model caches, and other state are preserved or properly restored. Test that changes don't inadvertently delete or corrupt user data.
- **Server-side debug visibility matters.** Changes to `infer --serve` should preserve persistent logging so background server runs remain debuggable without requiring an interactive launch.
- **Server control commands must verify reality, not assume it.** Start/stop helpers should confirm health or actual process exit before reporting success or removing pid/state files.
- **Persistent server/runtime toggles should be first-class config options.** If a setting is useful beyond one-off debugging, surface it through the config file and `flashchat` configuration wizard instead of leaving it env-only.
- **When a functional block or milestone is complete, consider prompting for a commit checkpoint.** Don’t interrupt active debugging for every small change, but when a coherent unit of work lands, ask whether it should be committed before moving on.

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
