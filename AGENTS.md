# README — What is this, and how is this possible ?

## History:

Back when Apple began its quest to free itself from reliance on CPU makers Motorola (68k), IBM (PowerPC), and Intel (x86) to provide the beating heart of its microcomputer offerings, with the flexible license framework of the fledgling ARM ecosystem, they bet hard on it, and it paid off.  But beyond the sheer compute power and efficiency, almost as an aside, it opened the door for them to enter the then-anemic handset computing workspace. This offered them freedom to bring many of the functions of its systems that were historically off-loaded to other 3rd party vendors in-house. This brought with it the ability to dispense with buying a 3rd party SSD processors from the likes of Samsung and Toshiba, instead performing all of those critical functions on-die within its own "A", then later "M" series processors. While a huge undertaking, so was ripping out the heart of their computers and replacing it with another, which Apple was no stranger to, having done so 3 times in the previous 2 decades.

One of the major benefits of this effort was bringing as many anciliaary functions under control of a processor it now had full control off. Driven primarily as a cost reduction effort, bringing NAND die control directly within control of the main CPU was one such effort. While it prevented easy upgrades of storage, it offered one major benefit besides not having to pay a 3rd party for their SSD management processor: S P E E D

What Apple has coined their "Apple Fabric", which is just a fancy name for their CPU performing all the NAND management that is traditionally reserved for a specialized processor like those from Sandforce (prior to their purchase by Seagate), Samsung, Marvel, and others, is now NAND chips connected as close as possible to the PCI bus so the CPU can manage it. This means near-wire speed for the Apple Fabric SSDs.

## How It Changes Everything

Instead of reliance on 3rd-party SSD processors managing NAND die control, the Apple M series CPU in today's "Apple Silicon" systems attains storage speed at near wire speed of the underlying architecture, resulting in the ability to attain speeds never before possible with secondary, intermediary NAND processors. And with that, comes the ability to reach read speeds that, while never able to reach parity with RAM, can, similary to Intel's all-but-defunct Optane memory systems, and now more comtemorary CTX "memory" can get close enough to RAM speed to be usable.

## What Is This Project

This project, envisioned by the original author Dan Woods, is the answer to the question, "What if instead of loading an entire Large Language Model in RAM the traditional way, we instead stream the individual LLM layers direclty to the GPU from storage?" Before Apple Fabric being ubiquotous across Apple's line up, include the base model Macbook Pro M5 I am typing this on, the idea was unheard of outside of esoteric non-volatile storage systems.

Using approximately 6GB of memory-resident space to store critical LLM metadata that requires very low-latency access for performance, when running, the 60 "expert" LLM layers of this project are streamed direclty from the underlying Apple Fabrid storage to the GPU for inference processing when each layer is activated by the respective prompt processing.  The result?  At over 200GB in size, the Quen3.5-397B-A17B-4bit LLM is historically reserved for big-iron systems in order to run. The model works by having 60 "experts" at the ready, each approximately 3.5GB in size, and based on the user prompt, will activate just a portion of those experts for each token being processed, streaming that experts data from the Apple Fabric to the GPU for processing.

The result? Over 3.5 tok/sec processing on a base model Macbook Pro M5. A feat unheard of ever before in history.

## Getting Started

Below is Dan Wood's original "readme" and paper, which I highly recommend you read, as it offers a fascinating peek into both the ingenuity of the human mind, as well as a wonderful demonstration of the capability multiplier that enlisting the power of machine learning offers to see that human ingenuity to fruition. 

Dan's original project was polished just enough to publish his whitepaper on his endevor, but was not ready for prime time regarding running by public, having fundemental issues with experts extraction and tokenizer problems. This project provides a means for the average joe to setup and run flashchat on their own laptop without much fanfare.  Using the power of Opencode and the stealth coding-specific machine learning model Big Pickle, within a few hours I was able to turn this academic effort into a (hopefully) turn-key project you can run yourself.

You can find more details in RUN.md (written entirely by Big Pickle through my guidance) but the meat of it should be to simply:
<!-- git clone https://github.com/fiveangle/flash-moe -->
cd flash-moe
./flashchat

And follow the prompts to setup and run.

## Requirements
Apple Silicon computer with at least 16GB of RAM
At least 415GB of free space on on the Apple Fabric SSD builtin to your Mac
xtools-cli
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
  Makefile          # Build system
  linenoise.c/h     # Line editing library

scripts/
  extract_weights.py        # Dispatch to model-specific non-expert extraction
  repack_experts.py         # Dispatch to model-specific expert repacking
  generate_expert_index.py  # Generate per-model expert_index.json
  export_tokenizer.py       # Export tokenizer vocabulary
  models/                   # Model-specific extraction/repack implementations

root/
  progress.py       # Results visualization
  assets/*.tsv      # Durable experiment/API metric logs
```

## Build Commands

```bash
# From metal_infer/ directory

# Build all targets
make

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
make full           # Full 60-layer forward (K=4)
make fullbench      # Full forward benchmark (3 iterations)

# Inference engine (full model)
./infer --prompt "Hello" --tokens 100

# With timing breakdown
./infer --prompt "Hello" --tokens 20 --timing

# 2-bit mode (faster but breaks tool calling)
./infer --prompt "Hello" --tokens 50 --2bit

# Single test files
clang -O2 test_lzfse.c -lcompression -o test_lzfse && ./test_lzfse
```

## Code Style

### General Principles
- **No comments unless explicitly requested** (per project convention)
- Write self-documenting code with clear naming
- Prefer early returns for error conditions
- Keep functions focused and single-purpose
- **Configuration policy:** shipped model metadata belongs in `assets/model_configs.json`; user settings belong in `~/.config/flashchat/config` and should select models by `MODEL` ID; generated per-model runtime artifacts belong in `<model>/flashchat/` so entire model snapshots can be moved or restored as a unit.
- **Do not add implicit config fallbacks.** Use `--config FILE` only as an explicit override, otherwise use `~/.config/flashchat/config`, environment overrides, and defaults derived from the bundled model registry.
- **Do not rely on current working directory for model registry lookup.** Shell, Python, and C/Objective-C callers should resolve the registry via `FLASHCHAT_MODEL_CONFIG` or the repo-root `assets/model_configs.json` path.
- **Always ask the user before modifying their system** (e.g., installing packages, changing config files, running system commands)
- **Setup/dependency installs must start with one explicit consent screen (`Y/x`) before any download/install work begins.** `X` must cancel cleanly with a clear "User cancelled" style message and instructions to re-run `flashchat` later.
- **When the user gives specific instructions for moving forward, ask if you should update the AGENTS.md file to ensure the instruction is adhered to.**
- **When testing interactive scripts (especially with piped input), if output is confusing or unreadable, improve the output formatting for clarity before continuing**
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
