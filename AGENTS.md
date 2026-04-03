# AGENTS.md — Flash-MoE Development Guide

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

root/
  repack_experts.py # Repack safetensors to binary
  progress.py       # Results visualization
  results.tsv       # Experiment log
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
- **Always ask the user before modifying their system** (e.g., installing packages, changing config files, running system commands)
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
