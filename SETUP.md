# Flashchat New-Install Setup Workflow

Developer reference documenting the full machinery behind a fresh `./flashchat` invocation. Not user-facing.

## Source Files

| File | Role |
|---|---|
| `flashchat` | Main bash CLI (~3200 lines). All setup orchestration lives here. |
| `lib/config.sh` | Config loading, model registry accessors, path computation (~570 lines). |
| `assets/model_configs.json` | Bundled model registry: dimensions, HF repo, quantization params, scripts. |
| `metal_infer/Makefile` | Binary compilation (`infer`, `chat`, `metal_infer`). |
| `scripts/extract_weights.py` | Dispatch to model-specific non-expert weight extraction. |
| `scripts/export_tokenizer.py` | BPE tokenizer vocabulary export to compact binary (`vocab.bin`). |
| `scripts/generate_expert_index.py` | Scan safetensors index to locate per-expert tensor offsets. |
| `scripts/repack_experts.py` | Dispatch to model-specific expert repacking into per-layer blobs. |
| `scripts/models/<model>-extract_weights.py` | Model-specific non-expert extraction. |
| `scripts/models/<model>-repack_experts.py` | Model-specific expert repacking. |
| `metal_infer/infer.m` | Full inference engine (~8500 lines). |
| `metal_infer/main.m` | Benchmark binary (~1800 lines). |
| `metal_infer/shaders.metal` | Metal compute kernels, compiled at runtime. |
| `metal_infer/tokenizer.h` | C single-header BPE tokenizer. |

## Entry Point: `main()` — flashchat:3084

```
./flashchat (no args)
  main()
    parse_global_opts "$@"   → handles -v, -q, --config, --model, -h
    if $# == 0 → interactive_menu()
    else → dispatch to command handler (chat, serve, prompt, etc.)
```

Global option flags (`VERBOSE`, `QUIET`, `CONFIG_FILE`, `FLASHCHAT_MODEL`) are set here, but `CONFIG_FILE` is a local variable consumed by downstream functions via `--config FILE` handling. See "Known Bug" below.

**Known bug:** `main()` processes `--config FILE` into a local variable `CONFIG_FILE`, but `flashchat_load_config()` in `lib/config.sh` does not read it. The `--config` override is only effective when explicitly checked by `flashchat` functions that look for `CONFIG_FILE` directly (e.g., `cmd_config`, `interactive_menu`). For other commands, the config is loaded from the standard priority chain, not from `--config`.

## Config Loading: `flashchat_load_config()` — lib/config.sh:302

Called by every command and `interactive_menu()`. Priority chain:

```
1. --config FILE override          (if FLASHCHAT_CONFIG_FILE_OVERRIDE is set)
2. ./flashchat.config              (project-local, in CWD)
3. ~/.config/flashchat/config      (user)
4. FLASHCHAT_* environment variables (override any matched key)
5. model_configs.json defaults     (registry-backed profile values)
6. Hardcoded FLASHCHAT_DEFAULT_*   (constants at top of config.sh)
```

On a fresh install with no config file:
- Steps 1-3 all miss → vars remain empty
- Step 4 applies any `FLASHCHAT_*` env vars
- Step 5: `MODEL` resolves to `default_model` from `assets/model_configs.json`; `MODEL_REPO` is looked up from registry; sampling profile values fill temperature, top_p, etc.
- Step 6: remaining empty vars get hardcoded defaults (SERVER_PORT=8000, TEMPERATURE=0.7, etc.)

## Path Computation: `_flashchat_compute_paths()` — lib/config.sh:276

Computes three critical paths from the loaded config:

| Variable | Computation | Example Value |
|---|---|---|
| `MODEL_PATH` | `_flashchat_detect_model_path(MODEL_REPO)` → scans `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/`, picks latest | `/Users/.../snapshots/abc123` |
| `WEIGHTS_DIR` | `flashchat_model_runtime_dir(MODEL, MODEL_PATH)` (or `FLASHCHAT_WEIGHTS_DIR` env override). Native non-4-bit variants use `$MODEL_PATH/flashchat/q<bits>`; other models use `$MODEL_PATH/flashchat` | `/Users/.../snapshots/abc123/flashchat/q8` |
| `EXPERTS_DIR` | `$WEIGHTS_DIR/packed_experts` | `/Users/.../snapshots/abc123/flashchat/q8/packed_experts` |

`_flashchat_detect_model_path()` — lib/config.sh:252:
- Escapes `/` to `--` in the repo string
- Lists snapshot directories, takes last one (most recent)
- Falls back to a placeholder if no snapshots found

## The Core Setup Orchestrator: `ensure_setup()` — flashchat:1974

This is the single function that guarantees the environment is ready for inference. Called by `interactive_menu()`, `cmd_chat()`, `cmd_prompt()`, `cmd_opencode()`.

```
ensure_setup(force=0)
  │
  ├─ set +e                        # disable exit-on-error for check functions
  ├─ flashchat_load_config()       # load/refresh config
  │
  ├─ [A] Xcode CLI                flashchat:1908-1962
  │   └─ if check_xcode_cli() fails → prompt_install_xcode_cli()
  │       (consent-gated, installs Homebrew if needed, xcode-select --install, polls up to 10min)
  │
  ├─ [B] Model Download           flashchat:1697-1741
  │   └─ while ! check_model_downloaded()
  │       if check_model_extracted() → break (safetensors exist, skip download)
  │       prompt_download_model() → hf download <repo> (or venv + huggingface-hub)
  │       flashchat_load_config() → re-compute paths after download
  │
  ├─ flashchat_load_config()       # re-load: MODEL_PATH now resolves
  │
  ├─ [C] Non-expert Weights       flashchat:1744-1768
  │   └─ if check_weights_extracted() fails → prompt_extract_weights()
  │       ensure_venv() → create metal_infer/.venv, pip install numpy
  │       run_python scripts/extract_weights.py --model-id <MODEL> --model <PATH> --output <WEIGHTS_DIR>
  │       → produces: model_weights.bin (~5.5GB) + model_weights.json
  │
  ├─ [D] Vocabulary               flashchat:1771-1796
  │   └─ if check_vocab_exported() fails → prompt_export_vocab()
  │       run_python scripts/export_tokenizer.py <MODEL_PATH>/tokenizer.json <WEIGHTS_DIR>/vocab.bin
  │       → produces: vocab.bin (~8MB, compact BPET binary format)
  │
  ├─ [E] Expert Weights           flashchat:1799-1876
  │   └─ if check_experts_extracted() fails → prompt_extract_experts()
  │       if expert_index.json is missing or points at a stale model path:
  │         run_python scripts/generate_expert_index.py --model <PATH> --output <WEIGHTS_DIR>
  │         → scans model.safetensors.index.json, writes expert → (shard, offset, size) mapping
  │       run_python scripts/repack_experts.py --model-id <MODEL> --index <expert_index.json>
  │       → produces: packed_experts/layer_XX.bin (and packed_mtp_experts/ for native MTP MoE models)
  │
  ├─ [F] Save Config              flashchat:1879-1905
  │   └─ if no config exists → prompt_save_config()
  │       → flashchat_create_default_config() (lib/config.sh:465)
  │       → writes ~/.config/flashchat/config with all current settings
  │
  └─ set -e                        # re-enable exit on error
```

Each step is gated by its check function. If the check passes, that step is skipped. Stages C-E share the same Python venv created at stage C.

## Check Functions

All return 0 (true/ready) or 1 (false/need setup).

| Function | File:Line | Checks |
|---|---|---|
| `check_xcode_cli()` | flashchat:1908 | `xcode-select -p` succeeds |
| `check_model_downloaded()` | flashchat:1432 | Selected model/quant runtime passes manifest, vocab, expert, dense, and MTP validation |
| `check_model_extracted()` | flashchat:1457 | `model-*.safetensors` or `model.safetensors` or `config.json` exist under MODEL_PATH |
| `check_weights_extracted()` | flashchat:1560 | `model_weights.bin` and `model_weights.json` exist under WEIGHTS_DIR |
| `check_vocab_exported()` | flashchat:1571 | `vocab.bin` exists under WEIGHTS_DIR |
| `check_experts_extracted()` | flashchat:1582 | MoE expert layers match the selected model; dense models validate any required MTP tensors instead |
| `check_binaries()` | flashchat:1521 | `metal_infer/infer` and `metal_infer/chat` exist |
| `check_binaries_current()` | flashchat:1534 | builds exist AND all source files are older than their binaries |

## Model Download: `prompt_download_model()` — flashchat:1697

Two code paths:

1. **Homebrew `hf` CLI** (preferred): `hf download <model_repo>` — uses HuggingFace Hub CLI
2. **Fallback venv** (if `hf` not found): creates `/tmp/hf_$$`, pip-installs `huggingface-hub`, runs `snapshot_download(repo_id=...)` via Python one-liner, removes venv

After download, re-runs `config_load` to recompute `MODEL_PATH`, then verifies safetensors exist.

## Non-Expert Weight Extraction

### Dispatcher: `scripts/extract_weights.py`

Reads `assets/model_configs.json` → finds `scripts.<model_id>.extract_weights` → dispatches to the model-specific script.

### Model-specific script (e.g., `scripts/models/qwen3.5-397B-A17B-extract_weights.py`)

For each safetensors shard:
1. Parse header to discover tensor names, dtypes, shapes
2. Identify tensors that are NOT expert MLP gate/up/down projections
3. Extract in native dtype (4-bit packed uint32, bf16 as uint16, f32 as float32)
4. Write contiguously to `model_weights.bin` (64-byte aligned)
5. Write `model_weights.json` mapping tensor_name → {offset, size, shape, dtype}

Output: ~5.5GB for both models.

## Vocabulary Export: `scripts/export_tokenizer.py`

1. Reads HuggingFace `tokenizer.json`
2. Extracts `model.vocab` (string→token_id), `model.merges` (BPE pairs), `added_tokens`
3. Writes binary with header `"BPET"`, version=2, then three sections: vocab entries, merge pairs, added tokens

Output: ~8MB `vocab.bin`. Read by `metal_infer/tokenizer.h` at inference time.

## Expert Index Generation: `scripts/generate_expert_index.py`

Runs when `expert_index.json` is missing or when its recorded `model_path` no longer matches the current snapshot location. This matters after a HuggingFace cache repo has been moved to or run from offload storage.

1. Reads `model.safetensors.index.json` — a manifest mapping tensor names to shard files
2. Uses regex to locate expert MLP tensors: `model.layers.<N>.mlp.switch_mlp.(gate|up|down)_proj.(weight|scales|biases)`
3. For each (layer, expert, component), records: safetensors shard filename, byte offset, byte size
4. Writes `expert_index.json`: `{layers: {N: {experts: {E: {file, offsets: {...}, sizes: {...}}}}}}`

## Expert Repacking

### Dispatcher: `scripts/repack_experts.py`

Same pattern as extract_weights: reads registry → dispatches to model-specific script, passing `--index <expert_index.json>`.

### Model-specific script (e.g., `scripts/models/qwen3.5-397B-A17B-repack_experts.py`)

For each layer (0..num_layers-1):
1. Read expert_index.json for that layer's expert locations
2. For each expert (0..num_experts-1):
   - Read 9 components in fixed order: gate_proj.{weight,scales,biases}, up_proj.{weight,scales,biases}, down_proj.{weight,scales,biases}
   - Each component read from the correct safetensors shard at the correct offset
3. Write all experts contiguously into `packed_experts/layer_XX.bin`

Output: one file per layer. 397B model: 60 layers × ~3.63 GB/layer = ~218 GB. 35B model: 40 layers × ~860 MB/layer = ~34 GB.

Key flag: `--layers 0-4` for partial repacking during testing. `--verify-only <layer>` for integrity checks without writing.

## Binary Compilation: `ensure_binaries()` — flashchat:1544

Called by `cmd_serve()`, `cmd_chat()`, `cmd_benchmark()`.

```bash
ensure_binaries()
  └─ check_binaries_current()     # exists AND source files haven't changed
       └─ if stale or missing:
            make -C "$SCRIPT_DIR" all chat
```

Makefile targets (metal_infer/Makefile):

| Target | Binary | Source |
|---|---|---|
| `all` | `metal_infer` | `main.m` |
| `all` | `infer` | `infer.m` |
| `chat` | `chat` | `chat.m` + `linenoise.c` |

All use: `clang -O2 -Wall -Wextra -fobjc-arc -framework Metal -framework Foundation -framework Accelerate -lpthread -lcompression`

Metal shaders (`shaders.metal`) are compiled at **runtime** via `MTLDevice newLibraryWithSource:`, not at build time.

## Python Environment: `ensure_venv()` — flashchat:1493

Created on first extraction. Reused across all Python scripts.

```bash
ensure_venv()
  └─ if FLASHCHAT_VENV exists and has numpy → return
  └─ python3 -m venv metal_infer/.venv
  └─ metal_infer/.venv/bin/pip install numpy
```

`run_python()` — flashchat:1516: wrapper that calls `ensure_venv()` then `$FLASHCHAT_VENV/bin/python "$@"`.

## Command-Entry Paths

How each flashchat command enters setup:

| Command | setup path | Notes |
|---|---|---|
| `flashchat` (no args) | `interactive_menu()` → checks config → `ensure_setup()` if missing | Full interactive wizard |
| `flashchat chat` | `main()` → `ensure_setup()` → `cmd_chat()` | Full setup, then TUI |
| `flashchat serve` | `main()` → `cmd_serve()` → checks model, `ensure_binaries()` | Skips `ensure_setup()`; binary-only gate |
| `flashchat prompt "..."` | `main()` → `ensure_setup()` → `cmd_prompt()` | Full setup, then API call |
| `flashchat opencode` | `main()` → `ensure_setup()` → `cmd_opencode()` | Full setup, then opens opencode |
| `flashchat benchmark` | `main()` → `cmd_benchmark()` → `ensure_binaries()` | Binary-only gate |
| `flashchat status/sessions/config/models/manage` | `main()` → `flashchat_load_config()` only | Read-only; no setup needed |

## Post-Setup State

After a successful `ensure_setup()`, the following directories and files exist:

```
~/.config/flashchat/
  config                        # User config file

~/.cache/huggingface/hub/
  models--mlx-community--<model>/
    snapshots/
      <snapshot>/
        model-00001-of-N.safetensors   # original HF model shards
        model.safetensors.index.json   # tensor → shard mapping
        tokenizer.json                 # BPE tokenizer
        flashchat/
          model_weights.bin            # non-expert weights (~5.5GB)
          model_weights.json           # tensor manifest
          vocab.bin                    # tokenizer vocab (~8MB)
          expert_index.json            # expert → safetensors mapping
          packed_experts/
            layer_00.bin .. layer_59.bin  # expert weights (~218GB for 397B)
          packed_mtp_experts/          # native MoE MTP expert weights when supported
          q8/                           # native 8-bit runtime variant for the same snapshot

<project>/metal_infer/
  .venv/                          # Python venv with numpy
  infer                           # compiled inference engine
  chat                            # compiled TUI client
  metal_infer                     # compiled benchmark binary
```

## Edge Cases & Notes

- **CMD3 compute** in `infer.m` opens packed expert files with O_RDONLY and uses `pread()` for random access per expert within each layer file. No mmap for expert data (mmap was 5x slower due to per-page faults on cold data).
- **2-bit quantization** repacking follows an identical flow through `scripts/repack_experts.py` dispatching to model-specific 2-bit repack scripts. The check functions use `EXPERTS_2BIT_DIR` when `QUANTIZATION=2bit`.
- **Stale binary detection** (`check_binaries_current`) compares modification times of each source file against its binary. This prevents running a stale binary after source changes. Only relevent after initial build.
- **Multi-model support** is registry-driven: new models are added to `assets/model_configs.json` with matching `scripts/models/<model>-extract_weights.py` and `scripts/models/<model>-repack_experts.py`. The `ensure_setup()` flow is model-agnostic.
