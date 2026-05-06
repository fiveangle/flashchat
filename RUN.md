# Running Flashchat

This guide covers how to run inference using the `flashchat` CLI wrapper.

## Quick Start

```bash
./flashchat
```

Running `flashchat` with no arguments launches an interactive menu where you can:
- Start a new chat session
- Resume an existing session
- Start the API server
- Configure settings
- Manage model storage
- View status

## Installation

### Prerequisites

Before running Flashchat, you need:

- An Apple Silicon Mac with at least 16GB of RAM
- Enough free internal SSD space for the selected model and generated expert data
- Xcode Command Line Tools for building the Metal inference binaries

On first run, `flashchat` will guide you through the rest:

- Create a Python virtual environment with NumPy for setup scripts
- Build the required binaries when needed (`infer`, `chat`)
- Download the model from HuggingFace (if not present)
- Create a default configuration file
- Prompt you through extracting weights and expert data

## Commands

### Interactive Mode

```bash
./flashchat
```

Launches an interactive menu with options for chat, server, configuration, and more.

### Chat

```bash
./flashchat chat                    # Start new chat session
./flashchat chat --resume <id>     # Resume existing session
```

Starts the interactive chat TUI. Automatically starts the server if not running.

### OpenCode Harness

```bash
./flashchat opencode
./flashchat opencode --port 8080
```

Starts the local Flashchat server if needed, checks for an OpenCode config at `~/.config/opencode/opencode.jsonc`, and launches `opencode` from the repo root.

### API Server

```bash
./flashchat serve                  # Start API server
./flashchat serve --stop           # Stop running server
./flashchat serve --stop --external # Stop an external Flashchat infer server on the configured port
./flashchat serve --port 8080      # Start on specific port
```

Starts the OpenAI-compatible HTTP server. Server runs persistently until stopped.

If the interactive menu shows `Running (external)`, Flashchat sees a listener on the configured server port but does not have a pid file for it. The `[O]n/[O]ff` menu option can offer to stop it, defaulting to no. The CLI equivalent is `./flashchat serve --stop --external`, which refuses to stop processes that do not look like Flashchat's `infer` server.

When `infer` runs in server mode, it also appends timestamped server activity to:

```text
~/.config/flashchat/logs/server.log
```

This is useful when `flashchat` starts the server in the background and you want to review request timing or errors afterward. You can tail it directly:

```bash
tail -f ~/.config/flashchat/logs/server.log
```

To override the log path for a single run:

```bash
FLASHCHAT_SERVER_LOG=/tmp/flashchat-server.log ./flashchat serve
```

Two server-side logging features can be enabled independently through the `flashchat` configuration wizard:

- `SERVER_DEBUG=1`
  - writes prompt/debug artifacts such as raw request bodies, assembled prompts, and final system prompts
- `SERVER_HTTP_LOG=1`
  - appends raw API traffic to:

```text
~/.config/flashchat/logs/http.log
```

This is useful for debugging frontend compatibility problems, SSE formatting, and unexpected request payloads without enabling the heavier prompt artifact dumps.

### Single Prompt

```bash
./flashchat prompt "Hello world"
./flashchat prompt "Explain quantum computing" --tokens 50
```

Runs a single prompt and prints the response.

### Benchmark

```bash
./flashchat benchmark              # Show available benchmarks
./flashchat benchmark run          # Single expert forward pass
./flashchat benchmark verify       # Metal vs CPU verification
./flashchat benchmark bench        # Single expert benchmark (10 iterations)
./flashchat benchmark moe          # MoE forward (K experts, single layer)
./flashchat benchmark moebench     # MoE benchmark (10 iterations)
./flashchat benchmark full         # Full model forward (K=4)
./flashchat benchmark fullbench   # Full benchmark (3 iterations)
```

Runs performance benchmarks. Uses configuration for model paths.

### Tool Template Debugging

```bash
./metal_infer/infer --render-request request.json --render-output debug/rendered-request
./metal_infer/infer --parse-tool-call tool_call.txt
make tool-template-smoke
```

The render path parses an OpenAI-compatible request and writes the exact native Qwen system prompt, conversation text, assembled prompt, and summary counts without loading model weights or starting the server. Use it to compare nanocode/opencode request logs with Flashchat's prompt rendering before debugging live model behavior.

### Configuration

```bash
./flashchat config                 # View configuration
./flashchat config --reset         # Re-run setup wizard (keeps sessions)
./flashchat config --full-reset    # Delete all data and start fresh
```

View and edit settings. If no config exists, defaults are used automatically. The reset option allows you to reconfigure while preserving chat sessions.

The configuration wizard selects from the models in `assets/model_configs.json` and shows the local setup state for each model, including downloaded HuggingFace files and generated files under `<model>/flashchat/`.

### Models

```bash
./flashchat models
```

Lists supported models and their local setup status. This command is read-only.

### Manage Model Storage

```bash
./flashchat manage
./flashchat manage --list
```

Manages local and offloaded model storage. The manage view shows each supported model's local/offloaded status, runtime readiness, original blob size, and total storage footprint.

Available storage actions:

- Remove original HuggingFace safetensors blobs after the generated runtime files are complete
- Delete a local model cache repo
- Offload a whole HuggingFace cache repo to `OFFLOAD_DIR`
- Fully reload an offloaded model back to the local HuggingFace cache
- Restore only the generated `<model>/flashchat/` runtime files from offload storage

Destructive actions require typing the exact model ID. Offload storage uses one global directory configured as `OFFLOAD_DIR` in `~/.config/flashchat/config` or overridden with `FLASHCHAT_OFFLOAD_DIR`.

### Status

```bash
./flashchat status
```

Shows system status including model, paths, server status, and generation settings.

### Sessions

```bash
./flashchat sessions              # List all sessions
./flashchat sessions --delete <id> # Delete a session
```

Manage chat sessions.

## Configuration

Configuration is loaded from (priority highest to lowest):

1. `--config FILE` (explicit override)
2. `~/.config/flashchat/config` (user)
3. Environment variables (`FLASHCHAT_*`)
4. Registry/default values

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASHCHAT_MODEL` | Supported model ID | `qwen3.6-35B-A3B` |
| `FLASHCHAT_MODEL_CONFIG` | Model registry path, including the default model and active setup scripts | `assets/model_configs.json` |
| `FLASHCHAT_MODEL_PATH` | Override model path | Auto-detected |
| `FLASHCHAT_OFFLOAD_DIR` | Unified root for offloaded HuggingFace model cache repos | unset |
| `FLASHCHAT_SERVER_PORT` | Server port | `8000` |
| `FLASHCHAT_SERVER_HOST` | Server host | `127.0.0.1` |
| `FLASHCHAT_WEIGHTS_DIR` | Weights directory | `<model>/flashchat` |
| `FLASHCHAT_EXPERTS_DIR` | Experts directory | `<model>/flashchat/packed_experts` |

### Example Config File

```bash
# ~/.config/flashchat/config

# Model Settings
MODEL="qwen3.6-35B-A3B"

# Storage Settings
OFFLOAD_DIR=""

# Generation Defaults
MAX_TOKENS="8192"
TEMPERATURE="0.7"
TOP_P="0.9"

# Server Settings
SERVER_PORT="8000"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="$HOME/.config/flashchat/logs/server.log"

# UI Settings
SHOW_THINKING="0"
COLOR_OUTPUT="1"
```

`SERVER_LOG_PATH` may be a `.log` file or a directory. Extensionless paths entered in the configuration wizard are treated as directories and will receive `server.log` plus debug artifacts when debug logging is enabled.

## API Endpoints

When running the server (`./flashchat serve`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (SSE streaming) |
| `/v1/responses` | POST | Responses API compatibility endpoint |
| `/v1` | GET | Lightweight service info / compatibility probe |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

The examples below assume the default `SERVER_HOST="127.0.0.1"` and `SERVER_PORT="8000"`.

### Example API Call

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-397b-a17b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### Responses API Example

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-397b-a17b",
    "input": "Summarize why Flashchat works on Apple Silicon.",
    "max_output_tokens": 256,
    "temperature": 0.2
  }'
```

### Reasoning Off Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-397b-a17b",
    "messages": [{"role": "user", "content": "Give a short answer only."}],
    "reasoning": false,
    "stream": false
  }'
```

## API Smoke Test

From the project root:

```bash
make cli-smoke
make api-smoke
make test
```

`make cli-smoke` runs the Flashchat CLI smoke test.

`make api-smoke` checks:
- `GET /health`
- `GET /v1`
- `GET /v1/models`
- `POST /v1/chat/completions` with and without streaming
- `POST /v1/responses` with and without streaming
- tool-call round trips for both endpoints

`make test` runs both smoke tests.

If nothing is already listening on the configured port, the script starts `metal_infer/infer --serve` automatically.

When enabled, it also appends lightweight timing rows to:

```bash
assets/api_perf_log.tsv
```

Each row records the date, branch, commit, endpoint scenario, request mode, duration, and derived stream tok/s when available. This is meant for spotting regressions over time, not for scientific benchmarking.
The log also records the hostname, hardware model, RAM size, and a compact CPU/GPU core summary so results from different Apple Silicon machines can be compared later.

## Harness Config

For OpenCode, a working provider entry looks like:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "flashchat": {
      "name": "flashchat",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "mlx-community/Qwen3.5-397B-A17B-4bit": {
          "name": "Qwen3.5-397B-A17B-4bit",
          "tools": true,
          "limit": {
            "context": 220000,
            "output": 16000
          }
        }
      },
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1"
      }
    }
  }
}
```

If you changed `SERVER_HOST` or `SERVER_PORT`, use those configured values in `baseURL`.

## Setup Artifacts

| File | Size | Description |
|------|------|-------------|
| `<model>/flashchat/model_weights.bin` | 5.5GB | Non-expert weights (mmap'd) |
| `<model>/flashchat/model_weights.json` | 371KB | Manifest for weight loading |
| `<model>/flashchat/vocab.bin` | 7.8MB | Tokenizer vocabulary |
| `<model>/flashchat/expert_index.json` | - | Safetensors expert lookup index |
| `<model>/flashchat/packed_experts/` | 218GB | Expert weights |
| `~/.config/flashchat/config` | - | User configuration |
| `~/.config/flashchat/sessions/` | - | Chat session history |
| `~/.config/flashchat/history` | - | Interactive prompt history |
| `~/.config/flashchat/system.md` | - | Optional custom system prompt |
