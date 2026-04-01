# Running Flash-MoE

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
- View status

## Installation

### Prerequisites

That's it! flashchat automatically:
- Creates a Python virtual environment with numpy on first run
- Builds the binaries when needed (infer, chat)

That's it! On first run, flashchat will automatically:
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

Starts the local Flash-MoE server if needed, checks for an OpenCode config at `~/.config/opencode/opencode.jsonc`, and launches `opencode` from the repo root.

### API Server

```bash
./flashchat serve                  # Start API server
./flashchat serve --stop           # Stop running server
./flashchat serve --port 8080      # Start on specific port
```

Starts the OpenAI-compatible HTTP server. Server runs persistently until stopped.

When `infer` runs in server mode, it also appends timestamped server activity to:

```text
~/.config/flash-moe/logs/server.log
```

This is useful when `flashchat` starts the server in the background and you want to review request timing or errors afterward. You can tail it directly:

```bash
tail -f ~/.config/flash-moe/logs/server.log
```

To override the log path for a single run:

```bash
FLASHMOE_SERVER_LOG=/tmp/flash-moe-server.log ./flashchat serve
```

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
./flashchat benchmark full         # Full 60-layer forward (K=4)
./flashchat benchmark fullbench   # Full benchmark (3 iterations)
```

Runs performance benchmarks. Uses configuration for model paths.

### Configuration

```bash
./flashchat config                 # View configuration
./flashchat config --reset         # Re-run setup wizard (keeps sessions)
./flashchat config --full-reset    # Delete all data and start fresh
```

View and edit settings. If no config exists, defaults are used automatically. The reset option allows you to reconfigure while preserving chat sessions.

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

1. `./flashmoe.config` (project-local)
2. `~/.config/flash-moe/config` (user)
3. Environment variables (`FLASHMOE_*`)
4. Hardcoded defaults

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASHMOE_MODEL_REPO` | HuggingFace repo | `mlx-community/Qwen3.5-397B-A17B-4bit` |
| `FLASHMOE_MODEL_PATH` | Override model path | Auto-detected |
| `FLASHMOE_QUANTIZATION` | 4bit or 2bit | `4bit` |
| `FLASHMOE_SERVER_PORT` | Server port | `8000` |
| `FLASHMOE_WEIGHTS_DIR` | Weights directory | `./metal_infer` |
| `FLASHMOE_EXPERTS_DIR` | Experts directory | `<model>/packed_experts` |

### Example Config File

```bash
# ~/.config/flash-moe/config

# Model Settings
MODEL_REPO="mlx-community/Qwen3.5-397B-A17B-4bit"

# Quantization: 4bit or 2bit
QUANTIZATION="4bit"

# Generation Defaults
MAX_TOKENS="8192"
TEMPERATURE="0.7"
TOP_P="0.9"

# Server Settings
SERVER_PORT="8000"
SERVER_HOST="127.0.0.1"

# UI Settings
SHOW_THINKING="0"
COLOR_OUTPUT="1"
```

## API Endpoints

When running the server (`./flashchat serve`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (SSE streaming) |
| `/v1/responses` | POST | Responses API compatibility endpoint |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

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
    "input": "Summarize why Flash-MoE works on Apple Silicon.",
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

From `metal_infer/`:

```bash
make api-smoke
```

This checks:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` with and without streaming
- `POST /v1/responses` with and without streaming
- tool-call round trips for both endpoints

If nothing is already listening on the configured port, the script starts `./infer --serve` automatically.

When enabled, it also appends lightweight timing rows to:

```bash
metal_infer/api_perf_log.tsv
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

## Setup Artifacts

| File | Size | Description |
|------|------|-------------|
| `metal_infer/model_weights.bin` | 5.5GB | Non-expert weights (mmap'd) |
| `metal_infer/model_weights.json` | 371KB | Manifest for weight loading |
| `metal_infer/vocab.bin` | 7.8MB | Tokenizer vocabulary |
| `<model>/packed_experts/` | 218GB | 4-bit expert weights |
| `<model>/packed_experts_2bit/` | 120GB | 2-bit expert weights |
| `~/.config/flash-moe/config` | - | User configuration |
| `~/.config/flash-moe/sessions/` | - | Chat session history |
