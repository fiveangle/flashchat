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

1. Download the model from HuggingFace (if not already downloaded):
   ```bash
   huggingface-cli download mlx-community/Qwen3.5-397B-A17B-4bit
   ```

2. Build the binaries:
   ```bash
   cd metal_infer && make
   ```

3. Create Python virtual environment:
   ```bash
   cd metal_infer
   python3 -m venv .venv
   .venv/bin/pip install numpy
   ```

### First Run

On first run, flashchat will automatically detect missing components and guide you through setup:

1. If model is not downloaded → prompt to download from HuggingFace
2. If weights are not extracted → prompt to extract (~5.5GB)
3. If vocabulary is not exported → prompt to export (~8MB)
4. If expert weights are not extracted → prompt to choose quantization mode:
   - 4-bit (~218GB) - Full quality, tool calling supported
   - 2-bit (~120GB) - Faster but breaks tool calling

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

### API Server

```bash
./flashchat serve                  # Start API server
./flashchat serve --stop           # Stop running server
./flashchat serve --port 8080      # Start on specific port
```

Starts the OpenAI-compatible HTTP server. Server runs persistently until stopped.

### Single Prompt

```bash
./flashchat prompt "Hello world"
./flashchat prompt "Explain quantum computing" --tokens 50
```

Runs a single prompt and prints the response.

### Configuration

```bash
./flashchat config                 # View configuration
```

View and edit settings. On first run, prompts to save configuration.

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
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

### Example API Call

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-397b-a17b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
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
