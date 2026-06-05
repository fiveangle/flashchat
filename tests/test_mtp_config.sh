#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEST_ROOT="debug/fresh-envs/$(date +%Y%m%d-%H%M%S)-mtp-config-smoke"
HOME_DIR="$TEST_ROOT/home"
CONFIG_JSON="$TEST_ROOT/model_configs.json"
mkdir -p "$HOME_DIR/.config/flashchat"

python3 - "$ROOT/assets/model_configs.json" "$CONFIG_JSON" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    data = json.load(f)

model = data["models"]["mlx-community-Qwen36-35B-A3B-4bit"]
model["mtp_default_predictions"] = 1
model["sampling_profiles"]["instruct"]["mtp_default_predictions"] = 3
data["server_defaults"] = {"mtp_default_predictions": 2}

with open(dst, "w") as f:
    json.dump(data, f, indent=2)
PY

run_get_mtp() {
    HOME="$HOME_DIR" FLASHCHAT_MODEL_CONFIG="$CONFIG_JSON" "$@"
}

profile_value=$(run_get_mtp bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MTP')
if [ "$profile_value" != "3" ]; then
    echo "FAIL: profile MTP default expected 3, got '$profile_value'" >&2
    exit 1
fi

cat > "$HOME_DIR/.config/flashchat/config" <<'EOF'
MODEL="mlx-community-Qwen36-35B-A3B-4bit"
SAMPLING_PROFILE="instruct"
MTP="0"
EOF

config_value=$(run_get_mtp bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MTP')
if [ "$config_value" != "0" ]; then
    echo "FAIL: config MTP override expected 0, got '$config_value'" >&2
    exit 1
fi

env_value=$(run_get_mtp env FLASHCHAT_MTP=2 bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MTP')
if [ "$env_value" != "2" ]; then
    echo "FAIL: env MTP override expected 2, got '$env_value'" >&2
    exit 1
fi

cat > "$HOME_DIR/.config/flashchat/config" <<'EOF'
MODEL="mlx-community-Qwen36-35B-A3B-4bit"
SAMPLING_PROFILE="instruct"
MTP="0"
MAX_TOKENS="1"
SERVER_PORT="19996"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="/tmp/flashchat-mtp-config-smoke-logs"
HUGGINGFACE_CACHE_DIR="/tmp/flashchat-mtp-config-smoke-hf"
OFFLOAD_DIR=""
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

{
    printf 'y'
    for _ in $(seq 1 12); do
        printf '\n'
    done
    printf 'auto\n'
    printf '\n'
    printf '\n'
} | HOME="$HOME_DIR" FLASHCHAT_MODEL_CONFIG="$CONFIG_JSON" ./flashchat config >/dev/null 2>&1

if ! grep -q '^MTP=""$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not write blank MTP for auto" >&2
    exit 1
fi

auto_value=$(run_get_mtp bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MTP')
if [ "$auto_value" != "3" ]; then
    echo "FAIL: auto MTP expected profile value 3, got '$auto_value'" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
del data["models"]["mlx-community-Qwen36-35B-A3B-4bit"]["sampling_profiles"]["instruct"]["mtp_default_predictions"]
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

server_default_value=$(run_get_mtp bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MTP')
if [ "$server_default_value" != "2" ]; then
    echo "FAIL: server MTP default expected 2, got '$server_default_value'" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
data["models"]["mlx-community-Qwen36-35B-A3B-4bit"]["sampling_profiles"]["instruct"]["mtp_default_predictions"] = 3
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

cat > "$HOME_DIR/.config/flashchat/config" <<'EOF'
MODEL="mlx-community-Qwen36-35B-A3B-4bit"
SAMPLING_PROFILE="custom"
MTP="2"
MAX_TOKENS="1"
SERVER_PORT="19996"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="/tmp/flashchat-mtp-config-smoke-logs"
HUGGINGFACE_CACHE_DIR="/tmp/flashchat-mtp-config-smoke-hf"
OFFLOAD_DIR=""
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

{
    printf 'y'
    printf '\n'
    printf '\n'
    printf '2\n'
    for _ in $(seq 1 12); do
        printf '\n'
    done
} | HOME="$HOME_DIR" FLASHCHAT_MODEL_CONFIG="$CONFIG_JSON" ./flashchat config >/dev/null 2>&1

if ! grep -q '^SAMPLING_PROFILE="thinking-coding"$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not save selected canned profile" >&2
    exit 1
fi
if ! grep -q '^MTP=""$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not preserve auto MTP for canned profile" >&2
    exit 1
fi
if ! grep -q '^TEMPERATURE="0.6"$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not write canned profile temperature" >&2
    exit 1
fi

echo "MTP config precedence smoke passed"
