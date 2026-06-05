#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEST_ROOT="$ROOT/debug/fresh-envs/$(date +%Y%m%d-%H%M%S)-profile-edit-config"
HOME_DIR="$TEST_ROOT/home"
CONFIG_JSON="$TEST_ROOT/model_configs.json"
mkdir -p "$HOME_DIR/.config/flashchat"

python3 - "$ROOT/assets/model_configs.json" "$CONFIG_JSON" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    data = json.load(f)
with open(dst, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

write_config() {
    local profile="${1:-custom}"
    local mtp="${2:-0}"
    cat > "$HOME_DIR/.config/flashchat/config" <<EOF
MODEL="mlx-community-Qwen36-35B-A3B-4bit"
MAX_TOKENS="1"
SAMPLING_PROFILE="$profile"
REASONING="0"
TEMPERATURE="0.7"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
SERVER_PORT="19994"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="$TEST_ROOT/logs"
HUGGINGFACE_CACHE_DIR="$TEST_ROOT/hf"
OFFLOAD_DIR=""
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
MTP="$mtp"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF
}

run_config_with_input() {
    HOME="$HOME_DIR" FLASHCHAT_MODEL_CONFIG="$CONFIG_JSON" ./flashchat config
}

write_config custom 0

add_output=$(
    {
        printf 'y'
        printf '\n'
        printf '\n'
        printf '0\n'
        printf 'quick-local\n'
        printf '\n'
        printf 'Quick Local\n'
        printf 'Fast local coding\n'
        printf '1\n'
        printf '0.55\n'
        printf '0.9\n'
        printf '30\n'
        printf '0.05\n'
        printf '0.2\n'
        printf '1.05\n'
        printf '2\n'
        for _ in $(seq 1 12); do printf '\n'; done
    } | run_config_with_input 2>&1
)

if ! echo "$add_output" | grep -q "\\[0\\] Add a profile"; then
    echo "FAIL: profile list did not show add option" >&2
    exit 1
fi
if ! echo "$add_output" | grep -q "\\[E\\] Edit a sampling profile"; then
    echo "FAIL: profile list did not show edit option" >&2
    exit 1
fi
if ! echo "$add_output" | grep -q "Sampling profile added."; then
    echo "FAIL: add profile flow did not report success" >&2
    exit 1
fi
if echo "$add_output" | awk 'seen && /Select sampling profile number or ID/ { found=1 } index($0, "Sampling profile added.") { seen=1 } END { exit found ? 0 : 1 }'; then
    echo "FAIL: add profile flow returned to profile selection after success" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
profile = data["models"]["mlx-community-Qwen36-35B-A3B-4bit"]["sampling_profiles"]["quick-local"]
assert profile["label"] == "Quick Local", profile
assert profile["description"] == "Fast local coding", profile
assert profile["reasoning"] == 1, profile
assert profile["temperature"] == 0.55, profile
assert profile["top_p"] == 0.9, profile
assert profile["top_k"] == 30, profile
assert profile["min_p"] == 0.05, profile
assert profile["presence_penalty"] == 0.2, profile
assert profile["repetition_penalty"] == 1.05, profile
assert profile["mtp_default_predictions"] == 2, profile
PY

if ! grep -q '^SAMPLING_PROFILE="quick-local"$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not save added profile selection" >&2
    exit 1
fi
if ! grep -q '^MTP="2"$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not use added profile MTP default" >&2
    exit 1
fi

edit_output=$(
    {
        printf 'y'
        printf '\n'
        printf '\n'
        printf 'E\n'
        printf 'quick-local\n'
        printf '\n'
        printf 'Quick Local Edited\n'
        printf '\n'
        printf '\n'
        printf '\n'
        printf '\n'
        printf '40\n'
        printf '\n'
        printf '\n'
        printf '\n'
        printf 'auto\n'
        for _ in $(seq 1 12); do printf '\n'; done
    } | run_config_with_input 2>&1
)

if ! echo "$edit_output" | grep -q "Sampling profile updated."; then
    echo "FAIL: edit profile flow did not report success" >&2
    exit 1
fi
if echo "$edit_output" | awk 'seen && /Select sampling profile number or ID/ { found=1 } index($0, "Sampling profile updated.") { seen=1 } END { exit found ? 0 : 1 }'; then
    echo "FAIL: edit profile flow returned to profile selection after success" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
profile = data["models"]["mlx-community-Qwen36-35B-A3B-4bit"]["sampling_profiles"]["quick-local"]
assert profile["label"] == "Quick Local Edited", profile
assert profile["top_k"] == 40, profile
assert "mtp_default_predictions" not in profile, profile
assert "mtp" not in profile, profile
PY

if ! grep -q '^SAMPLING_PROFILE="quick-local"$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not keep edited profile selected" >&2
    exit 1
fi
if ! grep -q '^MTP=""$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not clear edited profile MTP to automatic" >&2
    exit 1
fi

delete_output=$(
    {
        printf 'y'
        printf '\n'
        printf '\n'
        printf 'E\n'
        printf 'quick-local\n'
        printf 'quick-local\n'
        printf 'instruct\n'
        for _ in $(seq 1 12); do printf '\n'; done
    } | run_config_with_input 2>&1
)

if ! echo "$delete_output" | grep -q "Sampling profile deleted."; then
    echo "FAIL: delete profile flow did not report success" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
profiles = data["models"]["mlx-community-Qwen36-35B-A3B-4bit"]["sampling_profiles"]
assert "quick-local" not in profiles
PY

if ! grep -q '^SAMPLING_PROFILE="instruct"$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not continue after deleting profile" >&2
    exit 1
fi

echo "profile edit config smoke passed"
