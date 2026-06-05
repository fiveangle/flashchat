#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEST_ROOT="$ROOT/debug/fresh-envs/$(date +%Y%m%d-%H%M%S)-model-edit-config"
HOME_DIR="$TEST_ROOT/home"
CONFIG_JSON="$TEST_ROOT/model_configs.json"
mkdir -p "$HOME_DIR/.config/flashchat"

python3 - "$ROOT/assets/model_configs.json" "$CONFIG_JSON" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    data = json.load(f)
base = data["models"]["mlx-community-Qwen36-35B-A3B-4bit"].copy()
base["name"] = "Temporary Edit Model"
base["hf_repo"] = "example/Temporary-Edit-Model"
base["mtp_default_predictions"] = 1
data["models"]["temporary-edit-model"] = base
delete = base.copy()
delete["name"] = "Temporary Delete Model"
delete["hf_repo"] = "example/Temporary-Delete-Model"
data["models"]["temporary-delete-model"] = delete
with open(dst, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

cat > "$HOME_DIR/.config/flashchat/config" <<EOF
MODEL="mlx-community-Qwen36-35B-A3B-4bit"
MAX_TOKENS="1"
SAMPLING_PROFILE="custom"
REASONING="0"
TEMPERATURE="0.7"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE_PENALTY="1.5"
REPETITION_PENALTY="1.0"
SERVER_PORT="19997"
SERVER_HOST="127.0.0.1"
SERVER_LOG_PATH="$TEST_ROOT/logs"
HUGGINGFACE_CACHE_DIR="$TEST_ROOT/hf"
OFFLOAD_DIR=""
SERVER_DEBUG="0"
SERVER_HTTP_LOG="0"
SYSTEM_PROMPT_CACHE="1"
SYSTEM_PROMPT_CACHE_MAX_ENTRIES="2"
MTP="0"
SHOW_THINKING="0"
COLOR_OUTPUT="0"
EOF

run_config_with_input() {
    HOME="$HOME_DIR" FLASHCHAT_MODEL_CONFIG="$CONFIG_JSON" ./flashchat config
}

update_output=$(
    {
        printf 'y'
        printf 'e\n'
        printf 'temporary-edit-model\n'
        printf 'n\n'
        printf 'Edited Model Name\n'
        for _ in $(seq 1 10); do printf '\n'; done
        printf '6\n'
        printf '2\n'
        for _ in $(seq 1 80); do printf '\n'; done
    } | run_config_with_input 2>&1
)

if ! echo "$update_output" | grep -q "Delete registry entry ? \\[y/N\\]"; then
    echo "FAIL: edit flow did not ask delete question" >&2
    exit 1
fi

if ! echo "$update_output" | grep -q "Active experts (K / num_experts_per_tok) \\[8\\]"; then
    echo "FAIL: edit flow did not show active expert registry default" >&2
    exit 1
fi

if echo "$update_output" | awk 'seen && /Select model number or ID/ { found=1 } index($0, "Registry entry updated.") { seen=1 } END { exit found ? 0 : 1 }'; then
    echo "FAIL: edit flow returned to model selection after updating registry entry" >&2
    exit 1
fi

if ! echo "$update_output" | grep -q "Selected model: temporary-edit-model"; then
    echo "FAIL: edit flow did not continue with edited model" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
model = data["models"]["temporary-edit-model"]
assert model["name"] == "Edited Model Name", model["name"]
assert model["num_experts_per_tok"] == 6, model["num_experts_per_tok"]
assert model["mtp_default_predictions"] == 2, model["mtp_default_predictions"]
PY

if ! grep -q '^MODEL="temporary-edit-model"$' "$HOME_DIR/.config/flashchat/config"; then
    echo "FAIL: config wizard did not save edited model selection" >&2
    exit 1
fi

delete_output=$(
    {
        printf 'y'
        printf 'E\n'
        printf 'temporary-delete-model\n'
        printf 'y\n'
        for _ in $(seq 1 80); do printf '\n'; done
    } | run_config_with_input 2>&1
)

if ! echo "$delete_output" | grep -q "Registry entry deleted."; then
    echo "FAIL: delete flow did not report deletion" >&2
    exit 1
fi

python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
assert "temporary-delete-model" not in data["models"]
PY

echo "model edit config smoke passed"
