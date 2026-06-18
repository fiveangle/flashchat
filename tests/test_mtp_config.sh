#!/usr/bin/env bash
set -euo pipefail

# MTP precedence through lib/config.sh: profile default -> server default ->
# config override -> env override. (Wizard-side MTP handling is covered by
# the Python tests in tests/python/.)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEST_ROOT="debug/fresh-envs/$(date +%Y%m%d-%H%M%S)-mtp-config-smoke"
HOME_DIR="$TEST_ROOT/home"
CONFIG_JSON="$TEST_ROOT/model_configs.json"
mkdir -p "$HOME_DIR/.config/flashchat"
trap 'rm -rf "$TEST_ROOT"' EXIT

python3 - "$ROOT/assets/model_configs.json" "$CONFIG_JSON" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    data = json.load(f)

model = data["models"]["Qwen-Qwen36-35B-A3B"]
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
MODEL="Qwen-Qwen36-35B-A3B"
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

# Empty MTP in config falls back to the profile's default.
cat > "$HOME_DIR/.config/flashchat/config" <<'EOF'
MODEL="Qwen-Qwen36-35B-A3B"
SAMPLING_PROFILE="instruct"
MTP=""
EOF

auto_value=$(run_get_mtp bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MTP')
if [ "$auto_value" != "3" ]; then
    echo "FAIL: auto MTP expected profile value 3, got '$auto_value'" >&2
    exit 1
fi

# Without a profile default, the server default applies.
python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
del data["models"]["Qwen-Qwen36-35B-A3B"]["sampling_profiles"]["instruct"]["mtp_default_predictions"]
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

server_default_value=$(run_get_mtp bash -c 'source lib/config.sh; flashchat_load_config; flashchat_get MTP')
if [ "$server_default_value" != "2" ]; then
    echo "FAIL: server MTP default expected 2, got '$server_default_value'" >&2
    exit 1
fi

echo "MTP config precedence smoke passed"
