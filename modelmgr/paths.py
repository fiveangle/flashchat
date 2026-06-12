"""Filesystem locations: repo root, user config dir, HF cache, snapshots."""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_MANIFEST_DIR = os.path.join(REPO_ROOT, "assets", "models")

DEFAULT_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def config_dir() -> str:
    return os.environ.get("FLASHCHAT_CONFIG_DIR") or os.path.expanduser("~/.config/flashchat")


def config_file_path() -> str:
    # `flashchat --config FILE` exports this override; the Python core must
    # read the same file the bash launcher and the engine read.
    override = os.environ.get("FLASHCHAT_CONFIG_FILE_OVERRIDE")
    if override:
        return override
    return os.path.join(config_dir(), "config")


def state_file_path() -> str:
    return os.path.join(config_dir(), "models.state.json")


def user_manifest_dir() -> str:
    return os.path.join(config_dir(), "models.d")


def resolved_registry_path() -> str:
    return os.path.join(config_dir(), "resolved_models.json")


def hf_repo_dir_name(hf_repo: str) -> str:
    """'Qwen/Qwen3.6-35B-A3B' -> 'models--Qwen--Qwen3.6-35B-A3B'."""
    return "models--" + hf_repo.replace("/", "--")


def repo_root_dir(cache_dir: str, hf_repo: str) -> str:
    return os.path.join(cache_dir, hf_repo_dir_name(hf_repo))


def snapshot_dir(cache_dir: str, hf_repo: str) -> str | None:
    """Resolve the snapshot directory for a repo in an HF-style cache.

    Prefers the commit recorded in refs/main (what `hf download` maintains);
    falls back to the most recently modified snapshot when refs are absent
    (hand-built offload trees).
    """
    root = repo_root_dir(cache_dir, hf_repo)
    snapshots = os.path.join(root, "snapshots")
    if not os.path.isdir(snapshots):
        return None
    ref = os.path.join(root, "refs", "main")
    if os.path.isfile(ref):
        with open(ref) as f:
            commit = f.read().strip()
        candidate = os.path.join(snapshots, commit)
        if os.path.isdir(candidate):
            return candidate
    candidates = [
        os.path.join(snapshots, d)
        for d in os.listdir(snapshots)
        if os.path.isdir(os.path.join(snapshots, d))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def flashchat_dir(snapshot: str) -> str:
    return os.path.join(snapshot, "flashchat")


def shared_dir(snapshot: str) -> str:
    return os.path.join(flashchat_dir(snapshot), "shared")


def variant_dir(snapshot: str, variant_name: str) -> str:
    return os.path.join(flashchat_dir(snapshot), variant_name)


def dir_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


def free_space_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json_atomic(path: str, data, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")
    os.replace(tmp, path)
