"""download_hf step: fetch the original model snapshot from HuggingFace.

Uses huggingface_hub (resumable, hash-verified, populates the standard
hub cache layout) with the `hf` CLI as fallback. Replaces the old
throwaway-venv download hack in the bash launcher.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from . import StepContext


class DownloadError(RuntimeError):
    pass


def download_snapshot(hf_repo: str, cache_dir: str, progress=None,
                      allow_patterns=None) -> str:
    """Download (or resume) a repo snapshot; returns the snapshot path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return _download_via_cli(hf_repo, cache_dir)

    if progress:
        progress("download_hf", 0, 1, f"downloading {hf_repo}")
    return snapshot_download(
        repo_id=hf_repo,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )


def download_file(hf_repo: str, filename: str, cache_dir: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        snapshot = _download_via_cli(hf_repo, cache_dir, filename=filename)
        return os.path.join(snapshot, filename)
    return hf_hub_download(repo_id=hf_repo, filename=filename, cache_dir=cache_dir)


def _download_via_cli(hf_repo: str, cache_dir: str, filename: str | None = None) -> str:
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf:
        raise DownloadError(
            "huggingface_hub is not installed and no `hf` CLI was found; "
            "run the launcher once to set up the venv")
    cmd = [hf, "download", hf_repo]
    if filename:
        cmd.append(filename)
    env = dict(os.environ, HF_HUB_CACHE=cache_dir)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloadError(f"hf download failed: {result.stderr.strip()[-500:]}")
    from .. import paths
    snapshot = paths.snapshot_dir(cache_dir, hf_repo)
    if not snapshot:
        raise DownloadError(f"download reported success but no snapshot found for {hf_repo}")
    return snapshot


def run(ctx: StepContext, planned=None) -> None:
    cache_dir = ctx.options.get("cache_dir")
    if not cache_dir:
        raise DownloadError("download_hf requires options['cache_dir']")
    if ctx.dry_run:
        ctx.report("download_hf", 0, 1, f"would download {ctx.manifest.hf_repo}")
        return
    snapshot = download_snapshot(ctx.manifest.hf_repo, cache_dir, progress=ctx.progress)
    ctx.report("download_hf", 1, 1, snapshot)


def get_runner(name: str):
    return run
