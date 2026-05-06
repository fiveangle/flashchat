#!/usr/bin/env python3
"""Shared helpers for reading the Flashchat model registry."""

import json
import os
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[1]


def registry_path():
    return Path(os.environ.get("FLASHCHAT_MODEL_CONFIG", repo_root() / "assets" / "model_configs.json"))


def load_registry(path=None):
    path = Path(path) if path else registry_path()
    with open(path) as f:
        return json.load(f)


def default_model_id(registry=None, fallback="qwen3.6-35B-A3B"):
    registry = registry or load_registry()
    return registry.get("default_model") or fallback


def model_config(model_id, registry=None):
    registry = registry or load_registry()
    return registry.get("models", {}).get(model_id)


def model_ids(registry=None):
    registry = registry or load_registry()
    return list(registry.get("models", {}).keys())


def model_script_path(model_id, script_name, registry=None):
    registry = registry or load_registry()
    model = model_config(model_id, registry)
    if not model:
        return None
    script = model.get("scripts", {}).get(script_name)
    if not script:
        return None
    return repo_root() / script
