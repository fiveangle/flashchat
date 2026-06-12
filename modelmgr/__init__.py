"""Model management core for flashchat.

Owns the model registry (declarative per-model manifests + per-user state),
artifact recipes/steps, integrity verification, storage offload, and the
config/manage/onboarding TUIs. The bash `flashchat` launcher dispatches here
for everything except engine/server lifecycle.

This package layer (registry, manifests, resolved view, status) is
stdlib-only so it can run before the project venv exists; only the
extraction steps require numpy/huggingface_hub.
"""

__all__ = ["manifest", "registry", "resolved", "paths"]
