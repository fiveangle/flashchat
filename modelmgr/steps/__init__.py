"""Step protocol and registry.

A step is one unit of model preparation (export the tokenizer, pack the
experts, ...). Steps are shared, parameterized code: the manifest declares
WHICH steps a model's artifacts need; the step reads everything
model-specific from the manifest via its context. Implementations are
imported lazily because they need the venv (numpy/huggingface_hub) while
recipe *planning* must work with stdlib only.

Bump a step's version to invalidate previously-built outputs after a step
bugfix: recipes re-plan any artifact whose recorded step_version differs.
"""

import importlib
from dataclasses import dataclass, field

# name -> (module under modelmgr.steps, current version)
STEP_TABLE = {
    "download_hf":                ("download", 1),
    "export_tokenizer":           ("tokenizer", 1),
    "generate_expert_index":      ("expert_index", 1),
    "extract_weights":            ("extract_weights", 1),
    "repack_experts":             ("repack_experts", 1),
    "compile_native:non_experts": ("compile_native", 1),
    "compile_native:experts":     ("compile_native", 1),
    "compile_native:mtp_experts": ("compile_native", 1),
    "compile_native:bf16_mtp":    ("compile_native", 1),
    "materialize_shared":         ("materialize", 1),
}


def step_version(name: str) -> int:
    return STEP_TABLE[name][1]


def load_step(name: str):
    """Import and return the runner for a step: run(ctx, planned_step)."""
    module_name, _version = STEP_TABLE[name]
    module = importlib.import_module(f".{module_name}", __package__)
    return module.get_runner(name)


@dataclass
class StepContext:
    """Everything a step needs to run, independent of how it was invoked."""

    manifest: object            # modelmgr.manifest.Manifest
    variant_name: str | None    # None for shared-only steps
    snapshot: str               # HF snapshot dir (source safetensors live here)
    shared_dir: str
    variant_dir: str | None
    progress: object = None     # callable(phase, current, total, msg) or None
    dry_run: bool = False
    force: bool = False
    options: dict = field(default_factory=dict)  # e.g. {"bf16_mtp": True}

    def report(self, phase: str, current: int = 0, total: int = 0, msg: str = "") -> None:
        if self.progress:
            self.progress(phase, current, total, msg)

    @property
    def variant(self):
        return self.manifest.variant(self.variant_name) if self.variant_name else None
