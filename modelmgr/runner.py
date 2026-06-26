"""Recipe execution: run a planned step list with progress and estimates."""
from __future__ import annotations

import os
from dataclasses import dataclass

from . import configfile, offload, paths, recipes
from .artifacts import expert_pack_size
from .manifest import Manifest
from .steps import StepContext, load_step


@dataclass
class Estimate:
    bytes_to_write: int
    description: str


def estimate_step(manifest: Manifest, variant_name: str, step_name: str) -> Estimate:
    """Rough disk-cost preview for the benevolent confirmation prompts."""
    arch = manifest.architecture
    variant = manifest.variant(variant_name)
    if step_name in ("compile_native:experts", "repack_experts"):
        per_layer = expert_pack_size(arch["hidden_size"], arch["moe_intermediate_size"],
                                     variant.bits, variant.group_size) * arch["num_experts"]
        total = per_layer * arch["num_hidden_layers"]
        return Estimate(total, f"expert pack ({paths.human_bytes(total)})")
    if step_name == "compile_native:mtp_experts":
        per_layer = expert_pack_size(arch["hidden_size"], arch["moe_intermediate_size"],
                                     variant.bits, variant.group_size) * arch["num_experts"]
        total = per_layer * int((manifest.mtp or {}).get("num_hidden_layers", 1))
        return Estimate(total, f"MTP expert pack ({paths.human_bytes(total)})")
    if step_name in ("compile_native:non_experts", "extract_weights"):
        return Estimate(0, "non-expert weights")
    if step_name == "compile_native:bf16_mtp":
        return Estimate(0, "BF16 MTP weights (~tens of MB)")
    if step_name == "export_tokenizer":
        return Estimate(0, "tokenizer vocab (~8 MB)")
    return Estimate(0, step_name)


def plan_total_bytes(manifest: Manifest, variant_name: str, plan: recipes.Plan) -> int:
    return sum(estimate_step(manifest, variant_name, s.step).bytes_to_write
               for s in plan.steps)


def execute_plan(manifest: Manifest, variant_name: str, snapshot: str,
                 plan: recipes.Plan, progress=None, dry_run: bool = False,
                 options: dict | None = None,
                 output_snapshot: str | None = None) -> set[str]:
    """Run every planned step in order. Raises on the first failure —
    completed artifacts stay valid (each step commits its own manifest)."""
    if plan.needs_download:
        raise RuntimeError(
            "original model files are not local — download/restore them first")
    output_snapshot = output_snapshot or snapshot
    changed_scopes = set()
    for planned in plan.steps:
        ctx = StepContext(
            manifest=manifest,
            variant_name=variant_name if planned.scope != "shared" else
            (variant_name or next(iter(manifest.variants))),
            snapshot=snapshot,
            shared_dir=paths.shared_dir(output_snapshot),
            variant_dir=paths.variant_dir(output_snapshot, variant_name),
            progress=progress,
            dry_run=dry_run,
            options=options or {},
        )
        runner = load_step(planned.step)
        runner(ctx, planned)
        changed_scopes.add("shared" if planned.scope == "shared" else variant_name)
    if changed_scopes and not dry_run:
        offload.mark_artifact_scopes_dirty(
            manifest, configfile.get("OFFLOAD_DIR", ""), sorted(changed_scopes))
    return changed_scopes


def free_space_ok(snapshot: str, needed_bytes: int) -> tuple[bool, int]:
    free = paths.free_space_bytes(os.path.dirname(snapshot))
    return free >= needed_bytes, free
