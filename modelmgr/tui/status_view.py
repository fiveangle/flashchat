"""Status rendering: one block per base model with per-variant lines."""

from .. import paths
from ..registry import Registry
from ..status import ModelStatus, all_statuses, hf_cache_dir, offload_dir, selected_model
from . import common


def variant_glyph(status: ModelStatus, vname: str) -> str:
    v = status.variants[vname]
    if v.ready:
        size = f" ({paths.human_bytes(v.local_bytes)})" if v.local_bytes else ""
        return common.green(f"ready{size}")
    if v.offloaded:
        size = f" ({paths.human_bytes(v.offload_bytes)})" if v.offload_bytes else ""
        return common.yellow(f"offloaded{size}")
    line = status.summary_line(vname)
    if line == "not extracted":
        source_bytes = status.originals_bytes or status.offload_originals_bytes
        if source_bytes:
            line = f"{line} ({paths.human_bytes(source_bytes)} source)"
    if "attention" in line:
        return common.red(line)
    return common.yellow(line)


def print_model_block(index: int, status: ModelStatus, current=None) -> None:
    m = status.manifest
    marker = "*" if current and current[0].id == m.id else " "
    enabled = "" if status.enabled else common.dim(" (disabled)")
    user = common.dim(" (user-added)") if m.user_defined else ""
    print(f"[{index}]{marker} {common.bold(m.hf_repo)}{enabled}{user}")
    where = []
    if status.originals_local:
        where.append(f"originals local ({paths.human_bytes(status.originals_bytes)})")
    elif status.originals_offloaded:
        where.append(f"original weights offloaded ({paths.human_bytes(status.offload_originals_bytes)})")
    elif status.snapshot:
        where.append("originals not local")
    else:
        where.append("not downloaded")
    if status.archive != "none":
        where.append(f"archive: {status.archive}")
    print(f"     {' | '.join(where)}")
    for vname in m.variants:
        sel = " <- selected" if (current and current[0].id == m.id
                                 and current[1] == vname) else ""
        print(f"       {vname}: {variant_glyph(status, vname)}{sel}")


def print_status_list(registry: Registry, enabled_only: bool = False) -> list:
    statuses = all_statuses(registry, enabled_only=enabled_only)
    current = selected_model(registry)
    print(f"HF cache:    {hf_cache_dir()}")
    od = offload_dir()
    print(f"Offload dir: {od or common.dim('(not configured)')}")
    print()
    for i, status in enumerate(statuses, 1):
        print_model_block(i, status, current=current)
        print()
    return statuses


def print_header(registry: Registry) -> None:
    """One-line status for the launcher's interactive menu."""
    current = selected_model(registry)
    if not current:
        print("model: none configured")
        return
    manifest, vname = current
    from ..status import model_status
    status = model_status(registry, manifest)
    print(f"model: {manifest.name} [{vname}] — {status.summary_line(vname)}")
