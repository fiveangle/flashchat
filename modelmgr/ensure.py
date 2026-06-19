"""Launch-time orchestration: everything that runs before chat/serve.

Order matters:
1. migration (old layout must be converted before status checks)
2. onboarding (no config file -> first-run setup)
3. selected-model readiness (offer to build what is missing)
4. consolidated offload offer (quiet when the volume is unreachable)
5. resolved registry view refresh (the engine reads it)
"""
from __future__ import annotations

import os

from . import configfile, migrate, paths, resolved
from .registry import Registry, resolved_id
from .status import hf_cache_dir, offload_dir, selected_model


def _selected_snapshot(manifest) -> str | None:
    """Snapshot for the selected model, honoring the MODEL_PATH escape
    hatch (env beats config beats cache detection) — a launch-time redirect
    to an arbitrary snapshot, e.g. running directly from offload storage."""
    for override in (os.environ.get("FLASHCHAT_MODEL_PATH"),
                     configfile.load().get("MODEL_PATH")):
        if override and os.path.isdir(override):
            return override
    return paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)


def run(interactive: bool = True) -> bool:
    """Returns True when the configured model is ready to serve."""
    registry = Registry.load()

    if configfile.exists() and migrate.needed(registry):
        if not interactive:
            # Migration moves user data; never do it behind a quiet check.
            plan = migrate.build_plan(registry, hf_cache_dir(), offload_dir() or None)
            if not plan.empty:
                return False
            migrate.migrate_user_config(registry)
            migrate.bootstrap_state(registry, hf_cache_dir(), offload_dir() or None)
            registry = Registry.load()
        else:
            _run_migration(registry, interactive)
            registry = Registry.load()

    if not configfile.exists():
        if not interactive:
            print("no configuration found — run './flashchat config' first")
            return False
        from .tui import onboarding
        return onboarding.run(registry)

    selection = selected_model(registry)
    if selection is None:
        print("no models in the registry")
        return False
    manifest, variant_name = selection
    registry.state.enabled.setdefault(manifest.id, True)

    from .artifacts import variant_ready
    snapshot = _selected_snapshot(manifest)
    ready = bool(snapshot and variant_ready(manifest, variant_name, snapshot))
    if not ready and interactive:
        from .tui import build
        ready = build.ensure_variant_built(registry, manifest, variant_name)

    if interactive:
        from .tui.build import launch_time_offload_offer
        launch_time_offload_offer(registry)

    # Keep the engine-facing keys coherent with the selection.
    configfile.update({
        "MODEL": resolved_id(manifest, variant_name),
        "MODEL_BASE": manifest.id,
        "MODEL_VARIANT": variant_name,
        "MODEL_CONFIG": paths.resolved_registry_path(),
    })
    registry.state.save()
    resolved.write(registry)
    return ready


def _run_migration(registry: Registry, interactive: bool) -> None:
    cache = hf_cache_dir()
    od = offload_dir() or None
    plan = migrate.build_plan(registry, cache, od)
    if plan.empty:
        # Nothing on disk to convert; just stamp the layout version.
        migrate.migrate_user_config(registry)
        migrate.bootstrap_state(registry, cache, od)
        return
    print("\nFlashchat's artifact layout has changed: variant-independent files")
    print("(vocab, BF16 MTP weights) move to a shared/ dir and integrity")
    print("manifests are added. This is a one-time, in-place reorganization —")
    print("no re-extraction, originals untouched.")
    for snap_plan in plan.snapshots:
        if snap_plan.dedups or snap_plan.adopt_files:
            reclaimed = paths.human_bytes(snap_plan.bytes_reclaimed)
            print(f"  {snap_plan.manifest.name} ({snap_plan.location}): "
                  f"reclaims {reclaimed}, adopts {snap_plan.adopt_files} files")
        for conflict in snap_plan.conflicts:
            print(f"    note: {conflict}")
    print(f"Total space reclaimed: {paths.human_bytes(plan.bytes_reclaimed)}")
    if interactive:
        try:
            reply = input("Migrate now? [Y/n]: ").strip().lower()
        except EOFError:
            reply = "n"  # exhausted input is never consent
        if reply in ("n", "no"):
            print("Skipped — flashchat will ask again next launch "
                  "(or run './flashchat migrate').")
            return
    migrate.run(registry, cache, od)
    print("Migration complete.")


def shell_exports() -> str:
    """eval-able exports for the bash launcher's serve path."""
    registry = Registry.load()
    selection = selected_model(registry)
    lines = []
    if selection is None:
        lines.append("FC_READY=0")
    else:
        manifest, variant_name = selection
        snapshot = _selected_snapshot(manifest) or ""
        from .artifacts import variant_ready
        ready = bool(snapshot and variant_ready(manifest, variant_name, snapshot))
        resolved.write(registry)
        lines += [
            f"FC_MODEL_ID={_shquote(resolved_id(manifest, variant_name))}",
            f"FC_MODEL_BASE={_shquote(manifest.id)}",
            f"FC_MODEL_VARIANT={_shquote(variant_name)}",
            f"FC_MODEL_PATH={_shquote(snapshot)}",
            f"FC_MODEL_CONFIG={_shquote(paths.resolved_registry_path())}",
            f"FC_READY={'1' if ready else '0'}",
        ]
    return "\n".join(lines)


def _shquote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
