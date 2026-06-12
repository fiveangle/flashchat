"""modelmgr CLI — the management brain behind the bash launcher.

Subcommands:
  status / list      model + artifact status views
  onboard / ensure   first-run setup and launch-time orchestration
  config / manage    interactive wizards (offload/restore/regenerate live
                     in the manage TUI)
  migrate            one-time layout migration
  verify             artifact integrity checks (scriptable; --deep hashes)
  add-model          create a user manifest from a HuggingFace repo
  resolve            write the engine-facing registry view
  doctor             environment sanity checks
"""

import argparse
import os
import sys

from . import configfile, migrate, paths, resolved
from .registry import Registry, resolved_id
from .status import hf_cache_dir, offload_dir


def cmd_list(args) -> int:
    registry = Registry.load()
    for m in registry.manifests.values():
        flags = []
        if registry.is_enabled(m.id):
            flags.append("enabled")
        if m.user_defined:
            flags.append("user")
        if registry.default_model() is m:
            flags.append("default")
        variants = ", ".join(f"{v} -> {resolved_id(m, v)}" for v in m.variants)
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{m.id}  ({m.hf_repo}){suffix}")
        print(f"    variants: {variants}")
    return 0


def cmd_status(args) -> int:
    registry = Registry.load()
    from .tui import status_view
    if args.header:
        status_view.print_header(registry)
    else:
        status_view.print_status_list(registry, enabled_only=args.enabled)
    return 0


def cmd_resolve(args) -> int:
    if args.shell:
        from . import ensure
        print(ensure.shell_exports())
        return 0
    registry = Registry.load()
    out = resolved.write(registry, path=args.output, include_all=args.all)
    if not args.quiet:
        print(out)
    return 0


def cmd_ensure(args) -> int:
    from . import ensure
    return 0 if ensure.run(interactive=not args.non_interactive) else 1


def cmd_onboard(args) -> int:
    from .tui import onboarding
    return 0 if onboarding.run(Registry.load()) else 1


def cmd_config(args) -> int:
    registry = Registry.load()
    if not configfile.exists():
        from .tui import onboarding
        return 0 if onboarding.run(registry) else 1
    from .tui import config_wizard
    config_wizard.run(registry)
    return 0


def cmd_manage(args) -> int:
    from .tui import manage
    manage.run(Registry.load())
    return 0


def cmd_migrate(args) -> int:
    registry = Registry.load()
    cache = hf_cache_dir()
    od = offload_dir() or None
    plan = migrate.build_plan(registry, cache, od)
    if plan.empty and not migrate.needed(registry):
        print("nothing to migrate")
        return 0
    for snap_plan in plan.snapshots:
        if not snap_plan.dedups and not snap_plan.adopt_files:
            continue
        print(f"{snap_plan.manifest.name} ({snap_plan.location}, {snap_plan.snapshot}):")
        for action in snap_plan.dedups:
            print(f"  dedup {action.relpath}: keep {action.keep_from}, "
                  f"link {', '.join(action.duplicates) or '(none)'} "
                  f"-> reclaims {paths.human_bytes(action.bytes_reclaimed)}")
        if snap_plan.adopt_files:
            print(f"  adopt {snap_plan.adopt_files} files "
                  f"({paths.human_bytes(snap_plan.adopt_bytes)}) into integrity manifests")
        for conflict in snap_plan.conflicts:
            print(f"  conflict: {conflict}")
    print(f"total reclaimed: {paths.human_bytes(plan.bytes_reclaimed)}")
    if args.dry_run:
        return 0
    if not args.yes:
        try:
            if input("Proceed? [Y/n]: ").strip().lower() in ("n", "no"):
                return 1
        except EOFError:
            return 1  # exhausted input is never consent
    migrate.run(registry, cache, od, hash_now=args.hash)
    resolved.write(Registry.load())  # state changed (enabled models, default)
    print("migration complete")
    return 0


def cmd_verify(args) -> int:
    registry = Registry.load()
    from .artifacts import shared_status, variant_status
    from .status import selected_model

    if args.model:
        manifest = registry.get(args.model)
    else:
        selection = selected_model(registry)
        if selection is None:
            print("no model selected")
            return 1
        manifest = selection[0]
    snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
    if not snapshot:
        print(f"{manifest.id}: not local")
        return 1
    failures = 0
    scopes = [("shared", shared_status(manifest, snapshot, deep=args.deep))]
    scopes += [(v, variant_status(manifest, v, snapshot, deep=args.deep))
               for v in (args.variant.split(",") if args.variant else manifest.variants)]
    for scope, states in scopes:
        for s in states:
            mark = "OK" if s.state == "ok" else s.state
            if s.state == "missing" and not s.required:
                mark = "absent (optional)"
            print(f"  {scope}/{s.relpath}: {mark}"
                  + (f" ({s.detail})" if s.detail else ""))
            if not s.satisfied:
                failures += 1
    print("PASS" if failures == 0 else f"FAIL ({failures} problems)")
    return 0 if failures == 0 else 1


def cmd_add_model(args) -> int:
    import json

    from .addmodel import AddModelError, derive_manifest, save_user_manifest
    from .steps.download import DownloadError, download_file

    registry = Registry.load()
    try:
        config_path = download_file(args.repo, "config.json", hf_cache_dir())
        with open(config_path) as f:
            hf_config = json.load(f)
        manifest_dict = derive_manifest(args.repo, hf_config, registry,
                                        variants=args.variants.split(",") if args.variants else None)
        path = save_user_manifest(manifest_dict)
    except (AddModelError, DownloadError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"added {manifest_dict['id']}: {path}")
    state = registry.state
    state.enabled[manifest_dict["id"]] = True
    state.save()
    resolved.write(Registry.load())
    return 0


def cmd_doctor(args) -> int:
    registry = Registry.load()
    problems = 0
    print(f"config dir:   {paths.config_dir()}"
          + ("" if os.path.isdir(paths.config_dir()) else "  (missing)"))
    print(f"config file:  {'present' if configfile.exists() else 'absent (first run)'}")
    print(f"registry:     {len(registry.manifests)} models "
          f"({sum(1 for m in registry.manifests.values() if m.user_defined)} user-added)")
    print(f"layout:       v{registry.state.layout_version}"
          + ("" if not migrate.needed(registry) else "  (migration pending)"))
    cache = hf_cache_dir()
    print(f"HF cache:     {cache}" + ("" if os.path.isdir(os.path.expanduser(cache))
                                      else "  (missing)"))
    od = offload_dir()
    if od:
        from . import offload as offload_mod
        report = offload_mod.preflight(od)
        verdict = "ok" if report.ok else "; ".join(report.errors)
        print(f"offload dir:  {od}  ({verdict})")
        for w in report.warnings:
            print(f"              note: {w}")
        if not report.ok:
            problems += 1
    else:
        print("offload dir:  not configured")
    try:
        resolved.render(registry, include_all=True)
        print("resolved view: renders cleanly")
    except Exception as e:  # noqa: BLE001 — doctor reports, never crashes
        print(f"resolved view: ERROR {e}")
        problems += 1
    return 0 if problems == 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="modelmgr")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list known models and variants")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("status", help="model + artifact status")
    p.add_argument("--enabled", action="store_true", help="enabled models only")
    p.add_argument("--header", action="store_true", help="one-line summary")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("resolve", help="write the engine-facing registry view")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--all", action="store_true",
                   help="include every model regardless of enabled state")
    p.add_argument("--shell", action="store_true",
                   help="print eval-able FC_* exports for the launcher")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("ensure", help="launch-time setup/repair orchestration")
    p.add_argument("--non-interactive", action="store_true")
    p.set_defaults(func=cmd_ensure)

    p = sub.add_parser("onboard", help="first-run setup")
    p.set_defaults(func=cmd_onboard)

    p = sub.add_parser("config", help="interactive configuration wizard")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("manage", help="manual artifact/storage operations")
    p.set_defaults(func=cmd_manage)

    p = sub.add_parser("migrate", help="one-time layout migration")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--hash", action="store_true",
                   help="compute hash baselines during migration")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("verify", help="verify artifacts for a model")
    p.add_argument("--model", default=None, help="model id (default: selected)")
    p.add_argument("--variant", default=None, help="comma-separated variants")
    p.add_argument("--deep", action="store_true", help="full hash verification")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("add-model", help="add a model from HuggingFace")
    p.add_argument("repo", help="HF repo id, e.g. Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--variants", default=None, help="comma-separated, e.g. q4,q8")
    p.set_defaults(func=cmd_add_model)

    p = sub.add_parser("doctor", help="environment sanity checks")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
