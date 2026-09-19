"""Machine interface to modelmgr for native front ends (the menubar app).

Query commands print one JSON object. `run` commands stream NDJSON events,
one per line, and always finish with a `done` event:

  {"event": "progress", "phase": ..., "current": N, "total": N, "message": ...}
  {"event": "log", "message": ...}
  {"event": "artifact", ...}                   (verify)
  {"event": "done", "ok": bool, "message": ..., ...}

Operations never prompt: every choice the TUI asks for interactively is an
explicit argument here, and the caller confirms with the user beforehand.
SIGINT cancels a running operation the same way Ctrl-C does in the TUI.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

from . import (configfile, migrate, offload, paths, recipes, resolved, runner,
               settings)
from .artifacts import (shared_status, source_mtp_tensors_present,
                        template_supports_thinking, variant_ready, variant_status)
from .registry import Registry, resolved_id
from .server import guard_model_not_serving, is_server_running
from .status import all_statuses, hf_cache_dir, model_status, offload_dir, selected_model

API_SCHEMA = 1
RUNTIME_OVERHEAD_BYTES = 1 << 30


class ApiError(RuntimeError):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Event stream
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Emitter:
    def __init__(self, stream):
        self.stream = stream
        self._last_progress = 0.0

    def emit(self, event: str, **fields) -> None:
        self.stream.write(json.dumps({"event": event, **fields}) + "\n")
        self.stream.flush()

    def progress(self, phase: str, current: int = 0, total: int = 0, msg: str = "") -> None:
        now = time.monotonic()
        final = bool(total) and current >= total
        if not final and now - self._last_progress < 0.1:
            return
        self._last_progress = now
        self.emit("progress", phase=phase, current=current, total=total, message=msg)

    def log(self, message: str) -> None:
        message = _ANSI_RE.sub("", message).rstrip()
        if message.strip():
            self.emit("log", message=message.strip("\r"))


class _LogStream:
    """Stands in for sys.stdout during operations: prints become log events."""

    def __init__(self, emitter: Emitter):
        self.emitter = emitter
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while True:
            cut = min((i for i in (self._buf.find("\n"), self._buf.find("\r")) if i >= 0),
                      default=-1)
            if cut < 0:
                break
            self.emitter.log(self._buf[:cut])
            self._buf = self._buf[cut + 1:]
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        if self._buf:
            self.emitter.log(self._buf)
            self._buf = ""


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _variant_json(manifest, status, vname: str) -> dict:
    vs = status.variants[vname]
    variant = manifest.variant(vname)
    return {
        "name": vname,
        "bits": variant.bits,
        "resolved_id": vs.resolved_id,
        "ready": vs.ready,
        "offloaded": vs.offloaded,
        "local_bytes": vs.local_bytes,
        "offload_bytes": vs.offload_bytes,
        "summary": status.summary_line(vname),
        "missing": [{"relpath": s.relpath, "state": s.state, "detail": s.detail}
                    for s in vs.missing],
    }


def _profiles_json(manifest) -> list:
    out = []
    for name, p in manifest.sampling_profiles.items():
        out.append({
            "name": name,
            "label": p.get("label", name),
            "description": p.get("description", ""),
            "values": {cfg_key: str(p[key]) for key, cfg_key in settings.SAMPLING_KEYS
                       if key in p},
        })
    return out


def _model_json(registry: Registry, status, selection) -> dict:
    m = status.manifest
    selected = bool(selection and selection[0].id == m.id)
    return {
        "id": m.id,
        "name": m.name,
        "hf_repo": m.hf_repo,
        "enabled": status.enabled,
        "user_defined": m.user_defined,
        "suggested_default": m.suggested_default,
        "selected": selected,
        "selected_variant": selection[1] if selected else None,
        "default_variant": m.default_variant,
        "snapshot": status.snapshot,
        "originals_local": status.originals_local,
        "originals_bytes": status.originals_bytes,
        "originals_offloaded": status.originals_offloaded,
        "archive": status.archive,
        "offload_snapshot": status.offload_snapshot,
        "pending_offload_sync": offload.pending_scopes(m) if status.archive == "full" else [],
        "max_context": m.max_context,
        "num_experts_per_tok": m.num_experts_per_tok,
        "thinking_capable": m.thinking_capable,
        "mtp_capable": m.mtp_artifacts_required,
        "default_sampling_profile": m.default_sampling_profile,
        "sampling_profiles": _profiles_json(m),
        "variants": [_variant_json(m, status, v) for v in m.variants],
    }


def _effective_window(manifest) -> int:
    raw = configfile.get("CONTEXT_WINDOW")
    window = int(raw) if raw.isdigit() else settings.default_context_window(manifest)
    if manifest.max_context > 0:
        window = min(window, manifest.max_context)
    return window


def _kv_mode() -> str:
    mode = (configfile.get("KV_QUANT") or "off").lower()
    return mode if mode in ("off", "q8", "q4") else "off"


def memory_estimate(manifest, variant_name: str, snapshot: str | None) -> dict:
    """RAM the server needs resident: non-expert weights, the wired context
    cache, the expert pin cache cap, and fixed runtime overhead. Experts
    themselves stream from SSD and are not counted."""
    weights = 0
    if snapshot:
        path = os.path.join(paths.variant_dir(snapshot, variant_name), "model_weights.bin")
        if os.path.isfile(path):
            weights = os.path.getsize(path)
    window = _effective_window(manifest)
    kv = settings.kv_cache_bytes(manifest, window, _kv_mode())
    try:
        pin_gb = float(configfile.get("EXPERT_PIN_MAX_GB") or 0)
    except ValueError:
        pin_gb = 0.0
    pin = int(pin_gb * (1 << 30))
    return {
        "weights_bytes": weights,
        "kv_cache_bytes": kv,
        "context_window": window,
        "kv_quant": _kv_mode(),
        "expert_cache_max_bytes": pin,
        "overhead_bytes": RUNTIME_OVERHEAD_BYTES,
        "total_bytes": weights + kv + RUNTIME_OVERHEAD_BYTES,
    }


def state(check_offload: bool = True) -> dict:
    registry = Registry.load()
    selection = selected_model(registry)
    statuses = all_statuses(registry, check_offload=check_offload)
    config = {key: configfile.get(key) for key in
              list(settings.BY_KEY) + list(settings.MODEL_KEYS)}
    selected = None
    if selection:
        manifest, vname = selection
        status = next(s for s in statuses if s.manifest.id == manifest.id)
        snapshot = status.snapshot
        selected = {
            "model": manifest.id,
            "variant": vname,
            "resolved_id": resolved_id(manifest, vname),
            "ready": status.variants[vname].ready,
            "memory": memory_estimate(manifest, vname, snapshot),
            "max_active_experts": _runtime_max_active_experts(),
            "supports_thinking": template_supports_thinking(snapshot) if snapshot else None,
        }
    return {
        "schema": API_SCHEMA,
        "repo_root": paths.REPO_ROOT,
        "config_dir": paths.config_dir(),
        "config_file": paths.config_file_path(),
        "config_exists": configfile.exists(),
        "migration_needed": configfile.exists() and migrate.needed(registry),
        "hf_cache_dir": hf_cache_dir(),
        "offload_dir": offload_dir(),
        "server_running": is_server_running(),
        "config": config,
        "selected": selected,
        "models": [_model_json(registry, s, selection) for s in statuses],
        "settings": [s.to_json() for s in settings.ALL],
    }


def model_detail(model_id: str) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    status = model_status(registry, manifest)
    rows = []
    if status.snapshot:
        for s in shared_status(manifest, status.snapshot, want_optional=True):
            rows.append(_artifact_row("shared", s))
        for vname in manifest.variants:
            for s in variant_status(manifest, vname, status.snapshot, want_optional=True,
                                    want_mtp=configfile.mtp_enabled()):
                rows.append(_artifact_row(vname, s))
    return {"model": _model_json(registry, status, selected_model(registry)),
            "artifacts": rows}


def _artifact_row(scope: str, s) -> dict:
    return {"scope": scope, "relpath": s.relpath, "state": s.state,
            "detail": s.detail, "required": s.required, "satisfied": s.satisfied}


def _runtime_max_active_experts() -> int:
    from .tui.config_wizard import _runtime_max_active_experts as max_k
    return max_k()


def _manifest(registry: Registry, model_id: str):
    if model_id not in registry.manifests:
        raise ApiError(f"unknown model '{model_id}'", "unknown_model")
    return registry.manifests[model_id]


# ---------------------------------------------------------------------------
# Configuration writes
# ---------------------------------------------------------------------------

def _normalize(setting: settings.Setting, raw) -> str:
    value = "" if raw is None else str(raw).strip()
    if setting.clear_word and value.lower() == setting.clear_word.lower():
        value = ""
    if setting.kind == "mtp" and value.lower() == "auto":
        value = ""
    empty_ok = bool(setting.clear_word or setting.empty_title
                    or setting.kind in ("text", "path", "mtp"))
    if value == "":
        if not empty_ok:
            raise ApiError(f"{setting.title} needs a value", "invalid_value")
        return ""
    kind = setting.kind
    if kind == "bool":
        lowered = value.lower()
        if lowered in ("1", "true", "yes", "on"):
            return "1"
        if lowered in ("0", "false", "no", "off"):
            return "0"
        raise ApiError(f"{setting.title} must be on or off", "invalid_value")
    if kind in ("int", "float", "mtp"):
        try:
            number = int(value) if kind in ("int", "mtp") else float(value)
        except ValueError:
            noun = "a whole number" if kind != "float" else "a number"
            raise ApiError(f"{setting.title} must be {noun}", "invalid_value") from None
        if setting.minimum is not None and number < setting.minimum:
            raise ApiError(f"{setting.title} must be at least {setting.minimum:g}",
                           "invalid_value")
        if setting.maximum is not None and number > setting.maximum:
            raise ApiError(f"{setting.title} must be at most {setting.maximum:g}",
                           "invalid_value")
        if kind == "mtp" and number < 0:
            raise ApiError(f"{setting.title} must be 0 or more", "invalid_value")
        return value
    if kind == "choice" and setting.choices and value not in setting.choices:
        raise ApiError(f"{setting.title} must be one of: {', '.join(setting.choices)}",
                       "invalid_value")
    return value


def _profile_changes(manifest, name: str) -> dict:
    profile = manifest.sampling_profiles[name]
    out = {"SAMPLING_PROFILE": name}
    for key, cfg_key in settings.SAMPLING_KEYS:
        if key in profile:
            out[cfg_key] = str(profile[key])
    return out


def apply_settings(values: dict) -> dict:
    registry = Registry.load()
    selection = selected_model(registry)
    manifest = selection[0] if selection else None
    warnings: list = []
    changes: dict = {}
    for key, raw in values.items():
        setting = settings.BY_KEY.get(key)
        if setting is None:
            raise ApiError(f"unknown setting '{key}'", "unknown_setting")
        changes[key] = _normalize(setting, raw)

    if manifest is not None:
        _apply_model_rules(manifest, changes, warnings)
    if changes.get("OFFLOAD_DIR"):
        report = offload.preflight(changes["OFFLOAD_DIR"])
        warnings.extend(report.errors)
        warnings.extend(report.warnings)

    previous = configfile.load()
    changed = any(previous.get(k) != v for k, v in changes.items())
    configfile.initialize_defaults()
    configfile.update(changes)
    resolved.write(registry)
    return {"ok": True, "changed": changed, "values": changes, "warnings": warnings,
            "server_running": is_server_running()}


def _apply_model_rules(manifest, changes: dict, warnings: list) -> None:
    profile = changes.get("SAMPLING_PROFILE")
    if profile:
        if profile != "custom" and profile not in manifest.sampling_profiles:
            raise ApiError(f"unknown sampling profile '{profile}' for {manifest.name}",
                           "invalid_value")
        if profile != "custom":
            changes.update(_profile_changes(manifest, profile))
            p = manifest.sampling_profiles[profile]
            snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
            if (template_supports_thinking(snapshot) is False and
                    str(p.get("reasoning", "")).strip().lower() in ("1", "true", "on", "yes")):
                warnings.append(f"{manifest.name} does not support thinking mode; this "
                                "profile enables reasoning, which can produce malformed output.")

    window = changes.get("CONTEXT_WINDOW")
    if window:
        max_ctx = manifest.max_context
        if max_ctx > 0 and int(window) > max_ctx:
            warnings.append(f"Context window {window} exceeds the model maximum "
                            f"{max_ctx}; saved {max_ctx}.")
            window = str(max_ctx)
        changes["CONTEXT_WINDOW"] = "" if int(window) == settings.default_context_window(
            manifest) else window

    k = changes.get("ACTIVE_EXPERTS")
    if k:
        max_k = _runtime_max_active_experts()
        if int(k) > max_k:
            warnings.append(f"K={k} exceeds the runtime maximum {max_k}; saved {max_k}.")
            k = str(max_k)
        changes["ACTIVE_EXPERTS"] = "" if int(k) == manifest.num_experts_per_tok else k

    if "MTP" in changes and settings.mtp_value_enables(changes["MTP"]):
        snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
        supported = source_mtp_tensors_present(snapshot)
        if supported is None:
            supported = manifest.mtp_artifacts_required
        if not supported:
            warnings.append(f"{manifest.name} has no MTP head, so multi-token prediction "
                            "will have no effect.")


def select_model(model_id: str, variant_name: str | None) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    variant_name = variant_name or manifest.default_variant
    if variant_name not in manifest.variants:
        raise ApiError(f"{manifest.name} has no variant '{variant_name}'", "invalid_value")
    first_run = not configfile.exists()
    previous = selected_model(registry) if not first_run else None
    configfile.initialize_defaults()
    changes = {
        "MODEL": resolved_id(manifest, variant_name),
        "MODEL_BASE": manifest.id,
        "MODEL_VARIANT": variant_name,
    }
    current_profile = configfile.get("SAMPLING_PROFILE")
    switching_model = previous is None or previous[0].id != manifest.id
    if switching_model and current_profile != "custom" \
            and current_profile not in manifest.sampling_profiles:
        changes.update(_profile_changes(manifest, manifest.default_sampling_profile))
    elif first_run:
        changes.update(_profile_changes(manifest, manifest.default_sampling_profile))
    warnings: list = []
    window = configfile.get("CONTEXT_WINDOW")
    if window.isdigit() and manifest.max_context > 0 and int(window) > manifest.max_context:
        changes["CONTEXT_WINDOW"] = ""
        warnings.append(f"Context window reset to the default for {manifest.name} "
                        f"(max {manifest.max_context}).")
    configfile.update(changes)
    registry.state.enabled[manifest.id] = True
    if first_run:
        registry.state.default_model = manifest.id
        if registry.state.layout_version < 1:
            registry.state.layout_version = 1
    registry.state.save()
    resolved.write(registry)
    snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
    ready = bool(snapshot and variant_ready(manifest, variant_name, snapshot,
                                            want_mtp=configfile.mtp_enabled()))
    return {"ok": True, "ready": ready, "resolved_id": changes["MODEL"],
            "warnings": warnings, "server_running": is_server_running()}


def set_enabled(model_id: str, enabled: bool) -> dict:
    registry = Registry.load()
    _manifest(registry, model_id)
    registry.state.enabled[model_id] = enabled
    registry.state.save()
    resolved.write(registry)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Plans (previews the app shows before confirming an operation)
# ---------------------------------------------------------------------------

def _want_optional() -> bool:
    return configfile.get("MTP_BF16") == "1"


def _restore_json(plan) -> dict:
    return {"available": bool(plan.files or plan.links),
            "needed_bytes": plan.needed_bytes, "free_bytes": plan.free_bytes,
            "fits": plan.fits, "shortfall_bytes": plan.shortfall_bytes}


def plan_build(model_id: str, variant_name: str, repair: bool = False) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    manifest.variant(variant_name)
    cache, od = hf_cache_dir(), offload_dir()
    local = paths.snapshot_dir(cache, manifest.hf_repo)
    optional = _want_optional() or repair
    want_mtp = configfile.mtp_enabled()
    ready = bool(local and variant_ready(manifest, variant_name, local,
                                         want_optional=optional, want_mtp=want_mtp))
    sources = []
    if local and recipes.source_blobs_present(manifest, local):
        sources.append({"id": "local", "title": "Original files on this Mac"})
    runtime_restore = None
    if od:
        archive = offload.archive_state(manifest, od)
        osnap = paths.snapshot_dir(od, manifest.hf_repo)
        if osnap and recipes.source_blobs_present(manifest, osnap):
            sources.append({"id": "offload",
                            "title": f"Original files in offload storage ({od})"})
        if archive in ("originals", "full") and not (
                local and recipes.source_blobs_present(manifest, local)):
            rp = offload.plan_restore(manifest, cache, od, "originals")
            if rp.files:
                sources.append({"id": "restore-originals",
                                "title": "Restore original files from offload storage first",
                                "restore": _restore_json(rp)})
        if not ready and osnap and variant_ready(manifest, variant_name, osnap,
                                                 want_optional=optional, want_mtp=want_mtp):
            rp = offload.plan_restore(manifest, cache, od, "runtime", variant_name)
            runtime_restore = _restore_json(rp)
    sources.append({"id": "download-local",
                    "title": f"Download from HuggingFace to {cache}"})
    if od:
        sources.append({"id": "download-offload",
                        "title": f"Download from HuggingFace to offload storage ({od})"})

    target = local or os.path.join(paths.repo_root_dir(cache, manifest.hf_repo),
                                   "snapshots", "pending")
    plan = recipes.plan(manifest, variant_name, target, want_optional=optional,
                        want_mtp=want_mtp)
    steps = []
    for step in plan.steps:
        est = runner.estimate_step(manifest, variant_name, step.step)
        steps.append({"step": step.step, "scope": step.scope, "artifacts": step.artifacts,
                      "reason": step.reason, "detail": step.detail,
                      "description": est.description, "bytes": est.bytes_to_write})
    total = runner.plan_total_bytes(manifest, variant_name, plan)
    _ok, free = runner.free_space_ok(target, total)
    repair_info = None
    if repair and local:
        from .tui.manage import repair_targets
        create, _targets, labels = repair_targets(manifest, local, [variant_name])
        repair_info = {"create": sorted(set(create)), "rebuild": labels}
    return {
        "model": manifest.id, "variant": variant_name, "ready": ready and not repair,
        "local_snapshot": local, "needs_source": plan.needs_download,
        "sources": sources, "runtime_restore": runtime_restore,
        "steps": steps, "total_bytes": total, "free_bytes": free,
        "repair": repair_info,
        "serving_conflict": guard_model_not_serving(
            [resolved_id(manifest, v) for v in manifest.variants]),
    }


def plan_restore(model_id: str) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    od = offload_dir()
    if not od or offload.archive_state(manifest, od) == "none":
        return {"model": manifest.id, "available": False, "options": []}
    options = []
    for what, title in (("originals", "Original files only"),
                        ("runtime", "Runtime files only"),
                        ("full", "Everything")):
        options.append({"id": what, "title": title,
                        **_restore_json(offload.plan_restore(manifest, hf_cache_dir(), od, what))})
    return {"model": manifest.id, "available": True, "offload_dir": od, "options": options}


def plan_offload(model_id: str) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    status = model_status(registry, manifest)
    od = offload_dir()
    out = {"model": manifest.id, "offload_dir": od,
           "originals_bytes": status.originals_bytes,
           "any_ready": status.any_ready, "held_back": {}, "errors": [], "warnings": []}
    if not status.snapshot or not status.originals_local:
        out["errors"].append("There are no local original files to offload.")
    if not od:
        out["errors"].append("No offload folder is configured.")
    if status.snapshot:
        out["held_back"] = offload.local_scope_problems(manifest, status.snapshot)
    if od:
        report = offload.preflight(od)
        out["errors"].extend(report.errors)
        out["warnings"].extend(report.warnings)
    out["available"] = not out["errors"]
    return out


def plan_delete(model_id: str) -> dict:
    from .tui.manage import local_components
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    status = model_status(registry, manifest)
    components = []
    if status.snapshot:
        for item in local_components(manifest, status.snapshot):
            cid = "originals" if item["kind"] == "originals" else f"variant:{item['variant']}"
            components.append({"id": cid, "title": item["label"], "detail": item["detail"],
                               "bytes": item["size"],
                               "serving_conflict": guard_model_not_serving(item["ids"])})
        repo_root = os.path.dirname(os.path.dirname(status.snapshot))
        components.append({
            "id": "model", "title": "Entire local model",
            "detail": ("removes ALL local files and artifacts for this model"
                       + ("" if status.archive != "none" else
                          " — there is NO archive; you would need to re-download it")),
            "bytes": paths.dir_size_bytes(repo_root),
            "serving_conflict": guard_model_not_serving(
                [resolved_id(manifest, v) for v in manifest.variants])})
    return {"model": manifest.id, "archive": status.archive, "components": components}


def preflight_offload_dir(path: str) -> dict:
    report = offload.preflight(path)
    return {"ok": report.ok, "free_bytes": report.free_bytes,
            "symlinks": report.symlinks, "errors": report.errors,
            "warnings": report.warnings}


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _guard(manifest) -> None:
    err = guard_model_not_serving([resolved_id(manifest, v) for v in manifest.variants])
    if err:
        raise ApiError(err, "serving")


def _op_build(em: Emitter, model_id: str, variant_name: str, source: str,
              repair: bool) -> dict:
    from .steps.download import download_snapshot
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    manifest.variant(variant_name)
    _guard(manifest)
    cache, od = hf_cache_dir(), offload_dir()
    optional = _want_optional() or repair
    want_mtp = configfile.mtp_enabled()
    local_snapshot = paths.snapshot_dir(cache, manifest.hf_repo)

    if source == "restore-runtime":
        if not od:
            raise ApiError("no offload folder is configured")
        offload.restore_runtime_only(manifest, cache, od, progress=em.progress,
                                     variant_name=variant_name)
        local_snapshot = paths.snapshot_dir(cache, manifest.hf_repo)
        if not local_snapshot or not variant_ready(manifest, variant_name, local_snapshot,
                                                   want_optional=optional,
                                                   want_mtp=want_mtp):
            raise ApiError("restore finished but local verification failed")
        resolved.write(registry)
        return {"message": f"{manifest.name} [{variant_name}] is ready."}

    if source == "auto":
        if local_snapshot and recipes.source_blobs_present(manifest, local_snapshot):
            source = "local"
        elif od and recipes.source_blobs_present(
                manifest, paths.snapshot_dir(od, manifest.hf_repo)):
            source = "offload"
        else:
            source = "local"

    src = None
    if source == "local":
        src = local_snapshot
    elif source == "offload":
        if not od:
            raise ApiError("no offload folder is configured")
        src = paths.snapshot_dir(od, manifest.hf_repo)
    elif source == "restore-originals":
        if not od:
            raise ApiError("no offload folder is configured")
        offload.restore_originals(manifest, cache, od, progress=em.progress)
        src = paths.snapshot_dir(cache, manifest.hf_repo)
    elif source == "download-local":
        src = download_snapshot(manifest.hf_repo, cache, progress=em.progress)
    elif source == "download-offload":
        if not od:
            raise ApiError("no offload folder is configured")
        snap = download_snapshot(manifest.hf_repo, od, progress=em.progress)
        src = paths.snapshot_dir(od, manifest.hf_repo) or snap
    else:
        raise ApiError(f"unknown build source '{source}'")

    local_snapshot = paths.snapshot_dir(cache, manifest.hf_repo)
    if not local_snapshot:
        if src and os.path.commonpath([os.path.abspath(src),
                                       os.path.abspath(cache)]) == os.path.abspath(cache):
            local_snapshot = src
        elif src:
            local_snapshot = os.path.join(paths.repo_root_dir(cache, manifest.hf_repo),
                                          "snapshots", os.path.basename(os.path.normpath(src)))
        else:
            raise ApiError("the original model files are not available — choose a download "
                           "or restore source", "needs_source")

    if repair and os.path.isdir(local_snapshot):
        from .tui.manage import delete_artifact_targets, repair_targets
        _create, targets, labels = repair_targets(manifest, local_snapshot, [variant_name])
        if labels:
            em.log("Rebuilding broken artifacts: " + ", ".join(labels))
        delete_artifact_targets(manifest, local_snapshot, targets)

    plan = recipes.plan(manifest, variant_name, local_snapshot, want_optional=optional,
                        want_mtp=want_mtp)
    if plan.needs_download:
        if not recipes.source_blobs_present(manifest, src):
            raise ApiError("the selected source does not contain the original model files",
                           "needs_source")
        plan.needs_download = False

    changed: set = set()
    if not plan.empty:
        total = runner.plan_total_bytes(manifest, variant_name, plan)
        if total:
            ok, free = runner.free_space_ok(local_snapshot, total)
            if not ok:
                raise ApiError(f"not enough free disk space: need ~{paths.human_bytes(total)}, "
                               f"{paths.human_bytes(free)} free", "no_space")
        changed = runner.execute_plan(manifest, variant_name, src, plan,
                                      progress=em.progress, options={"cache_dir": cache},
                                      output_snapshot=local_snapshot)

    if not variant_ready(manifest, variant_name, local_snapshot,
                         want_optional=optional, want_mtp=want_mtp):
        raise ApiError("build finished but local verification failed — check the model's "
                       "artifact details")
    if changed and od and offload.archive_state(manifest, od) == "full":
        try:
            offload.sync_artifact_scopes(manifest, local_snapshot, od, sorted(changed),
                                         progress=em.progress)
        except offload.OffloadError as e:
            em.log(f"offload artifact sync deferred: {e}")
    resolved.write(registry)
    offload_suggested = bool(
        od and src == local_snapshot and manifest.id not in registry.state.never_offload
        and offload.blobs_size(local_snapshot))
    return {"message": f"{manifest.name} [{variant_name}] is ready.",
            "offload_suggested": offload_suggested}


def _op_restore(em: Emitter, model_id: str, what: str) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    od = offload_dir()
    if not od:
        raise ApiError("no offload folder is configured")
    fn = {"originals": offload.restore_originals, "runtime": offload.restore_runtime_only,
          "full": offload.restore_full}.get(what)
    if fn is None:
        raise ApiError(f"unknown restore selection '{what}'")
    restored = fn(manifest, hf_cache_dir(), od, progress=em.progress)
    resolved.write(registry)
    return {"message": f"Restored {paths.human_bytes(restored)}."}


def _op_offload(em: Emitter, model_id: str) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    status = model_status(registry, manifest)
    od = offload_dir()
    if not od:
        raise ApiError("no offload folder is configured")
    if not status.snapshot or not status.originals_local:
        raise ApiError("there are no local original files to offload")
    moved = offload.offload_model(manifest, status.snapshot, od, progress=em.progress)
    resolved.write(registry)
    return {"message": f"Offloaded {paths.human_bytes(moved)}."}


def _op_sync_offload(em: Emitter, model_id: str) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    status = model_status(registry, manifest)
    od = offload_dir()
    scopes = offload.pending_scopes(manifest)
    if not od or not status.snapshot or not scopes:
        return {"message": "The offload copy is already up to date."}
    synced = offload.sync_artifact_scopes(manifest, status.snapshot, od, scopes,
                                          progress=em.progress)
    return {"message": f"Updated the offload copy ({paths.human_bytes(synced)})."}


def _op_verify(em: Emitter, model_id: str) -> dict:
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
    if not snapshot:
        raise ApiError("the model is not stored on this Mac")
    corrupt, unhashed = 0, 0
    scopes = [("shared", lambda: shared_status(manifest, snapshot, deep=True,
                                               progress=em.progress))]
    scopes += [(v, (lambda v=v: variant_status(manifest, v, snapshot, deep=True,
                                               progress=em.progress)))
               for v in manifest.variants]
    for scope, check in scopes:
        em.progress("verify", 0, 0, scope)
        for s in check():
            em.emit("artifact", **_artifact_row(scope, s))
            if s.state in ("hash-mismatch", "size-mismatch", "invalid"):
                corrupt += 1
            elif s.state == "unhashed":
                unhashed += 1
    message = ("All hashed artifacts match." if not corrupt
               else f"{corrupt} corrupt artifact(s) — use Build / Repair to rebuild them.")
    return {"message": message, "corrupt": corrupt, "unhashed": unhashed}


def _op_hash(em: Emitter, model_id: str) -> dict:
    from .tui.manage import _backfill_hashes
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
    if not snapshot:
        raise ApiError("the model is not stored on this Mac")
    _backfill_hashes(manifest, snapshot, progress=em.progress)
    return {"message": "Hash baselines recorded."}


def _op_delete(em: Emitter, model_id: str, component: str, confirm: str) -> dict:
    import shutil
    from .tui.manage import delete_component, local_components
    registry = Registry.load()
    manifest = _manifest(registry, model_id)
    if confirm != manifest.id:
        raise ApiError("confirmation does not match the model id", "confirmation")
    snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
    if not snapshot:
        raise ApiError("the model is not stored on this Mac")
    if component == "model":
        _guard(manifest)
        repo_root = os.path.dirname(os.path.dirname(snapshot))
        size = paths.dir_size_bytes(repo_root)
        shutil.rmtree(repo_root)
        resolved.write(registry)
        return {"message": f"Deleted the local model ({paths.human_bytes(size)})."}
    for item in local_components(manifest, snapshot):
        cid = "originals" if item["kind"] == "originals" else f"variant:{item['variant']}"
        if cid != component:
            continue
        err = guard_model_not_serving(item["ids"])
        if err:
            raise ApiError(err, "serving")
        removed = delete_component(snapshot, item)
        resolved.write(registry)
        return {"message": f"Deleted {item['label']} ({paths.human_bytes(removed)})."}
    raise ApiError(f"nothing to delete for '{component}'")


def _op_add_model(em: Emitter, repo: str, generation_config: str | None) -> dict:
    from .addmodel import (AddModelError, derive_manifest, load_generation_config,
                           save_user_manifest)
    from .steps.download import DownloadError, download_file
    if not repo or "/" not in repo:
        raise ApiError("enter a HuggingFace model id like Qwen/Qwen3.6-35B-A3B",
                       "invalid_value")
    registry = Registry.load()
    cache = hf_cache_dir()
    em.progress("fetch", 0, 0, "config.json")
    try:
        config_path = download_file(repo, "config.json", cache)
        try:
            download_file(repo, "tokenizer_config.json", cache)
        except DownloadError:
            pass
    except DownloadError as e:
        raise ApiError(f"download failed: {e}", "download") from None
    with open(config_path) as f:
        hf_config = json.load(f)
    thinking = template_supports_thinking(paths.snapshot_dir(cache, repo))
    try:
        gen = load_generation_config(repo, cache,
                                     os.path.expanduser(generation_config)
                                     if generation_config else None)
        manifest_dict = derive_manifest(repo, hf_config, registry, thinking_capable=thinking,
                                        generation_config=gen)
    except AddModelError as e:
        code = "needs_generation_config" if not generation_config else "add_model"
        raise ApiError(str(e), code) from None
    path = save_user_manifest(manifest_dict)
    registry.state.enabled[manifest_dict["id"]] = True
    registry.state.save()
    resolved.write(Registry.load())
    return {"message": f"Added {manifest_dict['id']}.", "model": manifest_dict["id"],
            "manifest_path": path}


def _op_migrate(em: Emitter) -> dict:
    registry = Registry.load()
    migrate.run(registry, hf_cache_dir(), offload_dir() or None)
    return {"message": "Migration complete."}


def run_operation(args, events=None) -> int:
    saved_fd = None
    if events is None:
        # Events own the real stdout; stray writes from subprocesses go to stderr.
        saved_fd = os.dup(1)
        events = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
    em = Emitter(events)
    log_stream = _LogStream(em)
    sys.stdout = log_stream
    ok, result = False, {}
    try:
        op = args.operation
        if op == "build":
            result = _op_build(em, args.model, args.variant, args.source, args.repair)
        elif op == "restore":
            result = _op_restore(em, args.model, args.what)
        elif op == "offload":
            result = _op_offload(em, args.model)
        elif op == "sync-offload":
            result = _op_sync_offload(em, args.model)
        elif op == "verify":
            result = _op_verify(em, args.model)
        elif op == "hash":
            result = _op_hash(em, args.model)
        elif op == "delete":
            result = _op_delete(em, args.model, args.component, args.confirm)
        elif op == "add-model":
            result = _op_add_model(em, args.repo, args.generation_config)
        elif op == "migrate":
            result = _op_migrate(em)
        ok = True
    except ApiError as e:
        result = {"message": str(e), "code": e.code}
    except KeyboardInterrupt:
        result = {"message": "Cancelled.", "code": "cancelled"}
    except Exception as e:  # surfaced to the app instead of a traceback on a pipe
        result = {"message": f"{type(e).__name__}: {e}", "code": "exception"}
    finally:
        log_stream.close()
        sys.stdout = sys.__stdout__
    em.emit("done", ok=ok, **result)
    if saved_fd is not None:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")


def add_parser(sub) -> None:
    p = sub.add_parser("api", help="JSON interface for native front ends")
    api = p.add_subparsers(dest="api_command", required=True)

    q = api.add_parser("state", help="configuration, models, and settings schema")
    q.add_argument("--no-offload", action="store_true",
                   help="skip offload storage checks (fast path)")

    q = api.add_parser("model", help="per-artifact status for one model")
    q.add_argument("--model", required=True)

    q = api.add_parser("select", help="select the model and variant to serve")
    q.add_argument("--model", required=True)
    q.add_argument("--variant")

    q = api.add_parser("enable", help="enable or disable a model")
    q.add_argument("--model", required=True)
    q.add_argument("--enabled", choices=("0", "1"), required=True)

    q = api.add_parser("set", help="write settings from a JSON object")
    q.add_argument("--values", required=True)

    q = api.add_parser("plan", help="preview an operation")
    q.add_argument("kind", choices=("build", "restore", "offload", "delete"))
    q.add_argument("--model", required=True)
    q.add_argument("--variant")
    q.add_argument("--repair", action="store_true")

    q = api.add_parser("preflight-offload", help="check an offload folder")
    q.add_argument("--path", required=True)

    q = api.add_parser("run", help="run an operation, streaming NDJSON events")
    q.add_argument("operation", choices=("build", "restore", "offload", "sync-offload",
                                         "verify", "hash", "delete", "add-model", "migrate"))
    q.add_argument("--model")
    q.add_argument("--variant")
    q.add_argument("--source", default="auto",
                   choices=("auto", "local", "offload", "restore-originals",
                            "restore-runtime", "download-local", "download-offload"))
    q.add_argument("--repair", action="store_true")
    q.add_argument("--what", choices=("originals", "runtime", "full"))
    q.add_argument("--component")
    q.add_argument("--confirm", default="")
    q.add_argument("--repo")
    q.add_argument("--generation-config")


def main(args) -> int:
    if args.api_command == "run":
        return run_operation(args)
    try:
        cmd = args.api_command
        if cmd == "state":
            _print(state(check_offload=not args.no_offload))
        elif cmd == "model":
            _print(model_detail(args.model))
        elif cmd == "select":
            _print(select_model(args.model, args.variant))
        elif cmd == "enable":
            _print(set_enabled(args.model, args.enabled == "1"))
        elif cmd == "set":
            try:
                values = json.loads(args.values)
            except json.JSONDecodeError as e:
                raise ApiError(f"--values is not valid JSON: {e}") from None
            if not isinstance(values, dict):
                raise ApiError("--values must be a JSON object")
            _print(apply_settings(values))
        elif cmd == "plan":
            if args.kind == "build":
                registry = Registry.load()
                manifest = _manifest(registry, args.model)
                _print(plan_build(args.model, args.variant or manifest.default_variant,
                                  args.repair))
            elif args.kind == "restore":
                _print(plan_restore(args.model))
            elif args.kind == "offload":
                _print(plan_offload(args.model))
            else:
                _print(plan_delete(args.model))
        elif cmd == "preflight-offload":
            _print(preflight_offload_dir(os.path.expanduser(args.path)))
        return 0
    except ApiError as e:
        _print({"ok": False, "error": str(e), "code": e.code})
        return 1
