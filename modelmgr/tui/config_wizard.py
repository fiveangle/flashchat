"""[C]onfig wizard: model+variant selection, sampling, server, storage.

Port of the bash prompt_config_interactive, rebuilt on the registry: the
model list shows base models with per-variant status; selecting an unbuilt
variant offers to build it; add-model writes to models.d/ instead of
mutating the shipped registry.
"""

import os
import re

from .. import configfile, paths, resolved
from ..artifacts import source_mtp_tensors_present, template_supports_thinking
from ..registry import Registry, resolved_id
from ..status import all_statuses, hf_cache_dir, selected_model
from . import build, common, status_view

_SAMPLING_KEYS = (("temperature", "TEMPERATURE"), ("top_p", "TOP_P"),
                  ("top_k", "TOP_K"), ("min_p", "MIN_P"),
                  ("presence_penalty", "PRESENCE_PENALTY"),
                  ("repetition_penalty", "REPETITION_PENALTY"),
                  ("reasoning", "REASONING"))


def _runtime_max_active_experts() -> int:
    header = os.path.join(paths.REPO_ROOT, "metal_infer", "model_config.h")
    try:
        with open(header) as f:
            for line in f:
                m = re.match(r"\s*#define\s+MAX_K\s+(\d+)\b", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return 16


def run(registry: Registry) -> None:
    # Summary-first: `flashchat config` doubles as "show my settings";
    # nothing is touched unless the user opts into the wizard.
    _print_summary(registry)
    if not common.confirm("Start configuration wizard?"):
        return

    changes: dict = {}

    selection = _select_model(registry)
    if selection is None:
        return
    manifest, variant_name = selection
    changes.update({
        "MODEL": resolved_id(manifest, variant_name),
        "MODEL_BASE": manifest.id,
        "MODEL_VARIANT": variant_name,
        "CONFIG_SCHEMA_VERSION": "3",
    })
    registry.state.enabled[manifest.id] = True
    registry.state.save()

    changes["MAX_TOKENS"] = common.prompt(
        "Max response tokens", configfile.get("MAX_TOKENS", "8192"))

    changes.update(_select_sampling_profile(manifest))
    changes.update(_server_settings())
    changes.update(_storage_settings())
    changes.update(_advanced_settings(manifest))

    configfile.update(changes)
    resolved.write(registry)
    print(common.green("\nConfiguration saved."))

    from ..server import is_server_running
    if is_server_running():
        print("The running server will restart with the new configuration "
              "on the next request.")

    from ..artifacts import variant_ready
    from .. import paths
    from ..status import hf_cache_dir
    snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
    if not (snapshot and variant_ready(manifest, variant_name, snapshot)):
        print(f"\n{manifest.name} [{variant_name}] is not built yet.")
        if common.confirm("Build now?"):
            build.ensure_variant_built(registry, manifest, variant_name)


def _print_summary(registry: Registry) -> None:
    common.heading("Flashchat Configuration")
    current = selected_model(registry)
    if current:
        manifest, vname = current
        variant = manifest.variant(vname)
        print(f"Model: {manifest.name} ({manifest.hf_repo})")
        print(f"Variant: {vname} | Quantization: {variant.bits}-bit")
    else:
        print("Model: (none configured)")
    print(f"Sampling profile: {configfile.get('SAMPLING_PROFILE', '')}")
    print(f"Max response tokens: {configfile.get('MAX_TOKENS', '8192')}")
    print(f"Temperature: {configfile.get('TEMPERATURE', '')} | "
          f"Top-p: {configfile.get('TOP_P', '')} | Top-k: {configfile.get('TOP_K', '')}")
    print(f"Server: {configfile.get('SERVER_HOST', '127.0.0.1')}:"
          f"{configfile.get('SERVER_PORT', '8000')}")
    print(f"HuggingFace cache dir: {configfile.get('HUGGINGFACE_CACHE_DIR', '~/.cache/huggingface/hub')}")
    print(f"Offload dir: {configfile.get('OFFLOAD_DIR', '') or '(not configured)'}")
    print(f"System prompt cache: {configfile.get('SYSTEM_PROMPT_CACHE', '1')} "
          f"(max entries: {configfile.get('SYSTEM_PROMPT_CACHE_MAX_ENTRIES', '2')})")
    print(f"System prompt cache dir: {configfile.get('SYSTEM_PROMPT_CACHE_DIR', '') or '(model directory)'}")
    print(f"MTP: {configfile.get('MTP', '') or '(registry default)'}"
          f" | Show thinking: {configfile.get('SHOW_THINKING', '0')}")
    active = configfile.get("ACTIVE_EXPERTS", "")
    if current and current[0].num_experts_per_tok > 0:
        print(f"Active experts (K): {active or current[0].num_experts_per_tok}")
    print()


def _select_model(registry: Registry):
    common.heading("Model")
    statuses = all_statuses(registry)
    current = selected_model(registry)
    default_idx = None
    for i, status in enumerate(statuses, 1):
        if current and status.manifest.id == current[0].id:
            default_idx = i
        status_view.print_model_block(i, status, current=current)
        print()
    print("  [a] add a model from HuggingFace   [e] enable/disable models")
    choice = common.select_number(len(statuses), "Select model", default=default_idx,
                                  extras={"a": "add", "e": "enable"})
    if choice == "a":
        added = _add_model(registry)
        return _select_model(Registry.load()) if added else _select_model(registry)
    if choice == "e":
        _toggle_enabled(registry)
        return _select_model(registry)
    if choice is None:
        return current
    manifest = statuses[int(choice) - 1].manifest

    names = list(manifest.variants)
    if len(names) == 1:
        return manifest, names[0]
    print()
    status = statuses[int(choice) - 1]
    for i, v in enumerate(names, 1):
        print(f"  {i}) {v} — {status.summary_line(v)}")
    default = names.index(current[1]) + 1 if (current and current[0].id == manifest.id
                                              and current[1] in names) else \
        names.index(manifest.default_variant) + 1
    vchoice = common.select_number(len(names), "Variant", default=default)
    if vchoice is None:
        return None
    return manifest, names[int(vchoice) - 1]


def _toggle_enabled(registry: Registry) -> None:
    manifests = list(registry.manifests.values())
    for i, m in enumerate(manifests, 1):
        mark = common.green("enabled") if registry.is_enabled(m.id) else common.dim("disabled")
        print(f"  {i}) {m.name} — {mark}")
    choice = common.select_number(len(manifests), "Toggle which (empty to finish)")
    if choice is None:
        return
    m = manifests[int(choice) - 1]
    registry.state.enabled[m.id] = not registry.is_enabled(m.id)
    registry.state.save()
    resolved.write(registry)
    _toggle_enabled(registry)


def _add_model(registry: Registry) -> bool:
    from ..addmodel import AddModelError, derive_manifest, save_user_manifest
    from ..status import hf_cache_dir
    from ..steps.download import DownloadError, download_file

    common.heading("Add a Model")
    repo = common.prompt("HuggingFace model ID (e.g. Qwen/Qwen3.6-35B-A3B)")
    if not repo or "/" not in repo:
        print("Cancelled.")
        return False
    print("Fetching config.json from HuggingFace...")
    try:
        config_path = download_file(repo, "config.json", hf_cache_dir())
    except DownloadError as e:
        print(common.red(f"download failed: {e}"))
        return False
    import json
    with open(config_path) as f:
        hf_config = json.load(f)
    try:
        manifest_dict = derive_manifest(repo, hf_config, registry)
        path = save_user_manifest(manifest_dict)
    except AddModelError as e:
        print(common.red(str(e)))
        return False
    print(common.green(f"added {manifest_dict['id']} ({path})"))
    print(common.dim("Review the file to adjust variants or architecture overrides."))
    state = registry.state
    state.enabled[manifest_dict["id"]] = True
    state.save()
    return True


def _profile_enables_thinking(profile) -> bool:
    return str(profile.get("reasoning", "")).strip().lower() in ("1", "true", "on", "yes")


def _select_sampling_profile(manifest) -> dict:
    common.heading("Sampling profile")
    names = list(manifest.sampling_profiles)
    current = configfile.get("SAMPLING_PROFILE", manifest.default_sampling_profile)
    # Authoritative source-side check: does this model's chat template emit
    # <think> blocks? None = undeterminable (not downloaded) → stay quiet.
    supports_thinking = template_supports_thinking(
        paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo))
    for i, name in enumerate(names, 1):
        p = manifest.sampling_profiles[name]
        mark = " (current)" if name == current else ""
        note = ""
        if supports_thinking is False and _profile_enables_thinking(p):
            note = common.yellow("  ⚠ your selected model does not support reasoning mode!")
        print(f"  {i}) {p.get('label', name)}{mark}{note}")
        print(common.dim(f"     {p.get('description', '')} "
                         f"temp={p.get('temperature')} top_p={p.get('top_p')} "
                         f"reasoning={p.get('reasoning')}"))
    custom_idx = len(names) + 1
    custom_mark = " (current)" if current == "custom" else ""
    print(f"  {custom_idx}) custom — set each parameter yourself{custom_mark}")
    if current == "custom":
        default = custom_idx
    else:
        default = names.index(current) + 1 if current in names else 1
    choice = common.select_number(len(names) + 1, "Profile", default=default)
    if choice is None:
        return {}
    if int(choice) <= len(names):
        name = names[int(choice) - 1]
        profile = manifest.sampling_profiles[name]
        if supports_thinking is False and _profile_enables_thinking(profile):
            print(common.yellow(
                f"  CAUTION: {manifest.name} does not support thinking mode; your selected "
                f"profile '{profile.get('label', name)}' enables reasoning, which can produce "
                "malformed output."))
        out = {"SAMPLING_PROFILE": name}
        for key, cfg_key in _SAMPLING_KEYS:
            if key in profile:
                out[cfg_key] = str(profile[key])
        return out
    out = {"SAMPLING_PROFILE": "custom"}
    for key, cfg_key in _SAMPLING_KEYS:
        out[cfg_key] = common.prompt(key, configfile.get(cfg_key, ""))
    if manifest.num_experts_per_tok > 0:
        default_k = str(manifest.num_experts_per_tok)
        max_k = _runtime_max_active_experts()
        current_k = configfile.get("ACTIVE_EXPERTS", "")
        if current_k and current_k.isdigit() and int(current_k) > max_k:
            print(common.yellow(f"  saved K={current_k} exceeds runtime max {max_k}; using {max_k}"))
            current_k = str(max_k)
        current_k = current_k or default_k
        value = common.prompt(f"Active experts (K, default {default_k}, max {max_k})", current_k)
        if value.isdigit() and int(value) > max_k:
            print(common.yellow(f"  K={value} exceeds runtime max {max_k}; saving {max_k}"))
            value = str(max_k)
        out["ACTIVE_EXPERTS"] = "" if value == default_k else value
    return out


def _server_settings() -> dict:
    common.heading("Server")
    return {
        "SERVER_PORT": common.prompt("Port", configfile.get("SERVER_PORT", "8000")),
        "SERVER_HOST": common.prompt("Host", configfile.get("SERVER_HOST", "127.0.0.1")),
        "SERVER_LOG_PATH": common.prompt(
            "Log path", configfile.get("SERVER_LOG_PATH", "")),
    }


def _storage_settings() -> dict:
    common.heading("Storage")
    out = {
        "HUGGINGFACE_CACHE_DIR": common.prompt(
            "HuggingFace cache dir",
            configfile.get("HUGGINGFACE_CACHE_DIR", "~/.cache/huggingface/hub")),
    }
    od = common.prompt("Offload dir for archived models ('-' to disable)",
                       configfile.get("OFFLOAD_DIR", ""))
    out["OFFLOAD_DIR"] = "" if od == "-" else od
    if out["OFFLOAD_DIR"]:
        from .. import offload
        report = offload.preflight(out["OFFLOAD_DIR"])
        if report.ok:
            print(common.green(f"  offload dir OK "
                               f"({(report.free_bytes / 1024**3):.0f} GiB free"
                               f"{', symlinks supported' if report.symlinks else ''})"))
        else:
            for err in report.errors:
                print(common.yellow(f"  warning: {err}"))
            print(common.yellow("  saved anyway — fix the mount/permissions and "
                                "flashchat will use it when reachable"))
        for warning in report.warnings:
            print(common.yellow(f"  note: {warning}"))
    return out


def _advanced_settings(manifest) -> dict:
    if not common.confirm("Configure advanced options (debug, MTP, cache)?", default=False):
        return {}
    out = {}
    mtp_raw = ""
    for key, label, default in (
            ("SERVER_DEBUG", "Server debug logging (0/1)", "0"),
            ("SERVER_HTTP_LOG", "HTTP traffic log (0/1)", "0"),
            ("SYSTEM_PROMPT_CACHE", "System prompt cache (0/1)", "1"),
            ("SYSTEM_PROMPT_CACHE_MAX_ENTRIES", "Cache max entries", "2"),
            ("SYSTEM_PROMPT_CACHE_DIR", "External system prompt cache root (- for model directory)", ""),
            ("MTP", "Multi-token prediction (0=off, auto=registry default, N=batch)", "0"),
            ("MTP_BF16", "Use BF16 MTP predictor weights (0/1)", "0"),
            ("SHOW_THINKING", "Show thinking tokens (0/1)", "0"),
            ("COLOR_OUTPUT", "Color output (0/1)", "1")):
        value = common.prompt(label, configfile.get(key, default))
        if key == "MTP":
            mtp_raw = value
            if value.lower() == "auto":
                value = ""  # empty = registry/profile default (legacy semantics)
        if key == "SYSTEM_PROMPT_CACHE_DIR" and value == "-":
            value = ""
        out[key] = value
    _warn_if_mtp_unsupported(manifest, mtp_raw, out)
    return out


def _mtp_value_enables(raw: str) -> bool:
    """Mirror the engine's parse_mtp_predictions truthiness: 0/off/no/false and
    empty (registry default) do not actively request MTP; auto/on/yes and any
    positive integer do."""
    s = (raw or "").strip().lower()
    if s in ("", "0", "off", "no", "false"):
        return False
    if s in ("auto", "on", "yes", "true"):
        return True
    return s.isdigit() and int(s) > 0


def _warn_if_mtp_unsupported(manifest, mtp_raw: str, out: dict) -> None:
    """If the user is enabling MTP for a model whose checkpoint has no MTP head,
    say so and offer to turn it back off. Authoritative source-side safetensors
    scan, falling back to the manifest's declared capability when the model
    isn't downloaded."""
    if not _mtp_value_enables(mtp_raw):
        return
    snapshot = paths.snapshot_dir(hf_cache_dir(), manifest.hf_repo)
    supported = source_mtp_tensors_present(snapshot)
    confident = supported is not None
    if supported is None:
        supported = manifest.mtp_artifacts_required
    if supported:
        return
    if confident:
        print(common.yellow(
            f"\n  ⚠ {manifest.name} has no MTP head — its checkpoint ships no "
            "mtp.* tensors, so multi-token prediction will have no effect "
            "(the server decodes normally and logs it as inactive)."))
        if common.confirm("Disable MTP for this model?", default=True):
            out["MTP"] = "0"
    else:
        print(common.yellow(
            f"\n  ⚠ couldn't verify MTP support for {manifest.name} (model not "
            "downloaded); the registry lists no MTP head for it. If it truly has "
            "none, MTP will have no effect."))
        if common.confirm("Disable MTP for now?", default=False):
            out["MTP"] = "0"
