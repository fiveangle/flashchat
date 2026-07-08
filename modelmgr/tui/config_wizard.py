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


def _has_config_changes(changes: dict) -> bool:
    previous = configfile.load()
    missing = object()
    return any(previous.get(key, missing) != str(value)
               for key, value in changes.items())


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
        "CONFIG_SCHEMA_VERSION": "9",
    })
    registry.state.enabled[manifest.id] = True
    registry.state.save()

    changes["MAX_TOKENS"] = common.prompt(
        "Max response tokens", configfile.get("MAX_TOKENS", "8192"))

    cw = _context_window_setting(manifest)
    changes.update(cw)
    effective_window = (
        int(cw["CONTEXT_WINDOW"]) if cw["CONTEXT_WINDOW"]
        else (min(65536, manifest.max_context) if manifest.max_context > 0 else 65536))
    changes.update(_kv_quant_setting(manifest, effective_window))

    changes.update(_select_sampling_profile(manifest))
    changes.update(_server_settings())
    changes.update(_storage_settings())
    changes.update(_advanced_settings(manifest, variant_name, changes.get("ACTIVE_EXPERTS", "")))

    changed = _has_config_changes(changes)
    configfile.update(changes)
    resolved.write(registry)
    if changed:
        print(common.green("\nConfiguration saved."))
    else:
        print(common.green("\nNo configuration changes."))

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
    win = configfile.get("CONTEXT_WINDOW", "") or "65536 (default)"
    model_max = current[0].max_context if current else 0
    max_label = f" (model max {model_max})" if model_max > 0 else ""
    print(f"Context window: {win} tokens{max_label}")
    print(f"KV cache quantization: {configfile.get('KV_QUANT', '') or 'off'}")
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
        try:
            download_file(repo, "tokenizer_config.json", hf_cache_dir())
        except DownloadError:
            pass
    except DownloadError as e:
        print(common.red(f"download failed: {e}"))
        return False
    import json
    with open(config_path) as f:
        hf_config = json.load(f)
    try:
        thinking_capable = template_supports_thinking(paths.snapshot_dir(hf_cache_dir(), repo))
        manifest_dict = derive_manifest(repo, hf_config, registry,
                                        thinking_capable=thinking_capable)
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


def _context_window_setting(manifest) -> dict:
    """Top-level prompt: KV context window in tokens. Larger = more RAM
    (q8 KV ≈ 10 KiB/token; pair big windows with FLASHCHAT_KV_QUANT=q8 on small
    machines). Ships at 64K; clamps to the model's trained max_context."""
    max_ctx = manifest.max_context
    default_win = min(65536, max_ctx) if max_ctx > 0 else 65536
    current_win = configfile.get("CONTEXT_WINDOW", "")
    if current_win.isdigit() and max_ctx > 0 and int(current_win) > max_ctx:
        print(common.yellow(f"  saved context window {current_win} exceeds model max {max_ctx}; using {max_ctx}"))
        current_win = str(max_ctx)
    current_win = current_win or str(default_win)
    max_label = f", max {max_ctx}" if max_ctx > 0 else ""
    value = common.prompt(
        f"Context window (tokens, default {default_win}{max_label})", current_win)
    if value.isdigit() and max_ctx > 0 and int(value) > max_ctx:
        print(common.yellow(f"  context window {value} exceeds model max {max_ctx}; saving {max_ctx}"))
        value = str(max_ctx)
    return {"CONTEXT_WINDOW": "" if value == str(default_win) else value}


def _fmt_bytes(b: int) -> str:
    return f"{b / 2**30:.2f} GB" if b >= 2**30 else f"{b / 2**20:.0f} MB"


def _kv_quant_setting(manifest, window: int) -> dict:
    """KV cache quantization (off/q8/q4) as a numbered menu. Each option shows
    the wired GPU KV-buffer RAM it needs at the chosen window, computed from the
    model's KV geometry."""
    a = manifest.architecture
    n_kv = int(a.get("num_key_value_heads", 0) or 0)
    head_dim = int(a.get("head_dim", 0) or 0)
    n_full = int(a.get("num_hidden_layers", 0) or 0) // max(
        1, int(a.get("full_attention_interval", 1) or 1))
    kv_dim = n_kv * head_dim
    have_ram = kv_dim > 0 and n_full > 0
    # mode, GPU bytes/token across all full-attn layers (K + V [+ fp16 scales]), note
    modes = [
        ("off", 2 * kv_dim * 4,              "fp32, lossless"),
        ("q8",  2 * kv_dim + 2 * (n_kv * 2), "~lossless, best for large windows"),
        ("q4",  2 * (kv_dim // 2) + 2 * (n_kv * 2), "lossy, smallest"),
    ]
    names = [m[0] for m in modes]
    current = (configfile.get("KV_QUANT", "") or "off").lower()
    default_idx = names.index(current) + 1 if current in names else 1

    common.heading("KV cache quantization")
    if have_ram:
        print(f"GPU KV-buffer RAM at {window:,}-token window:\n")
    for i, (name, per_tok, note) in enumerate(modes, 1):
        ram = f"{_fmt_bytes(per_tok * n_full * window):>9}" if have_ram else ""
        mark = common.dim(" (current)") if name == current else ""
        print(f"  {i}) {name:<3} {ram}   {note}{mark}")

    choice = common.select_number(len(modes), "Select", default=default_idx)
    value = names[choice - 1] if choice else current
    return {"KV_QUANT": "" if value == "off" else value}


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


def _estimated_packed_expert_size(manifest, variant_name: str) -> int:
    arch = manifest.architecture
    hidden = int(arch.get("hidden_size", 0) or 0)
    intermediate = int(arch.get("moe_intermediate_size", 0) or 0)
    if hidden <= 0 or intermediate <= 0:
        return 0
    variant = manifest.variant(variant_name)
    bits = int(variant.bits or 0)
    group = int(variant.group_size or 0)
    if bits <= 0 or group <= 0:
        return 0

    def matrix_bytes(out_dim: int, in_dim: int) -> int:
        packed_weight = out_dim * in_dim * bits // 8
        groups = (in_dim + group - 1) // group
        scale_bias = out_dim * groups * 2 * 2
        return packed_weight + scale_bias

    return (matrix_bytes(intermediate, hidden) +
            matrix_bytes(intermediate, hidden) +
            matrix_bytes(hidden, intermediate))


def _expert_pin_guidance(manifest, variant_name: str, active_experts: str) -> None:
    k = manifest.num_experts_per_tok
    if not active_experts:
        active_experts = configfile.get("ACTIVE_EXPERTS", "")
    if active_experts and active_experts.isdigit():
        k = int(active_experts)
    layers = int(manifest.architecture.get("num_hidden_layers")
                 or manifest.architecture.get("num_layers") or 0)
    total_experts = int(manifest.architecture.get("num_experts", 0) or 0)

    if k <= 0 or layers <= 0:
        print(common.dim("  Set this as the number of complete experts to keep in RAM."))
        return

    slots_per_decode_pass = layers * k
    pool_slots = layers * total_experts if total_experts > 0 else 0
    expert_size = _estimated_packed_expert_size(manifest, variant_name)
    small = slots_per_decode_pass
    recommended = slots_per_decode_pass * 2
    large = slots_per_decode_pass * 4
    if expert_size > 0:
        print(common.dim(
            f"  Recommended: {recommended} experts ({paths.human_bytes(recommended * expert_size)}), "
            f"enough for about two generated tokens worth of K={k} expert choices."))
        print(common.dim(
            f"  Smaller: {small} experts ({paths.human_bytes(small * expert_size)}); "
            f"larger: {large} experts ({paths.human_bytes(large * expert_size)})."))
    else:
        print(common.dim(
            f"  Recommended: {recommended} experts, enough for about two generated tokens worth of K={k} expert choices."))
        print(common.dim(f"  Smaller: {small} experts; larger: {large} experts."))
    if pool_slots > 0:
        if expert_size > 0:
            print(common.dim(
                f"  Maximum useful value for this model: {pool_slots} experts "
                f"({paths.human_bytes(pool_slots * expert_size)})."))
        else:
            print(common.dim(f"  Maximum useful value for this model: {pool_slots} experts."))


def _advanced_settings(manifest, variant_name: str, active_experts: str = "") -> dict:
    if not common.confirm("Configure advanced options (debug, MTP, cache)?", default=False):
        return {}
    out = {}
    mtp_raw = ""
    # Entries: (KEY, prompt label, default, optional ONE-line dim help). The
    # label carries the value space; help exists only where the label alone
    # cannot explain the consequence. No headers, no blank lines — the section
    # reads as a serial choice log.
    for key, label, default, help_text in (
            ("SERVER_DEBUG", "Server debug logging to server.log (0/1)", "0", None),
            ("SERVER_HTTP_LOG", "HTTP request/response log to http.log (0/1)", "0", None),
            ("BATCH_PREFILL", "Batched prompt processing (0/1)", "1",
             "~4-6x faster prompt reading; responses unchanged in normal use."),
            ("PREFILL_CHUNK", "  ^ chunk size in tokens (8-4096)", "1024",
             "Bigger = faster but more working RAM: 1024 ~170 MB, 2048 ~340 MB."),
            ("ANE_PREFILL", "  ^ Neural Engine expert offload (0/1)", "1",
             "~25% faster on top of batching; auto-falls back to GPU if the hardware lacks a Neural Engine."),
            ("PREFILL_DEBUG", "  ^ debug (0=off, 1=chunk timings, 2=+state dump, slow)", "0", None),
            ("PREAD_PROFILE", "Disk-read timing log (empty=off, or a .tsv path)", "",
             "For diagnosing slow expert streaming; analyze with tools/pread_profile_analyze.py."),
            ("PREAD_PROFILE_CAP", "  ^ max recorded events before it stops", "2097152", None),
            ("EXPERT_PIN_MAX_EXPERTS", "Expert RAM cache target in complete experts (empty=use GiB cap)", "",
             None),
            ("EXPERT_PIN_MAX_GB", "  ^ maximum GiB cache limit (0 disables cache)", "4",
             None),
            ("EXPERT_PIN_AUTO_FRAC", "  ^ also capped to this fraction of free RAM (0.1-0.9)", "0.5", None),
            ("EXPERT_PIN_MLOCK", "  ^ lock against swap (0/1; keeps RAM from other apps)", "0", None),
            ("SYSTEM_PROMPT_CACHE", "System prompt cache (0/1)", "1",
             "Repeat requests skip re-reading the system prompt — big win for long agent prompts."),
            ("SYSTEM_PROMPT_CACHE_MAX_ENTRIES", "  ^ max saved prompts (1-64; entries can be tens of MB)", "2", None),
            ("SYSTEM_PROMPT_CACHE_DIR", "  ^ cache folder ('-' = beside the model)", "", None),
            ("MTP", "Multi-token prediction (0=off, auto=model default, 2+=batch size)", "0",
             "Lossless speculative decoding for models that ship a predictor head."),
            ("MTP_BF16", "  ^ BF16 predictor weights (0/1; more RAM, slightly better drafts)", "0", None),
            ("SHOW_THINKING", "Show thinking tokens (0/1)", "0", None),
            ("COLOR_OUTPUT", "Color output (0/1)", "1", None)):
        if help_text:
            print(common.dim(f"  {help_text}"))
        if key == "EXPERT_PIN_MAX_EXPERTS":
            _expert_pin_guidance(manifest, variant_name, active_experts)
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
