"""First-run onboarding: no config file -> pick a model, build it, save."""

from .. import configfile, resolved
from ..registry import Registry, resolved_id
from ..status import all_statuses
from . import build, common, status_view


def run(registry: Registry) -> bool:
    common.heading("Welcome to Flashchat — one-time setup")
    print("Pick a model to start with (you can enable more later):\n")

    statuses = all_statuses(registry)
    default_idx = 1
    for i, status in enumerate(statuses, 1):
        if status.manifest is registry.default_model():
            default_idx = i
        rec = common.green("  [recommended]") if status.manifest is registry.default_model() else ""
        status_view.print_model_block(i, status)
        if rec:
            print(rec)
        print()

    choice = common.select_number(len(statuses), "Select model", default=default_idx)
    if choice is None:
        print("Setup cancelled — run './flashchat config' any time.")
        return False
    manifest = statuses[int(choice) - 1].manifest

    variant_name = manifest.default_variant
    if len(manifest.variants) > 1:
        names = list(manifest.variants)
        print("\nVariants:")
        for i, v in enumerate(names, 1):
            mark = " (default)" if v == manifest.default_variant else ""
            print(f"  {i}) {v}{mark}")
        vchoice = common.select_number(len(names), "Variant",
                                       default=names.index(manifest.default_variant) + 1)
        if vchoice is None:
            return False
        variant_name = names[int(vchoice) - 1]

    if not build.ensure_variant_built(registry, manifest, variant_name):
        print("\nSetup incomplete — run './flashchat config' to finish later.")
        return False

    _save_selection(registry, manifest, variant_name)
    print(common.green("\nSetup complete. Start chatting with: ./flashchat chat"))
    return True


def _save_selection(registry: Registry, manifest, variant_name: str) -> None:
    profile_name = manifest.default_sampling_profile
    profile = manifest.sampling_profiles.get(profile_name, {})
    changes = {
        "MODEL": resolved_id(manifest, variant_name),
        "MODEL_BASE": manifest.id,
        "MODEL_VARIANT": variant_name,
        "SAMPLING_PROFILE": profile_name,
        "CONFIG_SCHEMA_VERSION": "2",
    }
    for key, cfg_key in (("temperature", "TEMPERATURE"), ("top_p", "TOP_P"),
                         ("top_k", "TOP_K"), ("min_p", "MIN_P"),
                         ("presence_penalty", "PRESENCE_PENALTY"),
                         ("repetition_penalty", "REPETITION_PENALTY"),
                         ("reasoning", "REASONING")):
        if key in profile:
            changes[cfg_key] = str(profile[key])
    configfile.update(changes)
    registry.state.enabled[manifest.id] = True
    registry.state.default_model = manifest.id
    if registry.state.layout_version < 1:
        registry.state.layout_version = 1  # fresh install: nothing to migrate
    registry.state.save()
    resolved.write(registry)
