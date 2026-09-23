# Model Management Framework

Developer reference for the `modelmgr` Python package — the management
brain behind the `./flashchat` launcher.

## Architecture

```
shipped manifests              user state
assets/models/*.json    ~/.config/flashchat/{models.d/*.json, models.state.json}
        \                      /
         modelmgr  (registry / recipes / steps / artifacts / offload / migrate / TUI)
        /                       \
~/.config/flashchat/resolved_models.json     ~/.config/flashchat/config
  (engine-facing, legacy flat schema)          (engine-facing, KEY="value")
        \                       /
         ./flashchat (bash launcher) -> metal_infer/infer --serve / chat
```

The C engine is untouched: it keeps parsing the flat legacy registry schema
(via `FLASHCHAT_MODEL_CONFIG`, see `metal_infer/model_config.h`) and the
config file. `modelmgr.resolved` renders that view from the
hierarchical manifests; `assets/model_configs.json` is **generated** output
(`make registry`, checked by `make registry-check`) kept as the first-run
fallback.

## Manifests: one file per base model

`assets/models/<id>.json` declares everything the framework needs:

- `hf_repo` — the HuggingFace source of truth
- `architecture` — engine dimensions (flow into the resolved view verbatim)
- `variants` — `q4`/`q8`/... each with quantization params, `legacy_ids`
  (old flat registry ids; preserve config/API compatibility), and an
  `artifacts` map
- `shared_artifacts` — variant-independent artifacts (vocab.bin, BF16 MTP
  weights) built once and symlinked into each variant dir
- `mtp` — MTP head metadata (`num_hidden_layers` presence means MTP
  artifacts are expected from native checkpoints)

Each artifact entry names the **step** that produces it (or `from_shared`).
Adding a model = writing one manifest (`./flashchat config` → add model does
this from the HF config.json automatically, into `~/.config/flashchat/models.d/`).
Code is only needed for architecturally novel models.

### Sampling belongs to the individual model

Each manifest owns `sampling_profiles` and `default_sampling_profile`; the resolved
registry forwards them to the configuration menu and server. Vocabulary size and
architecture do not establish suitable sampling settings.

Add Model reads the selected repository's `generation_config.json` and creates one
`model-default` profile. Temperature, top-p, and top-k must be explicitly supplied
for sampled generation; `do_sample: false` selects greedy generation. Missing
penalties use their disabled values, not values copied from another model. Reasoning
is enabled only when the source template is known to support it; importing a
non-thinking model does not create thinking profiles. Existing curated profiles in
shipped manifests remain unchanged.

When metadata is missing or outside the engine's supported ranges, import stops
with an explanation. Supply a model-specific JSON file at the menu prompt or with
`python -m modelmgr add-model OWNER/MODEL --generation-config FILE`. This is also
how to supply model-card recommendations when the repository metadata is incomplete.
Review the imported settings in the menu; model owners can curate additional named
profiles in their manifest. Existing user manifests are not silently overwritten.

A saved named profile that does not exist for the selected model falls back to that
model's declared default with a warning. Explicit `custom` settings and environment
or API overrides remain authoritative. Running servers need a restart to pick up
changed model metadata.

## Steps and recipes

`modelmgr/steps/` is a shared, parameterized step library:
`download_hf`, `export_tokenizer`, `generate_expert_index`,
`extract_weights`, `repack_experts`, `compile_native:{non_experts, experts,
mtp_experts, bf16_mtp}`, `materialize_shared`.

`recipes.plan(manifest, variant, snapshot)` derives the ordered step list by
checking every declared artifact against its integrity manifest — valid
artifacts are skipped, so plans converge to empty (idempotent). Bump a
step's version in `steps/STEP_TABLE` to invalidate its previous outputs
after a bugfix.

## Artifact integrity

Every artifact dir (`shared/`, `q4/`, ...) carries `.flashchat_artifacts.json`
with per-file size + sha256 + producing step. Hashes are computed **while
writing** (or page-cache-warm immediately after, for the pwrite-based expert
packers). Writers stream to `<name>.partial` and rename, so interruptions
never leave plausible-looking files.

- quick verify (status renders): existence + size + dimension cross-checks
- deep verify (`./flashchat verify --deep`, manage menu): full re-hash
- files adopted from pre-framework trees are `unhashed` until a baseline is
  computed (offered during deep verify)

## Disk layout v1

```
<snapshot>/flashchat/
  shared/   vocab.bin, bf16/mtp_weights.{bin,json}, .flashchat_artifacts.json
  q4/       model_weights.{bin,json}, packed_experts/, packed_mtp_experts/,
            vocab.bin -> ../shared/vocab.bin, bf16/ -> ../shared/bf16/,
            system_prompt_cache/ (engine-managed), .flashchat_artifacts.json
  q8/       same
```

`./flashchat migrate` converts legacy trees (per-variant duplicates, no
manifests) in place — one-time, covers the HF cache and the offload dir,
idempotent per snapshot, `--dry-run` to preview.

## Offload

`modelmgr/offload.py` replaces the old `mv`-based offload:
capability-probing preflight (writability/symlinks/space — surfaces NAS/USB
permission failures before any transfer), rsync-based model-directory refresh,
lightweight offload metadata, and local source-blob removal only after the
offload copy is present with matching size. Full hashing is reserved for
explicit deep/audit flows, not the default offload path. Archives never store
`system_prompt_cache` because it may contain prompt-derived user data.
Operations: offload model, restore full/runtime-only/originals.

Automation: after a successful build, and once per launch when the offload
volume is reachable, flashchat offers to offload complete models and later
refreshes pending artifact scopes (`shared`, `q4`, `q8`) when local runtime
artifacts changed while the offload copy was stale or unavailable. A build
marks a scope pending only when it actually changed files there, and the
launch offer first drops any pending scope whose offload copy already
matches. Both checks compare file size and modification time only (rsync's
quick check), so they never hash artifacts. The offer appears on every
interactive launch, from the menu or a subcommand alike, and never without
a terminal. `[M]anage`
is for everything manual: per-artifact verify/regenerate, variant deletion,
offload and restore.

## Testing

`make py-tests` runs `tests/python/` (manifest schema, golden resolved-view
vs the legacy registry + C-parser scoping emulation, recipe skip logic,
hash tamper detection, migration on fixture trees, offload resume/preflight).
`make registry-check` ensures `assets/model_configs.json` matches the
manifests.
