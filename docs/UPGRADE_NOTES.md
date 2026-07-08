# Upgrade Notes

Caveats to be aware of when updating Flashchat. Newest first.

## The 8-bit model id changed: `-8bit` → `-q8`

**Why.** Flashchat shows a model's quantizations as the base model name with the
quant as a parenthetical annotation — `Qwen3.6-35B-A3B (q4)` and
`Qwen3.6-35B-A3B (q8)` — instead of a suffixed name. The goal is to keep
Flashchat's ids from being confused with real Hugging Face repo ids: a bare
`-4bit`/`-8bit` suffix can be an actual HF repo, whereas `(q8)` is unmistakably
an annotation. (Flashchat ids also carry no `/`, so they can never be mistaken
for a valid HF `org/name` id.)

**The change.** The 8-bit build of Qwen3.6-35B-A3B, previously selectable as
`Qwen-Qwen36-35B-A3B-8bit`, is now `Qwen-Qwen36-35B-A3B-q8`. The 4-bit (default)
id is unchanged (`Qwen-Qwen36-35B-A3B`), so most configs need no action.

**Who is affected.** Only configs whose `MODEL` was set to the old 8-bit id.
After upgrading, that id no longer resolves: the launcher prints a "model not in
registry" warning and falls back to the default (4-bit) model at load.

**What to do.** Re-select the model once via `flashchat config` (pick the model,
then the q8 variant), or edit `~/.config/flashchat/config` and change
`MODEL="Qwen-Qwen36-35B-A3B-8bit"` to `MODEL="Qwen-Qwen36-35B-A3B-q8"`. On-disk
weights and prompt caches are keyed by base model + quant, not by this id, so
they are unaffected.
