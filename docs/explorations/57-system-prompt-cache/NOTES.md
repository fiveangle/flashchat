# 57 — System-prompt context caching

Date: 2026-03-17 (upstream origin) to 2026-06-18. Registered 2026-09-22.
Commits: `630b631` (upstream, Dan Woods), `ca4efaa` (persistent, 2026-05-06),
`f2e8ea6` (external cache directory, 2026-06-18).
Status: **Closed.** Shipped before it had an exploration ID.
Verdict: **Keep.**

## Hypothesis

Every request re-prefilled the same system prompt. Prefilling it once and
restoring a snapshot of the model state per request removes that cost from every
request's time to first token.

## What shipped

- **Upstream `630b631` (Dan Woods, 2026-03-17):** prefill the system prompt once at
  server start; snapshot the context cache, linear-attention state, and GPU
  delta-net state; restore the snapshot per request, so requests tokenize only the
  user turn and positions start after the system prompt.
- **Fork `ca4efaa` (2026-05-06):** persist the snapshot on disk so it survives
  restarts.
- **Fork `f2e8ea6` (2026-06-18):** allow an external cache directory. AGENTS.md
  now requires these caches to live under `<model>/flashchat/system_prompt_cache/`,
  validated against model dimensions and prompt hash, written atomically, and
  bounded in count.

## Measurement

Upstream's own claim in `630b631`: **~6 s TTFT saved per request**, on
Qwen3.5-397B-A17B. No fork-era A/B isolates this feature. It is labelled as the
implementing commit's claim, on a different model and machine.

## Conclusion

Keep. It is inside the 00 baseline, so its saving is counted and not available
again. Extensions are separate items: exact conversation reuse (26), partial
prefixes and branches (27), prepared prompt and schema caching (28), persistence
and restore latency (29), and prewarming known prefixes (41). Do not re-propose the
base snapshot.
