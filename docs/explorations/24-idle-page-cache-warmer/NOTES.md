# 24 — Warm unpinned expert pages between requests

Status: **DESIGN ONLY — not implemented or measured in the recovered record.**
Reconstructed on 2026-09-16; no new tests run.

## Hypothesis

During idle time between requests, read likely-needed, unpinned experts into
the operating system's file cache. The original idea uses expert frequency
counts: prioritize experts with high observed use that
do not fit in the explicit pin cache. It targets memory-constrained 16GB systems
where filesystem caching is an important remaining tier.

## Available evidence

- [Experiment 21, "Follow-ups not attempted"](../21-io-event-gating/NOTES.md)
  describes background LFU prefetch between requests and explicitly says it was
  not implemented. Commit `b803619` preserves that note.
- [The status board](../STATUS.md) reserves index 24 at P2 priority.
- No dedicated implementation commit, raw run, cache-hit measurement, or
  correctness result for this proposal was found in the inspected history.

Do not conflate this with [09's active-generation prefetch](../09-expert-prefetch/NOTES.md),
which measured poor prediction of the unpinned miss set, or with the old
page-cache/explicit-LRU infrastructure described by commits `706f83d` and
`fd26f2f`. Those are related background, not measurements of this idle-only design.

## Constraints and revisit criteria

Request boundaries might provide time to warm useful pages without extending
the current token's critical path, but warming can also evict useful data,
consume SSD bandwidth, and predict the next prompt poorly. No net benefit is
established. Inspect existing caching and request lifecycle first; do not add
a second inference workload to test this idea.

A future design would need a bounded byte/time budget, prompt handoff that stops
or yields warming immediately, and an explicit idle-window policy. It must not
silently disturb Dave's performance measurements. These are reconstruction-time
design cautions, not features or test results from the original proposal.
