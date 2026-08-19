# 10 — Prefill "regression" hunt (~2.2x vs Jul-8 fast rows)

Branch: main — investigation only. OUTCOME: **no regression exists; the
Jul-8 fast rows are unreliable. Case closed.**

## The anomaly

Durable bench rows for Qwen-Qwen36-35B-A3B q4 (same machine, same spec):
Jun 4–Jul 8 morning ~2.1–2.9s prefill; Jul 8 13:54–14:03 **0.86–1.12s**;
today's main ~1.9–2.6s.

## What was established (in order)

1. **Nothing unmerged**: ane-prefill tip (8bd1680) has zero commits main
   lacks. 4f2356a (the fast rows' HEAD) is an ancestor of main.
2. **The fast rows ran on a dirty tree**: the 13:51 debug session
   (docs/PREFILL_RELEASE_BENCHMARK_LOG.md) used FLASHCHAT_PREFILL_RELEASE,
   which only exists in 86c8846 — committed at 15:40, AFTER the 13:54 bench.
   So the fast rows came from uncommitted WIP whose exact state is
   unrecoverable.
3. **Bisect (bench_api per commit, isolated worktrees, unique ports after
   catching a port-squatter tainting control A)**:
   - 9fd7c83 + segfault fix: 1.86–2.36s
   - clean 4f2356a: 2.14–2.94s
   - clean 4f2356a + FLASHCHAT_ANE_PREFILL=1: 2.14–2.83s
   No clean committed tree in the window reproduces the fast rows. Sweep
   of intermediate commits aborted — pointless once both endpoints are
   equally slow.
4. **ANE is NOT silently failing on main** (new engagement logging, f94068d):
   40/40 chunks, 100% of rows on the Neural Engine, no fallback.

## Verdict

**Dismount.** The 0.86–1.12s rows cannot be reproduced by any committed
tree and came from an unrecoverable WIP state. Whether that WIP was a real
optimization that never landed or code that skipped work (and was therefore
fast AND wrong) can no longer be determined — the rows must be treated as
unreliable, and main's ~2s prefill is the true baseline. bench-report's
comparisons against those rows will show phantom "regressions"; discount
prefill deltas that reference 2026-07-08T13:54–14:03 rows.

## What the investigation did yield

- ANE engagement logging (permanent win — the observability gap that let
  this question fester is closed).
- tools/prefill_bisect.sh (hardened: unique port per run, bind-failure
  abort, segfault-fix cherry-pick for 86c8846+). Reusable for future
  per-commit perf questions.
- A real prefill optimization lead if prefill ever becomes a target: the
  phase breakdown for a 40-token prefill shows producer=1006ms +
  eval=795ms of 2328ms total, with 3,458 expert evals serving 12,480 rows
  (~3.6 rows/expert) — the ANE path runs one <=256-row slab per unique
  expert, so short prompts pay near-full-slab cost per expert. Batching
  sparse experts into shared slabs (or a small-N GPU fast path) is where a
  future prefill win lives. That is an optimization project, not a
  restoration.
