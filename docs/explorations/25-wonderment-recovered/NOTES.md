# 25 — Recovered bounded-experiment campaign from flashchat-work

Status: **HISTORICAL RECOVERY, not a new performance experiment.**
Original experiments: 2026-08-17; original commit: 2026-08-18.
Recovered: 2026-09-16. No benchmarks or engine changes made during recovery.

## Provenance and divergence

Source checkout: `/Users/speedster/dev/flashchat-work`, clean at
`a399937b4a63fe36e1f9f788986a244ebe689338` on `codex/decode-speed-wonderment`.
This is a separate Git checkout, not a registered worktree of the current repo.
The current checkout was at `b803619ff9c2af54e02c4cccb773653f5d1e66ee`, plus
uncommitted server/UI work and archive reconstruction.

The newest shared ancestor is `bb22cc56a86200e3835801dd144727a725ba6782`.
The other checkout has one unique commit, `a399937`; current HEAD has 20 commits
since that ancestor. Comparing all local and remote-tracking refs in both object
databases also found only that one other-tree-only commit. No network fetch was
performed. The other checkout had no uncommitted or untracked non-ignored files.

The original campaign called itself **10-wonderment-round**, colliding with this
archive's unrelated **10-prefill-regression**. Index 25 preserves the campaign
without overwriting or renumbering existing experiments. It does not imply the
experiments occurred after 24.

## Preserved primary evidence

- [ORIGINAL_NOTES.md](ORIGINAL_NOTES.md): verbatim committed notes, retaining
  their original heading and contemporary conclusions.
- [a399937.patch](a399937.patch): complete unique commit, including code,
  index changes, the missing working principle, and all **39 historical API
  performance rows** exactly as committed. Those rows are absent from the current
  `assets/api_perf_log.tsv`, checked by timestamp/model/scenario/metric identity.
  They remain archived here without changing status, labeling, values, or the
  main watchdog log. This patch is evidence, **not a patch to apply wholesale**.

## Results and representation in the current project

| Work | Other-tree evidence | Current representation |
|---|---|---|
| Protect active pin hits from eviction | 60-slot cache, 1,774 forced evictions in 80 tokens; output matched pinning off | Safety change landed separately in `6057f0c`; current reservation/admission logic has since evolved in 21 |
| Automatic linear-layer command fusion | Interleaved 100-token probes: off 9.92/10.40, fused 11.35/11.33 tokens/s | Same default-enabled benefit, but different control policy: `9332b18` keeps a first-class `FUSE_LINEAR` setting; other tree removes the switch entirely |
| Bypass filesystem caching (`F_NOCACHE`) | Short CLI probes looked faster; sustained API probes fell to 6.1–8.3 versus roughly 11.0–11.1 tokens/s with caching | General negative finding appears in 02; these particular probes and API rows were missing |
| Split expert reads | Concurrent fragments: 10.81 → 9.97; staged: 11.10/11.08 → 10.52/10.48 tokens/s, output identical | Reconstructed under 08; later positive implementation in 14 and negative event-gating probe in 21 are distinct tests |
| Bounded-experiment working principle | Cost models should rank probes, not veto cheap falsifiable measurements | This specific sentence was not carried into current AGENTS.md; current policy does call for bounded validation and avoiding noise-driven reruns |

These are historical reported results, not current performance guarantees.
Different prompts, memory/cache conditions, and runtime settings prevent direct
cross-session attribution. The old notes' claim that split reads should only be
retried with fewer submission boundaries is a contemporary conclusion, not a
restriction established by the later successful 14 implementation.

## What is and is not missing

No unique surviving engine feature requiring a port was found in the divergent
commit: pin safety is represented, and fusion differs in configurability rather
than missing capability. Both failed prototypes had already been removed before
`a399937`, so their implementation patches cannot be recovered from that commit.
The unique experiment notes and 39 API rows were the substantive archive gap.

Do not import the other checkout's whole tree: it predates the current prefill
work, split-I/O/residency changes, configuration improvements, and other fixes.
In particular its 07 notes still claim an unexplained prefill regression; this
tree corrected that claim via experiment 10 and commit `cdb14b8`.

The archived principle about allowing cheap, skeptical probes remains useful,
but does not authorize repeated baselines, disturbing production measurements,
or treating small single-digit differences as reliable wins. Follow current
AGENTS.md. Changes to code or working policy were not made as part of this audit.
