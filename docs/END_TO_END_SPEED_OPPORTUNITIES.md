# Flashchat exploration register

Initial review: **2026-09-16**, `develop` at `782f361`.

This is the cumulative register of Flashchat performance explorations: proposed,
in progress, completed, rejected, and superseded. It shares one permanent numeric
ID space with `docs/explorations/`, experiment branches, and commit headlines.
Use it to select work and retain its outcome, rather than creating a replacement
brainstorming list for each session.

The 2026-09-16 migration imported legacy IDs **00–25** and reserved **26–44** for
new proposals. Existing proposals keep their assigned IDs.

The end-to-end review covered the API implementation, Git history, exploration
notes, and relevant upstream implementations. **No new benchmarks were run for
that review or this migration.** New proposals are not measured speedup claims;
legacy findings describe their original workloads. Recheck implementation status
and source locations before starting work.

## Objective

Optimize elapsed time to a correct, useful result across the entire API and tool
loop. Include prompt preparation, cache reuse, generation, tool execution,
retries, delivery, and client rendering—not just decode tokens per second.

At 25 tokens/sec, eliminating 1,000 unnecessary generated tokens saves 40 seconds.
Increasing throughput from 25 to 30 tokens/sec saves about 6.7 seconds on those
same tokens. These are illustrative arithmetic, not benchmark predictions.

The strongest initial bets are:

1. Preserve conversation work already performed.
2. Remove repeated prompt preparation and avoidable memory pressure.
3. Reduce generation steps, failed calls, and tool round trips.

## Permanent IDs and evidence

- IDs identify an exploration, not its priority or completion order. Never renumber
  or reuse one after assignment, even if it is cancelled, rejected, or superseded.
- Before allocating an ID, check this register, numbered exploration directories,
  and Git history/branches. Allocate above the highest assigned or reserved number
  across all three. A proposal reserves its ID before it has a directory.
- Use the same number in `docs/explorations/<id>-<name>/`,
  `exp/<id>-<name>` branches, and `exp/<id>:` commit headlines. Preserve existing
  zero-padded IDs 00–09. Reuse the assigned ID when starting a registered proposal.
- Continue the same hypothesis under its existing ID. Give materially distinct
  follow-ups a new ID and link the relationship; do not erase the earlier outcome.
  Existing sub-attempts such as 21a/21b remain children of 21.
- Keep per-exploration `NOTES.md` and raw evidence in the existing archive. This
  register is the cross-exploration index and planning record, not a second raw
  results archive. Add the notes link when an exploration starts.
- Keep the unified table in permanent ID order; update status, outcome, and
  evidence independently. Completed or superseded work must not become untested
  merely because one measurement was omitted. Record that gap and a deliberate
  revisit condition instead.
- For every measured entry, record the control/candidate result, units, material
  workload/configuration, limitations, and evidence link. For mixed results, name
  each retained/rejected component and its result. State explicitly when there is
  no pending action; distinguish a revisit condition from an assigned next task.
- Append dated findings, method, correctness, measurements, limitations, and
  conclusion to exploration notes. Preserve negative and inconclusive results;
  never replace missing evidence with a projected gain.
- Distinguish **proposed**, **in progress**, **measured**, **kept/shipped**,
  **parked**, **rejected**, **superseded**, and **historical recovery**. A design
  estimate is not a measured result; a review rejection is not a measured failure.
- Missing legacy detail stays explicitly unknown. Do not invent methods, outcomes,
  or historical dates to fill the newer framework.

## Unified exploration table

**Reference audit: 2026-09-16.** All 45 rows were reviewed against their available
notes and the initial code review. Keep rows in permanent numeric order and update
them in place. Nothing is removed for failing or for being superseded.

**Status answers “does this remain work to select?”; outcome answers “what did we
learn?”** Missing measurement in one dimension does not make a completed or
superseded exploration untouched again.

- **Untested:** the proposed work remains unattempted. Suffixes **deferred** and
  **gated** make selection intent or prerequisites explicit; listing is not a
  commitment to pursue it.
- **Reviewed / parked:** investigated by review and set aside, without a measured
  implementation result. Reopen only for the stated reason.
- **Parked:** attempted or probed, with no current action unless the revisit
  condition is met.
- **Superseded:** handled by another numbered exploration; not an independent
  backlog item. Residual gaps remain visible without reopening the original row.
- **Closed:** the recorded investigation is complete, whether successful,
  unsuccessful, mixed, neutral, or archival. The decision column records what was
  kept/rejected and what could justify a later revisit.
- Use **In progress** when work starts. Preserve the last known outcome until new
  evidence warrants changing it.

Scan **Status**, not a missing timing result, for untested work. Read **Outcome**
and **Evidence** to find prior wins, losses, and interactions. Mixed outcomes name
the components separately; baseline/recovery entries do not count as speedups.

Evidence conventions: `A→B` means control to candidate; throughput is tokens/sec
unless stated otherwise. Most original decode probes used Qwen3.6-35B-A3B q4 on a
32 GB Mac17,2, with eight active experts; each row calls out material configuration
differences. Memory-pressure emulation is not real 16 GB hardware. Conditions,
methods, and provenance remain linked from each legacy title. These are historical
measurements, not fresh validation of current defaults. Do not add gains from
separate campaigns or attribute combined-version gains to one change. Small
single-digit differences are observations with noise, not precise promises.

**Measured speed change** is the compact scan column. Decode percentages mean
change in tokens/sec: **+** is faster, **−** is slower. Prefill entries explicitly
say **less/more time** because a latency reduction is beneficial and is not the
same percentage as a throughput increase. **Neutral** means no meaningful change
established; **— Not measured** never means zero gain. Mixed attempts retain
separate component results. Conditions and qualifications remain in the evidence
column; estimates and external speedups are excluded from the measured column.

No new benchmarks were run. Raw logs and the performance watchdog were preserved.
Percentages below are recomputed from the stated values where practical; discrepancies
with old rounded summaries are explicit. No missing historical measurement was invented.

| ID | Exploration | Status | Outcome | Measured speed change | Evidence / result and conditions | Decision / revisit criterion |
|---|---|---|---|---|---|---|
| 00 | [Baseline capture](explorations/00-baseline/NOTES.md) | Closed | Baseline | — Baseline only | Warm 32 GB Mac17,2; q4, q8 context cache, 8 active experts, pinning/MTP off: 150-token runs 16.42 / 17.33 / 17.37 tok/s; median **17.33 tok/s**. No optimization delta. | Retain as historical reference, not a current baseline. Pin-off was a test configuration, not the shipping default (12). |
| 01 | [Zero-copy expert pin cache](explorations/01-pin-zero-copy/NOTES.md) | Closed | Successful | **+10.7%** decode (4 GiB) | Warm 32 GB, 150-token medians: pin-off **17.33 → 19.18 tok/s** with 4 GiB zero-copy pinning (**+10.7%**). 8/12/16 GiB gave 18.98/18.77/18.48; no extra speed from larger arenas. Tested greedy output matched. | Keep zero-copy hits and active-slot protection (25, 21). Revisit sizing when the workload changes, not simply to maximize hit rate. |
| 02 | [RAM discipline](explorations/02-ram-discipline/NOTES.md) | Closed | Mixed | Locking: **+15–21%** pressure / **+6.3%** warm<br>Bypass: inconsistent; decay: no benefit | **Locking cache pages (mlock):** 3 GiB pinning under simulated 16 GB pressure gave **11.13→12.84, 10.37→12.28, 10.55→12.81 tok/s (+15–21%)**; warm 32 GB/4 GiB **21.44→22.78 (+6.3%)**. **Filesystem-cache bypass:** inconsistent. **Frequency decay at decode start:** no hit-rate or speed benefit. Notes headline says +15–24%; the listed round pairs support +15–21%. | Keep the locking capability with headroom; reject tested bypass/decay policies. No pending action. Revisit for materially different memory pressure or prompt/decode routing distributions. Emulation is not real 16 GB hardware; sustained bypass failure is also preserved in 25. |
| 03 | [Adaptive expert count](explorations/03-adaptive-k/NOTES.md) | Closed | Successful, quality trade-off | **+4.6%** decode; output differs | Reported interleaved 100-token A/B: **16.55/16.39 → 17.24/17.21 tok/s (+4.6%)**, mass=0.90/minimum 5; mean active experts 6.921 vs 8 (**13.5% fewer reads**). Mass=0.95 was +0.4%; 0.98 never triggered. Greedy output eventually diverged. | Keep opt-in, default off. Require task-quality evidence before broader adoption. Small throughput delta is workload-specific; fewer reads do not establish equal answer quality. |
| 04 | [Batched MTP verification](explorations/04-mtp-batched-default/NOTES.md) | Parked | Unsuccessful | **−25.8%** decode (pressure test) | Two-position MTP verification: **75.9% acceptance**, but warm decode **14.46 vs ~18+ tok/s**. Listed pressure-test pair **21.59→16.03 tok/s (−25.8%, 0.742x)**; notes summarize ~0.76x. Low-confidence fallback paid sequential verification on 39/58 iterations. | Do not enable this variant by default. Continue only through 20 with a cheaper fallback/drafting policy and total-cost evidence; high acceptance alone was insufficient. |
| 05 | [Matrix-vector occupancy remap](explorations/05-matvec-occupancy/NOTES.md) | Reviewed / parked | No measurement | — Not measured | Review rejected the premise that only one SIMD group was useful: the current kernel already assigns eight groups to output rows. No numbered implementation or timing result recovered; historical DEAD was a review judgment. | No active work. Reopen only for a concrete alternative mapping and evidence of a different bottleneck; do not repeatedly treat the original premise as untouched. |
| 06 | [Prefill I/O thread pool](explorations/06-io-threads-prefill/NOTES.md) | Superseded | Implemented through 13 | Prefill: not measured<br>Decode: neutral (13) | The proposed **4→8-worker pool** landed through 13 (`5af6a43`, configurable up to 16). Its decode check was **17.42→17.31 tok/s (neutral)**; decode uses a different dispatch path. **Isolated prefill benefit was never measured.** | Closed as a separate proposal; implementation and residual measurement gap belong to 13. Do not put 06 back into the untested queue. |
| 07 | [Linear-layer command fusion](explorations/07-cmd-fuse-linear/NOTES.md) | Closed | Successful | **+11.8%** decode (Stage A); Stage B neutral<br>With 8 GiB pinning: **29% more prefill time** | Stage A, interleaved 100 tokens: **16.29/16.19→18.15/18.15 tok/s (+11.8% from listed pairs; notes +11.7%)**; waits **11,920→7,920**. Stage B **18.15→18.21–18.22**, effectively neutral but kept; tested outputs matched. API prefill: fuse-only **2435→2399 ms**; adding 8 GiB pinning raised it to **3143 ms (+29% vs 2435)**. | Keep both stages. Attribute the gain to removed synchronization, not Stage B alone. Extra pinning trades decode reuse against prefill memory; revisit that trade for the actual workload (02, 12, 23). |
| 08 | [Early split expert reads](explorations/08-expert-read-split/NOTES.md) | Superseded | Unsuccessful prototypes | **−7.8% / −5.3%** decode (two prototypes) | Recovered early probes: concurrent fragments **10.81→9.97 tok/s (−7.8%)**; staged reads **11.10/11.08→10.52/10.48 (−5.3%)**. Output reportedly matched; prototypes removed. Later implementation 14 succeeded under different conditions. | Retain the failure history; use 14 as the implemented continuation. Do not infer why results reversed from unmatched cross-session speeds. Preserve prefix-read priority and scrutinize added GPU submissions (21). |
| 09 | [Predictive expert prefetch](explorations/09-expert-prefetch/NOTES.md) | Parked | Negative feasibility probe | — Speed not measured; coverage probe only | With pinning off, previous-token experts covered **32.8%** of reads; with pinning on, only **5.7% (4 GiB)** / **8.2% (3 GiB, pressure emulation)** of remaining misses were predictable. This measured predictor coverage, **not a prefetch throughput gain/loss**; no prefetch engine was built. | Reject the tested cheap predictors as planned; hot repeats were already pinned. Learned routing remains parked pending a concrete predictor, miss coverage, and target-hardware evidence. Old ~10% upside was an estimate. |
| 10 | [Prefill regression investigation](explorations/10-prefill-regression/NOTES.md) | Closed | Investigation resolved | — No reproducible speed change | Unreproducible dirty-tree prefill **0.86–1.12 s** versus clean-tree checks **1.86–2.94 s** caused a false comparison. ANE engagement was **40/40 chunks, no fallback**. No reproducible regression established; engagement logging and a bisect harness remained. | Do not use those fast historical rows as a regression baseline; preserve them. Sparse short-chunk cost identified here led to 11. No remaining restoration task. |
| 11 | [Hybrid ANE/GPU prefill](explorations/11-prefill-ane-crossover/NOTES.md) | Closed | Successful | **37–39% less prefill time** (65 tokens, GPU)<br>**6.5% less prefill time** (1429 tokens, ANE) | Qwen3.6-35B q4/32 GB: 65-token prefill **2592 ms (ANE)→1640/1573 ms (GPU), 37–39% less time**. At 285 tokens GPU was ~11–13% faster; at 1429 tokens **32596 ms (GPU)→30492 ms (ANE), −6.5%**. Interior 373–637-token differences were within ~5–7% noise. Tested hybrid output matched GPU-only. | Keep the **512-token chunk threshold**. Avoid tuning within the broad noisy crossover band. Revisit for a changed device/workload; long-prefill attention/recurrent work is a separate target. |
| 12 | [Expert pin-cache configuration A/B](explorations/12-pin-default-on/NOTES.md) | Closed | Existing benefit confirmed | **+34.3%** decode; existing benefit confirmed | 150-token medians, 32 GB, fusion on: pin-off **17.20/17.65**, 4 GiB+locking **22.98/23.81 tok/s**; means of block medians **17.425→23.395 (+34.3%)**. 3 GiB+locking **22.82**. Expert I/O **~0.46→0.17–0.21 ms**. | No new optimization or default change: pinning already shipped; local pin-off was personal test state. Do not add this gain to 01/02. The archived slots-with-zero-GB UX question is separate from this completed performance comparison. |
| 13 | [I/O concurrency for K=8](explorations/13-io-threads-k8/NOTES.md) | Closed | Neutral for decode | **Neutral** decode (−0.6%, noise)<br>Prefill: not measured | 4 vs 8 I/O workers, pin off, 150 tokens ×3: **17.42→17.31 tok/s (−0.6%, noise)**; expert I/O **0.462→0.465 ms**. Configurable pool default 8 landed. The tested decode path does not use that pool. | Keep implemented width control; no further decode probe. Prefill benefit is an acknowledged evidence gap, not an open duplicate of 06. Revisit only if a concrete prefill workload shows pool concurrency limiting progress. |
| 14 | [Split expert I/O](explorations/14-expert-read-split/NOTES.md) | Closed | Successful | **+6.5%** decode pin-off / **~+10%** pin-on | Interleaved, fusion and output-weight locking on: pin-off **17.61→18.75 tok/s (+6.5%)**; pin=4 GiB **~23.0→25.23 (~+10%)**. First 50+ generated token IDs matched in tested combinations. All-pin-hit layers bypass the split. | Keep shipped split I/O (`5af6a43`). Prefix reads precede suffix reads; suffix overlaps gate/up computation. Do not revive 08/21b scheduling without preserving that overlap; absolute rates are not comparable across campaigns. |
| 15 | [Output-projection weight residency](explorations/15-lm-head-resident/NOTES.md) | Closed | Successful | **+13.8% / +22.5%** decode (two blocks) | Warm 32 GB, 4 GiB locked expert cache and fusion: output-weight locking off **19.48 tok/s**; on-block medians **22.16 / 23.87 (+13.8% / +22.5%)**. Output projection **2.88→2.66–2.68 ms**; token stream matched. Implemented change also corrected mapped-file access advice. | Keep headroom-guarded residency. Recorded +15–20% was a rounded summary; listed blocks span +14–23%. Revisit if memory competition changes. No real 16 GB result or isolated attribution for each part of the implementation. |
| 16 | [Earlier expert I/O overlap](explorations/16-earlier-io-overlap/NOTES.md) | Superseded | Main approach covered by 14 | — No separate measurement; see 14 | Design review found no useful work between read start and wait on the fused path. The proposed split-read solution shipped as 14; cheap prediction was assessed in 09. **No separate 16 throughput result**; earlier top-K launch and learned prediction remained sketches. | Close this umbrella as a separate backlog item. Consider its residual ideas only with a new dependency/overlap hypothesis; do not conflate them with an unimplemented split-read feature. |
| 17 | [Full-attention command fusion](explorations/17-full-attn-fuse/NOTES.md) | Untested | No measurement | — Not measured | Design: extend command fusion to full-attention layers. Initial code review still finds fusion gated to linear attention. Historical **+3–8% is an estimate**, not a result; CPU position encoding/context append were the dependencies. | If selected, verify current full-attention dependencies, preserve state/operation order, and count removable submissions before an approved bounded measurement. |
| 18 | [Draft copied output from existing text](explorations/18-ngram-spec/NOTES.md) | Untested / deferred | No measurement | — Not measured | Prompt/history lookup could draft copied code with no separate model. No Flashchat performance result; cited external speedups are not evidence here. Client-supplied predicted output is a planning extension. | No current action planned. If selected later, require faithful acceptance/rollback and net savings after verification; fallback cannot guarantee zero overhead. |
| 19 | [Hot-expert co-location on disk](explorations/19-expert-hot-repack/NOTES.md) | Untested | No measurement | — Not measured | Design: co-locate frequently used experts on disk. No layout implementation or throughput result; more sequential I/O is a hypothesis, not an established property of the actual access stream. | If selected, establish benefit for real expert selections before adding repacking/migration complexity. Respect the registry-backed production format; the historical alternate-layout sketch is not implementation approval. |
| 20 | [MTP skip-spec and deeper batches](explorations/20-mtp-skip-spec/NOTES.md) | Untested / gated | No measurement | — Not measured | Follow-up to 04: avoid double verification on low-confidence drafts and try deeper batches. **1.2–1.6x was a projection**, not a result. Current code still contains the margin-gated batched-versus-production verification choice. | Requires a concrete lower-cost fallback, per-position acceptance/cost evidence, and the proposed real low-memory target. Drafting, decisions, and rollback still cost time; no never-slower guarantee. |
| 21 | [Pin reservation and event-gated GPU work](explorations/21-io-event-gating/NOTES.md) | Closed | Mixed | 21a: **+2.5%** decode pin-on; pin-off neutral<br>21b: **−2.7%** pin-on / **−42.3%** pin-off | **21a kept:** final isolated pin=4 GiB medians **24.71→25.32 tok/s (+2.5%)**, all three pairs favored it; pin-off **19.17→19.16**, neutral. **21b rejected:** raw-run medians with pinning **22.17→21.58 (−2.7%)**, pin-off **19.74→11.39 (−42.3%)**. Final tested outputs matched; only 21a shipped. | Keep direct reads into reserved slots; event-gating code removed. Revisit events only with prefix priority/suffix overlap preserved. +2.5% and −2.7% are small historical observations, not precise universal effects. Larger API gains included other changes; see audit notes below. |
| 22 | [GPU tail fusion](explorations/22-gpu-tail-fusion/NOTES.md) | Untested | No measurement | — Not measured | Design from 21: put final normalization/output projection in the final GPU command buffer. No implementation or correctness result recovered. **+1–3% was an estimate**, inside ordinary timing noise. | If selected, establish a removable boundary and correct dependencies. Evaluate submission reduction before interpreting small throughput deltas. |
| 23 | [OpenCode context and pin sizing](explorations/23-opencode-context-pin-sizing/NOTES.md) | Closed | Mixed sizing results | 4 GiB pin: **−35–36%** decode; 9.13 GiB: **~−3–6%**<br>Window sizing: neutral; fp32 vs q8: **~−24%**<br>Replay comparisons have confounders | Real ~16k OpenCode context, q4/32 GiB: evolving requests at **8 GiB pin=8.64/8.63**, **4 GiB=5.52/5.58 tok/s (−36/−35%)**, **9.13 GiB=8.14/8.37**, no win. Warm q8 64k/262k windows **9.82/9.90**, neutral; fp32/64k **7.47**. Sampled output/caps and cache warmth differ; not a lossless matched-output benchmark. | Retain 8 GiB/q8 for this workload; smaller arena thrashed, bigger hit rate did not improve latency. Requested 12 GiB resolved to 9.13, not a true 12 GiB test. Revisit when workload/memory changes; do not generalize to all machines. Cache-format observation is retained below. |
| 24 | [Idle page-cache warmer](explorations/24-idle-page-cache-warmer/NOTES.md) | Untested | No measurement | — Not measured | Design from 21: warm unpinned expert pages during idle time. No implementation or measurement; unlike 09 it targets idle gaps, not prediction during active decoding. | If selected, define bounded idle work with immediate handoff, then test whether useful reuse outweighs eviction/SSD contention. No automatic background work is authorized. |
| 25 | [Recovered wonderment campaign](explorations/25-wonderment-recovered/NOTES.md) | Closed | Historical recovery | Fusion: **~+11.6%** decode<br>Cache bypass: **~−25–45%** sustained decode<br>Recovered results; overlap prior entries | Recovered `a399937`, original notes and **39 API rows**. A 60-slot cache survived **1774 evictions/80 tokens** with matching output. Fusion **9.92/10.40→11.35/11.33 tok/s (~+11.6%)**. Cache bypass looked better in short CLI probes but sustained API fell to **6.1–8.3 vs ~11.0–11.1 tok/s**. Also preserves 08 losses. | Archive only: no unique surviving feature needs a port. Safety landed in `6057f0c`; fusion overlaps 07. Never sum duplicate gains. Revisit bypass only with sustained-request evidence; short-process wins hid lost filesystem reuse. |
| 26 | [Recover exact conversation-state reuse](explorations/26-conversation-state-reuse/NOTES.md) | Closed | Successful | **Observed −40.7% / −53.9% first-token wait** (turns 2/3), **confounded by visible Codex UI**<br>Cold turn: +2.5% | Qwen3.6 q4/q8, 32 GiB, one three-turn off/on comparison with unquantified UI contention: **13.430→7.965 s**, **18.104→8.344 s**; newly prefilled **611→312**, **916→312** tokens. Eleven live output comparisons plus state/config/cancellation checks passed. Standard cases recorded; post-case harness error and historical decode flag retained in notes. | Keep exact single-conversation reuse with a pre-assistant checkpoint, default on; commit approved. Client normalization may require replaying the latest assistant turn. No further benchmark run pending; revisit reproducible correctness/memory issues. Multi-conversation branching remains 27. |
| 27 | Reuse partial prefixes and conversation branches | Untested | No measurement | — Not measured | No measurement. Current memory snapshot is one whole system prefix; multiple conversations, partial prefixes, and branch checkpoints are the proposed extension. | Select checkpoint boundaries and a bounded memory budget. |
| 28 | [Cache prepared prompts, tokens, and schemas](explorations/28-prepared-prompt-cache/NOTES.md) | Closed; commit approved | Preparation gain measured; watchdog flags unresolved | **~3.9 ms saved per 26 KiB tool-prefix hit**; no demonstrated end-to-end gain | 20 paired CPU comparisons per case: 24-tool hits **3.912→0.035 ms**, misses **4.129→1.996 ms**; every output check passed. Standard q4/q8 watchdog retained 13 rows, with mixed decode −12–15% versus an older run; not explained by this probe. | Keep as a modest preparation optimization; Dave approved committing implementation and evidence on 2026-09-22. Preserve watchdog flags for future investigation. |
| 29 | Reduce cache persistence and restoration latency | Untested / design agreed | Snapshot sizes inspected; no optimization result | — No speed measurement | 2026-09-17: one Coder-Next prefix snapshot has 75.4 MiB of entirely zero CPU chunks and 73.7 MiB of zero GPU padding. This is a byte inspection, not demonstrated runtime memory or speed savings. Detailed findings below. | Prioritize actual-size snapshots, explicit zero chunks, and disk-backed saved state with bounded capture/restore workspace. CPU/GPU ownership tracking follows; compressed RAM tiers are deferred. |
| 30 | Recover request and prompt-staging memory | Untested | No measurement | — Not measured | No measurement. Server lacks a per-request autorelease pool and stages full-prompt embeddings before chunk processing. Memory growth and its speed impact remain unmeasured; transient GPU-buffer release already exists. | Audit object lifetimes and chunk-local embeddings. |
| 31 | Constrain tool generation before validation fails | Untested | No measurement | — Not measured | No measurement. Final tool validation plus a bounded repair attempt already exists (`520fb0f`); token-by-token grammar/schema constraints are the new proposal. | Choose supported grammar/schema coverage and correctness cases. |
| 32 | Batch predetermined output structure | Untested | No measurement | — Not measured | No measurement. Forced tool choice already inserts opening markup; skipping prediction of later uniquely determined structure is proposed. Model state must still process those tokens. | Identify forced spans and validate model-state advancement. |
| 33 | Return multiple independent tool calls per turn | Untested | No measurement | — Not measured | No measurement. Generation stops after the first valid call; emitting several independent calls per response requires compatible template and client behavior. | Establish model-template and client support. |
| 34 | Reduce unnecessary generated tokens | Untested | No measurement | — Not measured | No measurement. Reasoning toggles/budgets already exist; finer task policies, concise output and patch-oriented generation are proposed. Fewer tokens alone do not prove faster successful completion. | Evaluate task completion and quality, including retries. |
| 35 | Compact useful input and tool results | Untested | No measurement | — Not measured | No measurement. Client/tool-side result selection and compaction are proposed; they can remove useful information or invalidate cached prefixes. | Preserve information and prefix stability. |
| 36 | Coordinate competing and ancillary requests | Untested | No measurement | — Not measured | No measurement of scheduling extensions. HTTP separation (`5f3fdcd`) and disconnect cancellation (`fccc672`) already ship; queueing, priority and multi-conversation batching remain proposed. | Establish actual concurrent demand and latency priorities. |
| 37 | Reuse completed work after explicit retries | Untested | No measurement | — Not measured | No measurement. Exact completed-response reuse/reconnect is proposed; it is separate from existing cancellation and must not duplicate tool execution. | Define idempotency and delivery semantics. |
| 38 | Parse tool output incrementally | Untested | No measurement | — Not measured | No measurement. Current tool parsing searches accumulated output repeatedly; a stateful parser is proposed for large arguments. | Check large-argument parsing cost. |
| 39 | Reduce sampling and response-assembly overhead | Untested | No measurement | — Not measured | No measurement. CPU sampling, repeated scans and unnecessary streaming-response assembly are candidates; no demonstrated bottleneck or speedup. | Attribute CPU cost before changing algorithms. |
| 40 | Evict caches by reuse value and recomputation cost | Untested | No measurement | — Not measured | No measurement. Proposed retention policy values saved recomputation per byte; memory kept for conversation state can displace expert residency. | Account for competition with expert residency. |
| 41 | Prewarm known prefixes | Untested | No measurement | — Not measured | No measurement. Preparing known prefixes before demand moves work earlier; savings depend on reuse and absence of contention. | Determine when earlier work is likely to be reused. |
| 42 | Audit client rendering and stream buffering | Untested | No measurement | — Not measured | No measurement. Ordinary server text already streams; proxy/client buffering and transcript rendering are integration audit targets, not established causes of delay. | Compare server emission with visible client updates. |
| 43 | Sparse attention or context-state compression | Untested | No measurement | — Not measured | No measurement. Sparse attention/state compression beyond existing context-cache quantization is a research proposal; architecture and task quality constrain it. | Establish quality and architecture compatibility. |
| 44 | Better specialized speculative drafter | Untested | No measurement | — Not measured | No measurement. A specialized drafter beyond the current MTP head is proposed; no compatible artifact or favorable memory/verification budget established here. | Establish artifact compatibility, RAM cost, and verification economics. |

## Evidence audit notes and interactions

These notes preserve qualifications needed to interpret the table without
reconstructing the old campaign. They are an audit of existing evidence, not new
experiments.

- **02 — memory pressure:** the round-matched locking gains are 15.4%, 18.4%, and
  21.4% from the recorded control/locking pairs; the warm gain is 6.3%. The old
  +15–24% headline is broader than that table supports. Locking changed hit latency,
  not hit rate (70.9% at 3 GiB, 76.8% at 4 GiB). More pinned memory did not improve
  the emulated case because it competed with the file cache. Preserve the positive
  locking result, the inconsistent bypass result, and the null frequency-decay
  result separately. Source: [02 notes](explorations/02-ram-discipline/NOTES.md).
- **03 — acceptance of a quality trade-off:** +4.6% is the notes' explicitly
  interleaved pair comparison. Other saved short probes have different rates;
  they are not interchangeable matched controls. The observed read reduction is
  useful evidence, but neither it nor coherent output demonstrates unchanged
  task quality. Source: [03 notes](explorations/03-adaptive-k/NOTES.md).
- **04/20 — speculative economics:** 16.03/21.59 is 0.742x, while the historical
  narrative says ~0.76x. Both indicate a slowdown; the exact pair is retained in
  the register. Skipping verification after paying for a draft still incurs draft
  and decision cost, so the old “never worse” claim is not a guarantee. This also
  applies to 18. Source: [04 notes](explorations/04-mtp-batched-default/NOTES.md).
- **07/12/23 — cache capacity is workload-dependent:** 07's prefill penalty is
  3143/2435−1 = 29.1%, not the old +26% label. The measured 12 short-prompt win
  confirms existing pinning/locking rather than adding a new independent gain.
  In 23, reducing the arena caused 6,920 evictions and only 72.7% hits, while a
  larger 9.13 GiB arena reached 89.8% hits but still failed to beat 8 GiB latency.
  Optimize avoided work and elapsed time, not hit rate alone. Sources:
  [07](explorations/07-cmd-fuse-linear/NOTES.md),
  [12](explorations/12-pin-default-on/NOTES.md),
  [23](explorations/23-opencode-context-pin-sizing/NOTES.md).
- **15 — residency attribution:** saved `Generation:` summaries reproduce
  19.48 off and 22.16/23.87 on-block medians (+13.8%/+22.5%). The original
  +15–20% wording is approximate; the two blocks are retained rather than folded
  into a more precise single claim. The implementation also changed mapped-file
  access advice. Source: [15 notes](explorations/15-lm-head-resident/NOTES.md).
- **21 — final isolated result versus intermediate and combined gains:**
  [final A/B summary](explorations/21-io-event-gating/ab-final-summary.txt)
  supports +2.5% for the shipped pin reservation change, not the earlier +8%
  prototype comparison and not the +18–37% API changes versus an older commit
  containing fewer optimizations. For rejected event gating,
  [saved run summaries](explorations/21-io-event-gating/ab-events-summary.txt)
  give medians 22.17/21.58 with pinning (−2.7%) and 19.74/11.39 without (−42.3%).
  The notes incorrectly selected 22.10 as the first median and reported −4%.
  The large pin-off loss and the competition between read waves remain the key
  negative finding; the small pin-on change is noise-sensitive. No raw runs changed.
  Correctness also exposed a useful constraint: adding a finite-value guard to
  the shared combine kernel changed default-path numerics and generated output.
  A separate guarded variant restored the tested output match. Preserve operation
  order and verify supposedly inactive-path edits; equivalent-looking code was
  not sufficient evidence of identical behavior.
- **23 — long-context replay limitations:** the original 8 GiB server retained
  debug/profiler overhead, candidate sizes ran without it, sampling was 0.7, and
  some calls did not close by the 128-token cap. These are useful operational
  observations, not identical-output performance trials. Repeated-request 9.90
  tok/s must not replace evolving-request 8.63–8.64 as the control. Window capacity
  alone did not determine speed at the same ~16k actual position. A saved q8 prefix
  was rejected for fp32 and rebuilt; returning to q8 accepted the higher-precision
  entry. That is a retained cache-format observation, not a measured optimization
  or proven corruption. Revisit cache conversion only with a concrete compatibility
  question. Sources: [notes](explorations/23-opencode-context-pin-sizing/NOTES.md),
  [result records](explorations/23-opencode-context-pin-sizing/RESULTS.tsv).
- **08/14/21/25 — a failed mechanism is not a failed concept:** early split-read
  implementations lost time; later 14 succeeded; concurrent-wave event gating in
  21 lost again. Preserve useful prefix/suffix scheduling and inspect submission
  boundaries. There is no matched experiment proving one isolated difference
  explains the early-to-late reversal. The 25 archive also shows why a short CLI
  cache-bypass gain can reverse in a sustained API workload.

## Proposal details

### 18 — Draft copied output from existing text

Status: **Untested / deferred.** No current action planned. Planning updated 2026-09-16.

Edits and replacement files often reproduce text already in the prompt. Propose
those spans cheaply, verify them with the model, and generate normally around
changes. Long tool arguments containing source code are a promising workload.

The [original proposal](explorations/18-ngram-spec/NOTES.md) describes prompt lookup.
An explicit predicted-output
input from a cooperating client would extend it beyond opportunistic matching.
[llama.cpp documents text-history draft methods](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

Cheap drafting is not free verification. Streamed-expert verification costs,
rollback, and rejected drafts can erase the benefit or cause a slowdown. Preserve
the intended sampling behavior and measure complete request economics.

### 26 — Recover exact conversation-state reuse

**Functionally validated 2026-09-16; commit approved. Three-turn timings are confounded by visible Codex UI.** [Implementation, measurements, and limitations](explorations/26-conversation-state-reuse/NOTES.md). The initial findings below explain the proposal.

**Important finding:** continual conversation caching existed, but its active
session logic was subsequently removed. Commits `d965514` and `d5e5e71` introduced
and fixed it. Commit `bbaa549` removed the active-session logic during the API
rewrite.

At the initial review, `serve_loop` restores the system/tool snapshot, then
tokenizes and processes the conversation after it again.
`tokenize_continuation_turn` remains marked unused in
[infer.m](../metal_infer/infer.m).

After an answer or tool call, retain the exact token sequence and corresponding
model state. On the next request, reuse its unchanged prefix and process only new
material. An agent loop should usually ingest the latest tool result rather than
all previous searches, file contents, answers, and calls again.

Repeatedly processing a growing transcript can approach quadratic cumulative
prompt work across turns. Processing each addition once is much closer to linear.

Correctness is the central requirement. Hybrid models retain both attention
history and recurrent state. Subsequent request rendering can alter reasoning,
tool-argument order, and assistant-message formatting. Internal correction attempts
may not appear in the client transcript, and the last emitted token may not yet
have been forwarded through the model. A session ID alone is insufficient:
compare actual rendered tokens and restore a valid checkpoint.

**Initial priority: first investigation. This is recovery, not a new invention.**

### 27 — Reuse partial prefixes and conversation branches

The initial implementation holds one in-memory system snapshot. Extend reuse to:

- Stable tool definitions when later system instructions change.
- Repository instructions and other stable context before the changing dialogue.
- Recent conversations, so switching chats does not discard all their progress.
- Shared prefixes before edited messages or regenerated answers.

Start with one active conversation and a few strategic checkpoints. Introduce a
general prefix tree only if the workload justifies it. Recurrent state must be
saved at checkpoint boundaries; it cannot simply be truncated to an arbitrary
earlier token.

Useful references are [llama.cpp server caching](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
and [SGLang session-aware prefix caching](https://docs.sglang.io/docs/advanced_features/session_radix_cache).
Flashchat needs its own memory policy appropriate to unified memory and streamed
experts, rather than a direct transplant of a discrete-GPU design.

### 28 — Make cache hits skip prompt preparation

2026-09-21: implemented on `exp/28-prepared-prompt-cache`; see
[bounds, exact invalidation inputs, and correctness evidence](explorations/28-prepared-prompt-cache/NOTES.md).
2026-09-22: the approved standard watchdog completed without a demonstrated gain
and with unresolved regression flags. The [focused preparation test](explorations/28-prepared-prompt-cache/TEST.md)
then measured roughly **3.9 ms saved per 26 KiB tool-prefix hit** across 20 paired
comparisons, with exact-output checks passing. This is CPU preparation savings,
not a demonstrated whole-request or decode speedup.
The description below records the original opportunity.

Even an in-memory system-cache hit first rebuilds the system/tool prompt and
tokenizes it to count tokens. On a miss, the prefix is tokenized again as part of
the full prompt. Review `count_sys_prompt_tokens`, `tokenize_request_prompt`, and
the cache lookup in `serve_loop`.

Cache the prepared representation alongside the state snapshot:

- Rendered prompt bytes, token IDs, and token count.
- Reusable schema-validation structures.
- Unchanged message tokenization, with correct token-boundary handling.

Stabilize serialization so equivalent schemas do not create accidental misses.
Diagnose changing timestamps, tool order, and client metadata that break otherwise
reusable prefixes. Any normalization must be applied consistently before
inference; differently rendered prompts cannot be declared equivalent afterward.

Invalidate prepared data when its actual inputs change, including tokenizer,
template, tools, and system content.

### 29 — Reduce caching overhead on the answer path

For a cold system prefix, the current server captures a snapshot, compresses it,
writes it, flushes it to disk, and prunes the cache before continuing through the
conversation prompt. Relevant functions include `capture_system_prompt_snapshots`,
`save_system_prompt_disk_cache`, `write_cache_chunk`, and
`restore_system_prompt_snapshots`.

Candidates:

- Keep immediate reuse in memory and defer persistence until idle.
- Persist only prefixes that are reused or otherwise justify their write cost.
- Choose compression according to its actual write/read benefit.
- Avoid repeated copies of immutable attention-prefix data during restoration.

Background writing is not automatically a win: it can compete with expert reads
on the same SSD. Idle-time persistence may be preferable to concurrent writes.
Recurrent state still requires correct restoration, and deferred snapshots need
bounded, explicit ownership.

**The synchronous work is confirmed; the achievable savings are unmeasured.**

#### 2026-09-17 — Snapshot inspection and agreed disk-backed direction

No storage optimization has been implemented or benchmarked. The existing cache
`c8d59b92db36997d-v2.fcache` was expanded into individual chunks and every checksum
verified. It represents a 10,197-token Coder-Next q4 system/tool prefix with q8
attention storage. Its 192 payloads total 344.9 MiB uncompressed, 178.0 MiB stored.
Byte inspection found:

- All 36 CPU convolution chunks and 36 CPU recurrent chunks are entirely zero:
  75.375 MiB uncompressed, already only 55.4 KiB after LZFSE compression.
- The 36 GPU recurrent buffers have 72 MiB of zero tails beyond this model's
  dimensions; GPU convolution buffers add 1.6875 MiB of zero tails.
- All attention and GPU-state chunks contain some nonzero data. These are layer
  state buffers, not expert slots; expert routing does not leave attention layers
  unused. CPU and GPU state ownership can differ in hybrid execution.
- The recurrent-state chunk subset compresses from 224.4 to 71.7 MiB. This is a
  system-prefix observation, not a measured conversation-checkpoint compression ratio.

Current conversation checkpoints allocate approximately 187 MiB for the shipping
35B model, 224.4 MiB for Coder-Next, and 372.7 MiB for the registered 397B model when
CPU and GPU copies exist, plus a token ledger capped at four bytes per context
position. These are calculated allocations, not measured resident memory. They
share live attention history, retain only one checkpoint, and currently have no
memory-budget or pressure-eviction guard; invalidation alone does not free payloads.

Agreed implementation order:

1. Copy actual model dimensions instead of maximum GPU buffer capacity.
2. Represent known-zero chunks explicitly without payload allocations. Restore
   zero markers by clearing destinations, never by leaving previous state intact.
3. Prioritize disk backing: release the expanded system snapshot once safely
   persisted, then use the same storage mechanism for conversation checkpoints.
   Capture and restore one chunk at a time with bounded working memory; avoid a
   full expanded intermediate snapshot. Retain lossless validation and bounded
   disk usage, with recomputation when storage or validation fails.
4. Track authoritative convolution and recurrent state per layer to avoid
   unnecessary capture work. Do not infer ownership from one global GPU flag:
   convolution can run on CPU while recurrence runs on GPU. Do not require this
   tracking to complete before implementing disk backing.

Disk backing releases saved copies, not the live buffers used each token. The
current conversation checkpoint remains dependent on live attention history and
does not become restart-persistent merely by saving recurrent state to disk.
Compressed RAM tiers are deferred until real disk-backed usage provides evidence
for their value. Observe capture/restore overhead, reuse, workspace peaks, and
storage sizes without duplicating routine performance-watchdog runs. Correctness
checks must include restoration into nonzero destinations, execution-path handling,
and exact state comparisons. The 16 GiB target remains unvalidated.

### 30 — Recover request and prompt-staging memory

Two concrete candidates emerged:

1. **Per-request temporary-object lifetime.** The serving loop lacks a per-request
   Objective-C autorelease pool. Temporary objects can survive until the outer
   pool drains, despite automatic reference counting. Audit accumulation across
   requests and establish suitable draining boundaries.
2. **Chunk-local embeddings.** Prefill materializes embeddings for the entire
   prompt in `serve_embed_batch`, although model processing is chunked. Embedding
   only the current chunk would bound this staging allocation.

Also audit duplicate CPU/GPU state where one authoritative shared allocation
could suffice. The expected benefit is potentially indirect: more resident expert
data, less compression or paging, and steadier performance after long sessions.

These proposals are distinct from the prefill-buffer release work already shipped.
The memory impact still needs measurement. See
[Apple's autorelease-pool guidance](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html).

### 31 — Prevent malformed tool calls during generation

Today the server generates a call, validates it, and may spend another generation
attempt correcting it. Constrained decoding could disallow tokens that violate
the supported tool structure or schema before the call is complete.

Potentially avoid malformed syntax, unavailable function names, invalid enum
values, and some missing-field retries. Preserve final validation and the existing
correction path for violations beyond the decoder's supported constraints. A
syntactically valid command can still be the wrong command for the task.

[XGrammar supports mixed reasoning, text, and Qwen tool-call structures](https://xgrammar.mlc.ai/docs/latest/structural_tag/tool_calling_and_reasoning.html)
and provides an implementation reference. Evaluate time to a successful call,
including avoided retries and any masking overhead.

### 32 — Batch predetermined output structure

When the grammar permits only one continuation, process its fixed text together
instead of predicting punctuation one token at a time. Candidates include closing
tags, separators, and schema-determined text.

This is called jump-forward decoding.
[XGrammar exposes a method for finding forced strings](https://xgrammar.mlc.ai/docs/latest/api/python/grammar_matcher.html).
Those tokens still have to update model state: replace sequential generation with
batched processing, not omitted computation. Tokenization boundaries need care.

Forced tool choice already inserts the opening function prefix. The new proposal
extends that principle through the remaining structured output without inventing
semantic argument values.

### 33 — Return multiple independent tool calls per turn

The current generation loop stops after the first valid parsed tool call. Allowing
several independent calls in one response could eliminate entire model round
trips. Three independent file reads could be requested together and, with a
compatible client, executed concurrently before returning their results together.

The savings include repeated prompt processing, generation setup, and model
decision-making—even when the tools themselves are fast.

Establish model-template and client compatibility. Preserve dependencies between
calls and validate every call before emission. This is an API capability proposal;
Flashchat need not take ownership of tool execution.

### 34 — Reduce unnecessary generated tokens

Reasoning on/off already exists. Explore finer, visible policies for different
request types:

- Brief reasoning for routine dispatch and larger budgets for difficult decisions.
- Concise tool follow-ups rather than repeated narration of completed work.
- Patches or targeted replacements rather than complete file regeneration.
- Explicit stopping conditions for structured tasks.

Evaluate elapsed time to a correct completed task, including retries. A shorter
answer that causes a clarification or failed edit is slower overall. Conversely,
more initial reasoning—or a slower but more capable model—may finish sooner if it
avoids several failed attempts. These are quality-sensitive options, not silent
changes to default behavior.

### 35 — Keep input and tool results useful and compact

This primarily belongs in client/tool integration but directly affects API cost.
Prefer targeted file ranges, bounded search results, concise command summaries,
and retrievable references to large outputs. Deduplicate repeated tool results.

Compact old history when expected future savings exceed the cost of generating
and processing the summary. Rewriting early context destroys prefix reuse, so
stable append-only context with infrequent compaction may beat aggressive
shortening on every request.

Prompt caching avoids recomputing old context. Context reduction can additionally
reduce attention work during generation. Summarization and relevance filtering
change the information available and therefore require task-quality evaluation.

### 36 — Coordinate competing and ancillary requests

The dedicated HTTP thread and disconnect cancellation are already implemented.
Generation currently accepts one request at a time and rejects competitors with
`503`. See [SERVER.md](SERVER.md).

Extensions include a bounded queue with deadlines, interactive priority over
background summaries, cache-aware request selection, and cancellation of queued
work.

Continuous batching processes tokens from several conversations together and
might share weight reads. It can also worsen individual latency and consume more
state memory. Require evidence of concurrent demand before taking on that
architecture.

Handle ancillary tasks deliberately. The existing fixed-title shortcut shows
title generation has already interfered with real work. Deterministic handling or
a suitable small model could avoid occupying the main inference path, provided
its memory footprint does not hurt the primary model.

### 37 — Avoid repeating completed work after retries

An explicit idempotency key could let a client retrieve a completed response
instead of generating it again after a delivery failure. A bounded replay buffer
could support reconnecting where the client protocol permits it.

Cancellation saves abandoned work; idempotency avoids redoing completed work.
Start with exact identity and explicit retry intent. Similar-looking requests are
not necessarily interchangeable, and replay must not cause duplicate tool
execution.

At the application layer, unchanged file reads and deterministic tool results can
also be reused with appropriate version checks.

### 38 — Incremental tool parsing

Process each byte once instead of searching the accumulated buffer repeatedly. Most relevant to large tool arguments.

### 39 — Cheaper sampling and response assembly

Reduce CPU token selection, repeated string scans, and unnecessary final JSON construction for streaming chat. Attribute cost first; likely smaller than avoided model work.

### 40 — Cache eviction by recomputation cost per byte

Preserve expensive, frequently reused prefixes. Account for competition with expert residency.

### 41 — Prewarm known prefixes

Prepare a known system/tool prefix before the first interactive request. Moves work earlier; wastes work if unused.

### 42 — Client rendering and stream buffering

Reduce delay between server output and visible updates. Ordinary text already streams; inspect proxies, stream consumption, and transcript rendering.

### 43 — Sparse attention or context-state compression

Reduce long-context generation cost. Changes model behavior; substantial quality and architecture validation needed.

### 44 — Better specialized draft model

Improve speculative decoding beyond the current MTP head. Compatible artifacts, memory headroom, and favorable verification costs required.

## Related implementation history without separate new IDs

- System/tool prompt caching is implemented.
- Conversation caching was implemented, then its active-session path was removed
  during the API rewrite; 26 is recovery with stronger matching requirements.
- HTTP-thread separation and disconnect cancellation are implemented.
- Prefill transient-buffer release is implemented.
- Forced tool choice already inserts an opening function prefix.
- TensorOps prefill, GPU tail fusion, idle warming, and MTP follow-ups remain
  available at their documented stages. Some are designs or partially explored;
  others have negative measurements. Consult the
  [exploration archive](explorations/README.md) before proposing a repeat.

## Initial priorities — recorded 2026-09-16

1. Investigate 26 conversation-cache recovery and exact checkpoint semantics.
2. Investigate 28 prepared-prompt reuse and 30 memory lifetimes.
3. Assess 29 cache persistence/restoration costs alongside that work.
4. Investigate 31 constrained tool generation and 33 multiple calls per turn.
5. Select subsequent work using end-to-end evidence rather than decode throughput
   alone; 27 and 32 build naturally on the preceding capabilities.

Cost models should rank opportunities, not veto cheap falsifiable probes. Keep
experiments bounded and retain negative or inconclusive results.

## End-to-end validation framework

Keep canonical throughput checks, and add a small end-to-end workload set:

- Growing multi-turn conversations.
- Repeated tool loops.
- Alternating conversations.
- Edited history and regeneration.
- Large tool arguments.
- Explicit retries and delivery failures.

Measure first visible answer, first valid tool call, total task completion,
reprocessed prompt tokens, correction attempts, and memory growth. Attribute
request preparation, cache restoration, tokenization, prefill, reasoning,
generation, validation/retry, and delivery separately.

At the initial review, the canonical benchmark correctly ignores immediate role
headers but counts reasoning deltas as first output. A better time to first token
can therefore leave the user waiting just as long for an answer. The internal
request timer starts after cache restoration and tokenization, so those costs
also need separate attribution. See [bench_api.sh](../tests/bench_api.sh) and
`t_prefill` in [infer.m](../metal_infer/infer.m).

Follow [AGENTS.md](../AGENTS.md) when starting an experiment: obtain approval for
the specific benchmark scope, protect existing workloads, use the required quiet
desktop and health preflight, and record configuration and limitations. This
inventory does not authorize benchmarking or constitute performance sign-off.

## Register change history

- **2026-09-16 — Initial end-to-end review:** recorded 20 candidate descriptions.
  No benchmarks run.
- **2026-09-16 — Permanent-number migration:** imported legacy 00–25 and reserved
  new IDs 26–44. Updated archive entry points to
  consult this register before allocating IDs. Legacy notes and raw results remain
  in their original locations; no experiment was rerun or promoted by this migration.
- **2026-09-16 — Status-based organization:** grouped unattempted proposals
  together regardless of ID or age, separated selection from test status, and
  removed the special migration call-outs. IDs and historical evidence are unchanged.
- **2026-09-16 — Unified register:** consolidated all 45 entries into one table
  in permanent ID order, with an explicit outcome column. Successful, unsuccessful,
  mixed, neutral, untested, and reference entries stay together as their evidence
  evolves. Replaced the tested/untested split and the additional-proposals table;
  retained all descriptions and evidence links.
- **2026-09-16 — Reference audit:** reviewed every row against available notes,
  checked selected raw-run medians and arithmetic, separated status from outcome,
  and replaced vague wins/mixed labels with results, conditions, component
  decisions, and revisit criteria. Marked 06 and the main 16 proposal superseded,
  and 05 reviewed/parked. Corrected summary arithmetic without altering raw logs
  or the performance watchdog. Unmeasured dimensions remain explicit evidence gaps.
- **2026-09-16 — Speed-change scan column:** added compact measured gains/losses
  for every entry, distinguishing decode throughput from prefill elapsed time,
  neutral findings from missing measurements, and retained components from failed
  variants. Kept the detailed evidence and all permanent IDs unchanged.
- **2026-09-16 — Experiment 26 validated:** restored exact conversation reuse, recorded observed 40.7%/53.9% lower continuation first-token wait (confounded by visible Codex UI), functional results, and benchmark limitations in the exploration notes. Commit subsequently approved after the desktop-state correction.

- **2026-09-16 — Experiment 26 measurement correction:** Dave reported desktop restoration after the initial standard run, leaving Codex UI visible during both three-turn runs. Annotated all six conversation rows as confounded; preserved measurements, token-reuse evidence, and correctness results. Timing impact is unquantified. No reruns requested or performed.
