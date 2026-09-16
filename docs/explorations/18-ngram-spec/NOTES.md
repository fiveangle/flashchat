# 18 — Prompt-lookup / n-gram speculative decoding

Date: 2026-08-24  
Status: **NOT STARTED** (never attempted in this repo)

## Hypothesis

Draft candidate tokens from recent context (n-gram / prompt suffix match)
with **zero** extra model weights. Verify with one forward (optionally
batched union-expert path already in MTP code). High acceptance on code,
JSON, repetitive chat; low on open-ended prose.

Literature: prompt lookup decoding (PLD), REST, token recycling — often
1.5–2.5× on suitable workloads with vanilla transformers. MoE changes the
economics (verify still streams experts) but union I/O helps when drafts
share experts with the ground-truth path.

## Method

1. Maintain rolling suffix map of last N tokens → next-token candidates.
2. Draft up to B tokens; run batched verify (reuse `fused_batched_forward_N`
   / MTP verify infrastructure).
3. Accept longest correct prefix; never worse than plain decode if
   draft-fail → single forward (same skip-spec lesson as MTP).
4. Bench on code-heavy vs creative prompts separately.

## Risks

- Engineering couples to MTP verify path (good reuse, high complexity).
- Workload variance — must not regress creative chat.

## Verdict

Pending. Attractive because no draft model RAM on 16GB.
