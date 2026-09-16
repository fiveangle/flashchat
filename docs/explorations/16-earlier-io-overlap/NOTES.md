# 16 — Earlier expert I/O overlap with GPU

Date: 2026-08-24  
Status: **NOT STARTED**

## Hypothesis

Comments claim async pread overlaps “shared expert prep + next layer CMD1”.
Live decode path (`fused_layer_forward` ~9549):

1. top-K known
2. `async_pread_start`
3. optional memcpy of h_post / shared gate-up (skipped when GPU-resident)
4. `async_pread_wait` immediately
5. encode CMD3

On the fuse + GPU-resident path, step 3 is empty → **zero overlap**.
Expert I/O and GPU remain serial per layer.

## Options to create a real window

A. **Start miss preads as each expert index is selected** inside top-K loop
   (micro-gain: first experts start earlier by ~router tail time).

B. **Split-read (14)** — real overlap of down I/O with gate+up GPU.

C. **Speculative prev-set prefetch** — killed by exp/09 (5.7–8.2% ceiling
   with pin on). Do not rebuild.

D. **Learned probe prefetch** — parked in exp/09 (~10% 16GB upside).

E. **Double-buffer set B** — only helps if L+1 experts known early; needs
   predictor. main.m has the machinery for synthetic forwards only.

## Method

Profile with `FLASHCHAT_MTP_OVERLAP=1` + timing: measure
`async_pread_start → wait` gap today (expect <0.05 ms fused). Implement A
and/or land 14; remeasure gap and tok/s.

## Verdict

Pending. Prefer landing 14 over clever prefetch.
