# 54 — Context-cache quantization

Date: 2026-06-22. Registered 2026-09-22.
Commits: `54d9447` (implementation), `58c28f1` (config-wizard menu, 2026-06-23).
Status: **Closed.** Shipped before it had an exploration ID.
Verdict: **Keep.** The saving is memory, not tokens/sec.

## Hypothesis

The full-attention context cache was fp32. Quantizing it the way MTPLX does cuts
its RAM enough to matter on 16 GB machines, without changing greedy output.

## What shipped

`FLASHCHAT_KV_QUANT=off|q8|q4`, ported from MTPLX. Symmetric per-(token, kv_head)
quantization with one fp16 scale per `head_dim` vector; q4 packs two nibbles per
byte at a +8 offset. Quantize-on-write at the production append site, dequant-on-read
attention pipelines (`attn_scores_quant`, `attn_values_quant`), and a
precision-aware prompt cache whose snapshots carry scales (disk format v2).

MTP was excluded on purpose: the commit records that MTP buffers stay fp32 and adds
a fatal guard making `--mtp-*` mutually exclusive with KV quant, because the two
share fp32 buffers. That exclusion is proposal 48.

## Measurement

The implementing commit's own figures, on Qwen3.6-35B-A3B: context cache **~671 MB
→ ~170 MB (q8) / ~86 MB (q4) at 8K context**, with the default off byte-identical to
before. Not an independent A/B, and not a speed measurement.

## Conclusion

Keep. The 00 baseline runs a q8 context cache, so this saving is already counted
and is not available as further headroom. 43 covers compression beyond it; 48
covers the MTP path it excluded.
