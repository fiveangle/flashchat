# 56 — Upstream flash-moe campaign (pre-fork)

Date: 2026-03-16 to 2026-03-19 (upstream). Registered 2026-09-22.
Author of the original work: **Dan Woods**, upstream project `danveloper/flash-moe`.
Status: **Closed.** Historical recovery; no experiment was rerun.

## Provenance

Flashchat is a fork of `danveloper/flash-moe`. The 144 upstream commits in this
repository's history, all authored by Dan Woods, end at `3601d41` (2026-03-19). The
fork's own history begins at `b0865d8` (2026-03-26).

Upstream kept a detailed experiment record: a 58-experiment log, an I/O and GPU
investigation, two I/O plans, a results table, and a paper describing a
90-experiment trajectory. The fork deleted all of it on 2026-06-12 in `71fdcb7`
(model-management re-architecture), and nothing in the exploration archive
referenced it afterwards. It is restored here because it records measured
results, many of them negative, for mechanisms the fork has since proposed again.

The files under [`original/`](original/) are **byte-identical** to their upstream
Git blobs. Nothing in them was edited, including the attribution, which is recorded
here instead so the originals stay verifiable:

| Restored file | Source | Blob | Removed by |
|---|---|---|---|
| [original/docs/optimization-experiments-q4.md](original/docs/optimization-experiments-q4.md) | `3601d41` | `dbe6f2c970` | `71fdcb7` (fork) |
| [original/docs/io-and-gpu-exploration.md](original/docs/io-and-gpu-exploration.md) | `3601d41` | `fe2761dd28` | `71fdcb7` (fork) |
| [original/docs/plan-io-experiments.md](original/docs/plan-io-experiments.md) | `3601d41` | `be2610eba4` | `71fdcb7` (fork) |
| [original/docs/plan-async-pread-pipeline.md](original/docs/plan-async-pread-pipeline.md) | `3601d41` | `703099ffdb` | `71fdcb7` (fork) |
| [original/results.tsv](original/results.tsv) | `3601d41` | `0eeed6ccab` | `71fdcb7` (fork) |
| [original/metal_infer/results.tsv](original/metal_infer/results.tsv) | `3601d41` | `86cc254b1c` | `9fd8337` (fork) |
| [original/paper/flash_moe.tex](original/paper/flash_moe.tex) | `3601d41` | `022079f56e` | `71fdcb7` (fork) |
| [original/design_fast_expert_io.md](original/design_fast_expert_io.md) | `18d0a27^` | `2fc4b1bb8e` | `18d0a27` (upstream) |

`metal_infer/results.tsv` is restored as of `3601d41` because the fork edited it
once before deleting it; the restored copy is upstream-only. The other six
`docs/` and top-level files had no fork edits between `3601d41` and their removal.
`design_fast_expert_io.md` is an early Python-era design that upstream itself
retired in `18d0a27`. The paper's compiled PDF and auxiliary files, `progress.py`,
`progress.png`, and `train_predictor.py` are not restored; they remain in history
at `3601d41`.

No license file exists anywhere in this repository's history. These files are
restored at the repository owner's direction as part of the fork's own history,
with credit to Dan Woods.

To verify a restored file: `git hash-object <file>` must equal the blob above.

## Transfer caveats

Upstream measured a different model on different hardware, so none of these
numbers is a baseline for the fork:

- Model: Qwen3.5-397B-A17B, 4-bit and later 2-bit experts of ~3.9–7 MB each, K=4
  (pruned from the default 10). The fork's first-class target is Qwen3.6-35B-A3B
  q4, K=8, with much smaller experts.
- Hardware: M3 Max, 48 GB. The fork's measurements are on a 32 GB Mac17,2.
- Many rows ran on a single fixed prompt ("tea", "ocean") without repeats, and some
  predate the C/Metal engine (MLX and Python stages).
- Upstream's absolute tok/s is not comparable to any fork row.

A negative result here is strong evidence that the *mechanism* needs a specific
reason to behave differently on the fork's model and hardware before it is tried
again. It is not proof that it cannot.

## Upstream experiments mapped to the register

"Source" abbreviations: **Q4** = `optimization-experiments-q4.md`, **IO** =
`io-and-gpu-exploration.md`, **RT** = `results.tsv`, **Paper** = `flash_moe.tex`,
**Pipe** = `plan-async-pread-pipeline.md`. Figures are upstream's own.

### Kept upstream (still present in the fork's engine)

| Upstream experiment | Upstream result | Source | Register |
|---|---|---|---|
| Parallel pread + async GPU pipeline | peak 5.24 tok/s; TTFT 37% faster | `3e14484` | 06, 13 |
| Persistent expert pread pool instead of `dispatch_apply` | 3.97→3.81 ms/layer; 4.09→4.28 tok/s | RT | 06, 13 |
| Trust the OS: remove the Metal LRU cache and `F_NOCACHE` | 4.11→5.74 tok/s (+38%); compressor decompressions 60–130K/s → ~0 | `912762b`, Paper | 01, 02, 12 |
| FMA dequant kernel | 3.90→4.36 tok/s (commit says +12%; the log's own heading says +2.6%) | `3601d41`, Q4 | 05 |
| BLAS (Accelerate) delta-net | 0.28 vs 0.78 ms | `bf01c81` | none; inherited |
| Fused GPU delta-net by default | 3.90 vs 4.15 ms/layer, identical output | RT | none; inherited |
| GPU RoPE + fused CMD1+attention on full-attention layers | 5.53→4.76 ms/layer | `bab9956`, RT | 17 |
| GPU RMSNorm + residual + fused command buffers | peak 5.29 tok/s | `67d67ab` | 07 |
| GPU combine+norm in CMD3 | deferred wait 0.83→0 ms, but absorbed into cmd1_wait 0.49→1.78 ms; kept as a cleaner pipeline | RT | 07, 22 |
| System-prompt caching with per-request state restore | ~6 s TTFT saved per request | `630b631` | 57 |
| Session caching in the chat binary | second-turn prefill of new tokens only; ~98% of a 10-turn context not reprocessed (claim) | Paper | 26 |
| Batch prefill via `discard_deferred_experts()` | TTFT 5.6→2.6 s on a 16-token prompt | `3253ef2`, Paper | 55 (different mechanism) |
| 2 MB-aligned expert DMA buffers | +5% expert I/O | `672c81b` | 14 |
| C tokenizer | startup 3500→180 ms | Paper | none; inherited |

Upstream's batch prefill skipped expert work for every prompt token but the last,
so later layers saw hidden states without those tokens' expert contribution. That
is a different, approximate mechanism from 55's chunked batched prefill, which
computes every token and was greedy byte-matched against per-token prefill.

### Kept upstream, later reversed

| Upstream experiment | Upstream result | Source | Register |
|---|---|---|---|
| Tiered I/O: cold reads bypass the cache | +4% | `a8bbe9d` | 02 |
| `F_NOCACHE` with 2-bit experts | 5.68 tok/s (+3%), then 1% slower than trusting the OS (5.68 vs 5.74) | `5994c6b`, Paper | 02 |
| Indexed 2500-entry Metal expert cache (2-bit) | 3.98→4.35 tok/s vs 1500 entries; 3500+ regressed; removed by trust-the-OS | RT | 01, 12 |
| Fixed K=4 expert pruning (default 10) | 1.20 (K=10) → 1.91 (K=6) → 3.15 (K=4) tok/s, MLX stage; no observed loss at K=4 | Paper | 03 |
| 2-bit expert requantization | 5.55–5.74 tok/s sustained; reverted because tool-call JSON broke (quotes became backslashes) | RT, Q4 | none; AGENTS.md forbids reintroducing alternate low-precision runtime paths |

### Discarded upstream

| Upstream experiment | Upstream result | Source | Register |
|---|---|---|---|
| `mmap` expert files | 0.56 tok/s, 5x slower; ~244 page faults per 3.9 MB read | RT, Paper | none |
| Metal LRU cache at 1000–3000 entries | 2.24–1.99 tok/s vs 3.14 at 500 entries | IO | 01, 12 |
| Malloc zero-copy cache, 500 / 2581 entries | 2.77 / 2.10 tok/s | RT | 01 |
| Expert LRU cache (`--cache-entries`) | 44% hit rate, allocation overhead negated the gain | `bf4684a` | 01 |
| Speculative early routing | 53% accurate; 1.70 vs 2.75 tok/s (−38%) through cache pollution | `f2ed109`, RT | 09 |
| Temporal expert prediction + double-buffered prefetch | 25.6% hit; 3.18 vs 3.90 tok/s | Q4 | 09 |
| MLP routing predictor | 31% accurate, worse than the temporal baseline | Q4 | 09 |
| Two-pass overlapped routing | 6.0 tok/s but repetitive output: routing depends on updated hidden states | Paper | 09, 16 |
| Variable K in middle layers | K=0 blank output; K=2 or 3 collapsed; every layer needed K ≥ 4 | Paper | 03 |
| `F_RDADVISE` read hints | no improvement, added syscall overhead; as a prefetch during GPU compute, expert I/O −31% but cmd2_wait +73%, net 0% | `f2ed109`, Q4 | 06, 09, 16, 24 |
| `MADV_SEQUENTIAL` / `MADV_RANDOM` / disabling `F_RDAHEAD` | no effect / marginally worse / no effect | Paper | 02, 15 |
| Within-layer pread overlap | no improvement; only a 0.1 ms overlap window | Pipe | 16 |
| LZ4 expert compression (209→175 GB) | 3.55 tok/s (the log's −13%); LZFSE too slow; APFS transparent compression 2x slower | Q4 | none |
| One file per expert | 15% slower from VFS metadata overhead | Q4 | 19 |
| Expert file clustering by co-occurrence | 0%: scatter distance did not matter at 7 MB reads | Q4, RT | 19 |
| `dispatch_io` | −70% from `dispatch_data` overhead | Q4, RT | 06, 13 |
| `aio_read` | −7% | Q4 | 06, 13 |
| GPU private-buffer compression | isolated matvec −13.5%, but the blit made the pipeline −20% | Q4, RT | 05 |
| LUT dequant kernel (v5) | −2%: indirect register access serializes | Q4, RT | 05 |
| Vector-load dequant kernel (v4) | −3% from register pressure | Q4 | 05 |
| `extract_bits` intrinsic | neutral | Q4 | 05 |
| Spin-poll GPU wait | −23%: CPU spinning takes GPU thermal budget | Q4, RT | none |
| `addCompletedHandler` | neutral in practice | Q4 | none |
| Single encoder per expert (gate→up→SwiGLU→down) | 2% slower: less cross-expert parallelism | IO | 07, 22 |
| Fused gate+up+SwiGLU shader | slower; GPU preferred separate pipelined dispatches | `d9e91f3` | 07, 22 |
| Removing the CMD2→CPU→CMD3 bounce | 3.70→4.07 ms/layer; 4.40→3.99 tok/s | RT | 07 |
| GPU softmax + top-K routing | 3.84→3.96 ms/layer; 4.24→4.13 tok/s | RT | none |
| Fused GPU final-norm + lm_head tail | 3.93→4.01 ms/layer; 4.13→4.04 tok/s | RT | 22 |
| `MTLSharedEvent` async CMD1 / deferred-completion overlap | reverted: event overhead exceeded the savings | `1b7a2bb` | 21 |
| GPU attention scores at short context | reverted as too slow for short contexts | RT | 17 |
| Early GPU delta-net state | 195 MB per-layer buffers caused memory pressure; a later fused GPU delta-net was kept | Paper | none |
| C-extension pread alongside MLX (Python stage) | 18.8 GB/s standalone, 2.6 GB/s under GIL contention | Paper | none |

### Analysis without an implementation

| Upstream analysis | Finding | Source | Register |
|---|---|---|---|
| MTP speculative decoding | not implemented: ~1.75x expert I/O for ~1.7 tokens at 70% acceptance, "break-even at best" for SSD-streamed MoE | Q4 | 04, 20, 44 |
| Unified-memory constraint | SSD DMA and GPU compute share the memory controller, so background I/O during GPU compute costs more than it saves | Q4 | 09, 16, 24 |
| `preadv` versus `pread` (Python stage) | 14.1 vs 5.3 GB/s standalone | `design_fast_expert_io.md` | 06 |

## Conclusion

Keep as an archival assignment, like 25. No experiment was rerun, and no unique
surviving feature needs a port: every upstream keep listed above is still present
in the engine and therefore inside the fork's 00 baseline. The value is the
negative record. Before proposing an expert cache, prefetcher, I/O hint, file
layout, compression scheme, kernel variant, command-buffer fusion, or lower-bit
expert format, check the tables above and state why the fork's model and hardware
would change the upstream outcome.
