# 11 — Hybrid ANE/GPU prefill (sparse-slab crossover)

Branch: `exp/11-prefill-ane-crossover` (from main @ 05f9884). Started because
exp/10's phase data showed the ANE pipeline paying huge fixed costs on sparse
chunks — and prompted by the correction that prefill matters at EVERY prompt
length (each conversation turn re-prefills the whole history; only the system
prompt is snapshot-cached).

## Root cause (from exp/10 engagement logging)

The ANE path runs one <=256-row slab per unique routed expert. At short
chunk lengths, rows-per-expert is tiny (65-token chunk: ~4,900 rows across
~4,200 experts ≈ 4.9 rows/expert), so each eval is ~98% padding. Per-expert
fixed costs (GPU producer round-trip + ANE slab eval) dominate: at 65
tokens, ANE producer+eval = 2,218ms for the same 20,480 rows the GPU GEMMs
in 440ms.

## Measurements (Qwen3.6-35B q4, 32GB Mac17,2; fresh server, 3rd request;
# gemm/io = GPU-path expert phases; producer+eval = ANE expert phases)

| prompt tokens | GPU prefill | ANE prefill | winner | expert-phase ms (GPU vs ANE) |
|---|---|---|---|---|
| 65   | 1,640 / 1,573 | 2,592 | GPU −37% | 440 vs 2,218 |
| 285  | 5,952 / 6,240 | 6,722 / 7,215 | GPU −11..13% | ~1,580 vs ~3,580 |
| 725  | 15,678 / 15,728 | 14,860 / 14,995 | ANE −5% | 3,794 vs 4,304 |
| 1,429| 32,596 | 30,492 | ANE −6.5% | 7,450 vs 7,812 |

Why ANE wins long despite slower expert phase: producer (GPU) overlaps eval
(ANE), freeing the GPU for the batched attention/delta work that dominates
at length. Crossover ≈ 500–600 tokens per chunk.

## Change

`ane_min_chunk()` gate (default 512, `FLASHCHAT_ANE_MIN_CHUNK`, 0 = old
always-ANE behavior) checked BEFORE `ane_prefill_ready()` so short-prompt
sessions never pay ANE context compilation (~130–230ms). Chunks below the
threshold run the GPU path — which is also bit-faithful (int8 ANE
quantization no longer touches short prompts). Engagement status extended
with "short-chunk-gpu".

Hybrid verification (default env): 65→1,573ms GPU-path, 285→GPU-path,
725→14,881 ANE-path, 1,429→30,390 with a 40/80 ANE+GPU split across chunks
(full 1024-token chunks → ANE, 405-token tail → GPU). Temp-0 output:
hybrid default ≡ GPU-only (bit-identical).

## Config wiring (schema v11)

`ANE_MIN_CHUNK` through all 8 config.sh sites + flashchat bridge/signature/
status + wizard entry (with corrected ANE help text — "~25% faster" was
length-dependent) + serve banner hybrid line. Verified: isolated-HOME v10→
v11 migration, override 0, `config --show`, cli-smoke, 120 py-tests,
mtp-config-smoke.

## Durable bench (rows appended 2026-08-23)

technical 2,030ms / creative 1,858ms / worstcase 2,378ms TTFT, decode
16.3–18.8 tok/s, tool_call pass — no regression. Honest caveat: the bench's
warm requests re-serve the same short prompt with a snapshot-restored system
prompt, so TTFT there is dominated by fixed costs (restore+tokenize+first
token); the hybrid's prefill win (~400ms on those very requests) is real
but partially masked. The user-visible win is first-turn cold starts
(−40% TTFT at short prompts) and every long-prompt turn.

## Follow-ups (not blocking)

- 16GB-hardware validation of the 512 threshold (crossover could shift with
  slower SSD — expert io is a larger fraction there, shared by both paths).
- The bigger long-prefill cost center is now non-expert phases (~60% at
  725+ tokens: batched attention + delta-net) — separate project if prefill
  at length ever needs another push.
