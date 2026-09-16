# 00 — Baseline capture

Date: 2026-08-17
Machine: 32GB Apple Silicon (Mac17,2), model warm in page cache.
Config: user config as shipped — pin cache OFF (`EXPERT_PIN_MAX_GB=0`),
MTP=0, KV q8, batch prefill on, K=8, q4 variant.

## Hypothesis (cost model)

Qwen3.6-35B-A3B q4: 40 layers x 256 experts, K=8, expert = 1,769,472 B
(1.6875 MiB, exactly 432 pages) -> 540 MB expert reads/token. Non-expert
weights touched per token ~1.27 GB (incl. 286 MB lm_head). Per-layer decode
phases are strictly serial: CMD2 wait -> route -> expert pread wait -> CMD3.

On 16GB the same 1.8 GB/token working set can't be cached -> SSD streaming
dominates -> ~7-9 tok/s observed. Bytes/token is the wall; overlap is free
speed on top.

## Method

- `tools/quick_decode_bench.sh docs/explorations/00-baseline 150 3`
- `bash tests/bench_api.sh --model-id Qwen-Qwen36-35B-A3B` (durable rows in
  assets/api_perf_log.tsv, commit 9fd7c83)

## Observations

quick harness (150 tokens, 3 runs):

```
run1: decode=16.42 tok/s ttft=2696ms   (page cache still warming)
run2: decode=17.33 tok/s ttft=1841ms
run3: decode=17.37 tok/s ttft=1837ms
MEDIAN decode: 17.33 tok/s
layer-phase medians (ms): cmd1_wait=0.619 cmd2_wait=0.326 expert_io=0.375
                          cmd1_submit=0.008 cmd3_encode=0.021 total_layer=1.364
```

- 1.32 ms of 1.364 ms/layer is (cmd1_wait + cmd2_wait + expert_io) — strictly
  serial GPU/I/O phases. GPU compute per layer ~0.95ms, expert I/O ~0.38ms
  (page-cache warm; on 16GB cold this becomes ~13ms/layer at 5GB/s for
  experts alone).
- Short runs (24 tokens) understate steady state (11.9 tok/s) — always use
  >=150 tokens.

## Verdict

Baseline established: **17.33 tok/s median** (quick harness) on 32GB warm.
All paths compared against this number plus bench_api rows.
