I reviewed the live \[metal\_infer/infer.m]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m) path, with the commit you named as context. I did not run benchmarks; these are code-review findings and experiment candidates.

**Most Promising**

1. **Release or shrink prefill-only GPU buffers before decode.**\
   Batched prefill sizes several persistent `MTLBuffer`s from `FLASHCHAT_PREFILL_CHUNK`, including delta batch buffers, full-attn batch buffers, `buf_output`, static matmulN buffers, and CSR MoE buffers. They survive into decode even though decode mostly needs single-token or small `N<=8` capacity. This looks like a real RAM-recovery opportunity, likely hundreds of MB at `chunk=1024`, possibly more depending on routed CSR rows. Reclaiming it could directly feed OS expert page cache or a larger pin cache. Relevant areas: \[infer.m (line 2513)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:2513), \[infer.m (line 2842)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:2842), \[infer.m (line 2995)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:2995), \[infer.m (line 5670)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:5670).

2. **Quantize or window the MTP KV cache.**\
   Production KV supports fp32/q8/q4, but the MTP path still allocates fp32 K/V arrays and fp32 MTL buffers at full `GPU_KV_SEQ` when MTP is active. If MTP is on, this can erase some KV-quant RAM savings. A safer first step may be “MTP KV context window” smaller than production, then q8, then q4 if acceptance holds. Relevant areas: \[infer.m (line 517)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:517), \[infer.m (line 1265)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:1265), \[infer.m (line 2890)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:2890).

3. **Avoid full MTP KV memcpy on GPU attention.**\
   `mtp_try_gpu_attention` copies `ctx_len * kv_dim * sizeof(float)` into dedicated GPU buffers before attention. If GPU MTP attention becomes useful, this becomes O(context) host memcpy per draft. Incrementally mirroring appended MTP K/V into the GPU buffer, like production `kv_append_token`, would avoid that. Relevant area: \[infer.m (line 2668)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:2668).

4. **Make expert pin cache explicitly slot-based, not only GB-based.**\
   Internally it already floors GB to whole expert slots, so your instinct maps cleanly to the code. A user-facing `EXPERT_PIN_SLOTS` or `EXPERT_PIN_EXPERTS` would be more honest and predictable than “4 GB” when the actual result is “N whole experts.” The current cache logs slots after resolving, but config is still GB/fraction based. Relevant areas: \[infer.m (line 7142)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:7142), \[modelmgr/tui/config\_wizard.py (line 413)]\(/Users/speedster/dev/flashchat/modelmgr/tui/config\_wizard.py:413).

**Worth Exploring**\
5\. **Phase-aware pin-cache admission.**\
The LFU signal is cumulative `g_expert_freq`, and decode/prefill share the same counters. Long prefill could bias the cache toward prompt-routing experts that are less useful during generation. Consider resetting, decaying, or separately weighting admission once decode begins, especially since `g_overlap_in_decode` already distinguishes phases. Relevant areas: \[infer.m (line 9070)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:9070), \[infer.m (line 7221)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:7221).

6. **Do not hold the pin-cache mutex while copying entire experts.**\
   `expert_pin_lookup` copies the whole expert under `g_pin_mu`. With multi-expert async reads, pinned hits serialize on a large memcpy. A slot-refcount or copy-after-unlock scheme could reduce CPU-side contention. Relevant area: \[infer.m (line 7206)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:7206).

7. **Use pinned cache in chunked prefill too, or consciously avoid it.**\
   Decode uses `async_pread_start` with pin lookup/admit; chunked prefill’s `parallel_pread_experts_into` does plain preads. If the same experts appear in prompt and decode, prefill is currently warming only the OS cache, not the explicit pin arena. This might be intentional to avoid polluting decode cache, but it is a good A/B knob. Relevant areas: \[infer.m (line 5670)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:5670), \[infer.m (line 7487)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:7487).

8. **Reduce per-layer prefill heap churn.**\
   `batched_layer_forward_N` still allocates many large float buffers per layer/chunk (`res`, `nmd`, `qp`, `kk`, `vv`, `ao`, linear-attn temporaries). Reusable chunk scratch would not reduce steady-state RSS if kept forever, but paired with “free after prefill” it could reduce allocator churn and make the lifetime explicit. Relevant area: \[infer.m (line 5157)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:5157).

9. **Try bf16/fp16 staging for selected prefill intermediates.**\
   Many chunk intermediates are float32 even when model weights are fp16-ish/q4. The safest candidates are staging outputs that are immediately consumed by GPU/ANE kernels and not used for routing thresholds: ANE producer inputs/outputs, CSR MoE output accumulation, maybe `q_gate`/attention output staging. Routing logits/top-k and recurrent delta state are riskier.

10. **Precompute RoPE inv-frequencies.**\
    CPU K RoPE still recomputes `powf/cosf/sinf` per full-attn token/head. There is already a GPU RoPE table for Q, but K still does CPU scalar math before KV append. A persistent inv-freq table would remove repeated `powf`; not huge, but cheap and low risk. Relevant area: \[infer.m (line 4438)]\(/Users/speedster/dev/flashchat/metal\_infer/infer.m:4438).

**Suggested Experiment Order**

1. Add memory logging around prefill end: resident bytes for prefill staging, CSR buffers, matmulN statics, MTP KV, pin slots.
2. Implement “shrink prefill buffers to decode capacity after prompt” behind an env/config flag.
3. Add `EXPERT_PIN_SLOTS` as an override and log resolved slot count in config/serve.
4. A/B phase-aware pin admission with `make bench-api --model-id <id>` plus `make bench-report`.
5. Test q8 MTP KV or reduced MTP KV window only after acceptance and losslessness gates are easy to compare.

The biggest clean bet is dynamic prefill buffer reclamation. It matches your architecture note exactly: large RAM used for prefill, then mostly idle during generation, where RAM is more valuable as expert cache.


With MTP opt-in and effectively unused, I would remove MTP from the near-term optimization target list. It becomes a side branch: worth keeping correct, but not worth spending scarce RAM or engineering time on unless you decide to revive speculative decoding.

My updated recommendation:

**Highest Priority**

1. **Make prefill memory temporary.**\
   The most promising default-path win is still to treat large batched-prefill buffers as “setup workspace,” not permanent engine state. Prefill wants big chunk buffers; decode mostly wants expert cache and filesystem cache. After prompt processing finishes, shrink or release prefill-only Metal/CPU scratch buffers, then let that memory go back to the OS or use it to grow the expert pin cache.

2. **Change expert pin cache sizing from GB-first to expert-slot-first.**\
   Experts are fixed-size chunks, so “4 GB” is a fuzzy control. “Pin 512 experts” or “pin N experts per layer / total” is much clearer and maps to the hardware behavior better. Keep GB and free-RAM fraction as safety caps, but make the resolved unit whole expert slots.

3. **Make the pin cache decode-aware.**\
   Long prefill can train the cache on prompt experts, but the important latency path is token generation. I would either reset/decay expert frequency counters at the start of decode, or weight decode hits more heavily than prefill hits. This avoids spending pinned RAM on experts that were only hot during prompt ingestion.

**Medium Priority**\
4\. **Use freed prefill memory to resize the pin cache dynamically.**\
A nice architecture would be:

* During prefill: large prefill workspace, smaller expert pin cache.
* After prefill: release workspace, expand expert pin cache by whole expert slots.
* On next long prompt: optionally shrink pin cache back down if prefill needs room.

5. **Reduce lock hold time in the pin cache.**\
   Right now a cache hit copies a whole expert while holding the cache mutex. That can serialize otherwise parallel expert fetches. A slot reservation/refcount approach would let threads copy outside the lock.

6. **Clean up prefill heap churn.**\
   Chunked prefill allocates many large float32 temporary arrays per layer/chunk. Reusing a dedicated prefill scratch object would reduce allocator overhead and make later “free after prefill” much easier.

**Lower Priority**\
7\. **MTP memory improvements.**\
Since MTP is opt-in and currently not part of the normal fast path, I would not prioritize MTP KV quantization, MTP GPU KV mirroring, or MTP resident expert changes right now.

In short: I would focus on a phase-based memory model. Prefill is a bulk-processing phase; decode is an expert-streaming phase. The engine should aggressively hand memory from the first phase to the second. That matches Flashchat’s unusual bottleneck: once generation starts, every extra usable GB is probably more valuable as expert residency or OS file cache than as idle prefill workspace.
