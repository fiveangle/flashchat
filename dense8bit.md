# 8-bit DENSE inference bug (Qwen3.6-27B-8bit) — debugging handoff

> **OUTDATED (2026-06-09). Root cause found and it is NOT a kernel bug.** The
> 28.58GB q8 file exceeds Metal's `maxBufferLength`, so the whole-file
> `newBufferWithBytesNoCopy` returned nil and batched GPU paths encoded against
> a nil buffer → structured-but-wrong output. Proof: `FLASHCHAT_NO_BATCHED_PREFILL=1`
> produced the correct " Paris". The x_shared[4096] suspicion below is moot.
>
> **Do NOT attempt to wrap the file as segmented Metal windows**: Metal wires
> referenced buffers fully resident (not demand-paged) — wiring ~28.6GB on a
> 32GB machine hard-crashed the OS twice. Current fix: all GPU dispatch sites
> resolve weight pointers via `metal_weight_arg()` and fall back to CPU when no
> Metal buffer covers them (correct but slow); windows are opt-in
> (`FLASHCHAT_WF_WINDOWS=1`) and refused outright when file+4GB > RAM.
> q8-27B fundamentally cannot be GPU-resident on this 32GB box — use q4.

> Untracked scratch doc. Everything we learned chasing the gibberish output of
> the 8-bit **dense** 27B. The config refactor (commit `8a42fa5`) appears unrelated.

## TL;DR

The **27B-8bit** (dense, `hidden_size=5120`) produces coherent-but-wrong output
(foreign-language / dictionary text) for any prompt. Every other model works.
The bug is in the engine's **8-bit GPU matvec path that is only exercised when
`hidden_size > 4096`** — and the 27B is the only model with `hidden > 4096`.

## Symptom

Prompt `hi` →

```
inglesa
=== Forma adjetiva ===
1: femenino.
=== Etimología ===
=== gl ===
1 leng=gl:.
```

and on another run `晦气` ("bad luck"). It is **structured, grammatically-valid
text** (Wiktionary-style markup, real Spanish/Galician words) that is completely
unrelated to the prompt. That "coherent but wrong" signature ⇒ small *systematic*
numerical error in the forward pass derailing the logits into a high-probability
attractor (dictionary text is extremely common in training data), NOT gross
corruption / random token noise.

Also slow, much slower than previously. It was getting ~2.8t/s previously, now generates just a few words in 20+min.

## The decisive isolation matrix (owner-confirmed)

| | 4-bit | 8-bit |
|------|-------|-------|
| **MoE** (35B-A3B hidden=2048, 80B hidden=2048) | ✅ | ✅ |
| **dense** (27B hidden=5120) | ✅ | ❌ gibberish |

So the bug is in code that is **both dense-specific AND 8-bit-specific AND
hidden>4096**. The discriminator turned out to be **`hidden_size`**, not
dense-vs-MoE per se:

- `hidden ≤ 4096` (35B=2048, 80B=2048): single-token matvecs use `matvec_v3`.
- `hidden > 4096` (27B=5120): single-token matvecs route to **`matmulN`**
  (for gate/up via `gpu_batch_matvec`) and **`matvec_v3` with an oversized
  `in_dim`** (for QKV / linear `in_proj` via `fast_batch_matvec`).

27B-**4bit** uses the 4-bit versions of those kernels and works; 27B-**8bit**
uses the 8-bit versions and breaks.

## What has been RULED OUT (with evidence)

1. **The config refactor (`8a42fa5`).** The 4-bit 27B runs correctly through the
   exact same `--config`/`--model` launch path. Config is bit-agnostic.
2. **The q8 weights file.**
   - Structure: `manifest_end == file_size` exactly (no truncation).
   - Packing: q8 `down_proj.weight` is `[5120, 4352]` vs q4 `[5120, 2176]` — exactly
     2× columns ⇒ correctly 8-bit packed (4 values/u32 vs 8).
   - **Values**: dequantized q8 ≈ dequantized q4 (max abs diff ~0.0016 over 32
     values, mean magnitude ~0.010; q8 is the *finer* quantization, as expected).
   - ⇒ **Re-extracting will NOT help** (deterministic script → same correct bytes).
3. **`matmulN_8bit_v5`.** DISPROVEN by experiment: `FLASHCHAT_NO_MATVEC_MM=1`
   reroutes gate/up off `matmulN` onto `matvec_fast`, and the gibberish
   **persisted**. (Flag confirmed exported via `env | grep -i flash`.)
   NOTE: that flag only affects `gpu_batch_matvec`/`gpu_encode_batch_matvec`
   (gate/up). It does **not** touch `fast_batch_matvec` (QKV / linear in_proj),
   which is why the test wasn't a full exoneration of the v3 path.
4. **Bit-aware paths confirmed correct (kernel *selection* is bit-aware):**
   `embed_lookup` (branches on bits), `lm_head_forward` (→ `fast_dequant_matvec`,
   bit-aware), CPU `fast_dequant_matvec`, `gpu_batch_matvec` and
   `fast_batch_matvec` (both via the bit-aware `*_pipe` selectors), o_proj
   (uses `matvec_fast_pipe` after an earlier 8-bit fix — see its comment).

## CURRENT PRIME SUSPECT

**`dequant_matvec_8bit_v3`** running at **`in_dim = hidden = 5120`**, used by
**QKV (full-attn layers) and linear `in_proj` (48/64 layers)** via
`fast_batch_matvec` — the path `FLASHCHAT_NO_MATVEC_MM` does NOT reroute.

Specific smell: these v3 kernels cache the input in `threadgroup float
x_shared[4096]` and then do `for (i=lid; i<in_dim; i+=256) x_shared[i]=x[i];`.
At `in_dim=5120` that **writes (and later reads) x_shared[4096..5119] out of
bounds**. The memory log even records this class of bug being hit before
("matmul2 x_shared[4096] overflowed on 27B hidden=5120 — silent on 35B").

### The unresolved puzzle (READ THIS)

The 4-bit `dequant_matvec_4bit_v3` has the **identical** `x_shared[4096]` and the
**identical** overflow at `in_dim=5120`, yet the 27B-**4bit** runs coherently.
Static reading did **not** explain why the same overflow breaks 8-bit but not
4-bit. Hypotheses to test:
- The OOB write/read is "self-consistent" (writes x[4096..5119] into adjacent
  threadgroup memory, reads it back from the same place) and benign for 4-bit,
  but the Metal compiler lays out threadgroup memory differently for the 8-bit
  kernel so the OOB region clobbers something live.
- 8-bit's tighter per-element error budget tips an *already-marginal* result.
- The real defect is elsewhere in `matvec8_v3` (not the overflow) and only
  manifests at this dim — the kernels *looked* structurally symmetric on read
  (`packed_cols=in_dim/4`, `group_size/4`, `&0xFF`, `x_base=col*4`) so it's subtle.

Because reading didn't resolve it, **the next step must be measurement, not more
code reading.**

## NEXT STEPS (in priority order)

1. **Cheapest decisive test — kill the x_shared overflow.** Temporarily make the
   v3 kernels safe for `in_dim>4096`, rebuild, run 27B-8bit:
   - Option A: bump `threadgroup float x_shared[4096]` → `[8192]` in
     `dequant_matvec_4bit_v3`, `dequant_matvec_8bit_v3` (and the matmul2/matmulN
     v3/v5 kernels that also have `x_shared[4096]`), rebuild, test.
   - Option B: make `fast_batch_matvec` route `in_dim>4096` to `matvec_fast`
     (which has **no** x_shared — reads x directly) the way `gpu_batch_matvec`
     already does, rebuild, test.
   - If either makes 27B-8bit coherent → the overflow IS the bug; then figure out
     the 4-bit/8-bit asymmetry for the real root-cause writeup, and ship the fix
     (enlarge x_shared or route >4096 away from v3 everywhere).
2. **If overflow isn't it — per-layer numerical diff.** Instrument ONE 27B-8bit
   forward: dump each op's output (or per-layer hidden) and compare against a CPU
   reference dequant (known-correct, bit-aware). Find the FIRST op that diverges
   → pins the exact kernel. The engine already has MTP "verify"/trace debug
   scaffolding (`--mtp-verify-*`, `g_mtp_trace_*`, `mtp_trace_vector`) that could
   be adapted, or add a temporary per-layer `vec_rms`/argmax dump.
3. **Isolated kernel harness.** Feed a known (weight, scales, biases, x) to
   `dequant_matvec_8bit_v3` vs a CPU dequant of the same, compare — removes the
   whole-model noise.

## CODE MAP (grep by name; line numbers drift)

`metal_infer/shaders.metal`
- `dequant_matvec_4bit_v3`, `dequant_matvec_8bit_v3` — **suspect**; `x_shared[4096]`,
  overflow at in_dim=5120.
- `dequant_matvec_8bit_fast` — no x_shared, reads x directly (safe; this is what
  `NO_MATVEC_MM=1` routes gate/up to).
- `dequant_matmulN_4bit_v5`, `dequant_matmulN_8bit_v5` — matmulN (gate/up for
  hidden>4096); 8-bit one ruled out by the NO_MATVEC_MM test.
- `fused_gate_up_swiglu` — 4-bit hardcoded but **UNUSED** (no dispatch); ignore.

`metal_infer/infer.m`
- `dense_mlp_forward` — gate/up via `gpu_batch_matvec`, down via CPU
  `fast_dequant_matvec`. (dense-only path.)
- `fast_batch_matvec` — **always** `matvec_v3_pipe`; NO `in_dim>4096` guard; does
  NOT honor `FLASHCHAT_NO_MATVEC_MM`. Used by QKV and linear `in_proj`.
- `gpu_batch_matvec` / `gpu_encode_batch_matvec` — `use_v3` (in_dim≤4096) /
  `use_mm` (matmulN, in_dim>4096) / `matvec_fast`; honor `FLASHCHAT_NO_MATVEC_MM`.
- `matvec_v3_pipe`, `matvec_fast_pipe`, `matmuln_pipe` — bit-aware kernel selectors
  (`g_cfg.bits==8 ? *_8_* : *`).
- o_proj dispatch — uses `matvec_fast_pipe` (bit-aware; comment notes the prior
  8-bit fix where it had been hardcoded 4-bit).
- 8-bit pipeline creation + the `bits==8` required-pipeline guard (search
  `matmulN_8_v5`, `Required 8-bit Metal pipeline missing`).

## REPRO

Model: `Qwen-Qwen36-27B-8bit`; weights at
`~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/flashchat/q8/`

```bash
# WARNING: 27 GB on 32 GB RAM -> thrashes. Stop other servers / free RAM first.
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9

# broken (8-bit, hidden=5120):
FLASHCHAT_MODEL_PATH=$SNAP ./metal_infer/infer \
  --model-id Qwen-Qwen36-27B-8bit --model "$SNAP" \
  --prompt "The capital of France is" --tokens 8 --temperature 0

# baseline that WORKS (4-bit, same snapshot, q4):  expect "... Paris"
FLASHCHAT_MODEL_PATH=$SNAP ./metal_infer/infer \
  --model-id Qwen-Qwen36-27B --model "$SNAP" \
  --prompt "The capital of France is" --tokens 8 --temperature 0
```

## Weight-value verification script (proves q8 extraction is correct)

```python
import json, struct, numpy as np
SNAP="<.../flashchat>"
def bf16(u16): return struct.unpack('<f', struct.pack('<I', int(u16)<<16))[0]
def load(q,t): m=json.load(open(f'{SNAP}/{q}/model_weights.json')); return m.get('tensors',m)[t], f'{SNAP}/{q}/model_weights.bin'
def deq_row0(q, base, bits, gsz=64, ncols=32):
    wi,bn=load(q,base+'.weight'); si,_=load(q,base+'.scales'); bi,_=load(q,base+'.biases'); vpw=32//bits
    with open(bn,'rb') as f:
        f.seek(wi['offset']); w=np.frombuffer(f.read(4*(ncols//vpw)),np.uint32)
        f.seek(si['offset']); s=np.frombuffer(f.read(2*4),np.uint16)
        f.seek(bi['offset']); b=np.frombuffer(f.read(2*4),np.uint16)
    out=[]
    for c in range(ncols):
        g=c//gsz; val=(int(w[c//vpw])>>((c%vpw)*(32//vpw)))&((1<<bits)-1)
        out.append(val*bf16(s[g])+bf16(b[g]))
    return np.array(out)
base='model.layers.0.mlp.down_proj'
a=deq_row0('q4',base,4); b=deq_row0('q8',base,8)
print('max abs diff:', float(np.max(np.abs(a-b))))   # ~0.0016 -> q8 values correct
```

## Open questions to resolve for the writeup
- Why does the `x_shared[4096]` overflow at in_dim=5120 break 8-bit but not 4-bit?
  (If overflow turns out to be the cause.)
- Is the bug in QKV, in_proj, or both? (Both use `fast_batch_matvec`.)
- Final fix shape: enlarge x_shared to cover max in_dim, OR route `in_dim>4096`
  away from the x_shared kernels everywhere (consistent with `gpu_batch_matvec`).
