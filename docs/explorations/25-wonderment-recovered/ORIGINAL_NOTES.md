# 10 — Bounded unlikely experiments

Branch: `codex/decode-speed-wonderment`
Date: 2026-08-17

## Kept

- Zero-copy pin slots selected by the current layer are excluded from LFU
  eviction until that layer's misses have been admitted. A deliberately tiny
  60-slot cache forced 1,774 evictions during an 80-token run and remained
  byte-identical to pinning off.
- Lossless CMD1+CMD2 fusion is now automatic whenever its existing runtime
  invariants hold. Interleaved 100-token runs: off 9.92/10.40 tok/s, fused
  11.35/11.33 tok/s (about +11.6%).

## Rejected: `F_NOCACHE` expert streams

Short CLI runs looked promising. With a 3 GiB pin cache, normal expert reads
ran at 16.85/18.51 tok/s while `F_NOCACHE` reached 21.17/21.26 tok/s. Without
pinning, normal reads ran at 18.22/18.70 and `F_NOCACHE` at 19.70/19.69.
Outputs were byte-identical.

The canonical API benchmark reversed the result. `F_NOCACHE` produced mostly
6.1–8.3 tok/s, while the immediately following cached control produced about
11.0–11.1 tok/s in five of six decode scenarios. Bypassing the filesystem
cache helps the short process-local case but destroys useful expert reuse in
sustained server generation. The code was removed. Perf-log rows from
13:10–13:15 UTC are `F_NOCACHE=1`; rows from 13:16–13:20 UTC are the cached
control with automatic fusion.

## Rejected: split expert reads

The packed expert layout has a page-aligned 1.125 MiB gate/up prefix and a
0.5625 MiB down suffix. A prototype read them into separate Metal buffers and
overlapped the suffix read with gate/up GPU work.

- Concurrent fragment dispatch: 10.81 off vs 9.97 on.
- Staged prefix, then suffix concurrent with GPU: off 11.10/11.08 vs split
  10.52/10.48 tok/s.

Both schedules were byte-identical. The staged version reduced the loss, but
the extra Metal command-buffer boundary cost more than the hidden suffix I/O.
The prototype and its extra buffers were removed.

## Follow-up

Split reads are only worth revisiting if Metal shared events, a single-kernel
path, or another mechanism removes the extra submission boundary. Preserve
normal filesystem caching and focus next on reducing command boundaries,
improving explicit pin-cache policy, or eliminating bytes rather than globally
bypassing reuse.
