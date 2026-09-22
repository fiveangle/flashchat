# Design: `fast_expert_io` C Extension Module

## Problem Statement

The 397B-A17B model inference on M3 Max (48GB) is bottlenecked by expert weight
I/O. Current numbers:

- **234 cache misses/token** (at 61% hit rate, 600 expert activations/token)
- **~7.1 MB per expert** (3 projections x 3 attributes: weight/scales/biases)
- **~1.74 GB I/O per token**
- **Current throughput: ~1.4 GB/s** (Python mmap, single-threaded seek+read)
- **Current speed: 1.32 tok/s** (best result with 28GB LRU cache)

The Python I/O path (`_read_single_expert_attrs` in `stream_infer.py`) does:
1. Per-expert: 9 individual seek+read calls (3 projections x 3 attrs)
2. Coalescing within 64KB gap, but still per-expert threading
3. `ThreadPoolExecutor` with 4 workers per layer
4. `np.frombuffer` + `mx.array()` conversion per tensor
5. Python GIL contention between I/O threads and array construction

## Measured I/O Capabilities (M3 Max, 1TB SSD)

Benchmarked with 7MB expert-sized reads against the 397B safetensors files:

| Method | Threads | Throughput | Latency/read | Notes |
|--------|---------|-----------|-------------|-------|
| `pread` | 1 | 5.3 GB/s | 1,353 us | F_NOCACHE, random |
| `pread` | 1 | 7.7 GB/s | 960 us | Normal (page cache assists) |
| `preadv` 3-buf | 1 | 14.1 GB/s | 520 us | Scatter-gather, single syscall |
| `pread` | 2 | 13.4 GB/s | 547 us | F_NOCACHE, multi-file |
| `pread` | 4 | 20.5 GB/s | 358 us | F_NOCACHE, multi-file |
| `pread` | 8 | 35.4 GB/s | 207 us | F_NOCACHE, multi-file |
| `pread` | 4 | 27.8 GB/s | 264 us | page-cache warm, same file |
| `pread` | 8 | 43.2 GB/s | 170 us | page-cache warm, same file |

Key observations:
- **preadv is 2.6x faster than pread** for the same data (14.1 vs 5.3 GB/s) because
  it eliminates per-buffer syscall overhead for multi-tensor reads
- **Multi-threading scales near-linearly** up to 8 threads on NVMe
- **Cold reads from SSD**: ~5.3 GB/s single-thread, ~20 GB/s at 4 threads
- **Warm reads (page cache)**: ~7.7 GB/s single-thread, ~43 GB/s at 8 threads
- **macOS page size is 16KB** (ARM64), which is the DMA alignment boundary
- **Hardware spec: 17.5 GB/s sequential** (measured independently via `dd`)

## Performance Targets

Current I/O path: ~1.4 GB/s effective (Python overhead + GIL + single-threaded)

| Target | I/O BW | Projected tok/s | Speedup | How |
|--------|--------|----------------|---------|-----|
| Baseline | 1.4 GB/s | 1.32 | 1.0x | Current Python mmap |
| Conservative | 5.0 GB/s | 1.82 | 1.4x | C preadv, single thread |
| Realistic | 10.0 GB/s | 2.67 | 2.0x | C preadv + 4 pthreads |
| Aggressive | 14.0 GB/s | 3.08 | 2.3x | C preadv + readahead + async pipeline |
| Theoretical | 17.5 GB/s | 3.34 | 2.5x | Saturate SSD bandwidth |

(Assuming 200ms compute per token, 234 cache misses, 1.74 GB I/O per token)

---

## Architecture

```
                          Python (stream_infer.py)
                                  |
                     fast_expert_io.batch_read()
                     fast_expert_io.start_prefetch()
                     fast_expert_io.get_prefetch()
                                  |
                          ========|========
                          | C Extension  |
                          |              |
                    +-----+------+-------+----+
                    |            |             |
              ReadScheduler  IOWorkerPool  StagingBuffer
                    |            |             |
                    v            v             v
              Sort+Coalesce  pthreads     mmap/malloc
              ReadAhead      preadv()     page-aligned
              Align to 16KB  per-fd       zero-copy to MLX
                          ========|========
                                  |
                              NVMe SSD
```

### Three-Layer Design

**Layer 1: Python API** -- Thin CPython module with 4 functions.
Called from `generate_offload_selective()` in `stream_infer.py`.

**Layer 2: Read Scheduler** -- Sorts reads by file+offset, coalesces adjacent
reads, inserts readahead entries, aligns to page boundaries. Pure C, no I/O.

**Layer 3: I/O Worker Pool** -- Persistent pthread pool that executes the
scheduled reads via `preadv()`. Writes results into pre-allocated staging
buffers. Signals completion via pthread condition variable.

---

## API Surface

### 1. Module Initialization

```python
import fast_expert_io

# Called once at startup. Creates the worker pool and pre-allocates buffers.
fast_expert_io.init(
    num_workers=4,           # pthread pool size
    staging_buffer_mb=512,   # total staging buffer for readahead
    page_size=16384,         # macOS ARM64 page size
)
```

**C signature:**
```c
static PyObject* feio_init(PyObject* self, PyObject* args, PyObject* kwargs);
```

What it does:
- Spawns `num_workers` pthreads, each pinned to an efficiency core (via
  `pthread_set_qos_class_np(QOS_CLASS_UTILITY)` -- keeps I/O threads off
  performance cores that GPU/compute needs)
- Allocates a staging buffer pool: N page-aligned buffers of ~8MB each
  (one per in-flight expert read), obtained via `posix_memalign()`
- Opens and caches file descriptors for all safetensors files (passed in
  via a subsequent `register_files()` call or lazily on first use)

### 2. Register Safetensors Files

```python
# Called once at startup. Pre-parses headers and caches FDs.
fast_expert_io.register_files({
    "/path/to/model-00001.safetensors": {
        "tensor_name": (byte_offset, byte_length, dtype, shape),
        ...
    },
    ...
})
```

**C signature:**
```c
static PyObject* feio_register_files(PyObject* self, PyObject* args);
```

What it does:
- Opens each file with `open(path, O_RDONLY)`
- Sets `fcntl(fd, F_NOCACHE, 0)` -- we WANT the kernel page cache here,
  because our LRU cache operates at the expert-tensor level, and the kernel
  page cache provides a second-level cache for free. F_NOCACHE only makes
  sense if we fully manage our own caching and want to avoid polluting the
  page cache with one-shot reads. Since experts repeat, page cache helps.
  (Revisit this: F_NOCACHE might help for the 397B model where working set
  far exceeds physical memory and page cache thrashing hurts.)
- Stores the tensor metadata (offset, length, dtype, shape tuple) in a C
  hash table keyed by (file_index, tensor_name) for O(1) lookup
- Computes per-expert byte offset and stride for each fused tensor
  (shape[0] = num_experts, stride = product(shape[1:]) * dtype_size)

### 3. Batch Read (Synchronous)

```python
# Called once per layer per token. Reads all cache-missed experts in one batch.
# Returns a dict: expert_id -> {"gate_proj.weight": memoryview, ...}
results = fast_expert_io.batch_read(
    read_specs=[
        # (file_path, tensor_name, expert_index)
        ("/path/to/model-00005.safetensors",
         "language_model.model.layers.7.mlp.switch_mlp.gate_proj.weight", 42),
        ("/path/to/model-00005.safetensors",
         "language_model.model.layers.7.mlp.switch_mlp.gate_proj.scales", 42),
        # ... all 9 attrs for all missed experts in this layer
    ],
    readahead=2,    # also read experts +1, +2 adjacent in the fused tensor
)
```

**C signature:**
```c
static PyObject* feio_batch_read(PyObject* self, PyObject* args, PyObject* kwargs);
```

What it does internally:

**Phase 1: Schedule** (single thread, pure computation)
1. For each `(file, tensor, expert_idx)` in `read_specs`:
   - Look up the tensor metadata from the registered file table
   - Compute `offset = tensor_data_start + expert_idx * expert_stride`
   - Compute `length = expert_stride`
   - If `readahead > 0`, also enqueue reads for experts at indices
     `expert_idx+1`, `expert_idx+2`, etc. (clamped to `num_experts-1`),
     but ONLY if those adjacent indices aren't already in the read_specs
     or in the Python-side LRU cache (passed as a set)
2. Group reads by file descriptor
3. Within each file, sort reads by offset (ascending)
4. Coalesce reads whose gaps are <= 64KB into single larger reads:
   - If expert N and expert N+1 are both needed, they're adjacent in the
     fused tensor, so reading them as one 2x-sized `preadv` is free
   - Readahead experts are almost always adjacent, making this very effective
5. Align each read's start offset DOWN to the nearest 16KB page boundary
   and extend the end offset UP to the next 16KB boundary
6. Assign each coalesced read to a worker thread (round-robin by file to
   spread I/O across NVMe channels)

**Phase 2: Execute** (parallel, `num_workers` pthreads)
1. Each worker picks reads from its assigned queue
2. For each read: `preadv(fd, iov, iovcnt, aligned_offset)`
   - `iov` points into pre-allocated staging buffers
   - Single syscall reads all coalesced data into one contiguous buffer
3. Worker signals completion via `pthread_cond_signal`

**Phase 3: Unpack** (single thread, back in Python with GIL)
1. For each completed read, extract individual expert tensor data from
   the coalesced buffer using the known offsets
2. Return as Python dict of `{expert_id: {attr_key: memoryview}}` where
   each `memoryview` points directly into the staging buffer (zero-copy)
3. Caller (`stream_infer.py`) converts `memoryview` -> `mx.array()` and
   inserts into the ExpertCache

**Return value:**
```python
{
    42: {
        "gate_proj.weight": <memoryview of staging buffer, shape-aware>,
        "gate_proj.scales": <memoryview>,
        "gate_proj.biases": <memoryview>,
        "up_proj.weight": <memoryview>,
        # ... 9 total
    },
    # readahead experts also included if readahead > 0:
    43: { ... },
    44: { ... },
}
```

### 4. Async Prefetch (Pipeline Layer N+1 I/O During Layer N Compute)

```python
# Start reading experts for the NEXT layer while GPU computes current layer.
# Non-blocking -- returns immediately.
fast_expert_io.start_prefetch(
    read_specs=[...],  # same format as batch_read
    readahead=2,
    tag="layer_8",     # identifier for retrieval
)

# ... GPU computes layer 7 ...

# Retrieve prefetched data (blocks until I/O completes if not done yet).
results = fast_expert_io.get_prefetch(tag="layer_8")
```

**C signatures:**
```c
static PyObject* feio_start_prefetch(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* feio_get_prefetch(PyObject* self, PyObject* args);
```

What `start_prefetch` does:
1. Same scheduling as `batch_read` Phase 1
2. Dispatches to worker pool
3. Returns immediately (releases GIL during dispatch)
4. Workers execute reads in background

What `get_prefetch` does:
1. If workers are done: return results immediately
2. If workers are still running: release GIL, `pthread_cond_wait` until done
3. Unpack and return same format as `batch_read`

**Critical for the pipeline**: While GPU executes `mx.gather_qmm` for layer N
(~3-5ms compute per layer), we can be reading expert weights for layer N+1 from
SSD. This overlaps I/O with compute. At 60 layers, this means 59 layers get
free I/O overlap.

### 5. Cleanup

```python
fast_expert_io.shutdown()  # Join worker threads, close FDs, free buffers
```

---

## Internal Architecture

### Data Structures

```c
/* ---- File Registry ---- */
typedef struct {
    int fd;                     // open file descriptor
    char path[PATH_MAX];        // for error messages
    int nocache;                // whether F_NOCACHE is set
} RegisteredFile;

/* ---- Tensor Metadata (per fused tensor in a safetensors file) ---- */
typedef struct {
    int file_idx;               // index into RegisteredFile array
    off_t data_offset;          // absolute byte offset to start of tensor data
    int num_experts;            // shape[0] of the fused tensor
    size_t expert_stride;       // bytes per expert slice
    int expert_shape[4];        // shape[1:] (up to 3 dims, padded)
    int expert_ndim;            // number of dims in expert_shape
    int dtype;                  // enum: F16, BF16, U32, U8, etc.
    int elem_size;              // bytes per element
} TensorMeta;

/* ---- Single Read Request ---- */
typedef struct {
    int file_idx;
    off_t offset;               // page-aligned start offset
    size_t length;              // page-aligned length
    size_t true_offset;         // actual data start within the read buffer
    size_t true_length;         // actual data length
    int expert_idx;             // which expert this is for
    int tensor_id;              // index into TensorMeta array
    int is_readahead;           // 1 if this is a speculative readahead
    void* dest_buffer;          // pointer into staging buffer
} ReadRequest;

/* ---- Coalesced Read Group ---- */
typedef struct {
    int file_idx;
    off_t offset;               // page-aligned
    size_t length;              // page-aligned total
    void* buffer;               // staging buffer pointer
    ReadRequest* requests;      // array of individual requests within this group
    int num_requests;
    int complete;               // set to 1 by worker when done
} CoalescedRead;

/* ---- Worker Thread Context ---- */
typedef struct {
    pthread_t thread;
    int worker_id;
    CoalescedRead* work_queue;  // pointer to assigned reads
    int work_count;
    pthread_mutex_t* mutex;     // shared with coordinator
    pthread_cond_t* cond;       // shared with coordinator
    int* completed_count;       // atomic counter of finished reads
} WorkerCtx;

/* ---- Staging Buffer Pool ---- */
typedef struct {
    void* base;                 // posix_memalign'd base pointer
    size_t total_size;          // total allocation
    size_t used;                // current watermark
    // Simple bump allocator; reset after each batch_read/get_prefetch
} StagingPool;

/* ---- Module Global State ---- */
typedef struct {
    RegisteredFile* files;
    int num_files;
    TensorMeta* tensors;        // flat array, looked up by hash
    int num_tensors;
    WorkerCtx* workers;
    int num_workers;
    StagingPool staging;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int initialized;
    // Prefetch state
    CoalescedRead* prefetch_reads;
    int prefetch_count;
    int prefetch_complete;
    char prefetch_tag[64];
} ModuleState;
```

### Threading Model

```
Main Thread (Python)          Worker Threads (pthreads)
  |                              |  |  |  |
  |-- Schedule reads ----------->|  |  |  |   (mutex lock, set work queues)
  |                              |  |  |  |
  |-- Release GIL               |  |  |  |
  |    (Py_BEGIN_ALLOW_THREADS)  |  |  |  |
  |                              v  v  v  v
  |                            preadv() preadv() preadv() preadv()
  |                              |  |  |  |
  |                              v  v  v  v
  |<- pthread_cond_wait --------+--+--+--+   (all workers signal done)
  |
  |-- Reacquire GIL
  |    (Py_END_ALLOW_THREADS)
  |
  |-- Unpack results
  |-- Return to Python
```

The GIL is released during the entire I/O wait, allowing other Python threads
(if any) to run. More importantly, this avoids GIL contention between I/O
completion and the main thread.

Workers are **persistent** -- they spin on `pthread_cond_wait` between batches,
avoiding thread creation/destruction overhead (~50us per `pthread_create`).

### Worker Thread QoS

On macOS, we set worker threads to `QOS_CLASS_UTILITY` via
`pthread_set_qos_class_self_np()`. This tells the scheduler these are I/O-bound
background threads, keeping them off the performance cores that the GPU compute
and Metal command processing need. The efficiency cores are perfectly adequate
for issuing `preadv` syscalls (the actual I/O is done by the NVMe controller
via DMA, not by the CPU).

---

## macOS-Specific Optimizations

### 1. `preadv()` for Scatter-Gather Reads

Available on macOS (Darwin) via `<sys/uio.h>`. Single syscall to read data
into multiple non-contiguous buffers from a given file offset. This is the
single biggest win: our benchmarks show **2.6x throughput improvement** over
individual `pread()` calls (14.1 GB/s vs 5.3 GB/s for 7MB expert reads).

For a single expert, we can read all 3 projections' data (gate_proj, up_proj,
down_proj -- which are contiguous in the fused tensor) in one `preadv` call
with 3 iovec entries pointing to separate output buffers.

For coalesced reads (multiple adjacent experts), one `preadv` reads everything.

### 2. Page-Aligned I/O (16KB on ARM64)

All read offsets and lengths are aligned to `vm_page_size` (16384 bytes on
Apple Silicon). This ensures:
- Reads go through the optimal DMA path in the NVMe driver
- No partial-page reads that force the kernel to read-modify-write
- Buffer addresses from `posix_memalign(16384, ...)` satisfy DMA alignment

### 3. `F_NOCACHE` -- Conditional Use

`fcntl(fd, F_NOCACHE, 1)` bypasses the kernel's unified buffer cache (UBC).

**When to use it:** For the 397B model, total expert weights are ~198GB. The
page cache is ~30GB (48GB minus kernel/wired). With a 28GB user-space LRU
cache, every expert read would pollute the page cache with data we're already
caching ourselves, evicting other useful pages (attention weights, OS pages).

**When NOT to use it:** For the 122B model (61GB total), a significant fraction
fits in page cache + user LRU combined. Page cache hits are essentially free
(no syscall, no DMA -- just a memcpy from kernel wired memory).

**Decision:** Make F_NOCACHE configurable per `register_files()` call. Default
OFF for models < 2x DRAM, ON for models > 2x DRAM.

### 4. `madvise()` -- For Hybrid mmap+pread Strategy (Not Recommended)

We considered using `mmap` + `madvise(MADV_WILLNEED)` for readahead, but:
- `madvise` is advisory only -- no guarantee of timing
- Page faults from mmap reads go through the kernel fault path, adding ~10us
  per fault vs ~1us for `preadv` which uses the direct I/O path
- mmap doesn't give us control over buffer lifetime (the kernel decides when
  to evict pages)

**Verdict:** Pure `preadv()` is better. We manage our own buffers.

### 5. `dispatch_io` (GCD) -- Considered and Rejected

Grand Central Dispatch provides `dispatch_io_create(DISPATCH_IO_RANDOM, fd, ...)`
for kernel-scheduled async I/O. However:
- It's a block-based API, harder to integrate with CPython's GIL model
- Each `dispatch_io_read` callback runs on a GCD worker thread, creating
  unpredictable GIL acquisition patterns
- Our pthread pool + `preadv` approach already saturates the NVMe bandwidth
  at 4 threads (20+ GB/s), so GCD can't improve throughput
- GCD adds overhead for small I/O operations due to block allocation/dispatch
- The API is designed for Objective-C/Swift consumers, not C extension modules

**Verdict:** Not worth the complexity. pthreads + preadv is simpler and faster.

### 6. IOKit / NVMe Direct Access -- Considered and Rejected

Direct NVMe command submission via IOKit (`IONVMeController`) would bypass
the filesystem entirely. However:
- Requires root/entitlement for raw device access
- Must handle NVMe command queues, completion interrupts, buffer mapping
- Filesystem metadata (safetensors header offsets) wouldn't be available
- Risk of data corruption if filesystem and direct I/O conflict
- Our SSD already delivers 17.5 GB/s through the standard I/O stack

**Verdict:** Way too dangerous for a primary machine. Zero benefit -- the SSD
controller is the bottleneck, not the filesystem layer.

---

## Read Coalescing Strategy

### Adjacent Expert Optimization

In the fused safetensors tensor layout `[512, intermediate, hidden]`, experts
are contiguous in memory. Expert N starts at `base + N * expert_stride` and
expert N+1 starts at `base + (N+1) * expert_stride`.

When multiple experts from the same layer are needed:

```
Expert indices needed: [42, 43, 45, 100, 101, 102, 103]

After sorting by offset:
  42 @ offset 308MB  (7.1MB)
  43 @ offset 315MB  (7.1MB)  <- adjacent to 42, gap = 0
  45 @ offset 329MB  (7.1MB)  <- gap from 43 = 7.1MB (1 expert)
 100 @ offset 732MB  (7.1MB)
 101 @ offset 739MB  (7.1MB)  <- adjacent to 100
 102 @ offset 746MB  (7.1MB)  <- adjacent to 101
 103 @ offset 753MB  (7.1MB)  <- adjacent to 102

Coalesced groups (gap threshold = expert_stride, i.e. one expert):
  Group 1: [42, 43] -> single preadv at 308MB, 14.2MB
  Group 2: [45]     -> single pread at 329MB, 7.1MB
  Group 3: [100, 101, 102, 103] -> single preadv at 732MB, 28.4MB

With readahead=2:
  Group 1: [42, 43, 44] -> preadv at 308MB, 21.3MB (44 is readahead)
  Group 2: [45, 46, 47] -> preadv at 329MB, 21.3MB (46,47 are readahead)
  Group 3: [100, 101, 102, 103, 104, 105] -> preadv at 732MB, 42.6MB

Total: 3 syscalls instead of 7 (or 13 with readahead)
```

### Readahead Strategy

For each cache-missed expert at index N, we speculatively read experts at
indices N+1 and N+2 (configurable via `readahead` parameter).

**Why this works:**
1. Adjacent experts in the fused tensor are sequential on disk -- reading them
   costs almost nothing extra since the NVMe controller prefetches sequential
   data anyway
2. The LRU cache analysis shows 16-34% consecutive token overlap per layer,
   meaning recently-activated experts are likely to be reactivated
3. Readahead experts go straight into the LRU cache staging area. If they're
   used within the next few tokens, we avoid a cache miss. If not, they get
   evicted by normal LRU policy.

**Cost-benefit:**
- readahead=2 increases I/O volume by up to 2x per miss, but coalesced reads
  only add ~1 syscall latency (~50us) for sequential data
- At 17.5 GB/s sequential, reading an extra 14.2MB (2 experts) takes ~0.8ms
- If this prevents even 1 cache miss per layer on a future token, we save
  ~1.4ms (the cold-read cost of 1 expert at 5 GB/s)
- Net: readahead is profitable if hit rate > ~30%, which our LRU analysis
  suggests is achievable

---

## Memory Management

### Staging Buffer Design

```
  +------------------------------------------------------------------+
  |                    StagingPool (512 MB)                           |
  |  posix_memalign(16384, 512*1024*1024)                           |
  |                                                                  |
  |  [  batch_read buffer space  |  prefetch buffer space  ]        |
  |  ^                           ^                                   |
  |  bump_alloc for current      bump_alloc for prefetch             |
  |  batch_read() call           start_prefetch() call               |
  +------------------------------------------------------------------+
```

- Two halves: one for synchronous `batch_read`, one for async `start_prefetch`
- Bump allocator within each half: `alloc(size)` advances watermark, `reset()`
  zeros the watermark. No fragmentation, no malloc overhead.
- 512 MB total is enough for worst case: 234 misses * ~7.1 MB = ~1.66 GB...

Wait -- 1.66 GB doesn't fit in 512 MB. Let's recalculate:
- Per layer: ~4 misses (234 misses / 60 layers = 3.9 avg)
- Per layer I/O: 3.9 * 7.1 MB = ~28 MB
- With readahead=2: ~84 MB per layer
- Only one layer's reads are in-flight at a time (sync batch_read is per-layer)

So 512 MB is generous for per-layer batches. For the prefetch buffer (next
layer), another 256 MB. Total 512 MB for both halves is sufficient.

The staging buffers are **short-lived**: data is copied into `mx.array` and then
into the ExpertCache immediately after `batch_read` returns. The staging pool
is reset for the next layer.

### Zero-Copy Path to MLX

The ideal path is:
1. `preadv()` reads data into page-aligned staging buffer
2. Return `memoryview` (Python buffer protocol) pointing into staging buffer
3. Caller does `mx.array(np.frombuffer(memoryview, dtype=...))` or better,
   `mx.array(memoryview, dtype=mx.uint32)` if MLX supports it

Actually, MLX's `mx.array` from a numpy array does a copy into Metal-backed
memory anyway. So the real path is:

1. `preadv()` -> staging buffer (kernel DMA -> our buffer)
2. `np.frombuffer(staging_buffer_slice)` -> numpy view (zero-copy)
3. `mx.array(numpy_view)` -> Metal allocation (one copy, done by Metal)

The C extension can optionally do step 2 internally and return numpy arrays
directly, avoiding one Python-level indirection.

### BF16 Handling

Safetensors stores bfloat16 as raw 16-bit values. MLX needs them converted:
```c
// In C, this is a simple bit-shift (no FPU needed):
// bfloat16 -> float32: shift left 16 bits
uint16_t bf16_val = *(uint16_t*)src;
uint32_t f32_bits = (uint32_t)bf16_val << 16;
float f32_val = *(float*)&f32_bits;
```

We can do this in the C extension during the unpack phase, using NEON SIMD
for bulk conversion (~4 cycles per 8 elements via `vshll_n_u16`). This avoids
the current Python path: `np.frombuffer(uint16) -> astype(uint32) << 16 ->
view(float32) -> mx.array() -> astype(bfloat16)`.

However, MLX's `mx.array` with explicit dtype handles this efficiently already.
We should benchmark before adding NEON conversion -- it may not be the
bottleneck.

---

## Integration with stream_infer.py

### Minimal Changes Required

The integration point is in `generate_offload_selective()`, specifically the
per-layer expert loading loop (lines 1349-1549 of `stream_infer.py`).

Current code (simplified):
```python
# Per layer, after router gives us expert indices:
uncached_list = [idx for idx in unique_list if not expert_cache.has_expert(i, idx)]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(_read_single_expert_attrs, eidx, i, ...)
               for eidx in uncached_list]
    for future in futures:
        eidx, attrs, io_stats = future.result()
        for (proj_name, attr_name), arr in attrs.items():
            expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)
```

New code with fast_expert_io:
```python
# Per layer, after router gives us expert indices:
uncached_list = [idx for idx in unique_list if not expert_cache.has_expert(i, idx)]

if uncached_list:
    # Build read specs for all missed experts
    read_specs = []
    for eidx in uncached_list:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for attr in ("weight", "scales", "biases"):
                tensor_name = f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}.{attr}"
                filepath = expert_file_map.get(tensor_name)
                if filepath:
                    read_specs.append((filepath, tensor_name, eidx))

    # One C call replaces ThreadPoolExecutor + _read_single_expert_attrs
    results = fast_expert_io.batch_read(
        read_specs=read_specs,
        readahead=2,
    )

    for eidx, attrs in results.items():
        for attr_key, data in attrs.items():
            proj_name, attr_name = attr_key.split(".", 1)
            arr = mx.array(np.frombuffer(data, dtype=...))
            expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)
```

### Async Pipeline Integration

For the prefetch pipeline (reading layer N+1 during layer N compute):

```python
for i in range(num_layers):
    # Retrieve prefetched data for THIS layer (started during previous layer)
    if i > 0 and prefetch_active:
        prefetch_results = fast_expert_io.get_prefetch(tag=f"layer_{i}")
        # Insert prefetched experts into cache
        for eidx, attrs in prefetch_results.items():
            for attr_key, data in attrs.items():
                proj_name, attr_name = attr_key.split(".", 1)
                expert_cache.put_attr(i, eidx, proj_name, attr_name,
                                       mx.array(np.frombuffer(data, ...)))

    # ... run attention + router for layer i -> get expert indices ...
    # ... batch_read for any remaining misses ...
    # ... compute MoE for layer i ...

    # Start prefetching for layer i+1 (if we can predict which experts)
    # Problem: we don't know which experts layer i+1 will need until we
    # run its router. But we CAN prefetch based on LRU cache state:
    # experts that are NOT in the cache but were recently evicted are
    # likely to be needed again. Or: prefetch the same experts as this
    # layer (temporal locality across adjacent layers).
    if i < num_layers - 1:
        # Strategy: prefetch experts that THIS layer needed but cache
        # doesn't have for the next layer. Adjacent layers often share
        # expert activation patterns (~20-30% overlap observed).
        predicted_misses = predict_next_layer_misses(i, unique_list, expert_cache)
        if predicted_misses:
            specs = build_read_specs(i + 1, predicted_misses, expert_file_map)
            fast_expert_io.start_prefetch(read_specs=specs, readahead=1,
                                           tag=f"layer_{i+1}")
            prefetch_active = True
        else:
            prefetch_active = False
```

**Note on prefetch prediction:** Without running layer N+1's router, we can't
know exactly which experts it needs. Three viable prediction strategies:

1. **Same-expert prediction:** Assume layer N+1 needs similar experts to layer N.
   From the routing analysis, adjacent-layer overlap is ~20-30%.
2. **Frequency-based prediction:** Prefetch the top-K most frequently used experts
   for layer N+1 (from offline profiling data).
3. **No prediction, just readahead:** Skip cross-layer prefetch entirely, rely
   only on within-batch readahead (adjacent expert indices). This is simpler
   and still captures the main win (coalesced reads + multi-threaded preadv).

Recommendation: Start with option 3 (readahead only). Add cross-layer prefetch
later if profiling shows the I/O-compute overlap is significant.

---

## Expected Performance

### Breakdown of Improvements

| Optimization | Current | Projected | Source |
|-------------|---------|-----------|--------|
| Python overhead (GIL, ThreadPool, np.frombuffer) | ~200ms/token | ~20ms/token | Eliminate Python I/O path |
| Syscall overhead (9 seeks per expert) | ~9 syscalls/expert | ~1 preadv/expert | Coalesced reads |
| I/O parallelism (4 Python threads) | ~1.4 GB/s | ~10-14 GB/s | 4-8 pthreads, no GIL |
| Page alignment | Unaligned | 16KB aligned | posix_memalign + aligned offsets |
| Readahead (adjacent experts) | None | +2 per miss | Sequential SSD prefetch |
| Async pipeline (overlap I/O+compute) | Sequential | Overlapped | start_prefetch/get_prefetch |

### Projected Throughput

**Conservative estimate** (preadv + 4 threads, no readahead, no prefetch):
- I/O bandwidth: ~10 GB/s (based on measured 20 GB/s at 4 threads with
  F_NOCACHE on cold reads, halved for realistic conditions)
- I/O time per token: 1.74 GB / 10 GB/s = 174 ms
- Compute per token: ~200 ms
- Total: 374 ms/token = **2.67 tok/s** (2.0x improvement)

**Optimistic estimate** (preadv + 8 threads + readahead boosting cache to 70%):
- Cache misses drop from 234 to 180 (readahead prevents ~54 misses)
- I/O per token: 180 * 7.1 MB = 1.28 GB
- I/O bandwidth: ~14 GB/s
- I/O time: 91 ms
- Compute: ~200 ms
- Total: 291 ms/token = **3.44 tok/s** (2.6x improvement)

### 122B Model Projection

Currently at 3.99 tok/s with 86% cache hit rate at 500 tokens:
- Cache misses: 80 * 0.14 = 11.2 per token (80 = 48 layers * 8 active / 8...
  actually 48 layers * ~10 experts = 480 activations, 14% miss = 67 misses)
- With faster I/O: compute-bound rather than I/O-bound
- Projected: **5-6 tok/s** (limited by GPU compute, not I/O)

---

## Build System

### Using setuptools (setup.py)

```python
# setup.py
from setuptools import setup, Extension

fast_expert_io = Extension(
    'fast_expert_io',
    sources=['fast_expert_io.c'],
    extra_compile_args=[
        '-O3',                   # aggressive optimization
        '-march=armv8.4-a',      # M3 Max ARM ISA
        '-flto',                 # link-time optimization
        '-DPAGE_SIZE=16384',     # macOS ARM64 page size
    ],
    extra_link_args=[
        '-lpthread',             # POSIX threads
        '-flto',
    ],
)

setup(
    name='fast_expert_io',
    version='0.1.0',
    ext_modules=[fast_expert_io],
)
```

### Build commands

```bash
# Build in-place (for development)
cd /Users/danielwoods/Workspace/ane-research
uv run python setup.py build_ext --inplace

# Or with the uv project:
# Add to pyproject.toml under [build-system] if desired
```

### Alternative: Makefile (simpler, no setuptools)

```makefile
# Makefile
PYTHON := $(shell uv run python -c "import sys; print(sys.executable)")
PYTHON_INC := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))")

CC = clang
CFLAGS = -O3 -march=armv8.4-a -flto -fPIC -I$(PYTHON_INC) -DPAGE_SIZE=16384
LDFLAGS = -shared -lpthread -undefined dynamic_lookup

fast_expert_io.so: fast_expert_io.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f fast_expert_io.so

.PHONY: clean
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Worker thread crash corrupts model output | High | Robust error handling in C; worker threads never write to model memory |
| Memory leak in staging buffers | Medium | Bump allocator with explicit reset; valgrind/leaks testing |
| File descriptor exhaustion | Low | Pre-open FDs at init, cap at ~50 (one per safetensors shard) |
| preadv not available | None | Available on macOS since 10.12; we require macOS 26+ |
| Race condition in prefetch | Medium | mutex + condvar; prefetch results are write-once by workers |
| Staging buffer too small | Low | Dynamic sizing based on (misses * expert_size * (1+readahead)) |

---

## Implementation Phases

**Phase 1: Core batch_read** (highest value, lowest risk)
- init/shutdown, register_files, batch_read
- preadv with page alignment, 4 pthreads
- No readahead, no prefetch
- Expected: ~2x speedup over current Python path

**Phase 2: Readahead**
- Add readahead parameter to batch_read
- Coalesce adjacent experts into larger reads
- Expected: additional 10-20% from cache hit improvement

**Phase 3: Async prefetch pipeline**
- start_prefetch/get_prefetch
- Double-buffered staging pool
- Expected: additional 10-15% from I/O-compute overlap

**Phase 4: Tuning**
- F_NOCACHE experimentation
- Worker thread count optimization
- Coalescing gap threshold tuning
- NEON BF16 conversion (if profiling shows it matters)
