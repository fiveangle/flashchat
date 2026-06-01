/*
 * infer.m — Complete Qwen3.5-397B inference engine using Metal
 *
 * Full forward pass: embedding -> 60 transformer layers -> norm -> lm_head -> sample
 * Non-expert weights loaded from model_weights.bin (mmap'd at startup)
 * Expert weights loaded from packed_experts/ per layer per token (pread)
 *
 * Architecture: Qwen3.5-397B-A17B (MoE)
 *   - 60 layers: 45 linear attention (GatedDeltaNet) + 15 full attention
 *   - hidden_size=4096, head_dim=256, num_attention_heads=32, num_kv_heads=2
 *   - 512 experts/layer, 10 active (we use K=4 for speed)
 *   - Shared expert per layer (always active)
 *   - Linear attention: conv1d(kernel=4) + gated delta recurrence
 *   - Full attention: standard QKV + scaled dot product + RoPE
 *
 * Command buffer optimization (fused_layer_forward):
 *   Per-layer Metal command buffer structure:
 *     CMD1: attention input projections (3-4 dispatches, 1 commit)
 *     CPU:  attention compute (RoPE/softmax/delta-net)
 *     CMD2: o_proj + residual_add + rms_norm + routing + shared gate/up (8 encoders, 1 commit)
 *           GPU handles residual connection and post-attn norm internally,
 *           eliminating the CPU round-trip that previously split this into 2 cmd buffers.
 *     CPU:  softmax + top-K + pread all K experts (4 pthreads parallel)
 *     CMD3: all K expert forwards + shared SwiGLU + shared down
 *           + GPU-side combine + residual_add + rms_norm -> buf_input (DEFERRED commit)
 *           Batched encoding: 4 encoders for K experts + 2 shared + 3 combine = 9 total
 *   Total: 3 cmd buffers per layer. CMD3 is submitted async (commit without wait).
 *   GPU-side combine in CMD3: for non-last layers, CMD3 also computes:
 *     moe_combine_residual (weighted sum + residual + shared gate -> hidden)
 *     rms_norm (hidden -> buf_input using NEXT layer's input_norm weights)
 *   This allows the next layer's CMD1 to submit immediately without waiting
 *   for CMD3 completion — the GPU queue serializes CMD3(N-1) then CMD1(N).
 *   Saves ~0.83ms/layer deferred_wait + CPU combine + input_norm overhead.
 *   Multi-expert buffers (MAX_K=8 independent slots) allow all K expert
 *   forwards to be encoded into a single command buffer.
 *   Batched encoding: 2 encoders per expert (gate+up fused, SwiGLU+down fused)
 *   + 2 for shared expert = K*2 + 2 total encoders in CMD3.
 *   Double-buffered expert data (buf_multi_expert_data / data_B) for future
 *   async pread overlap with GPU compute.
 *
 * Build:  clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation -lpthread infer.m -o infer
 * Run:    ./infer --prompt "Explain relativity" --tokens 50
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/wait.h>
#include <compression.h>
#include <dirent.h>

#include "model_config.h"

#define FLASHCHAT_BUILD_STAMP __DATE__ " " __TIME__

ModelConfig g_cfg;

// ============================================================================
// Runtime constants (loaded from assets/model_configs.json into g_cfg)
// ============================================================================

// Pipeline and cache limits (not model-dependent)
#define MAX_SEQ_LEN 1048576
#define GPU_KV_SEQ  8192

// Default model path (overridden by --model flag)
#define MODEL_PATH_DEFAULT "/Volumes/usr/Users/speedster/dev/flashchat/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"

// ============================================================================
// Timing helper
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void server_log_timestamp(FILE *f) {
    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_now);
    fprintf(f, "[%s] ", ts);
}

static FILE *g_server_log = NULL;
static int g_server_log_fd = -1;
static FILE *g_server_http_log = NULL;
static char g_server_log_path[PATH_MAX] = {0};
static char g_server_log_dir[PATH_MAX] = {0};
static int g_server_debug_enabled = 0;
static int g_server_http_log_enabled = 0;
static int g_show_thinking_enabled = 0;
static int g_system_prompt_cache_enabled = 1;
static int g_system_prompt_cache_max_entries = 2;
static int g_mtp_predictions = -1;
static float g_default_temperature = 0.7f;
static float g_default_top_p = 0.8f;
static int g_default_top_k = 20;
static float g_default_min_p = 0.0f;
static float g_default_presence_penalty = 1.5f;
static float g_default_repetition_penalty = 1.0f;
static int g_default_reasoning_enabled = 1;
// When 1 (default), the sampler hard-overrides to greedy (temp=0) for any
// token generated *inside* a `<tool_call>...</tool_call>` block. Sampling
// diversity inside the rigid XML format has no upside (the format primer
// has high-confidence tokens) and a real downside (model occasionally
// picks `\n` instead of the function-name token, which we observed
// post-K=8 fix on long multi-turn nanocoder sessions). Override via the
// FLASHCHAT_TOOL_CALL_GREEDY=0 env var if you want sampled tool-call
// content (e.g. for diversifying tool parameter values across runs).
static int g_tool_call_greedy_enabled = 1;

static void server_log_emit(FILE *stream, const char *fmt, va_list ap) {
    va_list stream_ap;
    va_copy(stream_ap, ap);
    vfprintf(stream, fmt, stream_ap);
    fflush(stream);
    va_end(stream_ap);

    if (g_server_log) {
        va_list file_ap;
        va_copy(file_ap, ap);
        server_log_timestamp(g_server_log);
        vfprintf(g_server_log, fmt, file_ap);
        fflush(g_server_log);
        va_end(file_ap);
    }
}

static void server_logf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    server_log_emit(stdout, fmt, ap);
    va_end(ap);
}

static void server_log_errorf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    server_log_emit(stderr, fmt, ap);
    va_end(ap);
}

static void server_log_close(void) {
    if (g_server_http_log) {
        server_log_timestamp(g_server_http_log);
        fprintf(g_server_http_log, "[http] Logging stopped\n");
        fflush(g_server_http_log);
        fclose(g_server_http_log);
        g_server_http_log = NULL;
    }
    if (!g_server_log) return;
    server_log_timestamp(g_server_log);
    fprintf(g_server_log, "[serve] Logging stopped\n");
    fflush(g_server_log);
    g_server_log_fd = -1;
    fclose(g_server_log);
    g_server_log = NULL;
}

static int server_flag_enabled(const char *value) {
    return value && value[0] &&
           strcmp(value, "0") != 0 &&
           strcasecmp(value, "false") != 0 &&
           strcasecmp(value, "off") != 0 &&
           strcasecmp(value, "no") != 0;
}

static int parse_mtp_predictions(const char *value) {
    if (!value || !value[0]) return -1;
    if (strcasecmp(value, "auto") == 0) return -1;
    if (strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "off") == 0 ||
        strcasecmp(value, "no") == 0) {
        return 0;
    }
    if (strcasecmp(value, "true") == 0 ||
        strcasecmp(value, "on") == 0 ||
        strcasecmp(value, "yes") == 0) {
        return 1;
    }
    char *end = NULL;
    long v = strtol(value, &end, 10);
    if (end == value) return 0;
    if (v < 0) return -1;
    if (v > 16) return 16;
    return (int)v;
}

static void server_log_open(void) {
    if (g_server_log) return;

    const char *env_path = getenv("FLASHCHAT_SERVER_LOG");
    NSString *configured_path = nil;
    if (env_path && env_path[0]) {
        configured_path = [NSString stringWithUTF8String:env_path];
    } else {
        const char *home = getenv("HOME");
        if (!home || !home[0]) return;
        configured_path = [NSString stringWithFormat:@"%s/.config/flashchat/logs/server.log", home];
    }
    const char *env_debug = getenv("FLASHCHAT_SERVER_DEBUG");
    const char *env_http_log = getenv("FLASHCHAT_SERVER_HTTP_LOG");
    g_server_debug_enabled = server_flag_enabled(env_debug);
    g_server_http_log_enabled = server_flag_enabled(env_http_log);

    if (!configured_path || [configured_path length] == 0) return;

    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL is_dir = NO;
    BOOL path_exists = [fm fileExistsAtPath:configured_path isDirectory:&is_dir];
    if (!path_exists) {
        NSString *last = [configured_path lastPathComponent];
        if ([configured_path hasSuffix:@"/"] || [[last pathExtension] length] == 0) {
            is_dir = YES;
        }
    }

    NSString *log_dir = nil;
    NSString *log_path = nil;
    if (is_dir) {
        log_dir = configured_path;
        log_path = [configured_path stringByAppendingPathComponent:@"server.log"];
    } else {
        log_path = configured_path;
        log_dir = [log_path stringByDeletingLastPathComponent];
    }

    NSError *dir_error = nil;
    if (![fm createDirectoryAtPath:log_dir
       withIntermediateDirectories:YES
                        attributes:nil
                             error:&dir_error]) {
        fprintf(stderr, "WARNING: could not create server log dir %s: %s\n",
                [log_dir UTF8String],
                dir_error ? [[dir_error localizedDescription] UTF8String] : "unknown error");
        return;
    }

    const char *utf8_path = [log_path fileSystemRepresentation];
    FILE *f = fopen(utf8_path, "a");
    if (!f) {
        fprintf(stderr, "WARNING: could not open server log %s: %s\n",
                utf8_path, strerror(errno));
        return;
    }

    g_server_log = f;
    g_server_log_fd = fileno(f);
    strncpy(g_server_log_path, utf8_path, sizeof(g_server_log_path) - 1);
    g_server_log_path[sizeof(g_server_log_path) - 1] = '\0';
    strncpy(g_server_log_dir, [log_dir fileSystemRepresentation], sizeof(g_server_log_dir) - 1);
    g_server_log_dir[sizeof(g_server_log_dir) - 1] = '\0';
    server_log_timestamp(g_server_log);
    fprintf(g_server_log, "[serve] Logging started\n");
    server_log_timestamp(g_server_log);
    fprintf(g_server_log, "[serve] Build: %s\n", FLASHCHAT_BUILD_STAMP);
    server_log_timestamp(g_server_log);
    fprintf(g_server_log, "[serve] Show thinking: %s\n", g_show_thinking_enabled ? "enabled" : "disabled");
    server_log_timestamp(g_server_log);
    fprintf(g_server_log, "[serve] Sampling defaults: temp=%.3f top_p=%.3f top_k=%d min_p=%.3f presence=%.3f repetition=%.3f\n",
            g_default_temperature, g_default_top_p, g_default_top_k, g_default_min_p,
            g_default_presence_penalty, g_default_repetition_penalty);
    if (g_server_debug_enabled) {
        server_log_timestamp(g_server_log);
        fprintf(g_server_log, "[serve] Debug request dumping enabled\n");
    }
    fflush(g_server_log);

    if (g_server_http_log_enabled) {
        char http_log_path[PATH_MAX];
        snprintf(http_log_path, sizeof(http_log_path), "%s/http.log", g_server_log_dir);
        g_server_http_log = fopen(http_log_path, "a");
        if (!g_server_http_log) {
            server_log_timestamp(g_server_log);
            fprintf(g_server_log, "[serve] WARNING: could not open HTTP log %s: %s\n",
                    http_log_path, strerror(errno));
            fflush(g_server_log);
            g_server_http_log_enabled = 0;
        } else {
            server_log_timestamp(g_server_http_log);
            fprintf(g_server_http_log, "[http] Logging started\n");
            fflush(g_server_http_log);
        }
    }
    atexit(server_log_close);
}

static void server_http_log_block(const char *request_id, const char *direction,
                                  const char *label, const char *payload) {
    if (!g_server_http_log_enabled || !g_server_http_log) return;
    server_log_timestamp(g_server_http_log);
    fprintf(g_server_http_log, "[%s] %s %s\n", direction,
            request_id ? request_id : "-", label ? label : "");
    if (payload && payload[0]) {
        fwrite(payload, 1, strlen(payload), g_server_http_log);
        if (payload[strlen(payload) - 1] != '\n') fputc('\n', g_server_http_log);
    }
    fprintf(g_server_http_log, "\n");
    fflush(g_server_http_log);
}

static volatile sig_atomic_t g_server_shutdown_signal = 0;
static int g_server_listen_fd = -1;

static void serve_signal_handler(int signo) {
    const char *msg = "[signal] Shutdown requested\n";
    size_t len = sizeof("[signal] Shutdown requested\n") - 1;
    if (g_server_log_fd >= 0) write(g_server_log_fd, msg, len);
    write(STDERR_FILENO, msg, len);
    g_server_shutdown_signal = signo;
    if (g_server_listen_fd >= 0) {
        close(g_server_listen_fd);
        g_server_listen_fd = -1;
    }
}

static void install_serve_signal_handlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = serve_signal_handler;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

static void server_debug_write_text(const char *request_id, const char *suffix, const char *text) {
    if (!g_server_debug_enabled || !g_server_log_dir[0] || !request_id || !request_id[0] || !suffix || !suffix[0]) return;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s.%s", g_server_log_dir, request_id, suffix);
    FILE *f = fopen(path, "w");
    if (!f) {
        server_log_errorf("[serve] %s debug_write_failed path=%s error=%s\n",
                          request_id, path, strerror(errno));
        return;
    }
    if (text && text[0]) fwrite(text, 1, strlen(text), f);
    fclose(f);
    server_log_errorf("[serve] %s debug_dump=%s\n", request_id, path);
}

// ============================================================================
// Per-phase timing accumulators for fused_layer_forward
// Tracks time spent in each pipeline phase across all layers per token.
// Reset at token boundary, printed as summary.
// ============================================================================

typedef struct {
    double deferred_wait;    // waiting for previous CMD3 GPU
    double deferred_cpu;     // CPU readback + combine for deferred experts
    double input_norm;       // CPU RMS norm + CMD1 prep
    double cmd1_submit;      // CMD1 encode + commit
    double cmd1_wait;        // CMD1 waitUntilCompleted
    double cpu_attn;         // CPU attention compute (delta-net or full-attn)
    double cmd2_encode;      // CMD2 encode (o_proj + residual + norm + routing)
    double cmd2_wait;        // CMD2 commit + waitUntilCompleted
    double routing_cpu;      // CPU softmax + topK
    double spec_route;       // speculative early routing (gate matvec + topK)
    double expert_io;        // parallel pread + cache lookup
    double cmd3_encode;      // CMD3 encode experts + submit (deferred)
    double total;            // total per-layer time
    int count;               // number of layers timed
} LayerTimingAccum;

static LayerTimingAccum g_timing = {0};
static int g_timing_enabled = 0;

// Temporal prediction pipeline counters (declared early for timing_print access)
static int g_pred_enabled = 0;
static int g_pred_generating = 0;   // only set to 1 after prefill (predictions only help during generation)
static uint64_t g_pred_hits = 0;
static uint64_t g_pred_misses = 0;
static uint64_t g_pred_layers = 0;

// Routing data collection for training an expert predictor
// Binary format per sample: int32 layer_idx, int32 K, float32[4096] hidden, int32[K] expert_indices
static FILE *g_routing_log = NULL;
static int g_routing_log_samples = 0;

// LZ4 compressed expert support
// File format: [LZ4IndexEntry × 512] + [compressed blobs]
typedef struct {
    uint64_t offset;
    uint32_t comp_size;
    uint32_t raw_size;
} LZ4IndexEntry;

static LZ4IndexEntry *g_lz4_index[MAX_NUM_LAYERS];  // per-layer index (NULL if not using LZ4)
static void *g_lz4_comp_bufs[8];                 // pre-allocated compressed read buffers (MAX_K=8)
static int g_use_lz4 = 0;                        // auto-detected from packed_experts_lz4/

// ============================================================================
// Expert frequency tracking (diagnostic: --freq flag)
// ============================================================================

static int g_expert_freq[MAX_NUM_LAYERS][MAX_NUM_EXPERTS];  // activation count per (layer, expert)
static int g_freq_tracking = 0;  // enabled by --freq flag

// ----------------------------------------------------------------------------
// MTP expert-overlap instrumentation.
// Projects the SSD I/O a batched 2-position draft/verify forward would incur.
// A batched verify of [accepted_token, draft] loads, per layer, the UNION of
// the two tokens' active experts. When a draft is accepted (draft == real next
// token), that union is exactly the overlap between two consecutive real decode
// steps — which is what we measure here. Enabled when MTP is active or via
// FLASHCHAT_MTP_OVERLAP=1. Decode-phase only (gated on g_pred_generating).
// ----------------------------------------------------------------------------
static int  g_mtp_overlap_enabled = 0;
static int  g_overlap_in_decode = 0;             // 1 during auto-regressive decode (not prefill)
static int  g_overlap_prev[MAX_NUM_LAYERS][16];  // previous decode step's experts per layer
static int  g_overlap_prev_k[MAX_NUM_LAYERS];    // count per layer (0 = no previous step yet)
static long g_overlap_single_sum = 0;    // sum of per-layer K over counted step-pairs (1 token's load)
static long g_overlap_separate_sum = 0;  // sum of (K_prev + K_cur): two separate single-token forwards
static long g_overlap_union_sum = 0;     // sum of |prev ∪ cur|: one batched 2-position forward
static long g_overlap_intersect_sum = 0; // sum of |prev ∩ cur|
static int  g_overlap_layer_pairs = 0;   // number of per-layer step-pairs counted
static int g_cache_telemetry_enabled = 0;  // enabled by --cache-telemetry flag
static int g_think_budget = 2048; // max thinking tokens before force-emitting </think>

static const char *kApiModelId = NULL;

// ============================================================================
// System prompt hash for KV-cache snapshot deduplication
// ============================================================================

static uint64_t hash_string_djb2(const char *str) {
    uint64_t hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

// Tiered I/O: cold fds (F_NOCACHE) for first reads, warm fds (page cached) for repeats
static int *g_layer_fds_cold = NULL;    // [g_cfg.num_layers] cold fds (set in main)
static uint8_t g_expert_seen[MAX_NUM_LAYERS][MAX_NUM_EXPERTS / 8];  // bitset: seen before?

// Async pread state defined after InferPreadTask (see below)

__attribute__((unused))
static inline int expert_is_seen(int layer, int expert) {
    return (g_expert_seen[layer][expert >> 3] >> (expert & 7)) & 1;
}
__attribute__((unused))
static inline void expert_mark_seen(int layer, int expert) {
    g_expert_seen[layer][expert >> 3] |= (1 << (expert & 7));
}
// Pick fd for expert read. Currently: always use warm fd (OS page cache).
// Tiered I/O (cold F_NOCACHE for first reads) was tested but OS page cache
// without any bypass outperforms all custom caching strategies.
static inline int expert_pick_fd(int layer, int expert, int warm_fd) {
    (void)layer; (void)expert;
    return warm_fd;
}

static inline size_t active_expert_size(void) {
    return g_cfg.expert_size;
}
static int g_freq_total_tokens = 0;  // total tokens processed while tracking

typedef struct {
    uint64_t token_clock;
    uint64_t unique_experts_touched;
    uint64_t cold_misses;
    uint64_t eviction_misses;
    uint64_t evictions;
    uint64_t reuse_le_1;
    uint64_t reuse_le_4;
    uint64_t reuse_le_16;
    uint64_t reuse_le_64;
    uint64_t reuse_gt_64;
    uint64_t reuse_distance_sum;
    uint64_t reuse_distance_samples;
} CacheTelemetry;

static CacheTelemetry g_cache_telemetry = {0};
static uint8_t g_cache_seen[MAX_NUM_LAYERS][MAX_NUM_EXPERTS];
static uint64_t g_cache_last_touch_token[MAX_NUM_LAYERS][MAX_NUM_EXPERTS];
static uint64_t g_cache_last_evict_token[MAX_NUM_LAYERS][MAX_NUM_EXPERTS];

static void cache_telemetry_reset(void) {
    memset(&g_cache_telemetry, 0, sizeof(g_cache_telemetry));
    memset(g_cache_seen, 0, sizeof(g_cache_seen));
    memset(g_cache_last_touch_token, 0, sizeof(g_cache_last_touch_token));
    memset(g_cache_last_evict_token, 0, sizeof(g_cache_last_evict_token));
}

static void cache_telemetry_note_token(void) {
    if (!g_cache_telemetry_enabled) return;
    g_cache_telemetry.token_clock++;
}

static void cache_telemetry_touch(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= g_cfg.num_layers || expert_idx < 0 || expert_idx >= g_cfg.num_experts) return;
    if (!g_cache_seen[layer_idx][expert_idx]) {
        g_cache_seen[layer_idx][expert_idx] = 1;
        g_cache_telemetry.unique_experts_touched++;
    }
    g_cache_last_touch_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_miss(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= g_cfg.num_layers || expert_idx < 0 || expert_idx >= g_cfg.num_experts) return;
    if (!g_cache_seen[layer_idx][expert_idx]) {
        g_cache_telemetry.cold_misses++;
        g_cache_seen[layer_idx][expert_idx] = 1;
        g_cache_telemetry.unique_experts_touched++;
    } else {
        g_cache_telemetry.eviction_misses++;
        uint64_t dist = 0;
        if (g_cache_last_evict_token[layer_idx][expert_idx] > 0 &&
            g_cache_telemetry.token_clock >= g_cache_last_evict_token[layer_idx][expert_idx]) {
            dist = g_cache_telemetry.token_clock - g_cache_last_evict_token[layer_idx][expert_idx];
        }
        if (dist <= 1) g_cache_telemetry.reuse_le_1++;
        else if (dist <= 4) g_cache_telemetry.reuse_le_4++;
        else if (dist <= 16) g_cache_telemetry.reuse_le_16++;
        else if (dist <= 64) g_cache_telemetry.reuse_le_64++;
        else g_cache_telemetry.reuse_gt_64++;
        g_cache_telemetry.reuse_distance_sum += dist;
        g_cache_telemetry.reuse_distance_samples++;
    }
    g_cache_last_touch_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_evict(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= g_cfg.num_layers || expert_idx < 0 || expert_idx >= g_cfg.num_experts) return;
    g_cache_telemetry.evictions++;
    g_cache_last_evict_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_print(uint64_t hits, uint64_t misses) {
    if (!g_cache_telemetry_enabled) return;
    uint64_t total = hits + misses;
    fprintf(stderr, "\n=== Cache Telemetry ===\n");
    fprintf(stderr, "Tokens tracked: %llu\n", g_cache_telemetry.token_clock);
    fprintf(stderr, "Unique experts touched: %llu / %d (%.1f%%)\n",
            g_cache_telemetry.unique_experts_touched,
            g_cfg.num_layers * g_cfg.num_experts,
            100.0 * g_cache_telemetry.unique_experts_touched / (g_cfg.num_layers * g_cfg.num_experts));
    fprintf(stderr, "Miss breakdown: cold %llu (%.1f%% of misses), eviction %llu (%.1f%% of misses)\n",
            g_cache_telemetry.cold_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.cold_misses / misses : 0.0,
            g_cache_telemetry.eviction_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.eviction_misses / misses : 0.0);
    fprintf(stderr, "Evictions: %llu\n", g_cache_telemetry.evictions);
    fprintf(stderr, "Eviction reuse distance: <=1 tok %llu, <=4 %llu, <=16 %llu, <=64 %llu, >64 %llu",
            g_cache_telemetry.reuse_le_1,
            g_cache_telemetry.reuse_le_4,
            g_cache_telemetry.reuse_le_16,
            g_cache_telemetry.reuse_le_64,
            g_cache_telemetry.reuse_gt_64);
    if (g_cache_telemetry.reuse_distance_samples > 0) {
        fprintf(stderr, " (avg %.1f tok)\n",
                (double)g_cache_telemetry.reuse_distance_sum / g_cache_telemetry.reuse_distance_samples);
    } else {
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "Effective hit rate: %.1f%%\n",
            total > 0 ? 100.0 * hits / total : 0.0);
}

static void timing_reset(void) {
    memset(&g_timing, 0, sizeof(g_timing));
}

static void timing_print(void) {
    if (g_timing.count == 0) return;
    int n = g_timing.count;
    fprintf(stderr, "\n[timing] Per-layer breakdown (avg of %d layers, ms):\n", n);
    fprintf(stderr, "  deferred_wait:  %6.3f\n", g_timing.deferred_wait / n);
    fprintf(stderr, "  deferred_cpu:   %6.3f\n", g_timing.deferred_cpu / n);
    fprintf(stderr, "  input_norm:     %6.3f\n", g_timing.input_norm / n);
    fprintf(stderr, "  cmd1_submit:    %6.3f\n", g_timing.cmd1_submit / n);
    fprintf(stderr, "  cmd1_wait:      %6.3f\n", g_timing.cmd1_wait / n);
    fprintf(stderr, "  spec_route:     %6.3f\n", g_timing.spec_route / n);
    fprintf(stderr, "  cpu_attn:       %6.3f\n", g_timing.cpu_attn / n);
    fprintf(stderr, "  cmd2_encode:    %6.3f\n", g_timing.cmd2_encode / n);
    fprintf(stderr, "  cmd2_wait:      %6.3f\n", g_timing.cmd2_wait / n);
    fprintf(stderr, "  routing_cpu:    %6.3f\n", g_timing.routing_cpu / n);
    fprintf(stderr, "  expert_io:      %6.3f\n", g_timing.expert_io / n);
    fprintf(stderr, "  cmd3_encode:    %6.3f\n", g_timing.cmd3_encode / n);
    fprintf(stderr, "  total_layer:    %6.3f\n", g_timing.total / n);
    fprintf(stderr, "  sum_phases:     %6.3f\n",
            (g_timing.deferred_wait + g_timing.deferred_cpu + g_timing.input_norm +
             g_timing.cmd1_submit + g_timing.cmd1_wait + g_timing.spec_route +
             g_timing.cpu_attn +
             g_timing.cmd2_encode + g_timing.cmd2_wait + g_timing.routing_cpu +
             g_timing.expert_io + g_timing.cmd3_encode) / n);
    fprintf(stderr, "  cmd_buffers:    %d (3 per layer: CMD1+CMD2+CMD3)\n", n * 3);
    fprintf(stderr, "  sync_waits:     %d (2 per layer: CMD1+CMD2, CMD3 deferred)\n", n * 2);
    fprintf(stderr, "  gpu_encoders:   ~%d per layer (CMD1:3-4, CMD2:8-12, CMD3:~10)\n",
            22);  // approximate
    if (g_pred_enabled && g_pred_layers > 0) {
        uint64_t total = g_pred_hits + g_pred_misses;
        double hit_rate = total > 0 ? (double)g_pred_hits / total * 100.0 : 0;
        fprintf(stderr, "  [predict] hits=%llu misses=%llu rate=%.1f%% layers=%llu\n",
                g_pred_hits, g_pred_misses, hit_rate, g_pred_layers);
    }
}

// ============================================================================
// bf16 <-> f32 conversion (CPU side)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__attribute__((unused))
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ============================================================================
// JSON parser (minimal, for model_weights.json)
// ============================================================================

// We use NSJSONSerialization via ObjC since we already link Foundation

typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];  // "U32", "BF16", "F32"
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    int capacity;
} TensorManifest;

static TensorManifest *load_manifest(const char *json_path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:json_path]];
        if (!data) {
            fprintf(stderr, "ERROR: Cannot read %s\n", json_path);
            return NULL;
        }

        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                             options:0
                                                               error:&error];
        if (!root) {
            fprintf(stderr, "ERROR: JSON parse failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }

        NSDictionary *tensors = root[@"tensors"];
        if (!tensors) {
            fprintf(stderr, "ERROR: No 'tensors' key in manifest\n");
            return NULL;
        }

        TensorManifest *m = calloc(1, sizeof(TensorManifest));
        m->capacity = (int)[tensors count] + 16;
        m->tensors = calloc(m->capacity, sizeof(TensorInfo));
        m->num_tensors = 0;

        for (NSString *key in tensors) {
            NSDictionary *info = tensors[key];
            TensorInfo *t = &m->tensors[m->num_tensors];

            const char *name = [key UTF8String];
            t->name = strdup(name);
            t->offset = [info[@"offset"] unsignedLongLongValue];
            t->size = [info[@"size"] unsignedLongLongValue];

            NSArray *shape = info[@"shape"];
            t->ndim = (int)[shape count];
            for (int i = 0; i < t->ndim && i < 4; i++) {
                t->shape[i] = [shape[i] intValue];
            }

            const char *dtype = [info[@"dtype"] UTF8String];
            strncpy(t->dtype, dtype, 7);

            m->num_tensors++;
        }

        printf("[manifest] Loaded %d tensors from %s\n", m->num_tensors, json_path);
        return m;
    }
}

// Hash table for O(1) tensor lookup (replaces O(N) linear scan).
// FNV-1a hash, open addressing with linear probing.
#define TENSOR_HT_SIZE 8192  // power of 2, > 4x num_tensors (2092)

typedef struct {
    const char *key;     // tensor name (pointer into TensorInfo)
    TensorInfo *value;   // pointer to tensor info
} TensorHTEntry;

static TensorHTEntry tensor_ht[TENSOR_HT_SIZE];
static int tensor_ht_built = 0;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static void build_tensor_ht(TensorManifest *m) {
    if (tensor_ht_built) return;
    memset(tensor_ht, 0, sizeof(tensor_ht));
    for (int i = 0; i < m->num_tensors; i++) {
        uint32_t idx = fnv1a(m->tensors[i].name) & (TENSOR_HT_SIZE - 1);
        while (tensor_ht[idx].key) {
            idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
        }
        tensor_ht[idx].key = m->tensors[i].name;
        tensor_ht[idx].value = &m->tensors[i];
    }
    tensor_ht_built = 1;
}

static TensorInfo *find_tensor(TensorManifest *m, const char *name) {
    if (!tensor_ht_built) build_tensor_ht(m);
    uint32_t idx = fnv1a(name) & (TENSOR_HT_SIZE - 1);
    while (tensor_ht[idx].key) {
        if (strcmp(tensor_ht[idx].key, name) == 0) {
            return tensor_ht[idx].value;
        }
        idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
    }
    return NULL;
}

// ============================================================================
// Weight file: mmap'd binary blob
// ============================================================================

typedef struct {
    void *data;
    size_t size;
    TensorManifest *manifest;
} WeightFile;

typedef struct {
    int manifest_layers;
    int tensors_present;
    int packed_experts_present;
    int enabled;
    char packed_dir[PATH_MAX];
} MTPArtifacts;

typedef struct {
    uint16_t *pre_fc_norm_hidden_w;
    uint16_t *pre_fc_norm_embedding_w;
    uint32_t *fc_w; uint16_t *fc_s, *fc_b;
    uint16_t *layer_input_norm_w;
    uint32_t *q_w; uint16_t *q_s, *q_b;
    uint32_t *k_w; uint16_t *k_s, *k_b;
    uint32_t *v_w; uint16_t *v_s, *v_b;
    uint32_t *o_w; uint16_t *o_s, *o_b;
    uint16_t *q_norm_w;
    uint16_t *k_norm_w;
    uint16_t *post_attn_norm_w;
    uint32_t *gate_w; uint16_t *gate_s, *gate_b;
    uint32_t *sg_w;   uint16_t *sg_s, *sg_b;
    uint32_t *su_w;   uint16_t *su_s, *su_b;
    uint32_t *sd_w;   uint16_t *sd_s, *sd_b;
    uint32_t *seg_w;  uint16_t *seg_s, *seg_b;
    uint16_t *norm_w;
    int ready;
} MTPWeightCache;

static MTPWeightCache g_mtp_cache = {0};

// Persistent KV cache for the single MTP decoder layer's causal self-attention.
// Built incrementally over accepted (real) tokens during decode, capped at
// GPU_KV_SEQ. Plain arrays (the KVCache struct is defined later in the file).
static float *g_mtp_k_cache = NULL;  // [GPU_KV_SEQ * num_kv_heads * head_dim]
static float *g_mtp_v_cache = NULL;
static int    g_mtp_kv_len = 0;

static void cpu_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int dim, float eps);
static void fast_dequant_matvec(const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
                                const float *x, float *out, int out_dim, int in_dim, int group_size);
static void cpu_dequant_matvec(const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
                               const float *x, float *out, int out_dim, int in_dim, int group_size);
static float vec_rms(const float *v, int n);
static void apply_rotary_emb(float *q, float *k, int pos, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim);
static void cpu_softmax(float *x, int dim);
static void cpu_topk(const float *scores, int dim, int K, int *indices, float *values);
static void cpu_normalize_weights(float *weights, int K);
static void cpu_swiglu(const float *gate, const float *up, float *out, int dim);
static float cpu_sigmoid(float x);
static void cpu_vec_madd(float *dst, const float *src, float scale, int dim);
static void lm_head_forward(WeightFile *wf, const float *hidden, float *logits);
static int cpu_argmax(const float *x, int dim);
static void embed_lookup(WeightFile *wf, int token_id, float *out);

static WeightFile *open_weights(const char *bin_path, const char *json_path) {
    // mmap the binary file
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", bin_path, strerror(errno));
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;

    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
        return NULL;
    }

    // Advise sequential access
    madvise(data, size, MADV_SEQUENTIAL);

    TensorManifest *manifest = load_manifest(json_path);
    if (!manifest) {
        munmap(data, size);
        return NULL;
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = size;
    wf->manifest = manifest;

    printf("[weights] mmap'd %.2f GB from %s\n", size / 1e9, bin_path);
    return wf;
}

static void *get_tensor_ptr(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return NULL;
    }
    return (char *)wf->data + t->offset;
}

static TensorInfo *get_tensor_info(WeightFile *wf, const char *name) {
    return find_tensor(wf->manifest, name);
}

static void *get_tensor_ptr_optional(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    return t ? (char *)wf->data + t->offset : NULL;
}

static int manifest_has_tensor(WeightFile *wf, const char *name) {
    return get_tensor_info(wf, name) != NULL;
}

static int read_mtp_packed_layers(const char *model_path) {
    @autoreleasepool {
        char layout_path[PATH_MAX];
        snprintf(layout_path, sizeof(layout_path), "%s/flashchat/packed_mtp_experts/layout.json", model_path);
        NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:layout_path]];
        if (!data) return 0;
        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (!root) return 0;
        NSNumber *layers = root[@"num_layers"];
        return layers ? [layers intValue] : 0;
    }
}

static MTPArtifacts detect_mtp_artifacts(WeightFile *wf, const char *model_path) {
    MTPArtifacts mtp = {0};
    mtp.enabled = g_mtp_predictions > 0;
    snprintf(mtp.packed_dir, sizeof(mtp.packed_dir), "%s/flashchat/packed_mtp_experts", model_path);

    mtp.tensors_present =
        manifest_has_tensor(wf, "mtp.fc.weight") &&
        manifest_has_tensor(wf, "mtp.fc.scales") &&
        manifest_has_tensor(wf, "mtp.fc.biases") &&
        manifest_has_tensor(wf, "mtp.pre_fc_norm_hidden.weight") &&
        manifest_has_tensor(wf, "mtp.pre_fc_norm_embedding.weight") &&
        manifest_has_tensor(wf, "mtp.layers.0.input_layernorm.weight") &&
        manifest_has_tensor(wf, "mtp.layers.0.self_attn.q_proj.weight") &&
        manifest_has_tensor(wf, "mtp.layers.0.self_attn.q_proj.scales") &&
        manifest_has_tensor(wf, "mtp.layers.0.self_attn.q_proj.biases") &&
        manifest_has_tensor(wf, "mtp.layers.0.mlp.gate.weight") &&
        manifest_has_tensor(wf, "mtp.layers.0.mlp.gate.scales") &&
        manifest_has_tensor(wf, "mtp.layers.0.mlp.gate.biases") &&
        manifest_has_tensor(wf, "mtp.norm.weight");

    mtp.manifest_layers = mtp.tensors_present ? 1 : 0;

    int packed_layers = read_mtp_packed_layers(model_path);
    if (packed_layers > 0) {
        char layer_path[PATH_MAX];
        snprintf(layer_path, sizeof(layer_path), "%s/layer_00.bin", mtp.packed_dir);
        mtp.packed_experts_present = access(layer_path, R_OK) == 0;
    }

    return mtp;
}

static void build_mtp_cache(WeightFile *wf) {
    memset(&g_mtp_cache, 0, sizeof(g_mtp_cache));
    g_mtp_cache.pre_fc_norm_hidden_w = get_tensor_ptr_optional(wf, "mtp.pre_fc_norm_hidden.weight");
    g_mtp_cache.pre_fc_norm_embedding_w = get_tensor_ptr_optional(wf, "mtp.pre_fc_norm_embedding.weight");
    g_mtp_cache.fc_w = get_tensor_ptr_optional(wf, "mtp.fc.weight");
    g_mtp_cache.fc_s = get_tensor_ptr_optional(wf, "mtp.fc.scales");
    g_mtp_cache.fc_b = get_tensor_ptr_optional(wf, "mtp.fc.biases");
    g_mtp_cache.layer_input_norm_w = get_tensor_ptr_optional(wf, "mtp.layers.0.input_layernorm.weight");
    g_mtp_cache.q_w = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.q_proj.weight");
    g_mtp_cache.q_s = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.q_proj.scales");
    g_mtp_cache.q_b = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.q_proj.biases");
    g_mtp_cache.k_w = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.k_proj.weight");
    g_mtp_cache.k_s = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.k_proj.scales");
    g_mtp_cache.k_b = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.k_proj.biases");
    g_mtp_cache.v_w = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.v_proj.weight");
    g_mtp_cache.v_s = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.v_proj.scales");
    g_mtp_cache.v_b = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.v_proj.biases");
    g_mtp_cache.o_w = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.o_proj.weight");
    g_mtp_cache.o_s = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.o_proj.scales");
    g_mtp_cache.o_b = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.o_proj.biases");
    g_mtp_cache.q_norm_w = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.q_norm.weight");
    g_mtp_cache.k_norm_w = get_tensor_ptr_optional(wf, "mtp.layers.0.self_attn.k_norm.weight");
    g_mtp_cache.post_attn_norm_w = get_tensor_ptr_optional(wf, "mtp.layers.0.post_attention_layernorm.weight");
    g_mtp_cache.gate_w = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.gate.weight");
    g_mtp_cache.gate_s = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.gate.scales");
    g_mtp_cache.gate_b = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.gate.biases");
    g_mtp_cache.sg_w = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.gate_proj.weight");
    g_mtp_cache.sg_s = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.gate_proj.scales");
    g_mtp_cache.sg_b = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.gate_proj.biases");
    g_mtp_cache.su_w = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.up_proj.weight");
    g_mtp_cache.su_s = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.up_proj.scales");
    g_mtp_cache.su_b = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.up_proj.biases");
    g_mtp_cache.sd_w = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.down_proj.weight");
    g_mtp_cache.sd_s = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.down_proj.scales");
    g_mtp_cache.sd_b = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert.down_proj.biases");
    g_mtp_cache.seg_w = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert_gate.weight");
    g_mtp_cache.seg_s = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert_gate.scales");
    g_mtp_cache.seg_b = get_tensor_ptr_optional(wf, "mtp.layers.0.mlp.shared_expert_gate.biases");
    g_mtp_cache.norm_w = get_tensor_ptr_optional(wf, "mtp.norm.weight");
    g_mtp_cache.ready =
        g_mtp_cache.pre_fc_norm_hidden_w &&
        g_mtp_cache.pre_fc_norm_embedding_w &&
        g_mtp_cache.fc_w && g_mtp_cache.fc_s && g_mtp_cache.fc_b &&
        g_mtp_cache.layer_input_norm_w &&
        g_mtp_cache.q_w && g_mtp_cache.q_s && g_mtp_cache.q_b &&
        g_mtp_cache.k_w && g_mtp_cache.k_s && g_mtp_cache.k_b &&
        g_mtp_cache.v_w && g_mtp_cache.v_s && g_mtp_cache.v_b &&
        g_mtp_cache.o_w && g_mtp_cache.o_s && g_mtp_cache.o_b &&
        g_mtp_cache.q_norm_w && g_mtp_cache.k_norm_w &&
        g_mtp_cache.post_attn_norm_w &&
        g_mtp_cache.gate_w && g_mtp_cache.gate_s && g_mtp_cache.gate_b &&
        g_mtp_cache.sg_w && g_mtp_cache.sg_s && g_mtp_cache.sg_b &&
        g_mtp_cache.su_w && g_mtp_cache.su_s && g_mtp_cache.su_b &&
        g_mtp_cache.sd_w && g_mtp_cache.sd_s && g_mtp_cache.sd_b &&
        g_mtp_cache.seg_w && g_mtp_cache.seg_s && g_mtp_cache.seg_b &&
        g_mtp_cache.norm_w;

    if (g_mtp_cache.ready && !g_mtp_k_cache) {
        int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
        g_mtp_k_cache = calloc((size_t)GPU_KV_SEQ * kv_dim, sizeof(float));
        g_mtp_v_cache = calloc((size_t)GPU_KV_SEQ * kv_dim, sizeof(float));
    }
    g_mtp_kv_len = 0;
}

static int mtp_preflight_forward(WeightFile *wf) {
    (void)wf;
    if (!g_mtp_cache.ready) {
        fprintf(stderr, "[mtp] preflight failed: MTP weight cache is incomplete\n");
        return 1;
    }

    float *hidden = calloc(g_cfg.hidden_dim, sizeof(float));
    float *embedding = calloc(g_cfg.hidden_dim, sizeof(float));
    float *hidden_norm = calloc(g_cfg.hidden_dim, sizeof(float));
    float *embedding_norm = calloc(g_cfg.hidden_dim, sizeof(float));
    float *fc_in = calloc(g_cfg.hidden_dim * 2, sizeof(float));
    float *fc_out = calloc(g_cfg.hidden_dim, sizeof(float));
    if (!hidden || !embedding || !hidden_norm || !embedding_norm || !fc_in || !fc_out) {
        fprintf(stderr, "[mtp] preflight failed: allocation failure\n");
        free(hidden); free(embedding); free(hidden_norm); free(embedding_norm); free(fc_in); free(fc_out);
        return 1;
    }

    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        hidden[i] = sinf((float)i * 0.001f);
        embedding[i] = cosf((float)i * 0.001f);
    }

    cpu_rms_norm(hidden, g_mtp_cache.pre_fc_norm_hidden_w, hidden_norm, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
    cpu_rms_norm(embedding, g_mtp_cache.pre_fc_norm_embedding_w, embedding_norm, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
    // Fusion order matches Qwen3-Next MTP reference: cat([norm_embedding, norm_hidden]).
    memcpy(fc_in, embedding_norm, g_cfg.hidden_dim * sizeof(float));
    memcpy(fc_in + g_cfg.hidden_dim, hidden_norm, g_cfg.hidden_dim * sizeof(float));
    fast_dequant_matvec(g_mtp_cache.fc_w, g_mtp_cache.fc_s, g_mtp_cache.fc_b,
                        fc_in, fc_out, g_cfg.hidden_dim, g_cfg.hidden_dim * 2, g_cfg.group_size);

    float rms = vec_rms(fc_out, g_cfg.hidden_dim);
    int ok = isfinite(rms) && rms > 0.0f;
    fprintf(stderr, "[mtp] preflight fc_out_rms=%.6f\n", rms);

    free(hidden); free(embedding); free(hidden_norm); free(embedding_norm); free(fc_in); free(fc_out);
    return ok ? 0 : 1;
}

static int mtp_forward(WeightFile *wf, const char *model_path,
                       const float *input_hidden, const float *next_embedding,
                       float *out_hidden, int *out_draft_token, int log_details) {
    if (!g_mtp_cache.ready) {
        fprintf(stderr, "[mtp] forward failed: MTP weight cache is incomplete\n");
        return 1;
    }

    int rc = 1;
    int q_proj_dim = g_cfg.num_attn_heads * g_cfg.head_dim * 2;
    int q_dim = g_cfg.num_attn_heads * g_cfg.head_dim;
    int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    int heads_per_kv = g_cfg.num_attn_heads / g_cfg.num_kv_heads;

    float *hidden_norm = calloc(g_cfg.hidden_dim, sizeof(float));
    float *embedding_norm = calloc(g_cfg.hidden_dim, sizeof(float));
    float *fc_in = calloc(g_cfg.hidden_dim * 2, sizeof(float));
    float *x = calloc(g_cfg.hidden_dim, sizeof(float));
    float *residual = calloc(g_cfg.hidden_dim, sizeof(float));
    float *normed = calloc(g_cfg.hidden_dim, sizeof(float));
    float *q_proj = calloc(q_proj_dim, sizeof(float));
    float *k = calloc(kv_dim, sizeof(float));
    float *v = calloc(kv_dim, sizeof(float));
    float *q = calloc(q_dim, sizeof(float));
    float *q_gate = calloc(q_dim, sizeof(float));
    float *attn_out = calloc(q_dim, sizeof(float));
    float *attn_projected = calloc(g_cfg.hidden_dim, sizeof(float));
    float *h_post = calloc(g_cfg.hidden_dim, sizeof(float));
    float *gate_scores = calloc(g_cfg.num_experts, sizeof(float));
    float *shared_gate = calloc(g_cfg.shared_intermediate, sizeof(float));
    float *shared_up = calloc(g_cfg.shared_intermediate, sizeof(float));
    float *shared_act = calloc(g_cfg.shared_intermediate, sizeof(float));
    float *shared_out = calloc(g_cfg.hidden_dim, sizeof(float));
    float *moe_out = calloc(g_cfg.hidden_dim, sizeof(float));
    float *final_normed = calloc(g_cfg.hidden_dim, sizeof(float));
    if (!input_hidden || !next_embedding || !hidden_norm || !embedding_norm || !fc_in || !x ||
        !residual || !normed || !q_proj || !k || !v || !q || !q_gate || !attn_out ||
        !attn_projected || !h_post || !gate_scores || !shared_gate || !shared_up ||
        !shared_act || !shared_out || !moe_out || !final_normed) {
        fprintf(stderr, "[mtp] forward failed: allocation failure\n");
        goto cleanup;
    }

    cpu_rms_norm(input_hidden, g_mtp_cache.pre_fc_norm_hidden_w, hidden_norm, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
    cpu_rms_norm(next_embedding, g_mtp_cache.pre_fc_norm_embedding_w, embedding_norm, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
    // Fusion order matches Qwen3-Next MTP reference: cat([norm_embedding, norm_hidden]).
    memcpy(fc_in, embedding_norm, g_cfg.hidden_dim * sizeof(float));
    memcpy(fc_in + g_cfg.hidden_dim, hidden_norm, g_cfg.hidden_dim * sizeof(float));
    fast_dequant_matvec(g_mtp_cache.fc_w, g_mtp_cache.fc_s, g_mtp_cache.fc_b,
                        fc_in, x, g_cfg.hidden_dim, g_cfg.hidden_dim * 2, g_cfg.group_size);
    memcpy(residual, x, g_cfg.hidden_dim * sizeof(float));

    cpu_rms_norm(x, g_mtp_cache.layer_input_norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
    fast_dequant_matvec(g_mtp_cache.q_w, g_mtp_cache.q_s, g_mtp_cache.q_b,
                        normed, q_proj, q_proj_dim, g_cfg.hidden_dim, g_cfg.group_size);
    fast_dequant_matvec(g_mtp_cache.k_w, g_mtp_cache.k_s, g_mtp_cache.k_b,
                        normed, k, kv_dim, g_cfg.hidden_dim, g_cfg.group_size);
    fast_dequant_matvec(g_mtp_cache.v_w, g_mtp_cache.v_s, g_mtp_cache.v_b,
                        normed, v, kv_dim, g_cfg.hidden_dim, g_cfg.group_size);

    for (int h = 0; h < g_cfg.num_attn_heads; h++) {
        float *src = q_proj + h * (2 * g_cfg.head_dim);
        memcpy(q + h * g_cfg.head_dim, src, g_cfg.head_dim * sizeof(float));
        memcpy(q_gate + h * g_cfg.head_dim, src + g_cfg.head_dim, g_cfg.head_dim * sizeof(float));
    }
    for (int h = 0; h < g_cfg.num_attn_heads; h++) {
        float *qh = q + h * g_cfg.head_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < g_cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
        float inv_rms = 1.0f / sqrtf(sum_sq / g_cfg.head_dim + g_cfg.rms_norm_eps);
        for (int i = 0; i < g_cfg.head_dim; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(g_mtp_cache.q_norm_w[i]);
    }
    for (int h = 0; h < g_cfg.num_kv_heads; h++) {
        float *kh = k + h * g_cfg.head_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < g_cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
        float inv_rms = 1.0f / sqrtf(sum_sq / g_cfg.head_dim + g_cfg.rms_norm_eps);
        for (int i = 0; i < g_cfg.head_dim; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(g_mtp_cache.k_norm_w[i]);
    }
    int mtp_pos = g_mtp_kv_len;
    apply_rotary_emb(q, k, mtp_pos, g_cfg.num_attn_heads, g_cfg.num_kv_heads, g_cfg.head_dim, g_cfg.rotary_dim);

    // Append this (accepted) token's K/V to the persistent MTP KV cache, then run
    // real causal scaled-dot-product attention over the cached context. This
    // mirrors the backbone full-attention layer and replaces the earlier
    // context-free copy-V placeholder (which pinned RoPE to position 0).
    if (g_mtp_k_cache && g_mtp_v_cache && g_mtp_kv_len < GPU_KV_SEQ) {
        memcpy(g_mtp_k_cache + (size_t)g_mtp_kv_len * kv_dim, k, kv_dim * sizeof(float));
        memcpy(g_mtp_v_cache + (size_t)g_mtp_kv_len * kv_dim, v, kv_dim * sizeof(float));
        g_mtp_kv_len++;
    }
    const float *kc = g_mtp_k_cache ? g_mtp_k_cache : k;
    const float *vc = g_mtp_v_cache ? g_mtp_v_cache : v;
    int ctx_len = g_mtp_kv_len > 0 ? g_mtp_kv_len : 1;
    float attn_scale = 1.0f / sqrtf((float)g_cfg.head_dim);
    for (int h = 0; h < g_cfg.num_attn_heads; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * g_cfg.head_dim;
        float *scores = malloc(ctx_len * sizeof(float));
        for (int p = 0; p < ctx_len; p++) {
            const float *kp = kc + (size_t)p * kv_dim + kv_h * g_cfg.head_dim;
            float dot = 0.0f;
            for (int d = 0; d < g_cfg.head_dim; d++) dot += qh[d] * kp[d];
            scores[p] = dot * attn_scale;
        }
        cpu_softmax(scores, ctx_len);
        float *oh = attn_out + h * g_cfg.head_dim;
        for (int d = 0; d < g_cfg.head_dim; d++) oh[d] = 0.0f;
        for (int p = 0; p < ctx_len; p++) {
            const float *vp = vc + (size_t)p * kv_dim + kv_h * g_cfg.head_dim;
            for (int d = 0; d < g_cfg.head_dim; d++) oh[d] += scores[p] * vp[d];
        }
        free(scores);
    }
    for (int i = 0; i < q_dim; i++) {
        attn_out[i] *= cpu_sigmoid(q_gate[i]);
    }
    fast_dequant_matvec(g_mtp_cache.o_w, g_mtp_cache.o_s, g_mtp_cache.o_b,
                        attn_out, attn_projected, g_cfg.hidden_dim, q_dim, g_cfg.group_size);
    for (int i = 0; i < g_cfg.hidden_dim; i++) x[i] = residual[i] + attn_projected[i];

    cpu_rms_norm(x, g_mtp_cache.post_attn_norm_w, h_post, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
    fast_dequant_matvec(g_mtp_cache.gate_w, g_mtp_cache.gate_s, g_mtp_cache.gate_b,
                        h_post, gate_scores, g_cfg.num_experts, g_cfg.hidden_dim, g_cfg.group_size);
    fast_dequant_matvec(g_mtp_cache.sg_w, g_mtp_cache.sg_s, g_mtp_cache.sg_b,
                        h_post, shared_gate, g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
    fast_dequant_matvec(g_mtp_cache.su_w, g_mtp_cache.su_s, g_mtp_cache.su_b,
                        h_post, shared_up, g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
    fast_dequant_matvec(g_mtp_cache.seg_w, g_mtp_cache.seg_s, g_mtp_cache.seg_b,
                        h_post, &shared_act[0], 1, g_cfg.hidden_dim, g_cfg.group_size);
    float shared_gate_score = shared_act[0];

    cpu_softmax(gate_scores, g_cfg.num_experts);
    int expert_index = 0;
    float expert_weight = 1.0f;
    cpu_topk(gate_scores, g_cfg.num_experts, 1, &expert_index, &expert_weight);
    cpu_normalize_weights(&expert_weight, 1);

    char packed_path[PATH_MAX];
    snprintf(packed_path, sizeof(packed_path), "%s/flashchat/packed_mtp_experts/layer_00.bin", model_path);
    int fd = open(packed_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[mtp] forward failed: cannot open %s: %s\n", packed_path, strerror(errno));
        goto cleanup;
    }
    size_t esz = active_expert_size();
    void *expert_data = malloc(esz);
    if (!expert_data) {
        close(fd);
        fprintf(stderr, "[mtp] forward failed: expert allocation failure\n");
        goto cleanup;
    }
    ssize_t nread = pread(fd, expert_data, esz, (off_t)expert_index * esz);
    close(fd);
    if (nread != (ssize_t)esz) {
        memset(expert_data, 0, esz);
        expert_index = 0;
        fd = open(packed_path, O_RDONLY);
        nread = fd >= 0 ? pread(fd, expert_data, esz, 0) : -1;
        if (fd >= 0) close(fd);
    }
    if (nread == (ssize_t)esz) {
        uint32_t *gw = (uint32_t *)expert_data;
        uint16_t *gs_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size);
        uint16_t *gb_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size);
        uint32_t *uw = (uint32_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size);
        uint16_t *us_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size);
        uint16_t *ub_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size);
        uint32_t *dw = (uint32_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size);
        uint16_t *ds_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size + g_cfg.down_w_size);
        uint16_t *db_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size + g_cfg.down_w_size + g_cfg.down_s_size);
        float *eg = calloc(g_cfg.moe_intermediate, sizeof(float));
        float *eu = calloc(g_cfg.moe_intermediate, sizeof(float));
        float *ea = calloc(g_cfg.moe_intermediate, sizeof(float));
        float *eo = calloc(g_cfg.hidden_dim, sizeof(float));
        if (eg && eu && ea && eo) {
            cpu_dequant_matvec(gw, gs_p, gb_p, h_post, eg, g_cfg.moe_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
            cpu_dequant_matvec(uw, us_p, ub_p, h_post, eu, g_cfg.moe_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
            cpu_swiglu(eg, eu, ea, g_cfg.moe_intermediate);
            cpu_dequant_matvec(dw, ds_p, db_p, ea, eo, g_cfg.hidden_dim, g_cfg.moe_intermediate, g_cfg.group_size);
            cpu_vec_madd(moe_out, eo, expert_weight, g_cfg.hidden_dim);
        }
        free(eg); free(eu); free(ea); free(eo);
    }
    free(expert_data);

    cpu_swiglu(shared_gate, shared_up, shared_act, g_cfg.shared_intermediate);
    fast_dequant_matvec(g_mtp_cache.sd_w, g_mtp_cache.sd_s, g_mtp_cache.sd_b,
                        shared_act, shared_out, g_cfg.hidden_dim, g_cfg.shared_intermediate, g_cfg.group_size);
    float shared_weight = cpu_sigmoid(shared_gate_score);
    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        x[i] = x[i] + moe_out[i] + shared_out[i] * shared_weight;
    }
    cpu_rms_norm(x, g_mtp_cache.norm_w, final_normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);

    float fc_rms = vec_rms(residual, g_cfg.hidden_dim);
    float layer_rms = vec_rms(x, g_cfg.hidden_dim);
    float final_rms = vec_rms(final_normed, g_cfg.hidden_dim);
    if (log_details) {
        fprintf(stderr, "[mtp] decoder preflight fc_rms=%.6f layer_rms=%.6f final_rms=%.6f expert=%d\n",
                fc_rms, layer_rms, final_rms, expert_index);
    }
    if (out_hidden) {
        memcpy(out_hidden, final_normed, g_cfg.hidden_dim * sizeof(float));
    }
    if (out_draft_token) {
        *out_draft_token = -1;
    }
    if (get_tensor_info(wf, "lm_head.weight") &&
        get_tensor_info(wf, "lm_head.scales") &&
        get_tensor_info(wf, "lm_head.biases")) {
        float *logits = calloc(g_cfg.vocab_size, sizeof(float));
        if (logits) {
            lm_head_forward(wf, final_normed, logits);
            int draft_token = cpu_argmax(logits, g_cfg.vocab_size);
            if (out_draft_token) {
                *out_draft_token = draft_token;
            }
            if (log_details) {
                fprintf(stderr, "[mtp] decoder preflight draft_token=%d draft_logit=%.6f\n",
                        draft_token, logits[draft_token]);
            }
            free(logits);
        }
    } else if (log_details) {
        fprintf(stderr, "[mtp] decoder preflight draft logits skipped; lm_head tensors not present\n");
    }
    rc = (isfinite(final_rms) && final_rms > 0.0f) ? 0 : 1;

cleanup:
    free(hidden_norm); free(embedding_norm); free(fc_in); free(x);
    free(residual); free(normed); free(q_proj); free(k); free(v); free(q); free(q_gate);
    free(attn_out); free(attn_projected); free(h_post); free(gate_scores); free(shared_gate);
    free(shared_up); free(shared_act); free(shared_out); free(moe_out); free(final_normed);
    return rc;
}

static int mtp_preflight_decoder_layer(WeightFile *wf, const char *model_path) {
    float *hidden = calloc(g_cfg.hidden_dim, sizeof(float));
    float *embedding = calloc(g_cfg.hidden_dim, sizeof(float));
    float *out_hidden = calloc(g_cfg.hidden_dim, sizeof(float));
    if (!hidden || !embedding || !out_hidden) {
        fprintf(stderr, "[mtp] decoder preflight failed: allocation failure\n");
        free(hidden); free(embedding); free(out_hidden);
        return 1;
    }

    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        hidden[i] = sinf((float)i * 0.001f);
        embedding[i] = cosf((float)i * 0.001f);
    }

    int draft_token = -1;
    int rc = mtp_forward(wf, model_path, hidden, embedding, out_hidden, &draft_token, 1);
    float out_rms = vec_rms(out_hidden, g_cfg.hidden_dim);
    fprintf(stderr, "[mtp] reusable forward preflight out_rms=%.6f draft_token=%d\n",
            out_rms, draft_token);

    free(hidden);
    free(embedding);
    free(out_hidden);
    return rc;
}

static int mtp_can_shadow_draft(WeightFile *wf) {
    return g_mtp_predictions > 0 && g_mtp_cache.ready &&
           get_tensor_info(wf, "lm_head.weight") &&
           get_tensor_info(wf, "lm_head.scales") &&
           get_tensor_info(wf, "lm_head.biases") &&
           get_tensor_info(wf, "model.embed_tokens.weight") &&
           get_tensor_info(wf, "model.embed_tokens.scales") &&
           get_tensor_info(wf, "model.embed_tokens.biases");
}

static int mtp_shadow_draft_token(WeightFile *wf, const char *model_path,
                                  const float *backbone_hidden, int accepted_token,
                                  int *out_draft_token) {
    if (!mtp_can_shadow_draft(wf) || accepted_token < 0 || accepted_token >= g_cfg.vocab_size) {
        if (out_draft_token) *out_draft_token = -1;
        return -1;
    }

    float *embedding = calloc(g_cfg.hidden_dim, sizeof(float));
    float *mtp_hidden = calloc(g_cfg.hidden_dim, sizeof(float));
    if (!embedding || !mtp_hidden) {
        free(embedding);
        free(mtp_hidden);
        if (out_draft_token) *out_draft_token = -1;
        return -1;
    }

    embed_lookup(wf, accepted_token, embedding);
    int draft_token = -1;
    int rc = mtp_forward(wf, model_path, backbone_hidden, embedding, mtp_hidden, &draft_token, 0);
    free(embedding);
    free(mtp_hidden);
    if (out_draft_token) *out_draft_token = draft_token;
    return rc == 0 && draft_token >= 0 ? 0 : -1;
}

// Report measured expert overlap and project the I/O-bound speedup ceiling for a
// batched draft/verify forward, given the observed MTP draft acceptance rate.
static void mtp_overlap_report(int shadow_hits, int shadow_checks) {
    if (!g_mtp_overlap_enabled || g_overlap_layer_pairs == 0) return;

    double mean_single = (double)g_overlap_single_sum / g_overlap_layer_pairs;
    double overlap_frac = (double)g_overlap_intersect_sum / (double)g_overlap_single_sum;
    // Batched 2-position forward loads the union; two separate forwards load both
    // sets. union/separate is the I/O a batched verify costs vs. plain decoding of
    // the same two tokens (0.5 = perfect overlap/2x potential, 1.0 = break-even).
    double io_ratio = (double)g_overlap_union_sum / (double)g_overlap_separate_sum;

    fprintf(stderr, "[mtp] expert overlap: layer-pairs=%d mean_active=%.2f/layer "
                    "overlap=%.1f%% batched_io_vs_separate=%.3f\n",
            g_overlap_layer_pairs, mean_single, 100.0 * overlap_frac, io_ratio);

    if (shadow_checks > 0) {
        double A = (double)shadow_hits / (double)shadow_checks;
        // A batched forward of [accepted, draft] loads union experts and yields
        // (1+A) committed tokens on average (2 if accepted, 1 if rejected).
        // I/O-bound speedup vs. one single-token forward per token:
        //   single_per_tok / (union_per_batch / (1+A)) = (1+A) * single/union.
        double single_over_union = (double)g_overlap_single_sum / (double)g_overlap_union_sum;
        double projected = (1.0 + A) * single_over_union;
        fprintf(stderr, "[mtp] projection: acceptance=%.1f%% -> I/O-bound speedup ~%.2fx "
                        "(>1 worth building, ~1 break-even). GPU compute not modeled.\n",
                100.0 * A, projected);
    }
}

// ============================================================================
// Vocabulary for token decoding
// ============================================================================

typedef struct {
    char **tokens;   // token_id -> UTF-8 string
    int *lengths;    // token_id -> byte length
    int num_tokens;
} Vocabulary;

static Vocabulary *load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open vocab %s\n", path);
        return NULL;
    }

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "BPET", 4) != 0) {
        fprintf(stderr, "ERROR: Invalid vocab file (bad magic)\n");
        fclose(f);
        return NULL;
    }

    uint32_t version, vocab_size, num_merges, num_added;
    fread(&version, 4, 1, f);
    fread(&vocab_size, 4, 1, f);
    fread(&num_merges, 4, 1, f);
    fread(&num_added, 4, 1, f);

    // Total tokens = highest token ID in vocab
    // Vocab entries have IDs up to ~151936, added tokens use 248044-248077
    uint32_t total_tokens = 248078;  // max added token ID + 1

    Vocabulary *v = calloc(1, sizeof(Vocabulary));
    v->num_tokens = total_tokens;
    v->tokens = calloc(total_tokens, sizeof(char *));
    v->lengths = calloc(total_tokens, sizeof(int));

    // Read vocab entries
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t token_id;
        uint16_t byte_len;
        fread(&token_id, 4, 1, f);
        fread(&byte_len, 2, 1, f);
        if (byte_len > 0 && token_id < total_tokens) {
            v->tokens[token_id] = malloc(byte_len + 1);
            fread(v->tokens[token_id], 1, byte_len, f);
            v->tokens[token_id][byte_len] = '\0';
            v->lengths[token_id] = byte_len;
        } else {
            fseek(f, byte_len, SEEK_CUR);
        }
    }

    // Skip merges section
    for (uint32_t i = 0; i < num_merges; i++) {
        uint16_t len_a, len_b;
        fread(&len_a, 2, 1, f);
        fseek(f, len_a, SEEK_CUR);
        fread(&len_b, 2, 1, f);
        fseek(f, len_b, SEEK_CUR);
    }

    // Read added tokens
    for (uint32_t i = 0; i < num_added; i++) {
        uint32_t token_id;
        uint16_t byte_len;
        fread(&token_id, 4, 1, f);
        fread(&byte_len, 2, 1, f);
        if (byte_len > 0 && token_id < total_tokens) {
            v->tokens[token_id] = malloc(byte_len + 1);
            fread(v->tokens[token_id], 1, byte_len, f);
            v->tokens[token_id][byte_len] = '\0';
            v->lengths[token_id] = byte_len;
        } else {
            fseek(f, byte_len, SEEK_CUR);
        }
    }

    fclose(f);
    printf("[vocab] Loaded %d tokens\n", v->num_tokens);
    return v;
}

static const char *decode_token(Vocabulary *v, int token_id) {
    if (token_id < 0 || token_id >= v->num_tokens) {
        return "<unk>";
    }
    if (!v->tokens[token_id]) {
        return "<unk>";
    }
    
    const char *src = v->tokens[token_id];
    
    // Filter out special control tokens
    if (strcmp(src, "<|im_end|>") == 0) return "";
    if (strcmp(src, "<|im_start|>") == 0) return "";
    if (strcmp(src, "<|endoftext|>") == 0) return "";
    if (strcmp(src, "<think>") == 0) return "";
    if (strcmp(src, "") == 0) return "";
    if (strcmp(src, "<|im_start|>user") == 0) return "";
    if (strcmp(src, "<|im_start|>assistant") == 0) return "";
    if (strcmp(src, "<|im_start|>system") == 0) return "";
    
    // Replace UTF-8 encoded bytes with actual characters
    static char buf[256];
    char *dst = buf;
    const unsigned char *s = (const unsigned char *)src;
    
    while (*s && dst < buf + 250) {
        // 0xC4 0x8A = U+010A (newline in GPT-2 byte encoding)
        if (s[0] == 0xC4 && s[1] == 0x8A) { *dst++ = '\n'; s += 2; continue; }
        // 0xC4 0x8B = U+010B (tab)
        if (s[0] == 0xC4 && s[1] == 0x8B) { *dst++ = '\t'; s += 2; continue; }
        // 0xC4 0xA0 = U+0120 (space marker Ġ)
        if (s[0] == 0xC4 && s[1] == 0xA0) { *dst++ = ' '; s += 2; continue; }
        // 0xC4 0xA1 = U+0121 (ãŁ - often before punctuation)
        if (s[0] == 0xC4 && s[1] == 0xA1) { *dst++ = ' '; s += 2; continue; }
        // 0xC3 0xA2 0xC4 0xA2 0xC4 0xB6 = corrupted "âĢĶ" (double-encoded ellipsis/space)
        // Found in Qwen3 tokenizer vocab - convert to space
        if (s[0] == 0xC3 && s[1] == 0xA2 && s[2] == 0xC4 && s[3] == 0xA2 && s[4] == 0xC4 && s[5] == 0xB6) {
            *dst++ = ' '; s += 6; continue;
        }
        // 0xE2 0x80 0xA6 = U+2026 (ellipsis â€¦)
        if (s[0] == 0xE2 && s[1] == 0x80 && s[2] == 0xA6) { *dst++ = '.'; *dst++ = '.'; *dst++ = '.'; s += 3; continue; }
        // 0xE2 0x80 0xA0 = U+2020 (en space)
        if (s[0] == 0xE2 && s[1] == 0x80 && s[2] == 0xA0) { *dst++ = ' '; s += 3; continue; }
        // 0xE2 0x80 0x94 = U+2014 (em dash —)
        if (s[0] == 0xE2 && s[1] == 0x80 && s[2] == 0x94) { *dst++ = '-'; *dst++ = '-'; s += 3; continue; }
        // 0xE2 0x80 0x99 = U+2019 (right single quote ')
        if (s[0] == 0xE2 && s[1] == 0x80 && s[2] == 0x99) { *dst++ = '\''; s += 3; continue; }
        *dst++ = *s++;
    }
    *dst = '\0';
    
    return buf;
}

// ============================================================================
// Prompt tokens loader
// ============================================================================

typedef struct {
    uint32_t *ids;
    int count;
} PromptTokens;

static PromptTokens *load_prompt_tokens(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    fread(&pt->count, 4, 1, f);
    pt->ids = malloc(pt->count * sizeof(uint32_t));
    fread(pt->ids, 4, pt->count, f);
    fclose(f);
    return pt;
}

// ============================================================================
// C BPE tokenizer (replaces Python encode_prompt.py)
// ============================================================================
#define TOKENIZER_IMPL
#include "tokenizer.h"

static bpe_tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;

static void init_tokenizer(void) {
    if (g_tokenizer_loaded) return;
    
    const char *env_model_path = getenv("FLASHCHAT_MODEL_PATH");
    char model_vocab_fc[1024];
    char model_vocab[1024];
    if (env_model_path) {
        snprintf(model_vocab_fc, sizeof(model_vocab_fc), "%s/flashchat/vocab.bin", env_model_path);
        snprintf(model_vocab, sizeof(model_vocab), "%s/vocab.bin", env_model_path);
    }
    
    const char *paths[] = {
        env_model_path ? model_vocab_fc : NULL,
        env_model_path ? model_vocab : NULL,
        "tokenizer.bin",
        "metal_infer/tokenizer.bin",
        "vocab.bin",
        "metal_infer/vocab.bin",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        if (access(paths[i], R_OK) == 0) {
            if (bpe_load(&g_tokenizer, paths[i]) == 0) {
                g_tokenizer_loaded = 1;
                return;
            }
        }
    }
    fprintf(stderr, "WARNING: tokenizer.bin not found, tokenization will fail\n");
}

static PromptTokens *encode_prompt_text_to_tokens(const char *text) {
    init_tokenizer();
    if (!g_tokenizer_loaded) return NULL;

    // Allocate output buffer (generous: 4 tokens per character worst case)
    int max_ids = (int)strlen(text) * 4 + 256;
    uint32_t *ids = malloc(max_ids * sizeof(uint32_t));
    if (!ids) return NULL;

    int n = bpe_encode(&g_tokenizer, text, ids, max_ids);
    if (n < 0) { free(ids); return NULL; }

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = ids;
    pt->count = n;

    fprintf(stderr, "Tokens (%d): [", n);
    for (int i = 0; i < n && i < 20; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%u", ids[i]);
    }
    if (n > 20) fprintf(stderr, ", ...");
    fprintf(stderr, "]\n");

    return pt;
}

// ============================================================================
// CPU computation kernels
// ============================================================================

// 4-bit dequant matvec: out[out_dim] = W * x[in_dim]
// W is stored as packed uint32 (8 x 4-bit values per uint32)
// scales/biases are bfloat16 per group
static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;

    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int x_base = base_x + p * 8;

                for (int n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    acc += ((float)nibble * scale + bias) * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

// RMS normalization: out = x * w / rms(x)
static void cpu_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; i++) {
        float weight = bf16_to_f32(w_bf16[i]);
        out[i] = x[i] * inv_rms * weight;
    }
}

// SwiGLU: out = silu(gate) * up
static void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// Sigmoid
static float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softmax over a vector
static void cpu_softmax(float *x, int dim) {
    float max_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
        x[i] *= inv_sum;
    }
}

// Top-K: find K largest indices from scores[dim]
static void cpu_topk(const float *scores, int dim, int K, int *indices, float *values) {
    // Simple selection sort for small K
    // Initialize with -inf
    for (int k = 0; k < K; k++) {
        values[k] = -1e30f;
        indices[k] = 0;
    }

    for (int i = 0; i < dim; i++) {
        // Check if this score beats the smallest in our top-K
        int min_k = 0;
        for (int k = 1; k < K; k++) {
            if (values[k] < values[min_k]) min_k = k;
        }
        if (scores[i] > values[min_k]) {
            values[min_k] = scores[i];
            indices[min_k] = i;
        }
    }
}

// Normalize top-K weights to sum to 1
static void cpu_normalize_weights(float *weights, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += weights[k];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int k = 0; k < K; k++) weights[k] *= inv;
    }
}

// Element-wise add: dst += src
__attribute__((unused))
static void cpu_vec_add(float *dst, const float *src, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += src[i];
}

// Element-wise multiply-add: dst += scale * src
static void cpu_vec_madd(float *dst, const float *src, float scale, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += scale * src[i];
}

// Element-wise multiply: dst = a * b
__attribute__((unused))
static void cpu_vec_mul(float *dst, const float *a, const float *b, int dim) {
    for (int i = 0; i < dim; i++) dst[i] = a[i] * b[i];
}

// Copy
static void cpu_vec_copy(float *dst, const float *src, int dim) {
    memcpy(dst, src, dim * sizeof(float));
}

// Zero
__attribute__((unused))
static void cpu_vec_zero(float *dst, int dim) {
    memset(dst, 0, dim * sizeof(float));
}

// Argmax
static int cpu_argmax(const float *x, int dim) {
    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}

static float clampf_local(float x, float minv, float maxv) {
    if (x < minv) return minv;
    if (x > maxv) return maxv;
    return x;
}

static void apply_sampling_penalties(float *logits, int dim, const int *token_counts,
                                     float presence_penalty, float repetition_penalty) {
    if (!logits || !token_counts) return;
    int use_presence = fabsf(presence_penalty) > 0.000001f;
    int use_repetition = repetition_penalty > 0.0f && fabsf(repetition_penalty - 1.0f) > 0.000001f;
    if (!use_presence && !use_repetition) return;

    repetition_penalty = clampf_local(repetition_penalty, 0.01f, 10.0f);
    for (int i = 0; i < dim; i++) {
        if (token_counts[i] <= 0) continue;
        if (use_repetition) {
            if (logits[i] > 0.0f) logits[i] /= repetition_penalty;
            else logits[i] *= repetition_penalty;
        }
        if (use_presence) {
            logits[i] -= presence_penalty;
        }
    }
}

static int sample_top_p_temperature(const float *logits, int dim, float temperature,
                                    float top_p, int top_k, float min_p) {
    if (temperature <= 0.0f) {
        return cpu_argmax(logits, dim);
    }

    temperature = clampf_local(temperature, 0.01f, 5.0f);
    top_p = clampf_local(top_p, 0.01f, 1.0f);
    min_p = clampf_local(min_p, 0.0f, 1.0f);
    enum { SAMPLE_MAX_TOP_K = 1024 };
    int sample_top_k = top_k > 0 ? top_k : 256;
    if (sample_top_k > SAMPLE_MAX_TOP_K) sample_top_k = SAMPLE_MAX_TOP_K;
    if (sample_top_k > dim) sample_top_k = dim;

    int top_idx[SAMPLE_MAX_TOP_K];
    float top_logits[SAMPLE_MAX_TOP_K];
    int top_count = 0;

    for (int i = 0; i < dim; i++) {
        float val = logits[i];
        if (top_count < sample_top_k) {
            int j = top_count++;
            while (j > 0 && val > top_logits[j - 1]) {
                top_logits[j] = top_logits[j - 1];
                top_idx[j] = top_idx[j - 1];
                j--;
            }
            top_logits[j] = val;
            top_idx[j] = i;
            continue;
        }
        if (val <= top_logits[top_count - 1]) continue;
        int j = top_count - 1;
        while (j > 0 && val > top_logits[j - 1]) {
            top_logits[j] = top_logits[j - 1];
            top_idx[j] = top_idx[j - 1];
            j--;
        }
        top_logits[j] = val;
        top_idx[j] = i;
    }

    if (top_count == 0) return cpu_argmax(logits, dim);

    float max_logit = top_logits[0];
    float probs[SAMPLE_MAX_TOP_K];
    float sum = 0.0f;
    for (int i = 0; i < top_count; i++) {
        probs[i] = expf((top_logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f) return top_idx[0];
    for (int i = 0; i < top_count; i++) probs[i] /= sum;

    int min_p_limit = top_count;
    if (min_p > 0.0f) {
        float threshold = probs[0] * min_p;
        min_p_limit = 1;
        while (min_p_limit < top_count && probs[min_p_limit] >= threshold) {
            min_p_limit++;
        }
    }

    int limit = top_count;
    float cumulative = 0.0f;
    for (int i = 0; i < min_p_limit; i++) {
        cumulative += probs[i];
        if (cumulative >= top_p) {
            limit = i + 1;
            break;
        }
    }
    if (limit > min_p_limit) limit = min_p_limit;

    float renorm = 0.0f;
    for (int i = 0; i < limit; i++) renorm += probs[i];
    if (renorm <= 0.0f) return top_idx[0];
    float target = ((float)arc4random() / ((float)UINT32_MAX + 1.0f)) * renorm;
    float running = 0.0f;
    for (int i = 0; i < limit; i++) {
        running += probs[i];
        if (target <= running) {
            return top_idx[i];
        }
    }
    return top_idx[limit - 1];
}

static int pick_next_token(const float *logits, int dim, float temperature, float top_p,
                           int top_k, float min_p, float presence_penalty,
                           float repetition_penalty, const int *token_counts,
                           int reasoning_enabled) {
    float *tmp = NULL;
    const float *use_logits = logits;
    if (!reasoning_enabled || token_counts || fabsf(presence_penalty) > 0.000001f ||
        fabsf(repetition_penalty - 1.0f) > 0.000001f) {
        tmp = malloc((size_t)dim * sizeof(float));
        if (tmp) {
            memcpy(tmp, logits, (size_t)dim * sizeof(float));
            use_logits = tmp;
        }
    }
    if (tmp) {
        apply_sampling_penalties(tmp, dim, token_counts, presence_penalty, repetition_penalty);
        if (!reasoning_enabled) {
            tmp[g_cfg.think_start_token] = -1e30f;
            tmp[g_cfg.think_end_token] = -1e30f;
        }
    }
    int tok = sample_top_p_temperature(use_logits, dim, temperature, top_p, top_k, min_p);
    free(tmp);
    return tok;
}

static void seed_token_counts_from_prompt(int *token_counts, int vocab_size, PromptTokens *pt, int start) {
    if (!token_counts || !pt || !pt->ids) return;
    if (start < 0) start = 0;
    for (int i = start; i < pt->count; i++) {
        int tok = pt->ids[i];
        if (tok >= 0 && tok < vocab_size) token_counts[tok]++;
    }
}

// SiLU activation
static void cpu_silu(float *x, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// Conv1d depthwise: one step (for incremental inference)
// Input: conv_state[kernel_size-1][channels] + new_input[channels]
// Output: result[channels]
// Weight: [channels, kernel_size, 1] stored as bf16
// This is a depthwise conv1d: each channel is independent
static void cpu_conv1d_step(
    const float *conv_state,    // [(kernel_size-1) * channels] row-major
    const float *new_input,     // [channels]
    const uint16_t *weight_bf16, // [channels * kernel_size] flattened
    float *out,                 // [channels]
    int channels,
    int kernel_size
) {
    // For each channel, compute dot product of [conv_state..., new_input] with weight
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        // Process previous states from conv_state
        for (int k = 0; k < kernel_size - 1; k++) {
            float w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        // Process new input (last position in kernel)
        float w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    // Apply SiLU
    cpu_silu(out, channels);
}

// ============================================================================
// Metal context for GPU-accelerated matmuls
// ============================================================================

// Maximum number of batched matmul output slots.
// Used for encoding multiple matmuls into one command buffer.
#define MAX_BATCH_SLOTS 8

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v5;  // LUT dequant variant
    id<MTLComputePipelineState> matvec_fast;  // for in_dim > 4096
    id<MTLComputePipelineState> matmul2_v3;   // batched N=2 (MTP draft/verify)
    id<MTLComputePipelineState> matmulN_v3;    // batched N-wide (depth-N verify)
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> rms_norm_apply_bf16;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> swiglu;
    // GPU attention pipelines
    id<MTLComputePipelineState> attn_scores_pipe;
    id<MTLComputePipelineState> attn_softmax_pipe;
    id<MTLComputePipelineState> attn_values_pipe;
    id<MTLComputePipelineState> sigmoid_gate_pipe;
    // Reusable buffers for attention matmuls
    id<MTLBuffer> buf_input;     // input vector [g_cfg.hidden_dim or max projection input]
    id<MTLBuffer> buf_output;    // output vector [max projection output]
    id<MTLBuffer> wf_buf;        // the mmap'd weight file as a Metal buffer
    // Batched matmul output slots (preallocated, reused across dispatches)
    id<MTLBuffer> batch_out[MAX_BATCH_SLOTS];
    // Reusable buffers for expert computation (avoids per-expert alloc)
    // Legacy single-expert buffers (kept for gpu_expert_forward compat)
    id<MTLBuffer> buf_expert_data;   // holds one expert's packed weights (g_cfg.expert_size bytes)
    id<MTLBuffer> buf_expert_input;  // h_post input [g_cfg.hidden_dim floats]
    id<MTLBuffer> buf_expert_gate;   // gate_proj output [g_cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_up;     // up_proj output [g_cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_act;    // SwiGLU output [g_cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_out;    // down_proj output [g_cfg.hidden_dim floats]
    // Multi-expert buffers: K independent sets so all experts can be encoded
    // into a SINGLE command buffer (no per-expert commit+wait).
    // Each expert k uses slot [k].
    // Double-buffered: set A (data) for GPU compute, set B (data_B) for background pread.
    // Gate/up/act/out only need one set (GPU uses them after pread completes).
    #define MAX_K 8
    id<MTLBuffer> buf_multi_expert_data[MAX_K];   // [g_cfg.expert_size bytes] each — buffer set A
    id<MTLBuffer> buf_multi_expert_data_B[MAX_K]; // [g_cfg.expert_size bytes] each — buffer set B (prefetch)
    id<MTLBuffer> buf_multi_expert_gate[MAX_K];   // [g_cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_up[MAX_K];     // [g_cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_act[MAX_K];    // [g_cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_out[MAX_K];    // [g_cfg.hidden_dim floats]
    id<MTLBuffer> buf_multi_expert_input;         // [g_cfg.hidden_dim floats] (shared, read-only during dispatch)
    // Shared expert buffers for fused CMD2 (shared gate/up computed in CMD1,
    // SwiGLU + down_proj in CMD2 alongside routed experts)
    id<MTLBuffer> buf_shared_gate;   // [g_cfg.shared_intermediate floats]
    id<MTLBuffer> buf_shared_up;     // [g_cfg.shared_intermediate floats]
    id<MTLBuffer> buf_shared_act;    // [g_cfg.shared_intermediate floats] (SwiGLU output)
    id<MTLBuffer> buf_shared_out;    // [g_cfg.hidden_dim floats] (down_proj output)
    // Fused o_proj+norm+routing buffers (eliminates 1 cmd buffer per layer)
    id<MTLBuffer> buf_residual;     // [g_cfg.hidden_dim floats] holds residual for GPU add
    id<MTLBuffer> buf_h_mid;        // [g_cfg.hidden_dim floats] residual+oproj result
    id<MTLBuffer> buf_sum_sq;       // [1 float] for RMS norm reduction
    // GPU attention buffers (for full attention layers)
    id<MTLBuffer> buf_kv_k[MAX_FULL_ATTN_LAYERS];  // K cache per full-attn layer
    id<MTLBuffer> buf_kv_v[MAX_FULL_ATTN_LAYERS];  // V cache per full-attn layer
    id<MTLBuffer> buf_attn_q;       // [g_cfg.num_attn_heads * g_cfg.head_dim floats] all query heads
    id<MTLBuffer> buf_attn_scores;  // [g_cfg.num_attn_heads * MAX_SEQ_LEN floats] all heads' scores
    id<MTLBuffer> buf_attn_out;     // [g_cfg.num_attn_heads * g_cfg.head_dim floats] full attention output
    id<MTLBuffer> buf_attn_gate;    // [g_cfg.num_attn_heads * g_cfg.head_dim floats] sigmoid gate
    // CMD3 GPU-side combine buffers (weighted_sum + residual + norm on GPU)
    id<MTLComputePipelineState> moe_combine_residual;  // fused combine kernel
    id<MTLBuffer> buf_moe_hidden;     // [g_cfg.hidden_dim floats] GPU combine output (hidden state)
    id<MTLBuffer> buf_combine_params; // [10 floats] expert weights[8] + shared_gate_score + padding
    id<MTLBuffer> buf_cmd3_sum_sq;    // [1 float] for RMS norm reduction in CMD3
    // Shared event for CPU-GPU synchronization (async pipeline)
    id<MTLSharedEvent> pipeline_event;   // CPU signals when buf_input is ready
    uint64_t event_value;                // monotonically increasing event counter
    // GPU delta-net (gated_delta_net_step) and conv1d pipelines
    id<MTLComputePipelineState> delta_net_step;  // gated_delta_net_step kernel
    id<MTLComputePipelineState> conv1d_step;     // conv1d_step kernel
    id<MTLComputePipelineState> rms_norm_qk;     // per-head RMS normalize for q and k
    id<MTLComputePipelineState> compute_decay_beta; // g_decay and beta_gate for delta-net
    id<MTLComputePipelineState> gated_rms_norm;  // z-gated output normalization
    // Persistent GPU state buffers for linear attention layers
    id<MTLBuffer> buf_delta_state[MAX_LINEAR_LAYERS];   // [64*128*128] float per layer
    id<MTLBuffer> buf_conv_state[MAX_LINEAR_LAYERS];     // [3*12288] float per layer
    // Scratch buffers for delta-net inputs/outputs
    id<MTLBuffer> buf_delta_q;        // [2048] float
    id<MTLBuffer> buf_delta_k;        // [2048] float
    id<MTLBuffer> buf_delta_v;        // [8192] float
    id<MTLBuffer> buf_delta_g_decay;  // [64] float
    id<MTLBuffer> buf_delta_beta;     // [64] float
    id<MTLBuffer> buf_delta_output;   // [8192] float
    id<MTLBuffer> buf_conv_input;     // [12288] float
    id<MTLBuffer> buf_conv_output;    // [12288] float
} MetalCtx;

static MetalCtx *g_metal = NULL;

static MetalCtx *metal_setup(void) {
    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx); return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx); return NULL;
    }

    // Compile shaders from source
    NSError *error = nil;
    NSArray *paths = @[@"shaders.metal", @"metal_infer/shaders.metal"];
    NSString *src = nil;
    for (NSString *p in paths) {
        src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
        if (src) break;
    }
    if (!src) {
        fprintf(stderr, "ERROR: Cannot find shaders.metal\n");
        free(ctx); return NULL;
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_1;
    double t0 = now_ms();
    ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
    if (!ctx->library) {
        fprintf(stderr, "ERROR: Shader compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        free(ctx); return NULL;
    }
    printf("[metal] Shader compile: %.0f ms\n", now_ms() - t0);

    // Create pipelines
    id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "ERROR: shader '%s' not found\n", [name UTF8String]); return (id<MTLComputePipelineState>)nil; }
        NSError *e2 = nil;
        id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithFunction:fn error:&e2];
        if (!ps) { fprintf(stderr, "ERROR: pipeline '%s': %s\n", [name UTF8String], [[e2 localizedDescription] UTF8String]); }
        return ps;
    };

    ctx->matvec_v3     = makePipe(@"dequant_matvec_4bit_v3");
    ctx->matmul2_v3    = makePipe(@"dequant_matmul2_4bit_v3");
    ctx->matmulN_v3    = makePipe(@"dequant_matmulN_4bit_v3");
    ctx->matvec_v5     = makePipe(@"dequant_matvec_4bit_v5");  // LUT variant (no uint→float conversions)
    ctx->matvec_fast   = makePipe(@"dequant_matvec_4bit_fast");
    ctx->rms_norm_sum  = makePipe(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipe(@"rms_norm_apply");
    ctx->rms_norm_apply_bf16 = makePipe(@"rms_norm_apply_bf16");
    ctx->residual_add  = makePipe(@"residual_add");
    ctx->swiglu        = makePipe(@"swiglu_fused");
    ctx->attn_scores_pipe  = makePipe(@"attn_scores_batched");
    ctx->attn_softmax_pipe = makePipe(@"attn_softmax_batched");
    ctx->attn_values_pipe  = makePipe(@"attn_values_batched");
    ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");
    ctx->moe_combine_residual = makePipe(@"moe_combine_residual");
    ctx->delta_net_step    = makePipe(@"gated_delta_net_step");
    ctx->conv1d_step       = makePipe(@"conv1d_step");
    ctx->rms_norm_qk       = makePipe(@"rms_norm_qk");
    ctx->compute_decay_beta = makePipe(@"compute_decay_beta");
    ctx->gated_rms_norm    = makePipe(@"gated_rms_norm");
    if (!ctx->moe_combine_residual) fprintf(stderr, "[metal] WARNING: moe_combine_residual pipeline failed\n");
    if (!ctx->delta_net_step) fprintf(stderr, "[metal] WARNING: gated_delta_net_step pipeline failed (CPU fallback)\n");
    if (!ctx->conv1d_step)    fprintf(stderr, "[metal] WARNING: conv1d_step pipeline failed (CPU fallback)\n");
    if (!ctx->rms_norm_qk)       fprintf(stderr, "[metal] WARNING: rms_norm_qk pipeline failed (CPU fallback)\n");
    if (!ctx->compute_decay_beta) fprintf(stderr, "[metal] WARNING: compute_decay_beta pipeline failed (CPU fallback)\n");
    if (!ctx->gated_rms_norm)     fprintf(stderr, "[metal] WARNING: gated_rms_norm pipeline failed (CPU fallback)\n");

    if (!ctx->matvec_v3 || !ctx->matvec_fast) {
        fprintf(stderr, "ERROR: Required Metal pipeline missing\n");
        free(ctx); return NULL;
    }

    // Allocate reusable buffers (large enough for biggest projection)
    // Q proj output is 16384 floats, lm_head output is 248320 floats
    // o_proj input is 8192, linear attn out_proj input is 8192
    size_t max_out = g_cfg.vocab_size * sizeof(float);  // lm_head is largest
    size_t max_in = g_cfg.linear_total_value * sizeof(float);  // 8192 floats (linear_attn out_proj)
    if (max_in < (size_t)(g_cfg.num_attn_heads * g_cfg.head_dim) * sizeof(float)) {
        max_in = (size_t)(g_cfg.num_attn_heads * g_cfg.head_dim) * sizeof(float);  // o_proj input = 8192
    }
    if (max_in < (size_t)g_cfg.dense_intermediate * sizeof(float)) {
        max_in = (size_t)g_cfg.dense_intermediate * sizeof(float);  // dense down_proj input = 17408
    }
    ctx->buf_input  = [ctx->device newBufferWithLength:max_in  options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_out options:MTLResourceStorageModeShared];

    // Batched matmul output slots — each large enough for the biggest projection
    // q_proj = 16384 floats, qkv_proj = 12288, z_proj = 8192, o_proj = 4096
    // lm_head (248320) uses buf_output directly, not batched.
    {
        size_t slot_size = (size_t)(g_cfg.num_attn_heads * g_cfg.head_dim * 2) * sizeof(float);  // 16384 floats
        if (slot_size < (size_t)g_cfg.linear_conv_dim * sizeof(float))
            slot_size = (size_t)g_cfg.linear_conv_dim * sizeof(float);  // 12288 floats
        for (int i = 0; i < MAX_BATCH_SLOTS; i++) {
            ctx->batch_out[i] = [ctx->device newBufferWithLength:slot_size
                                                         options:MTLResourceStorageModeShared];
        }
    }

    // Expert computation buffers (reused across all experts and layers)
    ctx->buf_expert_data  = [ctx->device newBufferWithLength:g_cfg.expert_size
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_input = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_gate  = [ctx->device newBufferWithLength:g_cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_up    = [ctx->device newBufferWithLength:g_cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_act   = [ctx->device newBufferWithLength:g_cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_out   = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

    // Multi-expert buffers: K independent slots (double-buffered data)
    // Expert data buffers use 2MB-aligned backing memory for DMA efficiency.
    // The pread DMA controller transfers 3.6x faster with 2MB alignment vs 16KB.
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    size_t expert_alloc_size = (g_cfg.expert_size + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);  // round up to 2MB
    for (int k = 0; k < MAX_K; k++) {
        // 2MB-aligned allocation for optimal DMA throughput
        void *aligned_data = NULL, *aligned_data_b = NULL;
        posix_memalign(&aligned_data,   2*1024*1024, expert_alloc_size);
        posix_memalign(&aligned_data_b, 2*1024*1024, expert_alloc_size);
        memset(aligned_data, 0, expert_alloc_size);
        memset(aligned_data_b, 0, expert_alloc_size);
        ctx->buf_multi_expert_data[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data
                                                                       length:expert_alloc_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
        ctx->buf_multi_expert_data_B[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data_b
                                                                         length:expert_alloc_size
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:g_cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k]   = [ctx->device newBufferWithLength:g_cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k]  = [ctx->device newBufferWithLength:g_cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_out[k]  = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
    }

    // Shared expert buffers (for fused CMD2)
    ctx->buf_shared_gate = [ctx->device newBufferWithLength:g_cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up   = [ctx->device newBufferWithLength:g_cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_act  = [ctx->device newBufferWithLength:g_cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_out  = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // Fused o_proj+norm+routing buffers
    ctx->buf_residual = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid    = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq   = [ctx->device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    // CMD3 GPU-side combine buffers
    ctx->buf_moe_hidden    = [ctx->device newBufferWithLength:g_cfg.hidden_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:10 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_cmd3_sum_sq    = [ctx->device newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

    // GPU attention buffers
    {
        size_t kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;  // 512
        size_t kv_cache_size = GPU_KV_SEQ * kv_dim * sizeof(float);
        for (int i = 0; i < g_cfg.num_full_attn_layers; i++) {
            ctx->buf_kv_k[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
            ctx->buf_kv_v[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
        }
        ctx->buf_attn_q      = [ctx->device newBufferWithLength:g_cfg.num_attn_heads * g_cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_scores = [ctx->device newBufferWithLength:(size_t)g_cfg.num_attn_heads * GPU_KV_SEQ * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_out    = [ctx->device newBufferWithLength:g_cfg.num_attn_heads * g_cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_gate   = [ctx->device newBufferWithLength:g_cfg.num_attn_heads * g_cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        printf("[metal] GPU attention buffers: %d KV caches (%.1f MB each), scores buf %.1f MB\n",
               g_cfg.num_full_attn_layers, kv_cache_size / 1e6,
               (double)(g_cfg.num_attn_heads * MAX_SEQ_LEN * sizeof(float)) / 1e6);
    }

    // Persistent GPU state buffers for delta-net (linear attention layers)
    if (ctx->delta_net_step) {
        for (int i = 0; i < g_cfg.num_linear_layers; i++) {
            ctx->buf_delta_state[i] = [ctx->device newBufferWithLength:64*128*128*sizeof(float)
                                                               options:MTLResourceStorageModeShared];
            memset([ctx->buf_delta_state[i] contents], 0, 64*128*128*sizeof(float));
            ctx->buf_conv_state[i] = [ctx->device newBufferWithLength:3*12288*sizeof(float)
                                                              options:MTLResourceStorageModeShared];
            memset([ctx->buf_conv_state[i] contents], 0, 3*12288*sizeof(float));
        }
        // Scratch buffers for delta-net inputs/outputs (allocated once, reused)
        ctx->buf_delta_q       = [ctx->device newBufferWithLength:2048*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_k       = [ctx->device newBufferWithLength:2048*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_v       = [ctx->device newBufferWithLength:8192*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_g_decay = [ctx->device newBufferWithLength:64*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_beta    = [ctx->device newBufferWithLength:64*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_output  = [ctx->device newBufferWithLength:8192*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_conv_input    = [ctx->device newBufferWithLength:12288*sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_conv_output   = [ctx->device newBufferWithLength:12288*sizeof(float) options:MTLResourceStorageModeShared];
        printf("[metal] Delta-net GPU buffers: %d layers (%.1f MB state + %.1f MB scratch)\n",
               g_cfg.num_linear_layers,
               g_cfg.num_linear_layers * (64*128*128*4 + 3*12288*4) / 1e6,
               (2048+2048+8192+64+64+8192+12288+12288) * 4 / 1e6);
    }

    // Create shared event for CPU-GPU async pipeline
    ctx->pipeline_event = [ctx->device newSharedEvent];
    ctx->event_value = 0;

    printf("[metal] Inference pipelines ready (multi-expert[%d] + shared buffers allocated)\n", MAX_K);
    return ctx;
}

// Reset delta-net and conv GPU state buffers (call at start of new generation)
static void reset_delta_net_state(void) {
    if (!g_metal || !g_metal->delta_net_step) return;
    for (int i = 0; i < g_cfg.num_linear_layers; i++) {
        if (g_metal->buf_delta_state[i])
            memset([g_metal->buf_delta_state[i] contents], 0, 64*128*128*sizeof(float));
        if (g_metal->buf_conv_state[i])
            memset([g_metal->buf_conv_state[i] contents], 0, 3*12288*sizeof(float));
    }
}

// Wrap the mmap'd weight file as a Metal buffer (zero-copy on unified memory)
// mmap returns page-aligned addresses, Metal requires the same.
// On Apple Silicon, page size is 16KB.
static void metal_set_weights(MetalCtx *ctx, void *data, size_t size) {
    // Round size up to page boundary (16KB)
    size_t page_size = 16384;
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    ctx->wf_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                 length:aligned_size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!ctx->wf_buf) {
        fprintf(stderr, "WARNING: Cannot wrap weight file as Metal buffer (size=%.2f GB)\n",
                size / 1e9);
        fprintf(stderr, "  data=%p, aligned_size=%zu -- GPU matmul will fall back to CPU\n",
                data, aligned_size);
    } else {
        printf("[metal] Weight file wrapped as Metal buffer (%.2f GB)\n",
               aligned_size / 1e9);
    }
}

// GPU dequant matvec: out[out_dim] = W_4bit * x[in_dim]
// W_packed, scales, biases are pointers into mmap'd weight file
// x_f32 is CPU float array, result written back to out_f32
//
// We wrap the ENTIRE mmap'd weight file as a single Metal buffer and use
// byte offsets to point each shader argument at the right tensor.
// This avoids per-tensor buffer creation and the page-alignment constraint.
static void gpu_dequant_matvec(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x_f32, float *out_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    // Copy input to Metal buffer
    memcpy([ctx->buf_input contents], x_f32, in_dim * sizeof(float));

    size_t o_size = (size_t)out_dim * sizeof(float);

    // Compute offsets into the mmap'd weight buffer
    NSUInteger w_off = (NSUInteger)((const char *)W_packed - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales   - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases   - (const char *)[ctx->wf_buf contents]);

    // Ensure output buffer is large enough
    id<MTLBuffer> o_buf = ctx->buf_output;
    if (o_size > [o_buf length]) {
        o_buf = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    // v3 shader uses x_shared[4096], so can only handle in_dim <= 4096
    // For larger in_dim (e.g. o_proj with in_dim=8192), use matvec_fast
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
    [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
    [enc setBuffer:o_buf        offset:0     atIndex:4];
    [enc setBytes:&out_dim      length:4     atIndex:5];
    [enc setBytes:&in_dim       length:4     atIndex:6];
    [enc setBytes:&group_size   length:4     atIndex:7];

    if (use_v3) {
        // v3: tiled threadgroups, 256 threads, 8 rows per TG
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        // fast: one threadgroup per output row, 64 threads per TG
        NSUInteger tg_size = 64;
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back
    memcpy(out_f32, [o_buf contents], o_size);
}

// Batched N=2 dequant matmul: one weight read serves two input vectors.
// Foundation for MTP draft/verify. Requires in_dim <= 4096 (v3 x_shared limit).
static void gpu_dequant_matmulN(MetalCtx *ctx, const void *W_packed, const void *scales,
    const void *biases, const float *X, float *OUT, uint32_t out_dim, uint32_t in_dim,
    uint32_t group_size, uint32_t N);

static void gpu_dequant_matmul2(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x0_f32, const float *x1_f32, float *out0_f32, float *out1_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    // The matmul2 v3 kernel caches the input in x_shared[4096], so it only handles
    // in_dim <= 4096. For larger inputs (e.g. dense-27B hidden=5120, down_proj=17408)
    // route to matmulN, which reads X from device and handles any in_dim.
    if (in_dim > 4096) {
        float *X = malloc((size_t)2 * in_dim * sizeof(float));
        float *O = malloc((size_t)2 * out_dim * sizeof(float));
        memcpy(X, x0_f32, in_dim * sizeof(float));
        memcpy(X + in_dim, x1_f32, in_dim * sizeof(float));
        gpu_dequant_matmulN(ctx, W_packed, scales, biases, X, O, out_dim, in_dim, group_size, 2);
        memcpy(out0_f32, O, out_dim * sizeof(float));
        memcpy(out1_f32, O + out_dim, out_dim * sizeof(float));
        free(X); free(O);
        return;
    }
    static id<MTLBuffer> in0 = nil, in1 = nil, o0 = nil, o1 = nil;
    size_t i_size = (size_t)in_dim * sizeof(float);
    size_t o_size = (size_t)out_dim * sizeof(float);
    if (!in0 || (size_t)[in0 length] < i_size) in0 = [ctx->device newBufferWithLength:i_size options:MTLResourceStorageModeShared];
    if (!in1 || (size_t)[in1 length] < i_size) in1 = [ctx->device newBufferWithLength:i_size options:MTLResourceStorageModeShared];
    if (!o0  || (size_t)[o0 length]  < o_size) o0  = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
    if (!o1  || (size_t)[o1 length]  < o_size) o1  = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];

    memcpy([in0 contents], x0_f32, i_size);
    memcpy([in1 contents], x1_f32, i_size);

    NSUInteger w_off = (NSUInteger)((const char *)W_packed - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales   - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases   - (const char *)[ctx->wf_buf contents]);

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->matmul2_v3];
    [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off atIndex:2];
    [enc setBuffer:in0 offset:0 atIndex:3];
    [enc setBuffer:o0  offset:0 atIndex:4];
    [enc setBytes:&out_dim length:4 atIndex:5];
    [enc setBytes:&in_dim  length:4 atIndex:6];
    [enc setBytes:&group_size length:4 atIndex:7];
    [enc setBuffer:in1 offset:0 atIndex:8];
    [enc setBuffer:o1  offset:0 atIndex:9];
    uint32_t num_tgs = (out_dim + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    memcpy(out0_f32, [o0 contents], o_size);
    memcpy(out1_f32, [o1 contents], o_size);
}

// N-wide batched dequant matmul (N<=8): X is [N][in_dim], OUT is [N][out_dim],
// both host-side contiguous. One weight read serves all N tokens.
static void gpu_dequant_matmulN(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *X, float *OUT,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size, uint32_t N
) {
    static id<MTLBuffer> xbuf = nil, obuf = nil;
    size_t xs = (size_t)N * in_dim * sizeof(float);
    size_t os = (size_t)N * out_dim * sizeof(float);
    if (!xbuf || (size_t)[xbuf length] < xs) xbuf = [ctx->device newBufferWithLength:xs options:MTLResourceStorageModeShared];
    if (!obuf || (size_t)[obuf length] < os) obuf = [ctx->device newBufferWithLength:os options:MTLResourceStorageModeShared];
    memcpy([xbuf contents], X, xs);

    NSUInteger w_off = (NSUInteger)((const char *)W_packed - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales   - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases   - (const char *)[ctx->wf_buf contents]);

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->matmulN_v3];
    [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off atIndex:2];
    [enc setBuffer:xbuf offset:0 atIndex:3];
    [enc setBuffer:obuf offset:0 atIndex:4];
    [enc setBytes:&out_dim length:4 atIndex:5];
    [enc setBytes:&in_dim  length:4 atIndex:6];
    [enc setBytes:&group_size length:4 atIndex:7];
    [enc setBytes:&N length:4 atIndex:8];
    uint32_t num_tgs = (out_dim + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    memcpy(OUT, [obuf contents], os);
}

// Stage 1 microbench: does one weight read serve two tokens? Times two separate
// single-token matvecs vs one batched N=2 matmul on real weight matrices, and
// verifies the batched output matches. The ratio (batched / 2×single) is the
// per-layer cost multiplier that decides whether batched MTP verify can win.
static int mtp_bench_matmul(WeightFile *wf) {
    if (!g_metal || !g_metal->wf_buf) {
        fprintf(stderr, "[mtp-bench] requires GPU weight buffer\n");
        return 1;
    }
    struct { const char *w; const char *s; const char *b; uint32_t out_dim; uint32_t in_dim; const char *label; } cases[] = {
        { "lm_head.weight", "lm_head.scales", "lm_head.biases", (uint32_t)g_cfg.vocab_size, (uint32_t)g_cfg.hidden_dim, "lm_head" },
        { "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.q_proj.scales", "mtp.layers.0.self_attn.q_proj.biases",
          (uint32_t)(g_cfg.num_attn_heads * g_cfg.head_dim * 2), (uint32_t)g_cfg.hidden_dim, "q_proj" },
    };
    const int M = 100;
    for (int c = 0; c < 2; c++) {
        TensorInfo *wi = get_tensor_info(wf, cases[c].w);
        TensorInfo *si = get_tensor_info(wf, cases[c].s);
        TensorInfo *bi = get_tensor_info(wf, cases[c].b);
        if (!wi || !si || !bi) { fprintf(stderr, "[mtp-bench] %s: tensors missing, skip\n", cases[c].label); continue; }
        uint32_t out_dim = cases[c].out_dim, in_dim = cases[c].in_dim, gs = (uint32_t)g_cfg.group_size;
        if (in_dim > 4096) { fprintf(stderr, "[mtp-bench] %s: in_dim %u > 4096, skip\n", cases[c].label, in_dim); continue; }
        uint32_t *W = (uint32_t *)((char *)wf->data + wi->offset);
        uint16_t *S = (uint16_t *)((char *)wf->data + si->offset);
        uint16_t *B = (uint16_t *)((char *)wf->data + bi->offset);
        float *x0 = malloc(in_dim * sizeof(float)), *x1 = malloc(in_dim * sizeof(float));
        float *r0 = malloc(out_dim * sizeof(float)), *r1 = malloc(out_dim * sizeof(float));
        float *b0 = malloc(out_dim * sizeof(float)), *b1 = malloc(out_dim * sizeof(float));
        for (uint32_t i = 0; i < in_dim; i++) { x0[i] = sinf(i * 0.01f); x1[i] = cosf(i * 0.017f); }

        // Correctness: batched must match two singles.
        gpu_dequant_matvec(g_metal, W, S, B, x0, r0, out_dim, in_dim, gs);
        gpu_dequant_matvec(g_metal, W, S, B, x1, r1, out_dim, in_dim, gs);
        gpu_dequant_matmul2(g_metal, W, S, B, x0, x1, b0, b1, out_dim, in_dim, gs);
        double max_err = 0.0;
        for (uint32_t i = 0; i < out_dim; i++) {
            max_err = fmax(max_err, fabs(b0[i] - r0[i]));
            max_err = fmax(max_err, fabs(b1[i] - r1[i]));
        }

        for (int w = 0; w < 5; w++) {  // warmup
            gpu_dequant_matvec(g_metal, W, S, B, x0, r0, out_dim, in_dim, gs);
            gpu_dequant_matmul2(g_metal, W, S, B, x0, x1, b0, b1, out_dim, in_dim, gs);
        }
        double t = now_ms();
        for (int m = 0; m < M; m++) {
            gpu_dequant_matvec(g_metal, W, S, B, x0, r0, out_dim, in_dim, gs);
            gpu_dequant_matvec(g_metal, W, S, B, x1, r1, out_dim, in_dim, gs);
        }
        double t_two_single = (now_ms() - t) / M;
        t = now_ms();
        for (int m = 0; m < M; m++) {
            gpu_dequant_matmul2(g_metal, W, S, B, x0, x1, b0, b1, out_dim, in_dim, gs);
        }
        double t_batched = (now_ms() - t) / M;

        fprintf(stderr, "[mtp-bench] %-8s out=%u in=%u | 2x single=%.4f ms | batched2=%.4f ms | ratio=%.3f | max_err=%.2e\n",
                cases[c].label, out_dim, in_dim, t_two_single, t_batched, t_batched / t_two_single, max_err);
        free(x0); free(x1); free(r0); free(r1); free(b0); free(b1);
    }
    fprintf(stderr, "[mtp-bench] ratio < 1.0 means batching amortizes the weight read (lower = better; 0.5 = ideal 2x)\n");

    // Depth sweep on lm_head: per-token cost vs batch width N (the compute-side
    // case for deeper speculation — fixed weight read amortized over more tokens).
    TensorInfo *wi = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *si = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *bi = get_tensor_info(wf, "lm_head.biases");
    if (wi && si && bi) {
        uint32_t out_dim = (uint32_t)g_cfg.vocab_size, in_dim = (uint32_t)g_cfg.hidden_dim, gs = (uint32_t)g_cfg.group_size;
        uint32_t *W = (uint32_t *)((char *)wf->data + wi->offset);
        uint16_t *S = (uint16_t *)((char *)wf->data + si->offset);
        uint16_t *B = (uint16_t *)((char *)wf->data + bi->offset);
        double base_per_tok = 0.0;
        for (uint32_t N = 1; N <= 4; N++) {
            float *X = malloc((size_t)N * in_dim * sizeof(float));
            float *O = malloc((size_t)N * out_dim * sizeof(float));
            for (uint32_t n = 0; n < N; n++)
                for (uint32_t i = 0; i < in_dim; i++) X[n*in_dim+i] = sinf((i + n*7) * 0.01f);
            for (int w = 0; w < 5; w++) gpu_dequant_matmulN(g_metal, W, S, B, X, O, out_dim, in_dim, gs, N);
            const int M = 80;
            double t = now_ms();
            for (int m = 0; m < M; m++) gpu_dequant_matmulN(g_metal, W, S, B, X, O, out_dim, in_dim, gs, N);
            double per_call = (now_ms() - t) / M;
            double per_tok = per_call / N;
            if (N == 1) base_per_tok = per_tok;
            // Correctness: matmulN row n must equal gpu_dequant_matvec on X[n].
            float *ref = malloc(out_dim * sizeof(float)); double mxerr = 0;
            gpu_dequant_matvec(g_metal, W, S, B, X + (size_t)(N-1)*in_dim, ref, out_dim, in_dim, gs);
            for (uint32_t i = 0; i < out_dim; i++) mxerr = fmax(mxerr, fabs(ref[i] - O[(size_t)(N-1)*out_dim + i]));
            free(ref);
            fprintf(stderr, "[mtp-depth] lm_head N=%u | call=%.4f ms | per-token=%.4f ms | per-tok-ratio=%.3f | matmulN-vs-matvec max_err=%.2e\n",
                    N, per_call, per_tok, base_per_tok > 0 ? per_tok / base_per_tok : 1.0, mxerr);
            free(X); free(O);
        }
        fprintf(stderr, "[mtp-depth] per-tok-ratio is fraction of N=1 cost; lower = deeper batching amortizes more\n");
    }
    return 0;
}

// Wrapper: use GPU if available and weight buffer is set, CPU otherwise
static void fast_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    if (g_metal && g_metal->wf_buf) {
        gpu_dequant_matvec(g_metal, W, scales, biases, x, out,
                           (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)group_size);
    } else {
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
    }
}

// ============================================================================
// Batched GPU matmul: encode N independent matmuls sharing the same input
// into ONE command buffer, reducing dispatch overhead by N-1 round-trips.
// ============================================================================

typedef struct {
    const void *W;           // packed weights (pointer into mmap'd file)
    const void *scales;      // scales (pointer into mmap'd file)
    const void *biases;      // biases (pointer into mmap'd file)
    float *out_cpu;          // CPU output pointer (result copied here after GPU finishes)
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    int batch_slot;          // which batch_out[slot] to use for GPU output
} BatchMatvecSpec;

// Run N matmuls in a single command buffer. All share the same input vector.
// The input is copied once; all outputs go to preallocated batch_out slots.
static void gpu_batch_matvec(
    MetalCtx *ctx,
    const float *x_f32, uint32_t x_dim,  // shared input
    BatchMatvecSpec *specs, int num_specs
) {
    // Copy input once
    memcpy([ctx->buf_input contents], x_f32, x_dim * sizeof(float));

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy results back to CPU
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// ============================================================================
// Encode-only variants: add dispatches to an EXISTING command buffer.
// These do NOT commit — the caller batches multiple encode calls into one
// command buffer and commits once, eliminating per-dispatch overhead.
// ============================================================================

// Encode N matmuls into cmdbuf. Input must already be in ctx->buf_input.
static void gpu_encode_batch_matvec(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    BatchMatvecSpec *specs, int num_specs
) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
}

// Copy batch results from GPU buffers back to CPU pointers.
static void gpu_flush_batch_results(MetalCtx *ctx, BatchMatvecSpec *specs, int num_specs) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// Encode a single matvec reading from buf_expert_act into buf_expert_out,
// using weight pointers into the mmap'd weight file.
// Used for shared expert down_proj which reads from a different input than
// the attention projections.
static void gpu_encode_dequant_matvec_with_io_bufs(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    NSUInteger w_off = (NSUInteger)((const char *)W      - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales  - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases  - (const char *)[ctx->wf_buf contents]);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off atIndex:2];
    [enc setBuffer:in_buf      offset:0     atIndex:3];
    [enc setBuffer:out_buf     offset:0     atIndex:4];
    [enc setBytes:&out_dim     length:4     atIndex:5];
    [enc setBytes:&in_dim      length:4     atIndex:6];
    [enc setBytes:&group_size  length:4     atIndex:7];

    if (use_v3) {
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
}

// Encode one expert forward using multi-expert slot k.
// Expert data must already be in buf_multi_expert_data[k].
// Input must already be in buf_multi_expert_input.
__attribute__((unused))
static void gpu_encode_expert_forward_slot(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k  // slot index
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    gate_w_off = 0;
    gate_s_off = g_cfg.gate_w_size;
    gate_b_off = gate_s_off + g_cfg.gate_s_size;
    up_w_off   = gate_b_off + g_cfg.gate_b_size;
    up_s_off   = up_w_off + g_cfg.up_w_size;
    up_b_off   = up_s_off + g_cfg.up_s_size;
    down_w_off = up_b_off + g_cfg.up_b_size;
    down_s_off = down_w_off + g_cfg.down_w_size;
    down_b_off = down_s_off + g_cfg.down_s_size;
    id<MTLComputePipelineState> expert_pipe = ctx->matvec_v3;

    uint32_t gate_up_out = g_cfg.moe_intermediate;
    uint32_t gate_up_in  = g_cfg.hidden_dim;
    uint32_t down_out    = g_cfg.hidden_dim;
    uint32_t down_in     = g_cfg.moe_intermediate;
    uint32_t gs          = g_cfg.group_size;

    // gate_proj: data[k] -> gate[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj: data[k] -> up[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU: gate[k], up[k] -> act[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj: act[k] -> out[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Encode one expert forward using explicit data buffer (for double buffering).
// Expert data must already be in data_buf.
// Input must already be in buf_multi_expert_input.
// Uses slot k's gate/up/act/out scratch buffers.
__attribute__((unused))
static void gpu_encode_expert_forward_slot_buf(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k,                  // slot index (for gate/up/act/out scratch)
    id<MTLBuffer> data_buf  // expert weight data buffer (from either set A or B)
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    gate_w_off = 0;
    gate_s_off = g_cfg.gate_w_size;
    gate_b_off = gate_s_off + g_cfg.gate_s_size;
    up_w_off   = gate_b_off + g_cfg.gate_b_size;
    up_s_off   = up_w_off + g_cfg.up_w_size;
    up_b_off   = up_s_off + g_cfg.up_s_size;
    down_w_off = up_b_off + g_cfg.up_b_size;
    down_s_off = down_w_off + g_cfg.down_w_size;
    down_b_off = down_s_off + g_cfg.down_s_size;
    id<MTLComputePipelineState> expert_pipe = ctx->matvec_v3;

    uint32_t gate_up_out = g_cfg.moe_intermediate;
    uint32_t gate_up_in  = g_cfg.hidden_dim;
    uint32_t down_out    = g_cfg.hidden_dim;
    uint32_t down_in     = g_cfg.moe_intermediate;
    uint32_t gs          = g_cfg.group_size;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:gate_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:gate_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:up_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:up_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:down_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:down_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched expert encoding: encode K experts using 2 encoders per expert
// (gate+up fused, SwiGLU+down fused) + 2 for shared = K*2 + 2 encoders total.
// With K=4: 10 encoders (vs. old 4*K + 2 = 18 with per-operation encoding).
// Each expert gets its own encoder pair for GPU parallelism across experts.
// Within each encoder, gate+up (or SwiGLU+down) are serialized but share
// encoder creation overhead. Net win: fewer encoders, same parallelism.
static void gpu_encode_experts_batched(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int K,                       // number of experts to encode
    const int *valid,            // which experts are valid [MAX_K]
    id<MTLBuffer> __strong *expert_bufs   // per-expert weight data buffers [MAX_K]
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    gate_w_off = 0;
    gate_s_off = g_cfg.gate_w_size;
    gate_b_off = gate_s_off + g_cfg.gate_s_size;
    up_w_off   = gate_b_off + g_cfg.gate_b_size;
    up_s_off   = up_w_off + g_cfg.up_w_size;
    up_b_off   = up_s_off + g_cfg.up_s_size;
    down_w_off = up_b_off + g_cfg.up_b_size;
    down_s_off = down_w_off + g_cfg.down_w_size;
    down_b_off = down_s_off + g_cfg.down_s_size;
    id<MTLComputePipelineState> expert_pipe = ctx->matvec_v3;

    uint32_t gate_up_out = g_cfg.moe_intermediate;
    uint32_t gate_up_in  = g_cfg.hidden_dim;
    uint32_t down_out    = g_cfg.hidden_dim;
    uint32_t down_in     = g_cfg.moe_intermediate;
    uint32_t gs          = g_cfg.group_size;
    // Threadgroup count is based on out_dim; the kernel handles packed columns internally.
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;
    uint32_t swiglu_tgs  = (gate_up_out + 255) / 256;

    // Per-expert: Encoder A (gate+up), Encoder B (SwiGLU+down)
    // Separate encoders per expert enables GPU parallelism across experts.
    // Within each encoder, operations serialize (gate then up, SwiGLU then down).
    for (int k = 0; k < K; k++) {
        if (!valid[k]) continue;

        // Encoder A: gate_proj + up_proj (both read same input, write different outputs)
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            // gate_proj
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // up_proj (same encoder, serialized after gate — shares encoder overhead)
            [enc setBuffer:expert_bufs[k]                  offset:up_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:up_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Encoder B: SwiGLU + down_proj (SwiGLU depends on gate+up from Enc A)
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            // SwiGLU
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // down_proj (same encoder, serialized after SwiGLU)
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
            [enc setBytes:&down_out length:4 atIndex:5];
            [enc setBytes:&down_in  length:4 atIndex:6];
            [enc setBytes:&gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
}

// Encode one expert forward (gate+up+swiglu+down) into cmdbuf.
// Expert data must already be in buf_expert_data.
// Input must already be in buf_expert_input.
__attribute__((unused))
static void gpu_encode_expert_forward(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf
) {
    NSUInteger gate_w_off = 0;
    NSUInteger gate_s_off = g_cfg.gate_w_size;
    NSUInteger gate_b_off = gate_s_off + g_cfg.gate_s_size;
    NSUInteger up_w_off   = gate_b_off + g_cfg.gate_b_size;
    NSUInteger up_s_off   = up_w_off + g_cfg.up_w_size;
    NSUInteger up_b_off   = up_s_off + g_cfg.up_s_size;
    NSUInteger down_w_off = up_b_off + g_cfg.up_b_size;
    NSUInteger down_s_off = down_w_off + g_cfg.down_w_size;
    NSUInteger down_b_off = down_s_off + g_cfg.down_s_size;

    uint32_t gate_up_out = g_cfg.moe_intermediate;
    uint32_t gate_up_in  = g_cfg.hidden_dim;
    uint32_t down_out    = g_cfg.hidden_dim;
    uint32_t down_in     = g_cfg.moe_intermediate;
    uint32_t gs          = g_cfg.group_size;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched wrapper: takes N matmul specs sharing the same input, dispatches
// via GPU batch if available, otherwise falls back to CPU.
static void fast_batch_matvec(
    const float *x, uint32_t x_dim,
    BatchMatvecSpec *specs, int num_specs
) {
    if (g_metal && g_metal->wf_buf) {
        gpu_batch_matvec(g_metal, x, x_dim, specs, num_specs);
    } else {
        for (int i = 0; i < num_specs; i++) {
            BatchMatvecSpec *s = &specs[i];
            cpu_dequant_matvec(s->W, s->scales, s->biases, x, s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size);
        }
    }
}

// ============================================================================
// GPU expert forward: gate+up matvec -> SwiGLU -> down matvec
// All 3 matmuls + activation in a single command buffer submission.
// Expert data is copied into a reusable Metal buffer.
// ============================================================================

// expert_data_already_in_buffer: if true, expert data is already in buf_expert_data
//   (pread'd directly into it), skip the copy.
__attribute__((unused))
static void gpu_expert_forward(
    MetalCtx *ctx,
    const void *expert_data,     // g_cfg.expert_size bytes (may be buf_expert_data contents)
    const float *h_post,         // [g_cfg.hidden_dim] input
    float *expert_out,           // [g_cfg.hidden_dim] output
    int expert_data_already_in_buffer
) {
    // Expert layout offsets
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    gate_w_off = 0;
    gate_s_off = g_cfg.gate_w_size;
    gate_b_off = gate_s_off + g_cfg.gate_s_size;
    up_w_off   = gate_b_off + g_cfg.gate_b_size;
    up_s_off   = up_w_off + g_cfg.up_w_size;
    up_b_off   = up_s_off + g_cfg.up_s_size;
    down_w_off = up_b_off + g_cfg.up_b_size;
    down_s_off = down_w_off + g_cfg.down_w_size;
    down_b_off = down_s_off + g_cfg.down_s_size;
    id<MTLComputePipelineState> expert_pipe = ctx->matvec_v3;

    // Copy expert weights into Metal buffer only if not already there
    if (!expert_data_already_in_buffer) {
        memcpy([ctx->buf_expert_data contents], expert_data, active_expert_size());
    }
    memcpy([ctx->buf_expert_input contents], h_post, g_cfg.hidden_dim * sizeof(float));

    uint32_t gate_up_out = g_cfg.moe_intermediate;  // 1024
    uint32_t gate_up_in  = g_cfg.hidden_dim;        // 4096
    uint32_t down_out    = g_cfg.hidden_dim;        // 4096
    uint32_t down_in     = g_cfg.moe_intermediate;  // 1024
    uint32_t gs          = g_cfg.group_size;        // 64

    // Build one command buffer with all 4 dispatches:
    // 1. gate_proj matvec (h_post -> gate_out)
    // 2. up_proj matvec (h_post -> up_out)
    // 3. SwiGLU (gate_out, up_out -> act_out)
    // 4. down_proj matvec (act_out -> expert_out)

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // --- Dispatch 1: gate_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 2: up_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 3: SwiGLU(gate, up) -> act ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 4: down_proj [1024] -> [4096] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back to CPU
    memcpy(expert_out, [ctx->buf_expert_out contents], g_cfg.hidden_dim * sizeof(float));
}

// ============================================================================
// Rotary position embedding (for full attention layers)
// ============================================================================

static void apply_rotary_emb(float *q, float *k, int pos, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim) {
    // Apply RoPE to the first rotary_dim dimensions of each head
    // NON-TRADITIONAL (MLX default): pairs are (x[i], x[i + half_dim])
    // where half_dim = rotary_dim / 2
    int half = rotary_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(g_cfg.rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half]  = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(g_cfg.rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half]  = k0 * sin_a + k1 * cos_a;
        }
    }
}

// ============================================================================
// KV Cache for full attention layers
// ============================================================================

typedef struct {
    float *k_cache;  // [max_seq, num_kv_heads * head_dim]
    float *v_cache;  // [max_seq, num_kv_heads * head_dim]
    int len;         // current number of cached entries
} KVCache;

static KVCache *kv_cache_new(void) {
    KVCache *c = calloc(1, sizeof(KVCache));
    c->k_cache = calloc(MAX_SEQ_LEN * g_cfg.num_kv_heads * g_cfg.head_dim, sizeof(float));
    c->v_cache = calloc(MAX_SEQ_LEN * g_cfg.num_kv_heads * g_cfg.head_dim, sizeof(float));
    c->len = 0;
    return c;
}

static void kv_cache_free(KVCache *c) {
    if (c) {
        free(c->k_cache);
        free(c->v_cache);
        free(c);
    }
}

// ============================================================================
// Linear attention state (GatedDeltaNet recurrent state)
// ============================================================================

typedef struct {
    float *conv_state;  // [(kernel_size-1) * conv_dim] for conv1d
    float *ssm_state;   // [num_v_heads, head_v_dim, head_k_dim] recurrent state
} LinearAttnState;

static LinearAttnState *linear_attn_state_new(void) {
    LinearAttnState *s = calloc(1, sizeof(LinearAttnState));
    s->conv_state = calloc((g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim, sizeof(float));
    s->ssm_state = calloc(g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim, sizeof(float));
    return s;
}

static void linear_attn_state_free(LinearAttnState *s) {
    if (s) {
        free(s->conv_state);
        free(s->ssm_state);
        free(s);
    }
}

// ============================================================================
// Full attention layer forward (single token, incremental)
// ============================================================================

static int fa_debug_count = 0;

static float vec_rms(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum / n);
}

__attribute__((unused))
static void full_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,       // [g_cfg.hidden_dim] in/out
    KVCache *kv,
    int pos              // position in sequence
) {
    fa_debug_count++;
    int do_debug = 0;  // set to (fa_debug_count <= N) to enable debug

    char name[256];
    float *normed = malloc(g_cfg.hidden_dim * sizeof(float));
    float *residual = malloc(g_cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(residual, hidden, g_cfg.hidden_dim);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] layer=%d pos=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, pos, vec_rms(hidden, g_cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] normed_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(normed, g_cfg.hidden_dim), normed[0], normed[1], normed[2], normed[3], normed[4]);
    }

    // ---- QKV Projection ----
    // CRITICAL: Q projection outputs num_heads * head_dim * 2 = 16384
    // The second half is a sigmoid gate applied after attention
    int q_proj_dim = g_cfg.num_attn_heads * g_cfg.head_dim * 2;  // 32 * 256 * 2 = 16384
    int q_dim = g_cfg.num_attn_heads * g_cfg.head_dim;            // 32 * 256 = 8192
    int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;             // 2 * 256 = 512

    float *q_proj_out = calloc(q_proj_dim, sizeof(float));
    float *k = calloc(kv_dim, sizeof(float));
    float *v = calloc(kv_dim, sizeof(float));

    // Batch Q/K/V projections into a single GPU command buffer
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    uint32_t *qw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", layer_idx);
    uint16_t *qs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", layer_idx);
    uint16_t *qb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    uint32_t *kw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", layer_idx);
    uint16_t *ks = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", layer_idx);
    uint16_t *kb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    uint32_t *vw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", layer_idx);
    uint16_t *vs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", layer_idx);
    uint16_t *vb = get_tensor_ptr(wf, name);

    // Batch Q/K/V into one command buffer (3 dispatches, 1 commit)
    if (qw && qs && qb && kw && ks && kb && vw && vs && vb) {
        BatchMatvecSpec qkv_specs[3] = {
            { qw, qs, qb, q_proj_out, (uint32_t)q_proj_dim, g_cfg.hidden_dim, g_cfg.group_size, 0 },
            { kw, ks, kb, k,          (uint32_t)kv_dim,     g_cfg.hidden_dim, g_cfg.group_size, 1 },
            { vw, vs, vb, v,          (uint32_t)kv_dim,     g_cfg.hidden_dim, g_cfg.group_size, 2 },
        };
        fast_batch_matvec(normed, g_cfg.hidden_dim, qkv_specs, 3);
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] q_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                q_proj_out[0], q_proj_out[1], q_proj_out[2], q_proj_out[3], q_proj_out[4]);
    }

    // Split q_proj_out into queries and gate
    float *q = calloc(q_dim, sizeof(float));
    float *q_gate = calloc(q_dim, sizeof(float));
    for (int h = 0; h < g_cfg.num_attn_heads; h++) {
        float *src = q_proj_out + h * (2 * g_cfg.head_dim);
        memcpy(q + h * g_cfg.head_dim, src, g_cfg.head_dim * sizeof(float));
        memcpy(q_gate + h * g_cfg.head_dim, src + g_cfg.head_dim, g_cfg.head_dim * sizeof(float));
    }
    free(q_proj_out);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] v_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(v, kv_dim), v[0], v[1], v[2], v[3], v[4]);
        fprintf(stderr, "[FA-DBG] q_gate_rms=%.6f gate_first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(q_gate, q_dim), q_gate[0], q_gate[1], q_gate[2], q_gate[3], q_gate[4]);
        float gate_sigmoid_sum = 0.0f;
        for (int i = 0; i < q_dim; i++) {
            gate_sigmoid_sum += 1.0f / (1.0f + expf(-q_gate[i]));
        }
        fprintf(stderr, "[FA-DBG] gate_sigmoid_mean=%.6f\n", gate_sigmoid_sum / q_dim);
    }

    // ---- Q/K RMSNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    uint16_t *qnorm_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    uint16_t *knorm_w = get_tensor_ptr(wf, name);

    // Apply per-head Q norm
    if (qnorm_w) {
        for (int h = 0; h < g_cfg.num_attn_heads; h++) {
            float *qh = q + h * g_cfg.head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < g_cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / g_cfg.head_dim + g_cfg.rms_norm_eps);
            for (int i = 0; i < g_cfg.head_dim; i++) {
                qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
    }
    // Apply per-head K norm
    if (knorm_w) {
        for (int h = 0; h < g_cfg.num_kv_heads; h++) {
            float *kh = k + h * g_cfg.head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < g_cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / g_cfg.head_dim + g_cfg.rms_norm_eps);
            for (int i = 0; i < g_cfg.head_dim; i++) {
                kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }
    }


    // ---- RoPE ----
    apply_rotary_emb(q, k, pos, g_cfg.num_attn_heads, g_cfg.num_kv_heads, g_cfg.head_dim, g_cfg.rotary_dim);

    // ---- Update KV cache ----
    int cache_pos = kv->len;
    memcpy(kv->k_cache + cache_pos * kv_dim, k, kv_dim * sizeof(float));
    memcpy(kv->v_cache + cache_pos * kv_dim, v, kv_dim * sizeof(float));
    kv->len++;

    // ---- Scaled dot-product attention ----
    // GQA: g_cfg.num_attn_heads=32 heads, g_cfg.num_kv_heads=2 kv heads
    // Each group of 16 query heads shares 1 kv head
    int heads_per_kv = g_cfg.num_attn_heads / g_cfg.num_kv_heads;
    float scale = 1.0f / sqrtf((float)g_cfg.head_dim);

    float *attn_out = calloc(q_dim, sizeof(float));

    for (int h = 0; h < g_cfg.num_attn_heads; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * g_cfg.head_dim;

        // Compute attention scores for all cached positions
        float *scores = malloc(kv->len * sizeof(float));
        for (int p = 0; p < kv->len; p++) {
            float *kp = kv->k_cache + p * kv_dim + kv_h * g_cfg.head_dim;
            float dot = 0.0f;
            for (int d = 0; d < g_cfg.head_dim; d++) {
                dot += qh[d] * kp[d];
            }
            scores[p] = dot * scale;
        }

        // Softmax
        cpu_softmax(scores, kv->len);

        // Weighted sum of values
        float *oh = attn_out + h * g_cfg.head_dim;
        for (int p = 0; p < kv->len; p++) {
            float *vp = kv->v_cache + p * kv_dim + kv_h * g_cfg.head_dim;
            for (int d = 0; d < g_cfg.head_dim; d++) {
                oh[d] += scores[p] * vp[d];
            }
        }
        free(scores);
    }


    // ---- Apply sigmoid gate to attention output ----
    // MLX: return self.o_proj(output * mx.sigmoid(gate))
    // gate is reshaped to [B, L, num_heads*head_dim] = flat [q_dim]
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-q_gate[i]));
        attn_out[i] *= g;
    }

    // ---- Output projection ----
    float *attn_projected = calloc(g_cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    uint32_t *ow = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", layer_idx);
    uint16_t *os_ptr = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", layer_idx);
    uint16_t *ob = get_tensor_ptr(wf, name);
    if (ow && os_ptr && ob) fast_dequant_matvec(ow, os_ptr, ob, attn_out, attn_projected, g_cfg.hidden_dim, q_dim, g_cfg.group_size);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] attn_out_rms=%.6f o_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(attn_out, q_dim),
                attn_projected[0], attn_projected[1], attn_projected[2], attn_projected[3], attn_projected[4]);
    }

    // ---- Residual connection ----
    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_projected[i];
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] AFTER layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, g_cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    free(normed);
    free(residual);
    free(q);
    free(q_gate);
    free(k);
    free(v);
    free(attn_out);
    free(attn_projected);
}

// ============================================================================
// Stage 2: batched (N=2) full-attention block for MTP draft/verify.
// Processes two consecutive positions in one pass. The weight-bound q/k/v
// projections are shared via gpu_dequant_matmul2 (one weight read for both
// tokens); the cheap per-token tail (q/k norm, RoPE, attention, gate, o_proj,
// residual) mirrors full_attention_forward exactly so results match it. Token b
// (at pos_b = pos_a+1) attends to token a's freshly-appended K/V. o_proj uses
// two single matvecs (its in_dim=q_dim>4096 exceeds the matmul2 kernel limit).
// ============================================================================
static void attn_block_2(
    WeightFile *wf, int layer_idx,
    float *ha, float *hb,     // [hidden_dim] in/out for the two tokens
    KVCache *kv, int pos_a, int pos_b
) {
    char name[256];
    int q_proj_dim = g_cfg.num_attn_heads * g_cfg.head_dim * 2;
    int q_dim = g_cfg.num_attn_heads * g_cfg.head_dim;
    int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    int heads_per_kv = g_cfg.num_attn_heads / g_cfg.num_kv_heads;
    float scale = 1.0f / sqrtf((float)g_cfg.head_dim);

    float *res_a = malloc(g_cfg.hidden_dim * sizeof(float));
    float *res_b = malloc(g_cfg.hidden_dim * sizeof(float));
    memcpy(res_a, ha, g_cfg.hidden_dim * sizeof(float));
    memcpy(res_b, hb, g_cfg.hidden_dim * sizeof(float));

    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    float *na = malloc(g_cfg.hidden_dim * sizeof(float));
    float *nb = malloc(g_cfg.hidden_dim * sizeof(float));
    cpu_rms_norm(ha, norm_w, na, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
    cpu_rms_norm(hb, norm_w, nb, g_cfg.hidden_dim, g_cfg.rms_norm_eps);

    // ---- Batched Q/K/V projections (shared weight read across both tokens) ----
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    uint32_t *qw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", layer_idx);
    uint16_t *qs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", layer_idx);
    uint16_t *qb_ = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    uint32_t *kw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", layer_idx);
    uint16_t *ks = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", layer_idx);
    uint16_t *kb = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    uint32_t *vw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", layer_idx);
    uint16_t *vs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", layer_idx);
    uint16_t *vb = get_tensor_ptr(wf, name);

    float *qp_a = calloc(q_proj_dim, sizeof(float)), *qp_b = calloc(q_proj_dim, sizeof(float));
    float *k_a = calloc(kv_dim, sizeof(float)), *k_b = calloc(kv_dim, sizeof(float));
    float *v_a = calloc(kv_dim, sizeof(float)), *v_b = calloc(kv_dim, sizeof(float));
    gpu_dequant_matmul2(g_metal, qw, qs, qb_, na, nb, qp_a, qp_b, q_proj_dim, g_cfg.hidden_dim, g_cfg.group_size);
    gpu_dequant_matmul2(g_metal, kw, ks, kb, na, nb, k_a, k_b, kv_dim, g_cfg.hidden_dim, g_cfg.group_size);
    gpu_dequant_matmul2(g_metal, vw, vs, vb, na, nb, v_a, v_b, kv_dim, g_cfg.hidden_dim, g_cfg.group_size);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    uint16_t *qnorm_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    uint16_t *knorm_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    uint32_t *ow = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", layer_idx);
    uint16_t *os_ptr = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", layer_idx);
    uint16_t *ob = get_tensor_ptr(wf, name);

    // Per-token tail. Token a appended first (pos_a), so b (pos_b) attends to it.
    float *qp[2] = { qp_a, qp_b }; float *kk[2] = { k_a, k_b }; float *vv[2] = { v_a, v_b };
    float *out_h[2] = { ha, hb }; float *res[2] = { res_a, res_b };
    int pos[2] = { pos_a, pos_b };
    for (int t = 0; t < 2; t++) {
        float *q = calloc(q_dim, sizeof(float));
        float *q_gate = calloc(q_dim, sizeof(float));
        for (int h = 0; h < g_cfg.num_attn_heads; h++) {
            float *src = qp[t] + h * (2 * g_cfg.head_dim);
            memcpy(q + h * g_cfg.head_dim, src, g_cfg.head_dim * sizeof(float));
            memcpy(q_gate + h * g_cfg.head_dim, src + g_cfg.head_dim, g_cfg.head_dim * sizeof(float));
        }
        if (qnorm_w) {
            for (int h = 0; h < g_cfg.num_attn_heads; h++) {
                float *qh = q + h * g_cfg.head_dim; float ss = 0.0f;
                for (int i = 0; i < g_cfg.head_dim; i++) ss += qh[i] * qh[i];
                float inv = 1.0f / sqrtf(ss / g_cfg.head_dim + g_cfg.rms_norm_eps);
                for (int i = 0; i < g_cfg.head_dim; i++) qh[i] = qh[i] * inv * bf16_to_f32(qnorm_w[i]);
            }
        }
        if (knorm_w) {
            for (int h = 0; h < g_cfg.num_kv_heads; h++) {
                float *kh = kk[t] + h * g_cfg.head_dim; float ss = 0.0f;
                for (int i = 0; i < g_cfg.head_dim; i++) ss += kh[i] * kh[i];
                float inv = 1.0f / sqrtf(ss / g_cfg.head_dim + g_cfg.rms_norm_eps);
                for (int i = 0; i < g_cfg.head_dim; i++) kh[i] = kh[i] * inv * bf16_to_f32(knorm_w[i]);
            }
        }
        apply_rotary_emb(q, kk[t], pos[t], g_cfg.num_attn_heads, g_cfg.num_kv_heads, g_cfg.head_dim, g_cfg.rotary_dim);

        int cache_pos = kv->len;
        memcpy(kv->k_cache + (size_t)cache_pos * kv_dim, kk[t], kv_dim * sizeof(float));
        memcpy(kv->v_cache + (size_t)cache_pos * kv_dim, vv[t], kv_dim * sizeof(float));
        kv->len++;

        float *attn_out = calloc(q_dim, sizeof(float));
        for (int h = 0; h < g_cfg.num_attn_heads; h++) {
            int kv_h = h / heads_per_kv;
            float *qh = q + h * g_cfg.head_dim;
            float *scores = malloc(kv->len * sizeof(float));
            for (int p = 0; p < kv->len; p++) {
                float *kp = kv->k_cache + (size_t)p * kv_dim + kv_h * g_cfg.head_dim;
                float dot = 0.0f;
                for (int d = 0; d < g_cfg.head_dim; d++) dot += qh[d] * kp[d];
                scores[p] = dot * scale;
            }
            cpu_softmax(scores, kv->len);
            float *oh = attn_out + h * g_cfg.head_dim;
            for (int p = 0; p < kv->len; p++) {
                float *vp = kv->v_cache + (size_t)p * kv_dim + kv_h * g_cfg.head_dim;
                for (int d = 0; d < g_cfg.head_dim; d++) oh[d] += scores[p] * vp[d];
            }
            free(scores);
        }
        for (int i = 0; i < q_dim; i++) attn_out[i] *= 1.0f / (1.0f + expf(-q_gate[i]));

        float *attn_proj = calloc(g_cfg.hidden_dim, sizeof(float));
        if (ow && os_ptr && ob) fast_dequant_matvec(ow, os_ptr, ob, attn_out, attn_proj, g_cfg.hidden_dim, q_dim, g_cfg.group_size);
        for (int i = 0; i < g_cfg.hidden_dim; i++) out_h[t][i] = res[t][i] + attn_proj[i];
        free(q); free(q_gate); free(attn_out); free(attn_proj);
    }

    free(res_a); free(res_b); free(na); free(nb);
    free(qp_a); free(qp_b); free(k_a); free(k_b); free(v_a); free(v_b);
}

// Stage 4d-ii: batched (N=2) GPU linear-attention block for MTP verify.
// in_proj amortized via matmul2; the production 5-encoder delta-net recurrence is
// driven per-token (a updates buf_conv_state/buf_delta_state[li], then b reads them
// — faithful to production, recurrence is sequential anyway); out_proj amortized.
// hidden = residual + out_proj(gated_delta_output). Mirrors attn_block_2's contract.
// Snapshot of recurrent GPU state + KV lens for batched-verify reject-rollback.
typedef struct { float **delta; float **conv; int *kvlen; } GpuStateSnap;
__attribute__((unused))
static void linear_block_2(WeightFile *wf, int layer, float *ha, float *hb, GpuStateSnap *rb) {
    int Hd = g_cfg.hidden_dim, gsz = g_cfg.group_size;
    int li = layer - (layer + 1) / g_cfg.full_attn_interval;   // linear_layer_idx
    int qkv_dim = g_cfg.linear_conv_dim;
    int z_dim = g_cfg.linear_total_value;
    int nvh = g_cfg.linear_num_v_heads, nkh = g_cfg.linear_num_k_heads;
    char nm[256];
    #define LN(suf) (snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn." suf,layer), get_tensor_ptr(wf,nm))
    uint16_t *innorm = (snprintf(nm,sizeof(nm),"model.layers.%d.input_layernorm.weight",layer), get_tensor_ptr(wf,nm));
    uint32_t *qkvw=LN("in_proj_qkv.weight"); uint16_t *qkvs=LN("in_proj_qkv.scales"),*qkvb=LN("in_proj_qkv.biases");
    uint32_t *zw=LN("in_proj_z.weight");     uint16_t *zs=LN("in_proj_z.scales"),*zb=LN("in_proj_z.biases");
    uint32_t *bw=LN("in_proj_b.weight");     uint16_t *bs=LN("in_proj_b.scales"),*bb_=LN("in_proj_b.biases");
    uint32_t *aw=LN("in_proj_a.weight");     uint16_t *as_=LN("in_proj_a.scales"),*ab=LN("in_proj_a.biases");
    uint32_t *ow=LN("out_proj.weight");      uint16_t *os=LN("out_proj.scales"),*ob_=LN("out_proj.biases");
    uint16_t *conv1d_w = LN("conv1d.weight");
    float    *A_log    = (float *)LN("A_log");
    uint16_t *dt_bias  = LN("dt_bias");
    uint16_t *gnorm_w  = LN("norm.weight");
    #undef LN

    float *res_a = malloc(Hd*sizeof(float)), *res_b = malloc(Hd*sizeof(float));
    memcpy(res_a, ha, Hd*sizeof(float)); memcpy(res_b, hb, Hd*sizeof(float));
    float *na = malloc(Hd*sizeof(float)), *nb = malloc(Hd*sizeof(float));
    cpu_rms_norm(ha, innorm, na, Hd, g_cfg.rms_norm_eps);
    cpu_rms_norm(hb, innorm, nb, Hd, g_cfg.rms_norm_eps);

    // in_proj (amortized, in_dim=hidden<=4096)
    float *qkv_a=malloc(qkv_dim*sizeof(float)),*qkv_b=malloc(qkv_dim*sizeof(float));
    float *z_a=malloc(z_dim*sizeof(float)),*z_b=malloc(z_dim*sizeof(float));
    float *be_a=malloc(nvh*sizeof(float)),*be_b=malloc(nvh*sizeof(float));
    float *al_a=malloc(nvh*sizeof(float)),*al_b=malloc(nvh*sizeof(float));
    gpu_dequant_matmul2(g_metal, qkvw,qkvs,qkvb, na,nb, qkv_a,qkv_b, qkv_dim, Hd, gsz);
    gpu_dequant_matmul2(g_metal, zw,zs,zb,       na,nb, z_a,z_b,     z_dim,   Hd, gsz);
    gpu_dequant_matmul2(g_metal, bw,bs,bb_,      na,nb, be_a,be_b,   nvh,     Hd, gsz);
    gpu_dequant_matmul2(g_metal, aw,as_,ab,      na,nb, al_a,al_b,   nvh,     Hd, gsz);

    NSUInteger conv_off = (NSUInteger)((const char*)conv1d_w - (const char*)[g_metal->wf_buf contents]);
    NSUInteger alog_off = (NSUInteger)((const char*)A_log   - (const char*)[g_metal->wf_buf contents]);
    NSUInteger dtb_off  = (NSUInteger)((const char*)dt_bias - (const char*)[g_metal->wf_buf contents]);
    NSUInteger gn_off   = (NSUInteger)((const char*)gnorm_w - (const char*)[g_metal->wf_buf contents]);
    uint32_t conv_dim = g_cfg.linear_conv_dim, key_dim = g_cfg.linear_key_dim, value_dim = g_cfg.linear_value_dim;
    float inv_scale = 1.0f / sqrtf((float)g_cfg.linear_key_dim), eps = g_cfg.rms_norm_eps;
    uint32_t khpv = (uint32_t)(nvh / nkh);

    float *qkv_t[2] = { qkv_a, qkv_b }; float *z_t[2] = { z_a, z_b };
    float *be_t[2] = { be_a, be_b }; float *al_t[2] = { al_a, al_b };
    float *gated_a = malloc(z_dim*sizeof(float)), *gated_b = malloc(z_dim*sizeof(float));
    float *gated_t[2] = { gated_a, gated_b };

    for (int t = 0; t < 2; t++) {
        memcpy([g_metal->batch_out[0] contents], qkv_t[t], qkv_dim*sizeof(float));
        memcpy([g_metal->batch_out[1] contents], z_t[t],   z_dim*sizeof(float));
        memcpy([g_metal->batch_out[2] contents], be_t[t],  nvh*sizeof(float));
        memcpy([g_metal->batch_out[3] contents], al_t[t],  nvh*sizeof(float));
        id<MTLCommandBuffer> cmd = [g_metal->queue commandBuffer];
        // L1: conv1d_step
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_metal->conv1d_step];
          [e setBuffer:g_metal->buf_conv_state[li] offset:0 atIndex:0];
          [e setBuffer:g_metal->batch_out[0] offset:0 atIndex:1];
          [e setBuffer:g_metal->wf_buf offset:conv_off atIndex:2];
          [e setBuffer:g_metal->buf_conv_output offset:0 atIndex:3];
          [e setBytes:&conv_dim length:4 atIndex:4];
          [e dispatchThreadgroups:MTLSizeMake((conv_dim+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [e endEncoding]; }
        // L2: rms_norm_qk
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_metal->rms_norm_qk];
          [e setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];
          [e setBuffer:g_metal->buf_conv_output offset:g_cfg.linear_total_key*sizeof(float) atIndex:1];
          [e setBytes:&key_dim length:4 atIndex:2]; [e setBytes:&inv_scale length:4 atIndex:3];
          [e dispatchThreadgroups:MTLSizeMake(nkh,1,1) threadsPerThreadgroup:MTLSizeMake(key_dim,1,1)];
          [e endEncoding]; }
        // L3: compute_decay_beta
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_metal->compute_decay_beta];
          [e setBuffer:g_metal->batch_out[3] offset:0 atIndex:0];
          [e setBuffer:g_metal->batch_out[2] offset:0 atIndex:1];
          [e setBuffer:g_metal->wf_buf offset:alog_off atIndex:2];
          [e setBuffer:g_metal->wf_buf offset:dtb_off atIndex:3];
          [e setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
          [e setBuffer:g_metal->buf_delta_beta offset:0 atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(nvh,1,1)];
          [e endEncoding]; }
        // L4: gated_delta_net_step
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_metal->delta_net_step];
          [e setBuffer:g_metal->buf_delta_state[li] offset:0 atIndex:0];
          [e setBuffer:g_metal->buf_conv_output offset:0 atIndex:1];
          [e setBuffer:g_metal->buf_conv_output offset:g_cfg.linear_total_key*sizeof(float) atIndex:2];
          [e setBuffer:g_metal->buf_conv_output offset:2*g_cfg.linear_total_key*sizeof(float) atIndex:3];
          [e setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
          [e setBuffer:g_metal->buf_delta_beta offset:0 atIndex:5];
          [e setBuffer:g_metal->buf_delta_output offset:0 atIndex:6];
          [e setBytes:&khpv length:sizeof(khpv) atIndex:7];
          [e dispatchThreadgroups:MTLSizeMake(nvh,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
          [e endEncoding]; }
        // L5: gated_rms_norm -> batch_out[6]
        { id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
          [e setComputePipelineState:g_metal->gated_rms_norm];
          [e setBuffer:g_metal->buf_delta_output offset:0 atIndex:0];
          [e setBuffer:g_metal->batch_out[1] offset:0 atIndex:1];
          [e setBuffer:g_metal->wf_buf offset:gn_off atIndex:2];
          [e setBuffer:g_metal->batch_out[6] offset:0 atIndex:3];
          [e setBytes:&value_dim length:4 atIndex:4]; [e setBytes:&eps length:4 atIndex:5];
          [e dispatchThreadgroups:MTLSizeMake(nvh,1,1) threadsPerThreadgroup:MTLSizeMake(value_dim,1,1)];
          [e endEncoding]; }
        [cmd commit]; [cmd waitUntilCompleted];
        memcpy(gated_t[t], [g_metal->batch_out[6] contents], z_dim*sizeof(float));
        // Capture post-token-a recurrent state (before token b) for reject-rollback.
        if (rb && t == 0) {
            memcpy(rb->delta[li], [g_metal->buf_delta_state[li] contents], 64*128*128*sizeof(float));
            memcpy(rb->conv[li],  [g_metal->buf_conv_state[li] contents],  3*12288*sizeof(float));
        }
    }

    // out_proj (amortized, in_dim=linear_total_value=4096 fits matmul2)
    float *oa = malloc(Hd*sizeof(float)), *ob_out = malloc(Hd*sizeof(float));
    gpu_dequant_matmul2(g_metal, ow,os,ob_, gated_a,gated_b, oa,ob_out, Hd, z_dim, gsz);
    for (int i = 0; i < Hd; i++) { ha[i] = res_a[i] + oa[i]; hb[i] = res_b[i] + ob_out[i]; }

    free(res_a); free(res_b); free(na); free(nb);
    free(qkv_a); free(qkv_b); free(z_a); free(z_b); free(be_a); free(be_b); free(al_a); free(al_b);
    free(gated_a); free(gated_b); free(oa); free(ob_out);
}

// Dense-27B batched (N=2) MLP block: hidden += down(swiglu(gate(h_post), up(h_post)))
// for both tokens. Uses matmulN (not matmul2) since dense in_dims 5120/17408 exceed
// the matmul2 x_shared 4096 limit. No routing/experts/shared → no drift amplification.
static void dense_mlp_block_2(WeightFile *wf, int layer, float *ha, float *hb) {
    int H = g_cfg.hidden_dim, inter = g_cfg.dense_intermediate, gsz = g_cfg.group_size;
    char nm[256];
    #define DW(suf) (snprintf(nm,sizeof(nm),"model.layers.%d.mlp." suf,layer), get_tensor_ptr(wf,nm))
    uint16_t *post_w = (snprintf(nm,sizeof(nm),"model.layers.%d.post_attention_layernorm.weight",layer), get_tensor_ptr(wf,nm));
    uint32_t *gw=DW("gate_proj.weight"); uint16_t *gs=DW("gate_proj.scales"),*gb=DW("gate_proj.biases");
    uint32_t *uw=DW("up_proj.weight");   uint16_t *us=DW("up_proj.scales"),  *ub=DW("up_proj.biases");
    uint32_t *dw=DW("down_proj.weight"); uint16_t *ds=DW("down_proj.scales"),*db=DW("down_proj.biases");
    #undef DW
    float *hp = malloc((size_t)2*H*sizeof(float));   // [2][hidden] packed h_post
    cpu_rms_norm(ha, post_w, hp,     H, g_cfg.rms_norm_eps);
    cpu_rms_norm(hb, post_w, hp + H, H, g_cfg.rms_norm_eps);
    float *g2 = malloc((size_t)2*inter*sizeof(float)), *u2 = malloc((size_t)2*inter*sizeof(float)), *a2 = malloc((size_t)2*inter*sizeof(float));
    gpu_dequant_matmulN(g_metal, gw, gs, gb, hp, g2, inter, H, gsz, 2);
    gpu_dequant_matmulN(g_metal, uw, us, ub, hp, u2, inter, H, gsz, 2);
    cpu_swiglu(g2,         u2,         a2,         inter);
    cpu_swiglu(g2 + inter, u2 + inter, a2 + inter, inter);
    float *o2 = malloc((size_t)2*H*sizeof(float));
    gpu_dequant_matmulN(g_metal, dw, ds, db, a2, o2, H, inter, gsz, 2);
    for (int i = 0; i < H; i++) { ha[i] += o2[i]; hb[i] += o2[H + i]; }
    free(hp); free(g2); free(u2); free(a2); free(o2);
}

// Stage 2 unit test: attn_block_2 must match two sequential full_attention_forward
// calls (the oracle) on an identical pre-seeded KV cache, and should be faster
// (q/k/v projections amortized). Gated by --mtp-bench-attn.
static int mtp_bench_attn(WeightFile *wf) {
    if (!g_metal || !g_metal->wf_buf) { fprintf(stderr, "[mtp-attn] requires GPU\n"); return 1; }
    int layer = g_cfg.full_attn_interval - 1;   // first full-attention layer (0-indexed)
    int H = g_cfg.hidden_dim;
    int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    int cap = 64, P = 5;

    KVCache kv_o = {0}, kv_b = {0};
    kv_o.k_cache = calloc((size_t)cap * kv_dim, sizeof(float)); kv_o.v_cache = calloc((size_t)cap * kv_dim, sizeof(float));
    kv_b.k_cache = calloc((size_t)cap * kv_dim, sizeof(float)); kv_b.v_cache = calloc((size_t)cap * kv_dim, sizeof(float));

    for (int p = 0; p < P; p++) {
        float *h = malloc(H * sizeof(float));
        for (int i = 0; i < H; i++) h[i] = sinf((i + p * 13) * 0.01f);
        full_attention_forward(wf, layer, h, &kv_o, p);
        free(h);
    }
    memcpy(kv_b.k_cache, kv_o.k_cache, (size_t)cap * kv_dim * sizeof(float));
    memcpy(kv_b.v_cache, kv_o.v_cache, (size_t)cap * kv_dim * sizeof(float));
    kv_b.len = kv_o.len;

    float *ha = malloc(H * sizeof(float)), *hb = malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) { ha[i] = cosf(i * 0.013f); hb[i] = sinf(i * 0.019f + 1.0f); }
    int pa = kv_o.len, pb = kv_o.len + 1;

    float *ha_o = malloc(H * sizeof(float)), *hb_o = malloc(H * sizeof(float));
    float *ha_b = malloc(H * sizeof(float)), *hb_b = malloc(H * sizeof(float));
    memcpy(ha_o, ha, H * sizeof(float)); memcpy(hb_o, hb, H * sizeof(float));
    memcpy(ha_b, ha, H * sizeof(float)); memcpy(hb_b, hb, H * sizeof(float));

    full_attention_forward(wf, layer, ha_o, &kv_o, pa);
    full_attention_forward(wf, layer, hb_o, &kv_o, pb);
    attn_block_2(wf, layer, ha_b, hb_b, &kv_b, pa, pb);

    double err = 0.0;
    for (int i = 0; i < H; i++) { err = fmax(err, fabs(ha_o[i] - ha_b[i])); err = fmax(err, fabs(hb_o[i] - hb_b[i])); }

    const int M = 100;
    double t = now_ms();
    for (int m = 0; m < M; m++) {
        kv_o.len = P; memcpy(ha_o, ha, H * sizeof(float)); memcpy(hb_o, hb, H * sizeof(float));
        full_attention_forward(wf, layer, ha_o, &kv_o, pa);
        full_attention_forward(wf, layer, hb_o, &kv_o, pb);
    }
    double t_single = (now_ms() - t) / M;
    t = now_ms();
    for (int m = 0; m < M; m++) {
        kv_b.len = P; memcpy(ha_b, ha, H * sizeof(float)); memcpy(hb_b, hb, H * sizeof(float));
        attn_block_2(wf, layer, ha_b, hb_b, &kv_b, pa, pb);
    }
    double t_batched = (now_ms() - t) / M;

    fprintf(stderr, "[mtp-attn] layer=%d | 2x single=%.4f ms | batched=%.4f ms | ratio=%.3f | max_err=%.2e\n",
            layer, t_single, t_batched, t_batched / t_single, err);
    fprintf(stderr, "[mtp-attn] %s (tolerance ~1e-3 for fp reduction-order differences)\n",
            err < 1e-2 ? "PASS: batched matches oracle" : "FAIL: outputs diverge");

    free(kv_o.k_cache); free(kv_o.v_cache); free(kv_b.k_cache); free(kv_b.v_cache);
    free(ha); free(hb); free(ha_o); free(hb_o); free(ha_b); free(hb_b);
    return err < 1e-2 ? 0 : 1;
}

// ============================================================================
// Stage 3: batched (N=2) MoE block for MTP draft/verify.
// hidden = h_mid + Σ wₖ·expertₖ(norm(h_mid)) + shared(norm(h_mid))·σ(gate).
// Routed-expert and shared-expert matmuls are amortized via gpu_dequant_matmul2
// when both tokens hit the same expert; experts hit by only one token fall back
// to a single matvec. The expert weights are loaded once per unique expert in
// the union of the two tokens' routes (the I/O the real batched verify pays).
// ============================================================================
static int g_moe_last_union = 0;  // |union of the two tokens' routed experts| (diagnostic)

static void expert_slices(void *buf,
    uint32_t **gw, uint16_t **gs, uint16_t **gb,
    uint32_t **uw, uint16_t **us, uint16_t **ub,
    uint32_t **dw, uint16_t **ds, uint16_t **db) {
    char *p = buf;
    *gw = (uint32_t *)p; p += g_cfg.gate_w_size;
    *gs = (uint16_t *)p; p += g_cfg.gate_s_size;
    *gb = (uint16_t *)p; p += g_cfg.gate_b_size;
    *uw = (uint32_t *)p; p += g_cfg.up_w_size;
    *us = (uint16_t *)p; p += g_cfg.up_s_size;
    *ub = (uint16_t *)p; p += g_cfg.up_b_size;
    *dw = (uint32_t *)p; p += g_cfg.down_w_size;
    *ds = (uint16_t *)p; p += g_cfg.down_s_size;
    *db = (uint16_t *)p;
}

// Single-token routed expert FFN: out = down(swiglu(gate(x), up(x))).
static void routed_ffn_1(void *buf, const float *x, float *out) {
    int H = g_cfg.hidden_dim, MI = g_cfg.moe_intermediate, gsz = g_cfg.group_size;
    uint32_t *gw, *uw, *dw; uint16_t *gs, *gb, *us, *ub, *ds, *db;
    expert_slices(buf, &gw, &gs, &gb, &uw, &us, &ub, &dw, &ds, &db);
    float *eg = malloc(MI * sizeof(float)), *eu = malloc(MI * sizeof(float)), *ea = malloc(MI * sizeof(float));
    fast_dequant_matvec(gw, gs, gb, x, eg, MI, H, gsz);
    fast_dequant_matvec(uw, us, ub, x, eu, MI, H, gsz);
    cpu_swiglu(eg, eu, ea, MI);
    fast_dequant_matvec(dw, ds, db, ea, out, H, MI, gsz);
    free(eg); free(eu); free(ea);
}

// Batched routed expert FFN: one weight read serves both tokens.
static void routed_ffn_2(void *buf, const float *xa, const float *xb, float *oa, float *ob) {
    int H = g_cfg.hidden_dim, MI = g_cfg.moe_intermediate, gsz = g_cfg.group_size;
    uint32_t *gw, *uw, *dw; uint16_t *gs, *gb, *us, *ub, *ds, *db;
    expert_slices(buf, &gw, &gs, &gb, &uw, &us, &ub, &dw, &ds, &db);
    float *ega = malloc(MI*sizeof(float)), *egb = malloc(MI*sizeof(float));
    float *eua = malloc(MI*sizeof(float)), *eub = malloc(MI*sizeof(float));
    float *eaa = malloc(MI*sizeof(float)), *eab = malloc(MI*sizeof(float));
    gpu_dequant_matmul2(g_metal, gw, gs, gb, xa, xb, ega, egb, MI, H, gsz);
    gpu_dequant_matmul2(g_metal, uw, us, ub, xa, xb, eua, eub, MI, H, gsz);
    cpu_swiglu(ega, eua, eaa, MI);
    cpu_swiglu(egb, eub, eab, MI);
    gpu_dequant_matmul2(g_metal, dw, ds, db, eaa, eab, oa, ob, H, MI, gsz);
    free(ega); free(egb); free(eua); free(eub); free(eaa); free(eab);
}

// Compute post-attn norm + routing (softmax/topk/normalize) for one token.
static void moe_route(WeightFile *wf, int layer, const float *hidden, float *h_post, int *idx, float *wt) {
    int H = g_cfg.hidden_dim, K = g_cfg.num_experts_per_tok, gsz = g_cfg.group_size;
    char nm[256];
    snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", layer);
    uint16_t *post_w = get_tensor_ptr(wf, nm);
    cpu_rms_norm(hidden, post_w, h_post, H, g_cfg.rms_norm_eps);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", layer);   uint32_t *gw = get_tensor_ptr(wf, nm);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.scales", layer);   uint16_t *gs = get_tensor_ptr(wf, nm);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.biases", layer);   uint16_t *gb = get_tensor_ptr(wf, nm);
    float *scores = malloc(g_cfg.num_experts * sizeof(float));
    fast_dequant_matvec(gw, gs, gb, h_post, scores, g_cfg.num_experts, H, gsz);
    cpu_softmax(scores, g_cfg.num_experts);
    cpu_topk(scores, g_cfg.num_experts, K, idx, wt);
    cpu_normalize_weights(wt, K);
    free(scores);
}

// Shared expert + gate for one token: returns shared_out (gated), needs h_post.
static void shared_expert_1(WeightFile *wf, int layer, const float *h_post, float *shared_out) {
    int H = g_cfg.hidden_dim, SI = g_cfg.shared_intermediate, gsz = g_cfg.group_size;
    char nm[256];
    #define SW(suf) (snprintf(nm,sizeof(nm),"model.layers.%d.mlp." suf,layer), get_tensor_ptr(wf,nm))
    uint32_t *sgw = SW("shared_expert.gate_proj.weight"); uint16_t *sgs = SW("shared_expert.gate_proj.scales"), *sgb = SW("shared_expert.gate_proj.biases");
    uint32_t *suw = SW("shared_expert.up_proj.weight");   uint16_t *sus = SW("shared_expert.up_proj.scales"),   *sub = SW("shared_expert.up_proj.biases");
    uint32_t *sdw = SW("shared_expert.down_proj.weight"); uint16_t *sds = SW("shared_expert.down_proj.scales"), *sdb = SW("shared_expert.down_proj.biases");
    uint32_t *segw = SW("shared_expert_gate.weight");     uint16_t *segs = SW("shared_expert_gate.scales"),     *segb = SW("shared_expert_gate.biases");
    #undef SW
    float *sg = malloc(SI*sizeof(float)), *su = malloc(SI*sizeof(float)), *sa = malloc(SI*sizeof(float)), score = 0;
    fast_dequant_matvec(sgw, sgs, sgb, h_post, sg, SI, H, gsz);
    fast_dequant_matvec(suw, sus, sub, h_post, su, SI, H, gsz);
    cpu_swiglu(sg, su, sa, SI);
    fast_dequant_matvec(sdw, sds, sdb, sa, shared_out, H, SI, gsz);
    fast_dequant_matvec(segw, segs, segb, h_post, &score, 1, H, gsz);
    float w = cpu_sigmoid(score);
    for (int i = 0; i < H; i++) shared_out[i] *= w;
    free(sg); free(su); free(sa);
}

// Batched shared expert: always used by both tokens (guaranteed 100% overlap),
// so every matmul amortizes via matmul2.
static void shared_expert_2(WeightFile *wf, int layer, const float *hpa, const float *hpb,
                            float *out_a, float *out_b) {
    int H = g_cfg.hidden_dim, SI = g_cfg.shared_intermediate, gsz = g_cfg.group_size;
    char nm[256];
    #define SW(suf) (snprintf(nm,sizeof(nm),"model.layers.%d.mlp." suf,layer), get_tensor_ptr(wf,nm))
    uint32_t *sgw = SW("shared_expert.gate_proj.weight"); uint16_t *sgs = SW("shared_expert.gate_proj.scales"), *sgb = SW("shared_expert.gate_proj.biases");
    uint32_t *suw = SW("shared_expert.up_proj.weight");   uint16_t *sus = SW("shared_expert.up_proj.scales"),   *sub = SW("shared_expert.up_proj.biases");
    uint32_t *sdw = SW("shared_expert.down_proj.weight"); uint16_t *sds = SW("shared_expert.down_proj.scales"), *sdb = SW("shared_expert.down_proj.biases");
    uint32_t *segw = SW("shared_expert_gate.weight");     uint16_t *segs = SW("shared_expert_gate.scales"),     *segb = SW("shared_expert_gate.biases");
    #undef SW
    float *sga = malloc(SI*sizeof(float)), *sgb_ = malloc(SI*sizeof(float));
    float *sua = malloc(SI*sizeof(float)), *sub_ = malloc(SI*sizeof(float));
    float *saa = malloc(SI*sizeof(float)), *sab = malloc(SI*sizeof(float));
    float scorea = 0, scoreb = 0;
    gpu_dequant_matmul2(g_metal, sgw, sgs, sgb, hpa, hpb, sga, sgb_, SI, H, gsz);
    gpu_dequant_matmul2(g_metal, suw, sus, sub, hpa, hpb, sua, sub_, SI, H, gsz);
    cpu_swiglu(sga, sua, saa, SI);
    cpu_swiglu(sgb_, sub_, sab, SI);
    gpu_dequant_matmul2(g_metal, sdw, sds, sdb, saa, sab, out_a, out_b, H, SI, gsz);
    gpu_dequant_matmul2(g_metal, segw, segs, segb, hpa, hpb, &scorea, &scoreb, 1, H, gsz);
    float wa = cpu_sigmoid(scorea), wb = cpu_sigmoid(scoreb);
    for (int i = 0; i < H; i++) { out_a[i] *= wa; out_b[i] *= wb; }
    free(sga); free(sgb_); free(sua); free(sub_); free(saa); free(sab);
}

static void moe_block_1(WeightFile *wf, int layer, float *hidden, int packed_fd) {
    int H = g_cfg.hidden_dim, K = g_cfg.num_experts_per_tok;
    float *h_post = malloc(H * sizeof(float));
    int idx[64]; float wt[64];
    moe_route(wf, layer, hidden, h_post, idx, wt);
    float *moe_out = calloc(H, sizeof(float));
    void *buf = malloc(active_expert_size());
    float *eo = malloc(H * sizeof(float));
    for (int k = 0; k < K; k++) {
        if (pread(packed_fd, buf, active_expert_size(), (off_t)idx[k] * active_expert_size()) != (ssize_t)active_expert_size()) continue;
        routed_ffn_1(buf, h_post, eo);
        cpu_vec_madd(moe_out, eo, wt[k], H);
    }
    float *shared_out = malloc(H * sizeof(float));
    shared_expert_1(wf, layer, h_post, shared_out);
    for (int i = 0; i < H; i++) hidden[i] = hidden[i] + moe_out[i] + shared_out[i];
    free(h_post); free(moe_out); free(buf); free(eo); free(shared_out);
}

static void moe_block_2(WeightFile *wf, int layer, float *ha, float *hb, int packed_fd) {
    int H = g_cfg.hidden_dim, K = g_cfg.num_experts_per_tok;
    float *hpa = malloc(H*sizeof(float)), *hpb = malloc(H*sizeof(float));
    int ia[64], ib[64]; float wa[64], wb[64];
    moe_route(wf, layer, ha, hpa, ia, wa);
    moe_route(wf, layer, hb, hpb, ib, wb);
    float *moe_a = calloc(H, sizeof(float)), *moe_b = calloc(H, sizeof(float));
    void *buf = malloc(active_expert_size());
    float *oa = malloc(H*sizeof(float)), *ob = malloc(H*sizeof(float));

    // Union of the two routes: load each unique expert once.
    int seen[1024]; int n_union = 0; int uni[128];
    for (int e = 0; e < g_cfg.num_experts && e < 1024; e++) seen[e] = 0;
    for (int k = 0; k < K; k++) { if (!seen[ia[k]]) { seen[ia[k]] = 1; uni[n_union++] = ia[k]; } }
    for (int k = 0; k < K; k++) { if (!seen[ib[k]]) { seen[ib[k]] = 1; uni[n_union++] = ib[k]; } }
    g_moe_last_union = n_union;

    for (int u = 0; u < n_union; u++) {
        int e = uni[u];
        if (pread(packed_fd, buf, active_expert_size(), (off_t)e * active_expert_size()) != (ssize_t)active_expert_size()) continue;
        int ka = -1, kb = -1;
        for (int k = 0; k < K; k++) { if (ia[k] == e) ka = k; if (ib[k] == e) kb = k; }
        if (ka >= 0 && kb >= 0) {
            routed_ffn_2(buf, hpa, hpb, oa, ob);
            cpu_vec_madd(moe_a, oa, wa[ka], H);
            cpu_vec_madd(moe_b, ob, wb[kb], H);
        } else if (ka >= 0) {
            routed_ffn_1(buf, hpa, oa);
            cpu_vec_madd(moe_a, oa, wa[ka], H);
        } else if (kb >= 0) {
            routed_ffn_1(buf, hpb, ob);
            cpu_vec_madd(moe_b, ob, wb[kb], H);
        }
    }
    float *sa = malloc(H*sizeof(float)), *sb = malloc(H*sizeof(float));
    shared_expert_2(wf, layer, hpa, hpb, sa, sb);
    for (int i = 0; i < H; i++) { ha[i] = ha[i] + moe_a[i] + sa[i]; hb[i] = hb[i] + moe_b[i] + sb[i]; }
    free(hpa); free(hpb); free(moe_a); free(moe_b); free(buf); free(oa); free(ob); free(sa); free(sb);
}

// Stage 3 unit test: moe_block_2 must match two independent moe_block_1 calls.
static int mtp_bench_moe(WeightFile *wf, const char *model_path) {
    if (!g_metal || !g_metal->wf_buf) { fprintf(stderr, "[mtp-moe] requires GPU\n"); return 1; }
    int layer = 0, H = g_cfg.hidden_dim;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, layer);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[mtp-moe] cannot open %s\n", path); return 1; }

    float *ha = malloc(H*sizeof(float)), *hb = malloc(H*sizeof(float));
    for (int i = 0; i < H; i++) { ha[i] = sinf(i*0.011f); hb[i] = cosf(i*0.017f + 0.5f); }
    float *ha_o = malloc(H*sizeof(float)), *hb_o = malloc(H*sizeof(float));
    float *ha_b = malloc(H*sizeof(float)), *hb_b = malloc(H*sizeof(float));
    memcpy(ha_o, ha, H*sizeof(float)); memcpy(hb_o, hb, H*sizeof(float));
    memcpy(ha_b, ha, H*sizeof(float)); memcpy(hb_b, hb, H*sizeof(float));

    moe_block_1(wf, layer, ha_o, fd);
    moe_block_1(wf, layer, hb_o, fd);
    moe_block_2(wf, layer, ha_b, hb_b, fd);
    double err = 0.0;
    for (int i = 0; i < H; i++) { err = fmax(err, fabs(ha_o[i]-ha_b[i])); err = fmax(err, fabs(hb_o[i]-hb_b[i])); }

    const int M = 50;
    for (int w = 0; w < 3; w++) { memcpy(ha_o,ha,H*4); memcpy(hb_o,hb,H*4); moe_block_1(wf,layer,ha_o,fd); moe_block_1(wf,layer,hb_o,fd);
                                  memcpy(ha_b,ha,H*4); memcpy(hb_b,hb,H*4); moe_block_2(wf,layer,ha_b,hb_b,fd); }
    double t = now_ms();
    for (int m = 0; m < M; m++) { memcpy(ha_o,ha,H*sizeof(float)); memcpy(hb_o,hb,H*sizeof(float)); moe_block_1(wf,layer,ha_o,fd); moe_block_1(wf,layer,hb_o,fd); }
    double t_single = (now_ms()-t)/M;
    t = now_ms();
    for (int m = 0; m < M; m++) { memcpy(ha_b,ha,H*sizeof(float)); memcpy(hb_b,hb,H*sizeof(float)); moe_block_2(wf,layer,ha_b,hb_b,fd); }
    double t_batched = (now_ms()-t)/M;

    fprintf(stderr, "[mtp-moe] layer=%d K=%d union=%d/%d | 2x single=%.4f ms | batched=%.4f ms | ratio=%.3f | max_err=%.2e\n",
            layer, g_cfg.num_experts_per_tok, g_moe_last_union, 2 * g_cfg.num_experts_per_tok,
            t_single, t_batched, t_batched/t_single, err);
    fprintf(stderr, "[mtp-moe] %s (lower ratio = more expert overlap amortized)\n",
            err < 1e-2 ? "PASS: batched matches oracle" : "FAIL: outputs diverge");
    close(fd);
    free(ha); free(hb); free(ha_o); free(hb_o); free(ha_b); free(hb_b);
    return err < 1e-2 ? 0 : 1;
}

static void fused_layer_forward(WeightFile *wf, int layer_idx, float *hidden, KVCache *kv,
                                LinearAttnState *la_state, int pos, const void *mmap_base,
                                int K, int packed_fd);
static void complete_deferred_experts(void);
static void linear_attention_forward(WeightFile *wf, int layer_idx, float *hidden, LinearAttnState *state);

// 4b prerequisite: confirm the CPU-reference blocks (full_attention_forward /
// linear_attention_forward + moe_block_1) match the PRODUCTION fused_layer_forward
// for one layer on identical input. If they diverge, a batched forward built from
// the reference blocks would produce different tokens than normal decode.
static int mtp_verify_layer(WeightFile *wf, const char *model_path) {
    if (!g_metal || !g_metal->wf_buf) { fprintf(stderr, "[mtp-vlayer] requires GPU\n"); return 1; }
    int H = g_cfg.hidden_dim, K = g_cfg.num_experts_per_tok;
    int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    int full_layer = g_cfg.full_attn_interval - 1;   // first full-attention layer
    int lin_layer  = 0;                              // layer 0 is linear attention
    int layers[2] = { full_layer, lin_layer };
    const char *labels[2] = { "full-attn", "linear-attn" };
    int rc = 0;

    for (int t = 0; t < 2; t++) {
        int layer = layers[t]; int is_full = (t == 0);
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, layer);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "[mtp-vlayer] cannot open %s\n", path); rc = 1; continue; }
        struct stat st; void *mm = MAP_FAILED;
        if (fstat(fd, &st) == 0 && st.st_size > 0) mm = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

        float *H0 = malloc(H*sizeof(float));
        for (int i = 0; i < H; i++) H0[i] = sinf((i + layer*3) * 0.01f) * 0.5f;
        float *Hp = malloc(H*sizeof(float)), *Hc = malloc(H*sizeof(float));
        memcpy(Hp, H0, H*sizeof(float)); memcpy(Hc, H0, H*sizeof(float));

        // Production path: one fused layer + deferred-expert finalize.
        KVCache kvp = {0}; LinearAttnState *lsp = NULL;
        if (is_full) { kvp.k_cache = calloc((size_t)64*kv_dim, sizeof(float)); kvp.v_cache = calloc((size_t)64*kv_dim, sizeof(float)); }
        else lsp = linear_attn_state_new();
        fused_layer_forward(wf, layer, Hp, is_full ? &kvp : NULL, is_full ? NULL : lsp, 0,
                            mm != MAP_FAILED ? mm : NULL, K, fd);
        complete_deferred_experts();

        // Reference path: CPU attention block + moe_block_1.
        KVCache kvc = {0}; LinearAttnState *lsc = NULL;
        if (is_full) { kvc.k_cache = calloc((size_t)64*kv_dim, sizeof(float)); kvc.v_cache = calloc((size_t)64*kv_dim, sizeof(float));
                       full_attention_forward(wf, layer, Hc, &kvc, 0); }
        else { lsc = linear_attn_state_new(); linear_attention_forward(wf, layer, Hc, lsc); }
        moe_block_1(wf, layer, Hc, fd);

        double err = 0.0, ref = 0.0;
        for (int i = 0; i < H; i++) { err = fmax(err, fabs(Hp[i]-Hc[i])); ref = fmax(ref, fabs(Hp[i])); }
        double rel = ref > 0 ? err/ref : err;
        fprintf(stderr, "[mtp-vlayer] %-11s layer=%d | max_abs_err=%.3e | rel=%.3e | %s\n",
                labels[t], layer, err, rel, rel < 5e-2 ? "MATCH" : "DIVERGE");
        if (rel >= 5e-2) rc = 1;

        if (is_full) { free(kvp.k_cache); free(kvp.v_cache); free(kvc.k_cache); free(kvc.v_cache); }
        else { linear_attn_state_free(lsp); linear_attn_state_free(lsc); }
        if (mm != MAP_FAILED) munmap(mm, st.st_size);
        close(fd); free(H0); free(Hp); free(Hc);
    }
    fprintf(stderr, "[mtp-vlayer] MATCH => CPU reference blocks are faithful to production (safe to assemble batched forward)\n");
    return rc;
}

// 4d-ii scoping: is the CPU reference (full_attention_forward + moe_block_1) faithful
// to production fused_layer_forward at a DECODE position (len>=32, where production
// switches to GPU attention)? Prefill a full-attn layer's KV to len=40 via production,
// copy that KV to the reference, then compare both at pos 40. Same prefix K/V, so this
// isolates GPU-attention vs CPU-attention numerics. If MATCH, 4d-ii's full-attn layers
// can reuse attn_block_2 + moe_block_2.
static int mtp_verify_deep(WeightFile *wf, const char *model_path) {
    if (!g_metal || !g_metal->wf_buf) { fprintf(stderr, "[mtp-deep] requires GPU\n"); return 1; }
    int Hd = g_cfg.hidden_dim, kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim, K = g_cfg.num_experts_per_tok;
    int layer = g_cfg.full_attn_interval - 1;   // first full-attention layer (uses GPU attn at len>=32)
    int P = 40, cap = 128;
    char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, layer);
    int fd = open(path, O_RDONLY);
    void *mm = MAP_FAILED; struct stat st; if (fd>=0 && fstat(fd,&st)==0 && st.st_size>0) mm = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    KVCache kvp = {0}; kvp.k_cache = calloc((size_t)cap*kv_dim,sizeof(float)); kvp.v_cache = calloc((size_t)cap*kv_dim,sizeof(float));
    float *H = malloc(Hd*sizeof(float));
    for (int p = 0; p < P; p++) {
        for (int i = 0; i < Hd; i++) H[i] = sinf((i + p*7) * 0.01f) * 0.4f;
        fused_layer_forward(wf, layer, H, &kvp, NULL, p, mm!=MAP_FAILED?mm:NULL, K, fd);
        complete_deferred_experts();
    }
    // Reference KV = exact copy of the production-built K/V (no prefix drift).
    KVCache kvc = {0}; kvc.k_cache = malloc((size_t)cap*kv_dim*sizeof(float)); kvc.v_cache = malloc((size_t)cap*kv_dim*sizeof(float));
    memcpy(kvc.k_cache, kvp.k_cache, (size_t)cap*kv_dim*sizeof(float));
    memcpy(kvc.v_cache, kvp.v_cache, (size_t)cap*kv_dim*sizeof(float));
    kvc.len = kvp.len;

    float *Hp = malloc(Hd*sizeof(float)), *Hc = malloc(Hd*sizeof(float));
    for (int i = 0; i < Hd; i++) { Hp[i] = cosf(i*0.011f)*0.4f; Hc[i] = Hp[i]; }
    fused_layer_forward(wf, layer, Hp, &kvp, NULL, P, mm!=MAP_FAILED?mm:NULL, K, fd);
    complete_deferred_experts();
    full_attention_forward(wf, layer, Hc, &kvc, P);
    moe_block_1(wf, layer, Hc, fd);

    double err = 0, ref = 0;
    for (int i = 0; i < Hd; i++) { err = fmax(err, fabs(Hp[i]-Hc[i])); ref = fmax(ref, fabs(Hp[i])); }
    double rel = ref>0 ? err/ref : err;
    fprintf(stderr, "[mtp-deep] full-attn layer=%d at pos=%d (len=%d, GPU-attn regime) | max_err=%.3e rel=%.3e | %s\n",
            layer, P, kvc.len, err, rel, rel < 5e-2 ? "MATCH (reuse attn_block_2+moe_block_2)" : "DIVERGE (need GPU 2-query attn)");

    if (mm!=MAP_FAILED) munmap(mm, st.st_size); if (fd>=0) close(fd);
    free(kvp.k_cache); free(kvp.v_cache); free(kvc.k_cache); free(kvc.v_cache); free(H); free(Hp); free(Hc);
    return rel < 5e-2 ? 0 : 1;
}

// 4d-ii validation: linear_block_2 (+ moe_block_1) must match production
// fused_layer_forward for a LINEAR layer. Both share the GPU delta state
// (buf_delta_state[li]/buf_conv_state[li]); snapshot after prefix, run oracle
// (2 sequential production forwards), restore, run linear_block_2 (a->b) + MoE.
static int mtp_verify_linear(WeightFile *wf, const char *model_path) {
    if (!g_metal || !g_metal->wf_buf || !g_metal->delta_net_step) { fprintf(stderr, "[mtp-vlin] requires GPU delta-net\n"); return 1; }
    int Hd = g_cfg.hidden_dim, K = g_cfg.num_experts_per_tok;
    int layer = 0, li = layer - (layer + 1) / g_cfg.full_attn_interval;  // layer 0 is linear, li=0
    size_t dsz = 64*128*128*sizeof(float), csz = 3*12288*sizeof(float);
    char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, layer);
    int fd = open(path, O_RDONLY);
    void *mm = MAP_FAILED; struct stat st; if (fd>=0 && fstat(fd,&st)==0 && st.st_size>0) mm = mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    LinearAttnState *dummy = linear_attn_state_new();

    reset_delta_net_state();
    float *h = malloc(Hd*sizeof(float)); int P = 5;
    for (int p = 0; p < P; p++) {
        for (int i = 0; i < Hd; i++) h[i] = sinf((i + p*5) * 0.01f) * 0.4f;
        fused_layer_forward(wf, layer, h, NULL, dummy, p, mm!=MAP_FAILED?mm:NULL, K, fd);
        complete_deferred_experts();
    }
    // Snapshot post-prefix GPU state S.
    float *Sd = malloc(dsz), *Sc = malloc(csz);
    memcpy(Sd, [g_metal->buf_delta_state[li] contents], dsz);
    memcpy(Sc, [g_metal->buf_conv_state[li] contents], csz);

    float *Ha = malloc(Hd*sizeof(float)), *Hb = malloc(Hd*sizeof(float));
    for (int i = 0; i < Hd; i++) { Ha[i] = cosf(i*0.011f)*0.4f; Hb[i] = sinf(i*0.017f+0.7f)*0.4f; }

    // Oracle: two sequential production forwards from S.
    float *Hao = malloc(Hd*sizeof(float)), *Hbo = malloc(Hd*sizeof(float));
    memcpy(Hao, Ha, Hd*sizeof(float)); memcpy(Hbo, Hb, Hd*sizeof(float));
    fused_layer_forward(wf, layer, Hao, NULL, dummy, P,   mm!=MAP_FAILED?mm:NULL, K, fd); complete_deferred_experts();
    fused_layer_forward(wf, layer, Hbo, NULL, dummy, P+1, mm!=MAP_FAILED?mm:NULL, K, fd); complete_deferred_experts();

    // Restore S, run linear_block_2 (a->b) + per-token MoE.
    memcpy([g_metal->buf_delta_state[li] contents], Sd, dsz);
    memcpy([g_metal->buf_conv_state[li] contents], Sc, csz);
    float *Hat = malloc(Hd*sizeof(float)), *Hbt = malloc(Hd*sizeof(float));
    memcpy(Hat, Ha, Hd*sizeof(float)); memcpy(Hbt, Hb, Hd*sizeof(float));
    linear_block_2(wf, layer, Hat, Hbt, NULL);
    moe_block_1(wf, layer, Hat, fd);
    moe_block_1(wf, layer, Hbt, fd);

    double ea=0, eb=0, ref=0;
    for (int i = 0; i < Hd; i++) { ea=fmax(ea,fabs(Hao[i]-Hat[i])); eb=fmax(eb,fabs(Hbo[i]-Hbt[i])); ref=fmax(ref,fabs(Hao[i])); }
    double ra = ref>0?ea/ref:ea, rb = ref>0?eb/ref:eb;
    fprintf(stderr, "[mtp-vlin] linear layer=%d li=%d | tokenA rel=%.3e tokenB rel=%.3e | %s\n",
            layer, li, ra, rb, (ra<5e-2 && rb<5e-2) ? "MATCH (linear_block_2 faithful)" : "DIVERGE");

    if (mm!=MAP_FAILED) munmap(mm, st.st_size); if (fd>=0) close(fd);
    linear_attn_state_free(dummy);
    free(h); free(Sd); free(Sc); free(Ha); free(Hb); free(Hao); free(Hbo); free(Hat); free(Hbt);
    return (ra<5e-2 && rb<5e-2) ? 0 : 1;
}

// Reference single-token forward through all layers + norm + lm_head (CPU blocks).
static void ref_forward_1(WeightFile *wf, float *h, KVCache **kv, LinearAttnState **ls,
                          int *fds, int pos, float *logits) {
    int Hd = g_cfg.hidden_dim;
    for (int layer = 0; layer < g_cfg.num_layers; layer++) {
        int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
        if (is_full) full_attention_forward(wf, layer, h, kv[layer], pos);
        else         linear_attention_forward(wf, layer, h, ls[layer]);
        moe_block_1(wf, layer, h, fds[layer]);
    }
    uint16_t *norm_w = get_tensor_ptr(wf, "model.norm.weight");
    float *n = malloc(Hd * sizeof(float));
    cpu_rms_norm(h, norm_w, n, Hd, g_cfg.rms_norm_eps);
    lm_head_forward(wf, n, logits);
    free(n);
}

// Rollback state for a rejected draft: undo token b's contributions, leaving the
// state as if only token a (the accepted token) had been processed.
typedef struct {
    int kv_pre_len[MAX_NUM_LAYERS];  // full-attn: kv->len before token a
    float *conv[MAX_NUM_LAYERS];     // linear: conv_state snapshot after a, before b
    float *ssm[MAX_NUM_LAYERS];      // linear: ssm_state snapshot after a, before b
} VerifyRollback;

static int verify_conv_sz(void) { return (g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim; }
static int verify_ssm_sz(void)  { return g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim; }

static void verify_rollback_init(VerifyRollback *rb) {
    memset(rb, 0, sizeof(*rb));
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (((i + 1) % g_cfg.full_attn_interval) != 0) {
            rb->conv[i] = malloc(verify_conv_sz() * sizeof(float));
            rb->ssm[i]  = malloc(verify_ssm_sz()  * sizeof(float));
        }
    }
}
static void verify_rollback_free(VerifyRollback *rb) {
    for (int i = 0; i < g_cfg.num_layers; i++) { free(rb->conv[i]); free(rb->ssm[i]); }
}
// Undo token b: truncate KV (drop b's appended K/V) and restore linear states.
static void verify_rollback_apply(VerifyRollback *rb, KVCache **kv, LinearAttnState **ls) {
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (((i + 1) % g_cfg.full_attn_interval) == 0) {
            kv[i]->len = rb->kv_pre_len[i] + 1;  // keep a (appended first), drop b
        } else {
            memcpy(ls[i]->conv_state, rb->conv[i], verify_conv_sz() * sizeof(float));
            memcpy(ls[i]->ssm_state,  rb->ssm[i],  verify_ssm_sz()  * sizeof(float));
        }
    }
}

// 4b: full batched 2-position backbone forward. Layer-by-layer so token b sees
// token a's freshly-written KV (full attn) and a-updated recurrent state (linear).
// If rb != NULL, capture per-layer rollback state (post-a) so a rejected draft
// can be undone via verify_rollback_apply.
static void backbone_forward_2(WeightFile *wf, float *ha, float *hb,
                               KVCache **kv, LinearAttnState **ls, int *fds,
                               int pos_a, int pos_b, float *log_a, float *log_b,
                               VerifyRollback *rb) {
    int Hd = g_cfg.hidden_dim;
    for (int layer = 0; layer < g_cfg.num_layers; layer++) {
        int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
        if (is_full) {
            if (rb) rb->kv_pre_len[layer] = kv[layer]->len;
            attn_block_2(wf, layer, ha, hb, kv[layer], pos_a, pos_b);
        } else {
            // Linear recurrence is sequential: a updates the state, then b reads it.
            linear_attention_forward(wf, layer, ha, ls[layer]);
            if (rb) {
                memcpy(rb->conv[layer], ls[layer]->conv_state, verify_conv_sz() * sizeof(float));
                memcpy(rb->ssm[layer],  ls[layer]->ssm_state,  verify_ssm_sz()  * sizeof(float));
            }
            linear_attention_forward(wf, layer, hb, ls[layer]);
        }
        moe_block_2(wf, layer, ha, hb, fds[layer]);
    }
    uint16_t *norm_w = get_tensor_ptr(wf, "model.norm.weight");
    float *na = malloc(Hd * sizeof(float)), *nb = malloc(Hd * sizeof(float));
    cpu_rms_norm(ha, norm_w, na, Hd, g_cfg.rms_norm_eps);
    cpu_rms_norm(hb, norm_w, nb, Hd, g_cfg.rms_norm_eps);
    TensorInfo *wi = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *si = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *bi = get_tensor_info(wf, "lm_head.biases");
    uint32_t *W = (uint32_t *)((char *)wf->data + wi->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + si->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + bi->offset);
    gpu_dequant_matmul2(g_metal, W, S, B, na, nb, log_a, log_b, g_cfg.vocab_size, Hd, g_cfg.group_size);
    free(na); free(nb);
}

// 4b validation: backbone_forward_2 must equal two reference single forwards
// (same CPU blocks, just batched), on fresh caches at positions 0 and 1.
static int mtp_verify_forward(WeightFile *wf, const char *model_path) {
    if (!g_metal || !g_metal->wf_buf) { fprintf(stderr, "[mtp-vfwd] requires GPU\n"); return 1; }
    int Hd = g_cfg.hidden_dim, V = g_cfg.vocab_size, kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    int L = g_cfg.num_layers;

    int *fds = malloc(L * sizeof(int));
    KVCache **kvr = calloc(L, sizeof(KVCache*)), **kvb = calloc(L, sizeof(KVCache*));
    LinearAttnState **lsr = calloc(L, sizeof(LinearAttnState*)), **lsb = calloc(L, sizeof(LinearAttnState*));
    for (int i = 0; i < L; i++) {
        char p[PATH_MAX]; snprintf(p, sizeof(p), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, i);
        fds[i] = open(p, O_RDONLY);
        int is_full = ((i + 1) % g_cfg.full_attn_interval == 0);
        if (is_full) {
            kvr[i] = calloc(1, sizeof(KVCache)); kvr[i]->k_cache = calloc((size_t)64*kv_dim, sizeof(float)); kvr[i]->v_cache = calloc((size_t)64*kv_dim, sizeof(float));
            kvb[i] = calloc(1, sizeof(KVCache)); kvb[i]->k_cache = calloc((size_t)64*kv_dim, sizeof(float)); kvb[i]->v_cache = calloc((size_t)64*kv_dim, sizeof(float));
        } else { lsr[i] = linear_attn_state_new(); lsb[i] = linear_attn_state_new(); }
    }

    int tok_a = 100, tok_b = 200;  // arbitrary real token ids
    float *log_a_r = malloc(V*sizeof(float)), *log_b_r = malloc(V*sizeof(float));
    float *log_a_b = malloc(V*sizeof(float)), *log_b_b = malloc(V*sizeof(float));
    float *ha = malloc(Hd*sizeof(float)), *hb = malloc(Hd*sizeof(float));

    // Reference: two single forwards (a updates state at pos 0, then b at pos 1).
    embed_lookup(wf, tok_a, ha); ref_forward_1(wf, ha, kvr, lsr, fds, 0, log_a_r);
    embed_lookup(wf, tok_b, hb); ref_forward_1(wf, hb, kvr, lsr, fds, 1, log_b_r);

    // Batched: one 2-position forward on fresh caches.
    embed_lookup(wf, tok_a, ha); embed_lookup(wf, tok_b, hb);
    backbone_forward_2(wf, ha, hb, kvb, lsb, fds, 0, 1, log_a_b, log_b_b, NULL);

    double err_a = 0, err_b = 0;
    for (int i = 0; i < V; i++) { err_a = fmax(err_a, fabs(log_a_r[i]-log_a_b[i])); err_b = fmax(err_b, fabs(log_b_r[i]-log_b_b[i])); }
    int am_r = cpu_argmax(log_a_r, V), am_b = cpu_argmax(log_a_b, V);
    int bm_r = cpu_argmax(log_b_r, V), bm_b = cpu_argmax(log_b_b, V);
    fprintf(stderr, "[mtp-vfwd] tokenA argmax ref=%d batched=%d | logit max_err=%.3e\n", am_r, am_b, err_a);
    fprintf(stderr, "[mtp-vfwd] tokenB argmax ref=%d batched=%d | logit max_err=%.3e\n", bm_r, bm_b, err_b);
    int ok = (am_r == am_b) && (bm_r == bm_b) && err_a < 1e-2 && err_b < 1e-2;
    fprintf(stderr, "[mtp-vfwd] %s\n", ok ? "PASS: batched forward == reference (assembly correct)" : "FAIL: batched forward diverges");

    for (int i = 0; i < L; i++) {
        if (kvr[i]) { free(kvr[i]->k_cache); free(kvr[i]->v_cache); free(kvr[i]); }
        if (kvb[i]) { free(kvb[i]->k_cache); free(kvb[i]->v_cache); free(kvb[i]); }
        if (lsr[i]) linear_attn_state_free(lsr[i]);
        if (lsb[i]) linear_attn_state_free(lsb[i]);
        if (fds[i] >= 0) close(fds[i]);
    }
    free(fds); free(kvr); free(kvb); free(lsr); free(lsb);
    free(log_a_r); free(log_b_r); free(log_a_b); free(log_b_b); free(ha); free(hb);
    return ok ? 0 : 1;
}

// 4c-i: validate draft-reject rollback. After a batched verify forward of
// [t_next@P, draft@P+1], rolling back must leave the state identical to having
// processed only t_next@P (a single forward). Otherwise rejects corrupt output.
static int mtp_verify_rollback(WeightFile *wf, const char *model_path) {
    if (!g_metal || !g_metal->wf_buf) { fprintf(stderr, "[mtp-vroll] requires GPU\n"); return 1; }
    int Hd = g_cfg.hidden_dim, V = g_cfg.vocab_size, kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    int Ld = g_cfg.num_layers;
    int *fds = malloc(Ld * sizeof(int));
    KVCache **kvv = calloc(Ld, sizeof(KVCache*)), **kvg = calloc(Ld, sizeof(KVCache*));
    LinearAttnState **lsv = calloc(Ld, sizeof(LinearAttnState*)), **lsg = calloc(Ld, sizeof(LinearAttnState*));
    int cap = 64;
    for (int i = 0; i < Ld; i++) {
        char p[PATH_MAX]; snprintf(p, sizeof(p), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, i);
        fds[i] = open(p, O_RDONLY);
        if (((i + 1) % g_cfg.full_attn_interval) == 0) {
            kvv[i] = calloc(1, sizeof(KVCache)); kvv[i]->k_cache = calloc((size_t)cap*kv_dim, sizeof(float)); kvv[i]->v_cache = calloc((size_t)cap*kv_dim, sizeof(float));
            kvg[i] = calloc(1, sizeof(KVCache)); kvg[i]->k_cache = calloc((size_t)cap*kv_dim, sizeof(float)); kvg[i]->v_cache = calloc((size_t)cap*kv_dim, sizeof(float));
        } else { lsv[i] = linear_attn_state_new(); lsg[i] = linear_attn_state_new(); }
    }
    float *scratch = malloc(V * sizeof(float)), *h = malloc(Hd * sizeof(float));
    int prefix[3] = { 100, 200, 300 }; int P = 3;
    for (int pp = 0; pp < P; pp++) {
        embed_lookup(wf, prefix[pp], h); ref_forward_1(wf, h, kvv, lsv, fds, pp, scratch);
        embed_lookup(wf, prefix[pp], h); ref_forward_1(wf, h, kvg, lsg, fds, pp, scratch);
    }
    int t_next = 400, draft = 500;
    VerifyRollback rb; verify_rollback_init(&rb);
    float *ha = malloc(Hd*sizeof(float)), *hb = malloc(Hd*sizeof(float)), *la = malloc(V*sizeof(float)), *lb = malloc(V*sizeof(float));
    embed_lookup(wf, t_next, ha); embed_lookup(wf, draft, hb);
    backbone_forward_2(wf, ha, hb, kvv, lsv, fds, P, P + 1, la, lb, &rb);
    verify_rollback_apply(&rb, kvv, lsv);
    // Ground truth: single forward of t_next@P.
    embed_lookup(wf, t_next, h); ref_forward_1(wf, h, kvg, lsg, fds, P, scratch);

    double max_err = 0.0; int len_mismatch = 0;
    for (int i = 0; i < Ld; i++) {
        if (((i + 1) % g_cfg.full_attn_interval) == 0) {
            if (kvv[i]->len != kvg[i]->len) len_mismatch++;
            int n = kvg[i]->len * kv_dim;
            for (int j = 0; j < n; j++) {
                max_err = fmax(max_err, fabs(kvv[i]->k_cache[j] - kvg[i]->k_cache[j]));
                max_err = fmax(max_err, fabs(kvv[i]->v_cache[j] - kvg[i]->v_cache[j]));
            }
        } else {
            for (int j = 0; j < verify_conv_sz(); j++) max_err = fmax(max_err, fabs(lsv[i]->conv_state[j] - lsg[i]->conv_state[j]));
            for (int j = 0; j < verify_ssm_sz();  j++) max_err = fmax(max_err, fabs(lsv[i]->ssm_state[j]  - lsg[i]->ssm_state[j]));
        }
    }
    int ok = (len_mismatch == 0) && (max_err < 1e-3);
    fprintf(stderr, "[mtp-vroll] kv len mismatches=%d | state max_err=%.3e | %s\n",
            len_mismatch, max_err, ok ? "PASS: reject rollback restores a-only state" : "FAIL: rollback corrupts state");

    verify_rollback_free(&rb);
    for (int i = 0; i < Ld; i++) {
        if (kvv[i]) { free(kvv[i]->k_cache); free(kvv[i]->v_cache); free(kvv[i]); }
        if (kvg[i]) { free(kvg[i]->k_cache); free(kvg[i]->v_cache); free(kvg[i]); }
        if (lsv[i]) linear_attn_state_free(lsv[i]);
        if (lsg[i]) linear_attn_state_free(lsg[i]);
        if (fds[i] >= 0) close(fds[i]);
    }
    free(fds); free(kvv); free(kvg); free(lsv); free(lsg);
    free(scratch); free(h); free(ha); free(hb); free(la); free(lb);
    return ok ? 0 : 1;
}

#define MTP_IS_EOS(t) ((t) == g_cfg.eos_token_1 || (t) == g_cfg.eos_token_2)

// 4c-ii: the verify/accept decode loop. Runs baseline greedy and MTP-verify on
// the same prompt using the same CPU blocks, confirms identical output (lossless),
// and reports speedup + acceptance. (Absolute tok/s reflects the CPU-orchestrated
// block path, not the production GPU pipeline; the MTP/baseline RATIO is the
// meaningful number.)
static int mtp_generate(WeightFile *wf, const char *model_path, int max_new) {
    if (!g_metal || !g_metal->wf_buf || !g_mtp_cache.ready) {
        fprintf(stderr, "[mtp-gen] requires GPU + MTP weights (run with FLASHCHAT_MTP=1)\n"); return 1;
    }
    int Hd = g_cfg.hidden_dim, V = g_cfg.vocab_size, kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim, Ld = g_cfg.num_layers;
    int *fds = malloc(Ld * sizeof(int));
    KVCache **kv = calloc(Ld, sizeof(KVCache*));
    LinearAttnState **ls = calloc(Ld, sizeof(LinearAttnState*));
    int cap = 8192;
    for (int i = 0; i < Ld; i++) {
        char p[PATH_MAX]; snprintf(p, sizeof(p), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, i);
        fds[i] = open(p, O_RDONLY);
        if (((i + 1) % g_cfg.full_attn_interval) == 0) {
            kv[i] = calloc(1, sizeof(KVCache));
            kv[i]->k_cache = calloc((size_t)cap*kv_dim, sizeof(float));
            kv[i]->v_cache = calloc((size_t)cap*kv_dim, sizeof(float));
        } else ls[i] = linear_attn_state_new();
    }
    #define RESET_STATE() do { for (int i=0;i<Ld;i++){ if(kv[i]) kv[i]->len=0; if(ls[i]){ memset(ls[i]->conv_state,0,verify_conv_sz()*sizeof(float)); memset(ls[i]->ssm_state,0,verify_ssm_sz()*sizeof(float)); } } } while(0)

    int prompt[] = { 9707, 11, 358, 1079, 4378, 264, 13027, 729, 311 };  // arbitrary valid ids
    int np = (int)(sizeof(prompt) / sizeof(prompt[0]));
    float *h = malloc(Hd*sizeof(float)), *logits = malloc(V*sizeof(float));
    int *base_tok = malloc(max_new*sizeof(int)), *mtp_tok = malloc(max_new*sizeof(int));

    // ---- Baseline greedy (sequential single forwards) ----
    RESET_STATE();
    for (int p = 0; p < np; p++) { embed_lookup(wf, prompt[p], h); ref_forward_1(wf, h, kv, ls, fds, p, logits); }
    int tok = cpu_argmax(logits, V), pos = np, bn = 0;
    double t0 = now_ms();
    while (bn < max_new) {
        base_tok[bn++] = tok; if (MTP_IS_EOS(tok)) break;
        embed_lookup(wf, tok, h); ref_forward_1(wf, h, kv, ls, fds, pos, logits); pos++; tok = cpu_argmax(logits, V);
    }
    double base_ms = now_ms() - t0;

    // ---- MTP verify/accept ----
    RESET_STATE(); g_mtp_kv_len = 0;
    for (int p = 0; p < np; p++) { embed_lookup(wf, prompt[p], h); ref_forward_1(wf, h, kv, ls, fds, p, logits); }
    int t_next = cpu_argmax(logits, V);
    float *hcur = malloc(Hd*sizeof(float)); memcpy(hcur, h, Hd*sizeof(float));
    pos = np;
    VerifyRollback rb; verify_rollback_init(&rb);
    float *ha = malloc(Hd*sizeof(float)), *hb = malloc(Hd*sizeof(float)), *la = malloc(V*sizeof(float)), *lb = malloc(V*sizeof(float));
    int mn = 0, accepts = 0, checks = 0, fwd_calls = 0;
    double t1 = now_ms();
    while (mn < max_new) {
        if (MTP_IS_EOS(t_next)) { mtp_tok[mn++] = t_next; break; }
        int d = -1; mtp_shadow_draft_token(wf, model_path, hcur, t_next, &d);
        if (d < 0) {  // no draft: plain single step
            mtp_tok[mn++] = t_next;
            embed_lookup(wf, t_next, ha); ref_forward_1(wf, ha, kv, ls, fds, pos, la); fwd_calls++;
            memcpy(hcur, ha, Hd*sizeof(float)); pos++; t_next = cpu_argmax(la, V);
            continue;
        }
        embed_lookup(wf, t_next, ha); embed_lookup(wf, d, hb);
        backbone_forward_2(wf, ha, hb, kv, ls, fds, pos, pos+1, la, lb, &rb); fwd_calls++;
        int t_v1 = cpu_argmax(la, V);
        mtp_tok[mn++] = t_next; checks++;
        if (t_v1 == d) {  // accept draft
            accepts++;
            if (mn < max_new) mtp_tok[mn++] = d;
            // Advance the MTP KV for the accepted draft's position (input h(pos), token d)
            // so the MTP attention context has no gap. Each committed token must be
            // appended to the MTP KV exactly once, in order.
            { int dummy = -1; mtp_shadow_draft_token(wf, model_path, ha, d, &dummy); }
            memcpy(hcur, hb, Hd*sizeof(float)); t_next = cpu_argmax(lb, V); pos += 2;
        } else {          // reject: undo b, emit verified token next
            verify_rollback_apply(&rb, kv, ls);
            memcpy(hcur, ha, Hd*sizeof(float)); t_next = t_v1; pos += 1;
        }
    }
    double mtp_ms = now_ms() - t1;

    int n = bn < mn ? bn : mn, mism = 0;
    for (int i = 0; i < n; i++) if (base_tok[i] != mtp_tok[i]) { mism++; if (mism == 1) fprintf(stderr, "[mtp-gen] first mismatch at %d: base=%d mtp=%d\n", i, base_tok[i], mtp_tok[i]); }
    double acc = checks ? 100.0*accepts/checks : 0.0;
    fprintf(stderr, "[mtp-gen] baseline: %d tok in %.0f ms (%.2f tok/s)\n", bn, base_ms, 1000.0*bn/base_ms);
    fprintf(stderr, "[mtp-gen] mtp:      %d tok in %.0f ms (%.2f tok/s) | %d forwards | acceptance=%.1f%% (%d/%d)\n",
            mn, mtp_ms, 1000.0*mn/mtp_ms, fwd_calls, acc, accepts, checks);
    fprintf(stderr, "[mtp-gen] speedup=%.2fx | tokens/forward: base=1.00 mtp=%.2f\n",
            mtp_ms > 0 ? (base_ms/bn) / (mtp_ms/mn) : 0.0, (double)mn/fwd_calls);
    fprintf(stderr, "[mtp-gen] lossless check: %d/%d tokens match baseline | %s\n",
            n - mism, n, mism == 0 ? "PASS: identical output" : "FAIL: output diverges");

    verify_rollback_free(&rb);
    for (int i = 0; i < Ld; i++) { if (kv[i]){ free(kv[i]->k_cache); free(kv[i]->v_cache); free(kv[i]); } if (ls[i]) linear_attn_state_free(ls[i]); if (fds[i]>=0) close(fds[i]); }
    free(fds); free(kv); free(ls); free(h); free(logits); free(base_tok); free(mtp_tok); free(hcur); free(ha); free(hb); free(la); free(lb);
    #undef RESET_STATE
    return mism == 0 ? 0 : 1;
}

// Production single-token forward (reuses fused_layer_forward, faithful). h holds
// the token embedding on entry; on return h is the pre-norm hidden, logits filled.
static void prod_forward_1(WeightFile *wf, float *h, int pos, KVCache **kv,
                           LinearAttnState **ls, int *fds, void **mmaps, int K, float *logits) {
    for (int layer = 0; layer < g_cfg.num_layers; layer++) {
        int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
        fused_layer_forward(wf, layer, h, is_full ? kv[layer] : NULL, is_full ? NULL : ls[layer],
                            pos, mmaps[layer], K, fds[layer]);
    }
    complete_deferred_experts();
    uint16_t *nw = get_tensor_ptr(wf, "model.norm.weight");
    int Hd = g_cfg.hidden_dim;
    float *n = malloc(Hd * sizeof(float));
    cpu_rms_norm(h, nw, n, Hd, g_cfg.rms_norm_eps);
    lm_head_forward(wf, n, logits);
    free(n);
}

// GpuStateSnap typedef'd earlier (before linear_block_2).
static void gpu_snap_alloc(GpuStateSnap *s) {
    s->delta = calloc(g_cfg.num_linear_layers, sizeof(float*));
    s->conv  = calloc(g_cfg.num_linear_layers, sizeof(float*));
    s->kvlen = calloc(g_cfg.num_layers, sizeof(int));
    for (int i = 0; i < g_cfg.num_linear_layers; i++) {
        s->delta[i] = malloc(64*128*128 * sizeof(float));
        s->conv[i]  = malloc(3*12288 * sizeof(float));
    }
}
static void gpu_snap_free(GpuStateSnap *s) {
    for (int i = 0; i < g_cfg.num_linear_layers; i++) { free(s->delta[i]); free(s->conv[i]); }
    free(s->delta); free(s->conv); free(s->kvlen);
}
static int linear_idx_of(int layer) { return layer - (layer + 1) / g_cfg.full_attn_interval; }
static void gpu_snap_save(GpuStateSnap *s, KVCache **kv) {
    for (int layer = 0; layer < g_cfg.num_layers; layer++) {
        if (((layer + 1) % g_cfg.full_attn_interval) == 0) { s->kvlen[layer] = kv[layer]->len; }
        else { int li = linear_idx_of(layer);
            memcpy(s->delta[li], [g_metal->buf_delta_state[li] contents], 64*128*128*sizeof(float));
            memcpy(s->conv[li],  [g_metal->buf_conv_state[li] contents],  3*12288*sizeof(float)); }
    }
}
static void gpu_snap_restore(GpuStateSnap *s, KVCache **kv) {
    for (int layer = 0; layer < g_cfg.num_layers; layer++) {
        if (((layer + 1) % g_cfg.full_attn_interval) == 0) { kv[layer]->len = s->kvlen[layer]; }
        else { int li = linear_idx_of(layer);
            memcpy([g_metal->buf_delta_state[li] contents], s->delta[li], 64*128*128*sizeof(float));
            memcpy([g_metal->buf_conv_state[li] contents],  s->conv[li],  3*12288*sizeof(float)); }
    }
}
// Reject-rollback: restore state to token-a-only. Full layers: kvlen captured
// PRE-token-a, so a-only len = kvlen+1 (keep a, drop b). Linear: delta/conv were
// captured POST-token-a (in linear_block_2), so restore them directly.
static void batched_rollback_apply(GpuStateSnap *s, KVCache **kv) {
    for (int layer = 0; layer < g_cfg.num_layers; layer++) {
        if (((layer + 1) % g_cfg.full_attn_interval) == 0) { kv[layer]->len = s->kvlen[layer] + 1; }
        else { int li = linear_idx_of(layer);
            memcpy([g_metal->buf_delta_state[li] contents], s->delta[li], 64*128*128*sizeof(float));
            memcpy([g_metal->buf_conv_state[li] contents],  s->conv[li],  3*12288*sizeof(float)); }
    }
}

// 4d-ii: the amortized batched 2-position forward on faithful GPU primitives.
// full-attn = attn_block_2, linear = linear_block_2 (GPU delta-net), MoE = moe_block_2.
// All weight-bound matmuls are batched (matmul2); operates on production GPU state.
static void fused_layer_forward_2(WeightFile *wf, float *ha, float *hb,
                                  KVCache **kv, int *fds, int pos_a, int pos_b,
                                  float *log_a, float *log_b, GpuStateSnap *rb) {
    int Hd = g_cfg.hidden_dim;
    for (int layer = 0; layer < g_cfg.num_layers; layer++) {
        if (((layer + 1) % g_cfg.full_attn_interval) == 0) {
            if (rb) rb->kvlen[layer] = kv[layer]->len;  // pre-token-a KV len (rollback)
            attn_block_2(wf, layer, ha, hb, kv[layer], pos_a, pos_b);
        } else {
            linear_block_2(wf, layer, ha, hb, rb);      // captures post-a delta/conv if rb
        }
        if (g_cfg.num_experts == 0)
            dense_mlp_block_2(wf, layer, ha, hb);
        else
            moe_block_2(wf, layer, ha, hb, fds[layer]);
    }
    uint16_t *norm_w = get_tensor_ptr(wf, "model.norm.weight");
    float *na = malloc(Hd*sizeof(float)), *nb = malloc(Hd*sizeof(float));
    cpu_rms_norm(ha, norm_w, na, Hd, g_cfg.rms_norm_eps);
    cpu_rms_norm(hb, norm_w, nb, Hd, g_cfg.rms_norm_eps);
    TensorInfo *wi=get_tensor_info(wf,"lm_head.weight"), *si=get_tensor_info(wf,"lm_head.scales"), *bi=get_tensor_info(wf,"lm_head.biases");
    gpu_dequant_matmul2(g_metal, (uint32_t*)((char*)wf->data+wi->offset), (uint16_t*)((char*)wf->data+si->offset),
                        (uint16_t*)((char*)wf->data+bi->offset), na, nb, log_a, log_b, g_cfg.vocab_size, Hd, g_cfg.group_size);
    free(na); free(nb);
}

// 4d-ii validation: fused_layer_forward_2 must match two sequential production
// forwards (prod_forward_1) over ALL layers at decode positions — the end-to-end
// faithfulness check that catches fp compounding across 40 layers.
static int mtp_verify_forward2(WeightFile *wf, const char *model_path) {
    if (!g_metal || !g_metal->wf_buf || !g_metal->delta_net_step) { fprintf(stderr, "[mtp-vf2] requires GPU\n"); return 1; }
    int Hd = g_cfg.hidden_dim, V = g_cfg.vocab_size, kv_dim = g_cfg.num_kv_heads*g_cfg.head_dim, Ld = g_cfg.num_layers, K = g_cfg.num_experts_per_tok;
    int *fds = malloc(Ld*sizeof(int)); void **mmaps = calloc(Ld,sizeof(void*));
    KVCache **kv = calloc(Ld,sizeof(KVCache*)); LinearAttnState **ls = calloc(Ld,sizeof(LinearAttnState*));
    int cap = 256;
    for (int i=0;i<Ld;i++){ char p[PATH_MAX]; snprintf(p,sizeof(p),"%s/flashchat/packed_experts/layer_%02d.bin",model_path,i); fds[i]=open(p,O_RDONLY);
        if(fds[i]>=0){ struct stat st; if(fstat(fds[i],&st)==0&&st.st_size>0){void*m=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fds[i],0); if(m!=MAP_FAILED) mmaps[i]=m;} }
        if(((i+1)%g_cfg.full_attn_interval)==0){ kv[i]=calloc(1,sizeof(KVCache)); kv[i]->k_cache=calloc((size_t)cap*kv_dim,sizeof(float)); kv[i]->v_cache=calloc((size_t)cap*kv_dim,sizeof(float)); }
        else ls[i]=linear_attn_state_new(); }
    float *h=malloc(Hd*sizeof(float)), *scratch=malloc(V*sizeof(float));
    int P=35;
    reset_delta_net_state(); for(int i=0;i<Ld;i++) if(kv[i]) kv[i]->len=0;
    for(int p=0;p<P;p++){ for(int i=0;i<Hd;i++) h[i]=sinf((i+p*5)*0.01f)*0.4f; embed_lookup(wf,100+p,h); prod_forward_1(wf,h,p,kv,ls,fds,mmaps,K,scratch); }
    GpuStateSnap snap; gpu_snap_alloc(&snap); gpu_snap_save(&snap, kv);

    int ta=400, tb=500;
    float *Hao=malloc(Hd*sizeof(float)),*Hbo=malloc(Hd*sizeof(float)),*la_o=malloc(V*sizeof(float)),*lb_o=malloc(V*sizeof(float));
    embed_lookup(wf,ta,Hao); prod_forward_1(wf,Hao,P,kv,ls,fds,mmaps,K,la_o);
    embed_lookup(wf,tb,Hbo); prod_forward_1(wf,Hbo,P+1,kv,ls,fds,mmaps,K,lb_o);

    gpu_snap_restore(&snap, kv);
    float *Hat=malloc(Hd*sizeof(float)),*Hbt=malloc(Hd*sizeof(float)),*la_t=malloc(V*sizeof(float)),*lb_t=malloc(V*sizeof(float));
    embed_lookup(wf,ta,Hat); embed_lookup(wf,tb,Hbt);
    fused_layer_forward_2(wf,Hat,Hbt,kv,fds,P,P+1,la_t,lb_t,NULL);

    double ea=0,eb=0,ref=0; for(int i=0;i<V;i++){ ea=fmax(ea,fabs(la_o[i]-la_t[i])); eb=fmax(eb,fabs(lb_o[i]-lb_t[i])); ref=fmax(ref,fabs(la_o[i])); }
    int aa_o=cpu_argmax(la_o,V),aa_t=cpu_argmax(la_t,V),bb_o=cpu_argmax(lb_o,V),bb_t=cpu_argmax(lb_t,V);
    double ra=ref>0?ea/ref:ea, rb=ref>0?eb/ref:eb;
    fprintf(stderr,"[mtp-vf2] tokenA argmax prod=%d batched=%d rel=%.3e | tokenB argmax prod=%d batched=%d rel=%.3e\n", aa_o,aa_t,ra,bb_o,bb_t,rb);
    int ok = (aa_o==aa_t)&&(bb_o==bb_t);
    fprintf(stderr,"[mtp-vf2] %s (argmax match is what matters for lossless verify; rel err is fp drift over 40 layers)\n", ok?"PASS: batched forward argmax matches production":"DIVERGE: argmax differs");

    // ---- Amortization timing: batched 2-position forward vs two single forwards ----
    // (both warm now). This is the per-verify cost ratio that sets the speedup.
    const int MT = 3;
    double t = now_ms();
    for (int m = 0; m < MT; m++) {
        gpu_snap_restore(&snap, kv);
        embed_lookup(wf, ta, Hao); prod_forward_1(wf, Hao, P, kv, ls, fds, mmaps, K, la_o);
        embed_lookup(wf, tb, Hbo); prod_forward_1(wf, Hbo, P+1, kv, ls, fds, mmaps, K, lb_o);
    }
    double seq2 = (now_ms() - t) / MT;     // cost of 2 single forwards (2 tokens)
    t = now_ms();
    for (int m = 0; m < MT; m++) {
        gpu_snap_restore(&snap, kv);
        embed_lookup(wf, ta, Hat); embed_lookup(wf, tb, Hbt);
        fused_layer_forward_2(wf, Hat, Hbt, kv, fds, P, P+1, la_t, lb_t, NULL);
    }
    double batched2 = (now_ms() - t) / MT;  // cost of 1 batched forward (2 positions)
    double cost_ratio = batched2 / seq2;
    fprintf(stderr, "[mtp-vf2] timing: 2x single=%.1f ms | batched2=%.1f ms | cost_ratio=%.3f\n", seq2, batched2, cost_ratio);
    for (double A = 0.55; A <= 0.85; A += 0.15) {
        double speedup = (1.0 + A) * seq2 / (2.0 * batched2);  // (1+A) tokens per batched forward
        fprintf(stderr, "[mtp-vf2] projected speedup @ acceptance %.0f%% = %.2fx\n", A*100, speedup);
    }

    gpu_snap_free(&snap);
    for(int i=0;i<Ld;i++){ if(kv[i]){free(kv[i]->k_cache);free(kv[i]->v_cache);free(kv[i]);} if(ls[i]) linear_attn_state_free(ls[i]); if(mmaps[i]){struct stat st; fstat(fds[i],&st); munmap(mmaps[i],st.st_size);} if(fds[i]>=0) close(fds[i]); }
    free(fds); free(mmaps); free(kv); free(ls); free(h); free(scratch);
    free(Hao); free(Hbo); free(la_o); free(lb_o); free(Hat); free(Hbt); free(la_t); free(lb_t);
    return ok?0:1;
}

// 4d-i: verify/accept loop on the PRODUCTION forward (faithful hiddens), verify
// done as two sequential production forwards (no amortization yet — validates
// acceptance recovery, GPU-state rollback, and losslessness before the batched
// kernels add speedup in 4d-ii).
static int mtp_generate_gpu(WeightFile *wf, const char *model_path, int max_new) {
    if (!g_metal || !g_metal->wf_buf || !g_mtp_cache.ready) { fprintf(stderr, "[mtp-gpu] requires GPU + MTP\n"); return 1; }
    int Hd = g_cfg.hidden_dim, V = g_cfg.vocab_size, kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim, Ld = g_cfg.num_layers, K = g_cfg.num_experts_per_tok;
    int *fds = malloc(Ld*sizeof(int)); void **mmaps = calloc(Ld, sizeof(void*));
    KVCache **kv = calloc(Ld, sizeof(KVCache*)); LinearAttnState **ls = calloc(Ld, sizeof(LinearAttnState*));
    for (int i = 0; i < Ld; i++) {
        char p[PATH_MAX]; snprintf(p, sizeof(p), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, i);
        fds[i] = open(p, O_RDONLY); mmaps[i] = NULL;
        if (fds[i] >= 0) { struct stat st; if (fstat(fds[i], &st)==0 && st.st_size>0){ void *m=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fds[i],0); if(m!=MAP_FAILED) mmaps[i]=m; } }
        if (((i + 1) % g_cfg.full_attn_interval) == 0) { kv[i]=calloc(1,sizeof(KVCache)); kv[i]->k_cache=calloc((size_t)GPU_KV_SEQ*kv_dim,sizeof(float)); kv[i]->v_cache=calloc((size_t)GPU_KV_SEQ*kv_dim,sizeof(float)); }
        else ls[i] = linear_attn_state_new();
    }
    int prompt[] = { 9707, 11, 358, 1079, 4378, 264, 13027, 729, 311 }; int np = (int)(sizeof(prompt)/sizeof(prompt[0]));
    float *h = malloc(Hd*sizeof(float)), *logits = malloc(V*sizeof(float));
    int *base_tok = malloc(max_new*sizeof(int)), *mtp_tok = malloc(max_new*sizeof(int));

    // baseline greedy
    reset_delta_net_state(); for (int i=0;i<Ld;i++) if(kv[i]) kv[i]->len=0;
    for (int pp=0; pp<np; pp++){ embed_lookup(wf, prompt[pp], h); prod_forward_1(wf,h,pp,kv,ls,fds,mmaps,K,logits); }
    int tok = cpu_argmax(logits,V), pos=np, bn=0; double t0=now_ms();
    while (bn<max_new){ base_tok[bn++]=tok; if(MTP_IS_EOS(tok))break; embed_lookup(wf,tok,h); prod_forward_1(wf,h,pos,kv,ls,fds,mmaps,K,logits); pos++; tok=cpu_argmax(logits,V);}
    double base_ms = now_ms()-t0;

    // MTP verify (sequential faithful forwards + GPU rollback)
    reset_delta_net_state(); for (int i=0;i<Ld;i++) if(kv[i]) kv[i]->len=0; g_mtp_kv_len=0;
    for (int pp=0; pp<np; pp++){ embed_lookup(wf, prompt[pp], h); prod_forward_1(wf,h,pp,kv,ls,fds,mmaps,K,logits); }
    int t_next=cpu_argmax(logits,V); float *hcur=malloc(Hd*sizeof(float)); memcpy(hcur,h,Hd*sizeof(float)); pos=np;
    GpuStateSnap snap; gpu_snap_alloc(&snap);
    float *ha=malloc(Hd*sizeof(float)), *hb=malloc(Hd*sizeof(float)), *la=malloc(V*sizeof(float)), *lb=malloc(V*sizeof(float));
    int mn=0, accepts=0, checks=0; double t1=now_ms();
    while (mn<max_new){
        if (MTP_IS_EOS(t_next)){ mtp_tok[mn++]=t_next; break; }
        int d=-1; mtp_shadow_draft_token(wf, model_path, hcur, t_next, &d);
        if (d<0){ mtp_tok[mn++]=t_next; embed_lookup(wf,t_next,ha); prod_forward_1(wf,ha,pos,kv,ls,fds,mmaps,K,la); memcpy(hcur,ha,Hd*sizeof(float)); pos++; t_next=cpu_argmax(la,V); continue; }
        embed_lookup(wf,t_next,ha); prod_forward_1(wf,ha,pos,kv,ls,fds,mmaps,K,la);
        gpu_snap_save(&snap, kv);
        int t_v1=cpu_argmax(la,V); mtp_tok[mn++]=t_next; checks++;
        embed_lookup(wf,d,hb); prod_forward_1(wf,hb,pos+1,kv,ls,fds,mmaps,K,lb);
        if (t_v1==d){ accepts++; if(mn<max_new) mtp_tok[mn++]=d; { int dm=-1; mtp_shadow_draft_token(wf,model_path,ha,d,&dm); } memcpy(hcur,hb,Hd*sizeof(float)); t_next=cpu_argmax(lb,V); pos+=2; }
        else { gpu_snap_restore(&snap, kv); memcpy(hcur,ha,Hd*sizeof(float)); t_next=t_v1; pos+=1; }
    }
    double mtp_ms=now_ms()-t1;
    int n = bn<mn?bn:mn, mism=0;
    for (int i=0;i<n;i++) if(base_tok[i]!=mtp_tok[i]){ mism++; if(mism==1) fprintf(stderr,"[mtp-gpu] first mismatch at %d: base=%d mtp=%d\n",i,base_tok[i],mtp_tok[i]); }
    double accpct = checks?100.0*accepts/checks:0.0;
    fprintf(stderr,"[mtp-gpu] baseline: %d tok %.0f ms (%.2f tok/s)\n", bn, base_ms, 1000.0*bn/base_ms);
    fprintf(stderr,"[mtp-gpu] mtp(seq verify): %d tok %.0f ms (%.2f tok/s) | acceptance=%.1f%% (%d/%d)\n", mn, mtp_ms, 1000.0*mn/mtp_ms, accpct, accepts, checks);
    fprintf(stderr,"[mtp-gpu] lossless: %d/%d match | %s\n", n-mism, n, mism==0?"PASS":"FAIL");
    fprintf(stderr,"[mtp-gpu] NOTE: verify is sequential (2 forwards/step) so no speedup yet; validates acceptance + GPU rollback + losslessness\n");

    gpu_snap_free(&snap);
    for (int i=0;i<Ld;i++){ if(kv[i]){free(kv[i]->k_cache);free(kv[i]->v_cache);free(kv[i]);} if(ls[i]) linear_attn_state_free(ls[i]); if(mmaps[i]) { struct stat st; fstat(fds[i],&st); munmap(mmaps[i], st.st_size);} if(fds[i]>=0) close(fds[i]); }
    free(fds); free(mmaps); free(kv); free(ls); free(h); free(logits); free(base_tok); free(mtp_tok); free(hcur); free(ha); free(hb); free(la); free(lb);
    return mism==0 ? 0 : 1;
}

// Dense MTP draft: produce one draft token from the dense MTP head given the
// backbone hidden (hcur) and the accepted token. Context-free self-attention
// (single token, copy-V — like the 35B which still hit 55-83%). Dense MLP head.
static int dense_mtp_draft(WeightFile *wf, const float *hcur, int t_next) {
    int H = g_cfg.hidden_dim, V = g_cfg.vocab_size, inter = g_cfg.dense_intermediate, gsz = g_cfg.group_size;
    int qpd = g_cfg.num_attn_heads * g_cfg.head_dim * 2, qd = g_cfg.num_attn_heads * g_cfg.head_dim;
    int kvd = g_cfg.num_kv_heads * g_cfg.head_dim, hpk = g_cfg.num_attn_heads / g_cfg.num_kv_heads;
    #define MG(n) get_tensor_ptr(wf, "mtp." n)
    uint16_t *pfh=MG("pre_fc_norm_hidden.weight"), *pfe=MG("pre_fc_norm_embedding.weight"), *mnorm=MG("norm.weight");
    uint32_t *fcw=MG("fc.weight"); uint16_t *fcs=MG("fc.scales"),*fcb=MG("fc.biases");
    uint16_t *iln=MG("layers.0.input_layernorm.weight"), *paln=MG("layers.0.post_attention_layernorm.weight");
    uint16_t *qn=MG("layers.0.self_attn.q_norm.weight"), *kn=MG("layers.0.self_attn.k_norm.weight");
    uint32_t *qw=MG("layers.0.self_attn.q_proj.weight"); uint16_t *qs=MG("layers.0.self_attn.q_proj.scales"),*qb=MG("layers.0.self_attn.q_proj.biases");
    uint32_t *kw=MG("layers.0.self_attn.k_proj.weight"); uint16_t *ks=MG("layers.0.self_attn.k_proj.scales"),*kb=MG("layers.0.self_attn.k_proj.biases");
    uint32_t *vw=MG("layers.0.self_attn.v_proj.weight"); uint16_t *vs=MG("layers.0.self_attn.v_proj.scales"),*vb=MG("layers.0.self_attn.v_proj.biases");
    uint32_t *ow=MG("layers.0.self_attn.o_proj.weight"); uint16_t *os=MG("layers.0.self_attn.o_proj.scales"),*ob=MG("layers.0.self_attn.o_proj.biases");
    uint32_t *mgw=MG("layers.0.mlp.gate_proj.weight"); uint16_t *mgs=MG("layers.0.mlp.gate_proj.scales"),*mgb=MG("layers.0.mlp.gate_proj.biases");
    uint32_t *muw=MG("layers.0.mlp.up_proj.weight");   uint16_t *mus=MG("layers.0.mlp.up_proj.scales"),  *mub=MG("layers.0.mlp.up_proj.biases");
    uint32_t *mdw=MG("layers.0.mlp.down_proj.weight"); uint16_t *mds=MG("layers.0.mlp.down_proj.scales"),*mdb=MG("layers.0.mlp.down_proj.biases");
    #undef MG
    if (!fcw || !mgw) return -1;
    float eps = g_cfg.rms_norm_eps;
    float *emb=malloc(H*4),*hn=malloc(H*4),*en=malloc(H*4),*fc_in=malloc(2*H*4),*x=malloc(H*4),*res=malloc(H*4),*normed=malloc(H*4);
    embed_lookup(wf, t_next, emb);
    cpu_rms_norm(hcur, pfh, hn, H, eps); cpu_rms_norm(emb, pfe, en, H, eps);
    memcpy(fc_in, en, H*4); memcpy(fc_in+H, hn, H*4);
    fast_dequant_matvec(fcw,fcs,fcb, fc_in, x, H, 2*H, gsz);
    memcpy(res, x, H*4);
    cpu_rms_norm(x, iln, normed, H, eps);
    float *qp=malloc(qpd*4),*k=malloc(kvd*4),*v=malloc(kvd*4),*q=malloc(qd*4),*qg=malloc(qd*4),*ao=calloc(qd,4),*ap=malloc(H*4);
    fast_dequant_matvec(qw,qs,qb,normed,qp,qpd,H,gsz);
    fast_dequant_matvec(kw,ks,kb,normed,k,kvd,H,gsz);
    fast_dequant_matvec(vw,vs,vb,normed,v,kvd,H,gsz);
    for (int h=0;h<g_cfg.num_attn_heads;h++){ float*s=qp+h*2*g_cfg.head_dim; memcpy(q+h*g_cfg.head_dim,s,g_cfg.head_dim*4); memcpy(qg+h*g_cfg.head_dim,s+g_cfg.head_dim,g_cfg.head_dim*4); }
    for (int h=0;h<g_cfg.num_attn_heads;h++){ float*qh=q+h*g_cfg.head_dim,ss=0; for(int i=0;i<g_cfg.head_dim;i++)ss+=qh[i]*qh[i]; float inv=1.f/sqrtf(ss/g_cfg.head_dim+eps); for(int i=0;i<g_cfg.head_dim;i++)qh[i]=qh[i]*inv*bf16_to_f32(qn[i]); }
    for (int h=0;h<g_cfg.num_kv_heads;h++){ float*kh=k+h*g_cfg.head_dim,ss=0; for(int i=0;i<g_cfg.head_dim;i++)ss+=kh[i]*kh[i]; float inv=1.f/sqrtf(ss/g_cfg.head_dim+eps); for(int i=0;i<g_cfg.head_dim;i++)kh[i]=kh[i]*inv*bf16_to_f32(kn[i]); }
    apply_rotary_emb(q, k, 0, g_cfg.num_attn_heads, g_cfg.num_kv_heads, g_cfg.head_dim, g_cfg.rotary_dim);
    for (int h=0;h<g_cfg.num_attn_heads;h++){ int kvh=h/hpk; memcpy(ao+h*g_cfg.head_dim, v+kvh*g_cfg.head_dim, g_cfg.head_dim*4); }
    for (int i=0;i<qd;i++) ao[i]*=cpu_sigmoid(qg[i]);
    fast_dequant_matvec(ow,os,ob, ao, ap, H, qd, gsz);
    for (int i=0;i<H;i++) x[i]=res[i]+ap[i];
    float *hp=malloc(H*4),*g=malloc(inter*4),*u=malloc(inter*4),*a=malloc(inter*4),*o=malloc(H*4);
    cpu_rms_norm(x, paln, hp, H, eps);
    fast_dequant_matvec(mgw,mgs,mgb, hp, g, inter, H, gsz);
    fast_dequant_matvec(muw,mus,mub, hp, u, inter, H, gsz);
    cpu_swiglu(g,u,a,inter);
    fast_dequant_matvec(mdw,mds,mdb, a, o, H, inter, gsz);
    for (int i=0;i<H;i++) x[i]+=o[i];
    float *fn=malloc(H*4),*logits=malloc(V*4);
    cpu_rms_norm(x, mnorm, fn, H, eps);
    lm_head_forward(wf, fn, logits);
    int draft = cpu_argmax(logits, V);
    free(emb);free(hn);free(en);free(fc_in);free(x);free(res);free(normed);
    free(qp);free(k);free(v);free(q);free(qg);free(ao);free(ap);
    free(hp);free(g);free(u);free(a);free(o);free(fn);free(logits);
    return draft;
}

// THE FINISH LINE: dense MTP speculative decode end-to-end. Baseline greedy vs
// MTP draft/verify (batched 2-position forward), lossless check + real tok/s.
// Reject uses a single re-forward to advance state (simple+correct; proper
// post-a rollback is a later optimization).
static int mtp_generate_dense(WeightFile *wf, const char *model_path, int max_new) {
    if (!g_metal || !g_metal->wf_buf || g_cfg.num_experts != 0) { fprintf(stderr, "[mtp-dense] requires dense GPU model\n"); return 1; }
    (void)model_path;
    int Hd=g_cfg.hidden_dim, V=g_cfg.vocab_size, kv_dim=g_cfg.num_kv_heads*g_cfg.head_dim, Ld=g_cfg.num_layers, K=4;
    int *fds=malloc(Ld*sizeof(int)); void **mmaps=calloc(Ld,sizeof(void*));
    KVCache **kv=calloc(Ld,sizeof(KVCache*)); LinearAttnState **ls=calloc(Ld,sizeof(LinearAttnState*));
    int cap=8192;
    for(int i=0;i<Ld;i++){ fds[i]=-1; if(((i+1)%g_cfg.full_attn_interval)==0){ kv[i]=calloc(1,sizeof(KVCache)); kv[i]->k_cache=calloc((size_t)cap*kv_dim,4); kv[i]->v_cache=calloc((size_t)cap*kv_dim,4);} else ls[i]=linear_attn_state_new(); }
    int prompt[]={9707,11,358,1079,4378,264,13027,729,311}; int np=(int)(sizeof(prompt)/sizeof(prompt[0]));
    float *h=malloc(Hd*4),*logits=malloc(V*4); int *base_tok=malloc(max_new*4),*mtp_tok=malloc(max_new*4);
    #define IS_EOS(t) ((t)==g_cfg.eos_token_1||(t)==g_cfg.eos_token_2)

    // baseline
    reset_delta_net_state(); for(int i=0;i<Ld;i++) if(kv[i]) kv[i]->len=0;
    for(int p=0;p<np;p++){ embed_lookup(wf,prompt[p],h); prod_forward_1(wf,h,p,kv,ls,fds,mmaps,K,logits);}
    int tok=cpu_argmax(logits,V),pos=np,bn=0; double t0=now_ms();
    while(bn<max_new){ base_tok[bn++]=tok; if(IS_EOS(tok))break; embed_lookup(wf,tok,h); prod_forward_1(wf,h,pos,kv,ls,fds,mmaps,K,logits); pos++; tok=cpu_argmax(logits,V);}
    double base_ms=now_ms()-t0;

    // MTP dense verify
    reset_delta_net_state(); for(int i=0;i<Ld;i++) if(kv[i]) kv[i]->len=0;
    for(int p=0;p<np;p++){ embed_lookup(wf,prompt[p],h); prod_forward_1(wf,h,p,kv,ls,fds,mmaps,K,logits);}
    int t_next=cpu_argmax(logits,V); float *hcur=malloc(Hd*4); memcpy(hcur,h,Hd*4); pos=np;
    GpuStateSnap snap; gpu_snap_alloc(&snap);
    float *ha=malloc(Hd*4),*hb=malloc(Hd*4),*la=malloc(V*4),*lb=malloc(V*4);
    int mn=0,accepts=0,checks=0; double t1=now_ms();
    while(mn<max_new){
        if(IS_EOS(t_next)){ mtp_tok[mn++]=t_next; break; }
        int d=dense_mtp_draft(wf,hcur,t_next);
        if(d<0){ mtp_tok[mn++]=t_next; embed_lookup(wf,t_next,ha); prod_forward_1(wf,ha,pos,kv,ls,fds,mmaps,K,la); memcpy(hcur,ha,Hd*4); pos++; t_next=cpu_argmax(la,V); continue; }
        embed_lookup(wf,t_next,ha); embed_lookup(wf,d,hb);
        fused_layer_forward_2(wf,ha,hb,kv,fds,pos,pos+1,la,lb,&snap);  // captures post-a rollback state
        int t_v1=cpu_argmax(la,V); mtp_tok[mn++]=t_next; checks++;
        if(t_v1==d){ accepts++; if(mn<max_new) mtp_tok[mn++]=d; memcpy(hcur,hb,Hd*4); t_next=cpu_argmax(lb,V); pos+=2; }
        else { batched_rollback_apply(&snap,kv); memcpy(hcur,ha,Hd*4); t_next=t_v1; pos++; }  // drop b, keep a — no re-forward
    }
    double mtp_ms=now_ms()-t1;
    int n=bn<mn?bn:mn,mism=0; for(int i=0;i<n;i++) if(base_tok[i]!=mtp_tok[i]){mism++; if(mism==1)fprintf(stderr,"[mtp-dense] first mismatch @%d base=%d mtp=%d\n",i,base_tok[i],mtp_tok[i]);}
    double acc=checks?100.0*accepts/checks:0;
    fprintf(stderr,"[mtp-dense] baseline: %d tok %.0f ms (%.2f tok/s)\n",bn,base_ms,1000.0*bn/base_ms);
    fprintf(stderr,"[mtp-dense] mtp:      %d tok %.0f ms (%.2f tok/s) | acceptance=%.1f%% (%d/%d)\n",mn,mtp_ms,1000.0*mn/mtp_ms,acc,accepts,checks);
    fprintf(stderr,"[mtp-dense] SPEEDUP=%.2fx | lossless: %d/%d match %s\n", base_ms>0&&mtp_ms>0?(1000.0*mn/mtp_ms)/(1000.0*bn/base_ms):0, n-mism,n, mism==0?"PASS":"FAIL");
    gpu_snap_free(&snap);
    for(int i=0;i<Ld;i++){ if(kv[i]){free(kv[i]->k_cache);free(kv[i]->v_cache);free(kv[i]);} if(ls[i])linear_attn_state_free(ls[i]); }
    free(fds);free(mmaps);free(kv);free(ls);free(h);free(logits);free(base_tok);free(mtp_tok);free(hcur);free(ha);free(hb);free(la);free(lb);
    #undef IS_EOS
    return mism==0?0:1;
}

// ============================================================================
// Linear attention layer forward (GatedDeltaNet, single token, incremental)
// ============================================================================

// RMS norm without weights (just normalize)
static void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms;
}

// RMSNormGated: out = rms_norm(x) * silu(z)
static void cpu_rms_norm_gated(const float *x, const float *z, const uint16_t *w_bf16,
                                float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = bf16_to_f32(w_bf16[i]);
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

static int linear_attn_bypass = 0;  // set to 1 to skip linear attention (identity)
static int gpu_linear_attn_enabled = 1;  // fused GPU delta-net path (can disable via --cpu-linear)

__attribute__((unused))
static void linear_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [g_cfg.hidden_dim] in/out
    LinearAttnState *state
) {
    // If bypass is enabled, just pass through (identity)
    if (linear_attn_bypass) {
        (void)wf; (void)layer_idx; (void)state;
        return;
    }

    int la_debug = 0;

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, g_cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    char name[256];
    float *normed = malloc(g_cfg.hidden_dim * sizeof(float));
    float *residual = malloc(g_cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(residual, hidden, g_cfg.hidden_dim);

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);

    // ---- Batch QKV + Z + B + A projections (4 matmuls, 1 command buffer) ----
    int qkv_dim = g_cfg.linear_conv_dim;  // 12288
    float *qkv = calloc(qkv_dim, sizeof(float));
    int z_dim = g_cfg.linear_total_value;  // 8192
    float *z = calloc(z_dim, sizeof(float));
    float *beta = calloc(g_cfg.linear_num_v_heads, sizeof(float));
    float *alpha = calloc(g_cfg.linear_num_v_heads, sizeof(float));

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    uint32_t *qkv_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", layer_idx);
    uint16_t *qkv_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", layer_idx);
    uint16_t *qkv_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    uint32_t *z_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", layer_idx);
    uint16_t *z_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", layer_idx);
    uint16_t *z_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    uint32_t *b_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", layer_idx);
    uint16_t *b_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", layer_idx);
    uint16_t *b_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    uint32_t *a_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", layer_idx);
    uint16_t *a_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", layer_idx);
    uint16_t *a_b = get_tensor_ptr(wf, name);

    if (qkv_w && qkv_s && qkv_b && z_w && z_s && z_b &&
        b_w && b_s && b_b && a_w && a_s && a_b) {
        BatchMatvecSpec la_specs[4] = {
            { qkv_w, qkv_s, qkv_b, qkv,   (uint32_t)qkv_dim,         g_cfg.hidden_dim, g_cfg.group_size, 0 },
            { z_w,   z_s,   z_b,   z,      (uint32_t)z_dim,           g_cfg.hidden_dim, g_cfg.group_size, 1 },
            { b_w,   b_s,   b_b,   beta,   (uint32_t)g_cfg.linear_num_v_heads, g_cfg.hidden_dim, g_cfg.group_size, 2 },
            { a_w,   a_s,   a_b,   alpha,  (uint32_t)g_cfg.linear_num_v_heads, g_cfg.hidden_dim, g_cfg.group_size, 3 },
        };
        fast_batch_matvec(normed, g_cfg.hidden_dim, la_specs, 4);
    }

    // ---- Conv1d step ----
    // conv_state holds last (kernel_size-1) inputs for each of the conv_dim channels
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", layer_idx);
    uint16_t *conv_w = get_tensor_ptr(wf, name);

    float *conv_out = calloc(qkv_dim, sizeof(float));
    if (conv_w) {
        cpu_conv1d_step(state->conv_state, qkv, conv_w, conv_out,
                        qkv_dim, g_cfg.linear_conv_kernel_dim);
    }

    // Update conv state: shift left, append new input
    memmove(state->conv_state, state->conv_state + qkv_dim,
            (g_cfg.linear_conv_kernel_dim - 2) * qkv_dim * sizeof(float));
    memcpy(state->conv_state + (g_cfg.linear_conv_kernel_dim - 2) * qkv_dim, qkv,
           qkv_dim * sizeof(float));

    // ---- Split conv_out into q, k, v ----
    // q: [num_k_heads * head_k_dim] = [2048]
    // k: [num_k_heads * head_k_dim] = [2048]
    // v: [num_v_heads * head_v_dim] = [8192]
    float *lin_q = conv_out;  // first g_cfg.linear_total_key elements
    float *lin_k = conv_out + g_cfg.linear_total_key;  // next g_cfg.linear_total_key
    float *lin_v = conv_out + 2 * g_cfg.linear_total_key;  // rest = g_cfg.linear_total_value

    // ---- RMS normalize q and k (bare, no weights) ----
    // q: scale = key_dim^(-0.5), normalize per head then scale by key_dim^(-1.0)
    // Actually from the code:
    //   inv_scale = k.shape[-1] ** -0.5 = head_k_dim^(-0.5) = 128^(-0.5)
    //   q = (inv_scale**2) * rms_norm(q) = (1/128) * rms_norm(q)
    //   k = inv_scale * rms_norm(k) = (1/sqrt(128)) * rms_norm(k)
    float inv_scale = 1.0f / sqrtf((float)g_cfg.linear_key_dim);

    for (int h = 0; h < g_cfg.linear_num_k_heads; h++) {
        float *qh = lin_q + h * g_cfg.linear_key_dim;
        cpu_rms_norm_bare(qh, qh, g_cfg.linear_key_dim, 1e-6f);
        float q_scale = inv_scale * inv_scale;  // inv_scale^2 = 1/head_k_dim
        for (int d = 0; d < g_cfg.linear_key_dim; d++) qh[d] *= q_scale;
    }
    for (int h = 0; h < g_cfg.linear_num_k_heads; h++) {
        float *kh = lin_k + h * g_cfg.linear_key_dim;
        cpu_rms_norm_bare(kh, kh, g_cfg.linear_key_dim, 1e-6f);
        for (int d = 0; d < g_cfg.linear_key_dim; d++) kh[d] *= inv_scale;
    }

    // ---- Gated delta net recurrence ----
    // From gated_delta.py:
    //   g = exp(-exp(A_log) * softplus(a + dt_bias))   -- per-head decay
    //   beta_gate = sigmoid(b)                          -- per-head beta (NO dt_bias)
    //   For each v_head:
    //     state = state * g                             -- decay
    //     kv_mem = sum(state * k, axis=key_dim)         -- predict v from state
    //     delta = (v - kv_mem) * beta_gate              -- error signal
    //     state = state + outer(delta, k)               -- update state
    //     output = sum(state * q, axis=key_dim)         -- read from state

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", layer_idx);
    float *A_log = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *dt_bias_bf16 = get_tensor_ptr(wf, name);

    float *out_values = calloc(g_cfg.linear_total_value, sizeof(float));  // [num_v_heads * head_v_dim]

    int k_heads_per_v = g_cfg.linear_num_v_heads / g_cfg.linear_num_k_heads;  // 64/16 = 4

    // Precompute per-head decay (g) and beta
    float g_decay[g_cfg.linear_num_v_heads];
    float beta_gate[g_cfg.linear_num_v_heads];
    for (int vh = 0; vh < g_cfg.linear_num_v_heads; vh++) {
        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        float a_val = alpha[vh];
        float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
        float A_val = A_log ? expf(A_log[vh]) : 1.0f;
        float softplus_val = logf(1.0f + expf(a_val + dt_b));  // softplus(a + dt_bias)
        g_decay[vh] = expf(-A_val * softplus_val);

        // beta = sigmoid(b)  (just b, NO dt_bias)
        beta_gate[vh] = cpu_sigmoid(beta[vh]);
    }

    for (int vh = 0; vh < g_cfg.linear_num_v_heads; vh++) {
        int kh = vh / k_heads_per_v;  // which k head this v head maps to

        float g = g_decay[vh];
        float b_gate = beta_gate[vh];

        // state is [head_v_dim, head_k_dim]
        float *S = state->ssm_state + vh * g_cfg.linear_value_dim * g_cfg.linear_key_dim;
        float *v_h = lin_v + vh * g_cfg.linear_value_dim;
        float *k_h = lin_k + kh * g_cfg.linear_key_dim;

        // Step 1: Decay state
        for (int vi = 0; vi < g_cfg.linear_value_dim; vi++) {
            for (int ki = 0; ki < g_cfg.linear_key_dim; ki++) {
                S[vi * g_cfg.linear_key_dim + ki] *= g;
            }
        }

        // Step 2: Compute kv_mem[vi] = sum_ki(S[vi,ki] * k[ki])
        // Then delta[vi] = (v[vi] - kv_mem[vi]) * beta
        // Then state[vi,ki] += k[ki] * delta[vi]
        for (int vi = 0; vi < g_cfg.linear_value_dim; vi++) {
            float kv_mem = 0.0f;
            for (int ki = 0; ki < g_cfg.linear_key_dim; ki++) {
                kv_mem += S[vi * g_cfg.linear_key_dim + ki] * k_h[ki];
            }
            float delta = (v_h[vi] - kv_mem) * b_gate;
            for (int ki = 0; ki < g_cfg.linear_key_dim; ki++) {
                S[vi * g_cfg.linear_key_dim + ki] += k_h[ki] * delta;
            }
        }

        // Step 3: Output: y[vi] = sum_ki(S[vi,ki] * q[ki])
        float *q_h = lin_q + kh * g_cfg.linear_key_dim;
        float *o_h = out_values + vh * g_cfg.linear_value_dim;
        for (int vi = 0; vi < g_cfg.linear_value_dim; vi++) {
            float sum = 0.0f;
            for (int ki = 0; ki < g_cfg.linear_key_dim; ki++) {
                sum += S[vi * g_cfg.linear_key_dim + ki] * q_h[ki];
            }
            o_h[vi] = sum;
        }
    }

    // ---- RMSNormGated: out = rms_norm(out_values_per_head) * silu(z_per_head) * weight ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", layer_idx);
    uint16_t *gated_norm_w = get_tensor_ptr(wf, name);

    float *gated_out = calloc(g_cfg.linear_total_value, sizeof(float));
    for (int vh = 0; vh < g_cfg.linear_num_v_heads; vh++) {
        float *oh = out_values + vh * g_cfg.linear_value_dim;
        float *zh = z + vh * g_cfg.linear_value_dim;
        float *gh = gated_out + vh * g_cfg.linear_value_dim;
        if (gated_norm_w) {
            cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, g_cfg.linear_value_dim, g_cfg.rms_norm_eps);
        } else {
            memcpy(gh, oh, g_cfg.linear_value_dim * sizeof(float));
        }
    }

    // ---- Output projection: [value_dim=8192] -> [hidden_dim=4096] ----
    float *attn_out = calloc(g_cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", layer_idx);
    uint32_t *out_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", layer_idx);
    uint16_t *out_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", layer_idx);
    uint16_t *out_b = get_tensor_ptr(wf, name);
    if (out_w && out_s && out_b) {
        fast_dequant_matvec(out_w, out_s, out_b, gated_out, attn_out, g_cfg.hidden_dim,
                            g_cfg.linear_total_value, g_cfg.group_size);
    }

    // ---- Residual ----
    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_out[i];
    }

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] AFTER layer=%d out_proj_rms=%.6f gated_rms=%.6f hidden_rms=%.6f\n",
                layer_idx, vec_rms(attn_out, g_cfg.hidden_dim),
                vec_rms(gated_out, g_cfg.linear_total_value),
                vec_rms(hidden, g_cfg.hidden_dim));
    }

    free(normed);
    free(residual);
    free(qkv);
    free(z);
    free(beta);
    free(alpha);
    free(conv_out);
    free(out_values);
    free(gated_out);
    free(attn_out);
}

// ============================================================================
// MoE forward (routing + expert computation + shared expert)
// ============================================================================

static int moe_debug_count = 0;

__attribute__((unused))
static void moe_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,         // [g_cfg.hidden_dim] in/out
    const char *model_path __attribute__((unused)),
    int K,                 // number of active experts (e.g. 4)
    int packed_fd          // fd for this layer's packed expert file (-1 if not available)
) {
    moe_debug_count++;
    int moe_debug = 0;  // set to (moe_debug_count <= N) to enable debug
    int moe_dump = 0;

    char name[256];
    float *h_post = malloc(g_cfg.hidden_dim * sizeof(float));
    float *h_mid = malloc(g_cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(h_mid, hidden, g_cfg.hidden_dim);

    // ---- Post-attention LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, h_post, g_cfg.hidden_dim, g_cfg.rms_norm_eps);

    // ---- Batch routing gate + shared expert gate/up + shared_expert_gate (4 matmuls, 1 commit) ----
    float *gate_scores = calloc(g_cfg.num_experts, sizeof(float));
    float *shared_gate = calloc(g_cfg.shared_intermediate, sizeof(float));
    float *shared_up = calloc(g_cfg.shared_intermediate, sizeof(float));
    float shared_gate_score = 0.0f;

    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", layer_idx);
    uint32_t *gate_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", layer_idx);
    uint16_t *gate_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", layer_idx);
    uint16_t *gate_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    uint32_t *sgw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
    uint16_t *sgs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
    uint16_t *sgb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    uint32_t *suw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
    uint16_t *sus = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
    uint16_t *sub = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    uint32_t *seg_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", layer_idx);
    uint16_t *seg_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", layer_idx);
    uint16_t *seg_b = get_tensor_ptr(wf, name);

    // All 4 matmuls share h_post as input -- batch into one command buffer
    if (gate_w && gate_s && gate_b && sgw && sgs && sgb &&
        suw && sus && sub && seg_w && seg_s && seg_b) {
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)g_cfg.num_experts,        g_cfg.hidden_dim, g_cfg.group_size, 0 },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size, 1 },
            { suw,    sus,    sub,    shared_up,           (uint32_t)g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size, 2 },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            g_cfg.hidden_dim, g_cfg.group_size, 3 },
        };
        fast_batch_matvec(h_post, g_cfg.hidden_dim, moe_specs, 4);
    }

    // Softmax routing scores
    cpu_softmax(gate_scores, g_cfg.num_experts);

    // Top-K expert selection
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, g_cfg.num_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] routing: K=%d experts=[", K);
        for (int k = 0; k < K; k++) fprintf(stderr, "%d(%.4f)%s", expert_indices[k], expert_weights[k], k<K-1?",":"");
        fprintf(stderr, "]\n");
    }

    // ---- Routed expert computation ----
    float *moe_out = calloc(g_cfg.hidden_dim, sizeof(float));

    if (packed_fd >= 0) {
        float *expert_out = malloc(g_cfg.hidden_dim * sizeof(float));

        size_t esz = active_expert_size();
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;

            if (g_metal && g_metal->buf_expert_data) {
                // GPU path: pread directly into Metal buffer, run gate+up+swiglu+down on GPU
                void *expert_buf_ptr = [g_metal->buf_expert_data contents];
                ssize_t nread = pread(packed_fd, expert_buf_ptr, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    continue;
                }

                gpu_expert_forward(g_metal, expert_buf_ptr, h_post, expert_out, 1 /*already in buffer*/);
            } else {
                // CPU fallback
                void *expert_data = malloc(esz);
                ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    free(expert_data);
                    continue;
                }

                uint32_t *gw = (uint32_t *)expert_data;
                uint16_t *gs_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size);
                uint16_t *gb_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size);
                uint32_t *uw = (uint32_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size);
                uint16_t *us_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size);
                uint16_t *ub_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size);
                uint32_t *dw = (uint32_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size);
                uint16_t *ds_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size + g_cfg.down_w_size);
                uint16_t *db_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size + g_cfg.down_w_size + g_cfg.down_s_size);

                float *gate_proj_out = malloc(g_cfg.moe_intermediate * sizeof(float));
                float *up_proj_out = malloc(g_cfg.moe_intermediate * sizeof(float));
                float *act_out = malloc(g_cfg.moe_intermediate * sizeof(float));

                cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                                   g_cfg.moe_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
                cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                                   g_cfg.moe_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
                cpu_swiglu(gate_proj_out, up_proj_out, act_out, g_cfg.moe_intermediate);
                cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out,
                                   g_cfg.hidden_dim, g_cfg.moe_intermediate, g_cfg.group_size);

                free(gate_proj_out);
                free(up_proj_out);
                free(act_out);
                free(expert_data);
            }

            // Accumulate weighted
            if (moe_dump) {
                fprintf(stderr, "[MOE-DUMP] expert[%d] out_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        eidx, vec_rms(expert_out, g_cfg.hidden_dim),
                        expert_out[0], expert_out[1], expert_out[2], expert_out[3], expert_out[4]);
            }
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], g_cfg.hidden_dim);
        }

        free(expert_out);
    }

    // ---- Shared expert SwiGLU (gate_proj + up_proj already computed above) ----
    float *shared_out = calloc(g_cfg.hidden_dim, sizeof(float));
    float *shared_act = calloc(g_cfg.shared_intermediate, sizeof(float));
    cpu_swiglu(shared_gate, shared_up, shared_act, g_cfg.shared_intermediate);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] layer=%d h_post_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(h_post, g_cfg.hidden_dim), h_post[0], h_post[1], h_post[2], h_post[3], h_post[4]);
        fprintf(stderr, "[MOE-DUMP] gate_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_gate, g_cfg.shared_intermediate),
                shared_gate[0], shared_gate[1], shared_gate[2], shared_gate[3], shared_gate[4]);
        fprintf(stderr, "[MOE-DUMP] up_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_up, g_cfg.shared_intermediate),
                shared_up[0], shared_up[1], shared_up[2], shared_up[3], shared_up[4]);
        fprintf(stderr, "[MOE-DUMP] swiglu_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_act, g_cfg.shared_intermediate),
                shared_act[0], shared_act[1], shared_act[2], shared_act[3], shared_act[4]);
    }

    // shared_expert down_proj (separate dispatch — different input than h_post)
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    uint32_t *sdw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
    uint16_t *sds = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
    uint16_t *sdb = get_tensor_ptr(wf, name);
    if (sdw && sds && sdb) {
        fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out, g_cfg.hidden_dim,
                            g_cfg.shared_intermediate, g_cfg.group_size);
    }

    // ---- Shared expert gate (sigmoid) -- already computed above ----
    float shared_weight = cpu_sigmoid(shared_gate_score);

    // Scale shared expert output
    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (moe_debug) {
        fprintf(stderr, "[MOE-DBG] layer=%d h_mid_rms=%.4f moe_rms=%.4f shared_rms=%.4f shared_gate=%.4f hidden_rms=%.4f\n",
                layer_idx, vec_rms(h_mid, g_cfg.hidden_dim), vec_rms(moe_out, g_cfg.hidden_dim),
                vec_rms(shared_out, g_cfg.hidden_dim), shared_weight,
                vec_rms(hidden, g_cfg.hidden_dim));
    }

    free(h_post);
    free(h_mid);
    free(gate_scores);
    free(moe_out);
    free(shared_out);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}

// ============================================================================
// Embedding lookup (4-bit quantized)
// ============================================================================

static void embed_lookup(WeightFile *wf, int token_id, float *out) {
    // Embedding: weight[vocab_size, hidden_dim/8] (U32), scales[vocab_size, groups], biases[vocab_size, groups]
    // For embedding lookup, we just need one row.
    // But the embedding is quantized: each row has hidden_dim/8 uint32 values (packed 4-bit)
    // plus scales and biases per group

    TensorInfo *w_info = get_tensor_info(wf, "model.embed_tokens.weight");
    TensorInfo *s_info = get_tensor_info(wf, "model.embed_tokens.scales");
    TensorInfo *b_info = get_tensor_info(wf, "model.embed_tokens.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embedding tensors not found\n");
        memset(out, 0, g_cfg.hidden_dim * sizeof(float));
        return;
    }

    // w shape: [248320, 512] U32 -> each row has 512 uint32 = 4096 packed 4-bit values
    int packed_cols = w_info->shape[1];  // 512
    int num_groups = s_info->shape[1];   // 64

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    const uint32_t *w_row = W + (size_t)token_id * packed_cols;
    const uint16_t *s_row = S + (size_t)token_id * num_groups;
    const uint16_t *b_row = B + (size_t)token_id * num_groups;

    int group_size = g_cfg.hidden_dim / num_groups;  // 4096/64 = 64
    int packed_per_group = group_size / 8;     // 8

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * 8;

            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (float)nibble * scale + bias;
            }
        }
    }
}

// ============================================================================
// LM head (logits projection)
// ============================================================================

static void lm_head_forward(WeightFile *wf, const float *hidden, float *logits) {
    // lm_head: [hidden_dim=4096] -> [vocab_size=248320]
    // This is a HUGE matmul. For 248320 output dims, it will be slow on CPU.
    // Optimization: only compute top candidates

    TensorInfo *w_info = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *s_info = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *b_info = get_tensor_info(wf, "lm_head.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: lm_head tensors not found\n");
        return;
    }

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    // Full matmul — use GPU if available (248320 output rows!)
    fast_dequant_matvec(W, S, B, hidden, logits, g_cfg.vocab_size, g_cfg.hidden_dim, g_cfg.group_size);
}

// ============================================================================
// Parallel I/O infrastructure for expert pread (from proven main.m pattern)
// ============================================================================

#define NUM_IO_THREADS 4  // 4 threads for K=4 experts (one per expert)

typedef struct {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
    ssize_t result;
    const void *mmap_base;  // if non-NULL, memcpy from mmap instead of pread
    // LZ4 compression fields (set by caller when reading compressed experts)
    void *lz4_comp_buf;     // if non-NULL: pread into this, then LZ4 decompress into dst
    uint32_t lz4_comp_size; // compressed size to read from disk
} InferPreadTask;

typedef struct {
    InferPreadTask *tasks;
    int num_tasks;
    int thread_id;
} InferPreadThreadArg;

__attribute__((unused))
static void *infer_pread_thread_fn(void *arg) {
    InferPreadThreadArg *ta = (InferPreadThreadArg *)arg;
    for (int i = ta->thread_id; i < ta->num_tasks; i += NUM_IO_THREADS) {
        InferPreadTask *t = &ta->tasks[i];
        t->result = pread(t->fd, t->dst, t->size, t->offset);
    }
    return NULL;
}

// ============================================================================
// Persistent I/O Thread Pool — eliminates pthread_create/join per layer
// ============================================================================

typedef struct {
    pthread_t threads[NUM_IO_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    InferPreadTask *tasks;
    int num_tasks;
    int tasks_completed;
    int generation;          // incremented each dispatch — workers wait for new gen
    volatile int shutdown;
} IOThreadPool;

static IOThreadPool g_io_pool;
static int g_io_pool_initialized = 0;

static void *io_pool_worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    int my_gen = 0;
    pthread_mutex_lock(&g_io_pool.mutex);
    while (1) {
        while (g_io_pool.generation == my_gen && !g_io_pool.shutdown)
            pthread_cond_wait(&g_io_pool.work_ready, &g_io_pool.mutex);
        if (g_io_pool.shutdown) break;
        my_gen = g_io_pool.generation;

        // Snapshot work for this generation
        int num_tasks = g_io_pool.num_tasks;
        InferPreadTask *tasks = g_io_pool.tasks;
        pthread_mutex_unlock(&g_io_pool.mutex);

        // Process assigned tasks (stride by thread count)
        for (int i = tid; i < num_tasks; i += NUM_IO_THREADS) {
            InferPreadTask *t = &tasks[i];
            if (t->lz4_comp_buf && t->lz4_comp_size > 0) {
                // LZ4 path: read compressed from SSD, decompress into dst
                ssize_t nr = pread(t->fd, t->lz4_comp_buf, t->lz4_comp_size, t->offset);
                if (nr == (ssize_t)t->lz4_comp_size) {
                    size_t dec = compression_decode_buffer(
                        t->dst, t->size, t->lz4_comp_buf, t->lz4_comp_size,
                        NULL, COMPRESSION_LZ4);
                    t->result = (ssize_t)dec;
                } else {
                    t->result = -1;
                }
            } else {
                t->result = pread(t->fd, t->dst, t->size, t->offset);
            }
        }

        pthread_mutex_lock(&g_io_pool.mutex);
        g_io_pool.tasks_completed++;
        if (g_io_pool.tasks_completed == NUM_IO_THREADS)
            pthread_cond_signal(&g_io_pool.work_done);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
    return NULL;
}

static void io_pool_init(void) {
    if (g_io_pool_initialized) return;
    pthread_mutex_init(&g_io_pool.mutex, NULL);
    pthread_cond_init(&g_io_pool.work_ready, NULL);
    pthread_cond_init(&g_io_pool.work_done, NULL);
    g_io_pool.shutdown = 0;
    g_io_pool.generation = 0;
    g_io_pool.tasks = NULL;
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_create(&g_io_pool.threads[i], NULL, io_pool_worker, (void*)(intptr_t)i);
    g_io_pool_initialized = 1;
}

static dispatch_queue_t g_io_gcd_queue = NULL;

static void io_pool_dispatch(InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.tasks = tasks;
    g_io_pool.num_tasks = num_tasks;
    g_io_pool.tasks_completed = 0;
    g_io_pool.generation++;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    while (g_io_pool.tasks_completed < NUM_IO_THREADS) {
        pthread_cond_wait(&g_io_pool.work_done, &g_io_pool.mutex);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
}

// ---- Async expert pread pipeline ----
// Starts pread on background GCD threads immediately after routing.
// The pread overlaps with shared expert prep + next layer's CMD1+attn+CMD2.
// Wait for completion right before CMD3 needs the expert data.
typedef struct {
    InferPreadTask tasks[MAX_K];
    int num_tasks;
    int valid[MAX_K];
    dispatch_group_t group;
    int active;
} AsyncPreadState;
static AsyncPreadState g_async_pread = {0};

static void async_pread_start(int packed_fd, int *expert_indices, int K,
                               id<MTLBuffer> __strong *dst_bufs, const void *mmap_base) {
    (void)mmap_base;
    size_t esz = active_expert_size();
    g_async_pread.num_tasks = K;
    g_async_pread.active = 1;
    if (!g_async_pread.group) g_async_pread.group = dispatch_group_create();

    for (int k = 0; k < K; k++) {
        g_async_pread.tasks[k].fd = packed_fd;
        g_async_pread.tasks[k].dst = [dst_bufs[k] contents];
        g_async_pread.tasks[k].offset = (off_t)expert_indices[k] * esz;
        g_async_pread.tasks[k].size = esz;
        g_async_pread.tasks[k].result = 0;
    }

    // Fire off parallel preads on GCD — returns immediately
    static dispatch_queue_t io_q = NULL;
    if (!io_q) io_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int k = 0; k < K; k++) {
        InferPreadTask *t = &g_async_pread.tasks[k];
        dispatch_group_async(g_async_pread.group, io_q, ^{
            t->result = pread(t->fd, t->dst, t->size, t->offset);
        });
    }
}

static void async_pread_wait(void) {
    if (!g_async_pread.active) return;
    dispatch_group_wait(g_async_pread.group, DISPATCH_TIME_FOREVER);
    for (int k = 0; k < g_async_pread.num_tasks; k++) {
        g_async_pread.valid[k] = (g_async_pread.tasks[k].result == (ssize_t)active_expert_size());
    }
    g_async_pread.active = 0;
}

static void io_pool_shutdown(void) {
    if (!g_io_pool_initialized) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.shutdown = 1;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    pthread_mutex_unlock(&g_io_pool.mutex);
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_join(g_io_pool.threads[i], NULL);
    pthread_mutex_destroy(&g_io_pool.mutex);
    pthread_cond_destroy(&g_io_pool.work_ready);
    pthread_cond_destroy(&g_io_pool.work_done);
    g_io_pool_initialized = 0;
}

// Parallel pread of K experts into Metal buffers using pthreads.
// Returns number of successfully loaded experts, sets valid[] flags.
__attribute__((unused))
static int parallel_pread_experts(
    int packed_fd,
    int *expert_indices,
    int K,
    int *valid,  // [MAX_K] output: 1 if expert loaded successfully
    const void *mmap_base  // mmap'd layer file (NULL to use pread)
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        tasks[k].fd = packed_fd;
        tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
        tasks[k].result = 0;
        tasks[k].mmap_base = mmap_base;
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, esz);
        }
    }
    return loaded;
}

// ============================================================================
// Parallel pread into explicit buffer set (for double buffering).
// Same as parallel_pread_experts but reads into caller-specified MTLBuffers.
// ============================================================================
__attribute__((unused))
static int parallel_pread_experts_into(
    int packed_fd,
    int *expert_indices,
    int K,
    id<MTLBuffer> __strong *dst_bufs,  // target Metal buffers (set A or B)
    int *valid  // [MAX_K] output: 1 if expert loaded successfully
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        tasks[k].fd = packed_fd;
        tasks[k].dst = [dst_bufs[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
        tasks[k].result = 0;
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, esz);
        }
    }
    return loaded;
}

// ============================================================================
// Expert LRU Cache: keeps recently-used expert Metal buffers in GPU memory.
//
// Key: (layer_idx, expert_idx) -> Metal buffer containing 7.08MB expert data.
// On cache HIT:  skip pread entirely, use the cached Metal buffer for GPU dispatch.
// On cache MISS: pread into a new/evicted Metal buffer, insert into cache.
// LRU eviction:  when cache is full, evict the least recently used entry.
//
// Memory budget: 2000 entries * 7.08MB = 14.2GB. With 5.5GB non-expert weights
// + 14.2GB cache = 19.7GB total. Fits in 48GB with room for OS.
//
// Unlike Python/MLX where LRU caching caused Metal heap pressure and slower
// mx.eval(), here Metal buffers ARE the cache -- no conversion overhead.
// ============================================================================

typedef struct {
    int layer_idx;
    int expert_idx;
    id<MTLBuffer> buffer;    // Metal buffer holding g_cfg.expert_size bytes
    uint64_t last_used;      // monotonic counter for LRU ordering
} ExpertCacheEntry;

typedef struct {
    ExpertCacheEntry *entries;
    int max_entries;
    int num_entries;
    int used_entries;
    int entry_idx[MAX_NUM_LAYERS][MAX_NUM_EXPERTS];
    uint64_t access_counter; // monotonic, incremented on every access
    id<MTLDevice> device;    // for allocating new Metal buffers
    // Stats
    uint64_t hits;
    uint64_t misses;
} ExpertLRUCache;

static ExpertLRUCache *g_expert_cache = NULL;

// Speculative early routing stats
static uint64_t g_spec_route_attempts = 0;   // total speculative routing attempts
static uint64_t g_spec_route_hits = 0;        // correctly predicted experts (found in cache at real routing time)
static uint64_t g_spec_route_preloads = 0;    // async preloads initiated (cache misses at speculation time)

// ---- Temporal prediction pipeline ----
// Stores previous token's expert routing per layer. On the next token,
// predicted experts are preloaded into buf_multi_expert_data_B during CMD1_wait
// idle time. After routing, hits use buf_B, misses sync-pread into buf_A.
// Different from previous failed speculative attempts:
//   - Loads into scratch buffers (no cache pollution)
//   - Uses CMD1_wait idle time (no additional CPU cost)
//   - Only sync-preads misses (not all K experts)
static int g_pred_experts[60][MAX_K];              // previous token's expert indices per layer
static int g_pred_count[60];                       // how many experts stored per layer
static int g_pred_valid = 0;                       // 1 after first token completes (predictions available)
// g_pred_enabled, g_pred_hits, g_pred_misses, g_pred_layers declared near timing (line ~163)

static ExpertLRUCache *expert_cache_new(id<MTLDevice> device, int max_entries) {
    ExpertLRUCache *cache = calloc(1, sizeof(ExpertLRUCache));
    cache->entries = calloc(max_entries, sizeof(ExpertCacheEntry));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->device = device;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < g_cfg.num_layers; l++) {
        for (int e = 0; e < g_cfg.num_experts; e++) {
            cache->entry_idx[l][e] = -1;
        }
    }
    // Pre-allocate ALL Metal buffers at startup (avoids allocation overhead at runtime)
    size_t esz = active_expert_size();
    double t_prealloc = now_ms();
    for (int i = 0; i < max_entries; i++) {
        cache->entries[i].buffer = [device newBufferWithLength:esz
                                                      options:MTLResourceStorageModeShared];
        cache->entries[i].layer_idx = -1;
        cache->entries[i].expert_idx = -1;
        cache->entries[i].last_used = 0;
        if (!cache->entries[i].buffer) {
            fprintf(stderr, "WARNING: expert_cache: pre-alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
    }
    cache->num_entries = max_entries; // All slots pre-allocated (but empty keys)
    printf("[expert_cache] Initialized: max_entries=%d (%.1f GB budget), pre-alloc %.0f ms\n",
           max_entries, (double)max_entries * esz / 1e9, now_ms() - t_prealloc);
    return cache;
}

static void expert_cache_free(ExpertLRUCache *cache) {
    if (!cache) return;
    printf("[expert_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    // Metal buffers released by ARC when entries are freed
    free(cache->entries);
    free(cache);
}

// Lookup: returns the cached Metal buffer if found, otherwise NULL.
// On hit, updates the LRU timestamp.
static id<MTLBuffer> expert_cache_lookup(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    int idx = cache->entry_idx[layer_idx][expert_idx];
    if (idx >= 0) {
        cache->entries[idx].last_used = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer_idx, expert_idx);
        return cache->entries[idx].buffer;
    }
    cache->misses++;
    cache_telemetry_miss(layer_idx, expert_idx);
    return nil;
}

// Insert: adds a new entry. If the cache is full, evicts the LRU entry.
// Returns the Metal buffer to pread into (either newly allocated or evicted+reused).
static id<MTLBuffer> expert_cache_insert(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    id<MTLBuffer> buf = nil;

    int existing = cache->entry_idx[layer_idx][expert_idx];
    if (existing >= 0) {
        cache->entries[existing].last_used = ++cache->access_counter;
        return cache->entries[existing].buffer;
    }

    // Find a slot: first try an unused slot (layer_idx == -1), then LRU evict
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }
    if (target >= 0) {
        // Unused pre-allocated slot
        buf = cache->entries[target].buffer;
        cache->entries[target].layer_idx = layer_idx;
        cache->entries[target].expert_idx = expert_idx;
        cache->entries[target].last_used = ++cache->access_counter;
        cache->entry_idx[layer_idx][expert_idx] = target;
        return buf;
    }

    // Cache full: find LRU entry (smallest last_used)
    int lru_idx = 0;
    uint64_t min_used = cache->entries[0].last_used;
    for (int i = 1; i < cache->num_entries; i++) {
        if (cache->entries[i].last_used < min_used) {
            min_used = cache->entries[i].last_used;
            lru_idx = i;
        }
    }

    // Reuse the evicted entry's Metal buffer (same size, no realloc needed)
    int old_layer = cache->entries[lru_idx].layer_idx;
    int old_expert = cache->entries[lru_idx].expert_idx;
    cache_telemetry_evict(old_layer, old_expert);
    if (old_layer >= 0 && old_expert >= 0) {
        cache->entry_idx[old_layer][old_expert] = -1;
    }
    buf = cache->entries[lru_idx].buffer;
    cache->entries[lru_idx].layer_idx = layer_idx;
    cache->entries[lru_idx].expert_idx = expert_idx;
    cache->entries[lru_idx].last_used = ++cache->access_counter;
    cache->entry_idx[layer_idx][expert_idx] = lru_idx;
    return buf;
}

// ============================================================================
// Malloc-based expert frequency cache.
// Stores expert data in regular malloc'd memory (not Metal buffers) to avoid
// GPU memory pressure. On hit, memcpy to Metal scratch buffer. Much larger
// capacity than Metal buffer LRU cache at the cost of one memcpy per hit.
// ============================================================================

typedef struct {
    void **data;           // [max_entries] page-aligned malloc'd g_cfg.expert_size buffers
    id<MTLBuffer> __strong *metal_bufs;  // [max_entries] zero-copy Metal buffer wrappers
    int *layer_idx;        // [max_entries] layer index for each entry
    int *expert_idx;       // [max_entries] expert index for each entry
    uint64_t *last_used;   // [max_entries] monotonic counter for LRU
    int max_entries;
    int num_entries;
    int used_entries;
    int entry_idx[MAX_NUM_LAYERS][MAX_NUM_EXPERTS];
    uint64_t access_counter;
    uint64_t hits;
    uint64_t misses;
} MallocExpertCache;

static MallocExpertCache *g_malloc_cache = NULL;

static MallocExpertCache *malloc_cache_init(int max_entries, id<MTLDevice> device) {
    MallocExpertCache *cache = calloc(1, sizeof(MallocExpertCache));
    cache->data = calloc(max_entries, sizeof(void *));
    cache->metal_bufs = (__strong id<MTLBuffer> *)calloc(max_entries, sizeof(id<MTLBuffer>));
    cache->layer_idx = calloc(max_entries, sizeof(int));
    cache->expert_idx = calloc(max_entries, sizeof(int));
    cache->last_used = calloc(max_entries, sizeof(uint64_t));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < g_cfg.num_layers; l++) {
        for (int e = 0; e < g_cfg.num_experts; e++) {
            cache->entry_idx[l][e] = -1;
        }
    }

    size_t esz = active_expert_size();
    printf("[malloc_cache] Initializing: %d entries (%.1f GB) with zero-copy Metal wrappers\n",
           max_entries, (double)max_entries * esz / 1e9);
    double t_start = now_ms();

    size_t page_size = (size_t)getpagesize();
    // Round expert size up to page boundary for newBufferWithBytesNoCopy
    size_t aligned_size = (esz + page_size - 1) & ~(page_size - 1);

    for (int i = 0; i < max_entries; i++) {
        // Page-aligned allocation for zero-copy Metal buffer
        void *buf = NULL;
        if (posix_memalign(&buf, page_size, aligned_size) != 0 || !buf) {
            fprintf(stderr, "WARNING: malloc_cache: alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
        memset(buf, 0, aligned_size);
        cache->data[i] = buf;

        // Create zero-copy Metal buffer wrapping the malloc'd memory
        // nil deallocator = Metal doesn't free the memory
        cache->metal_bufs[i] = [device newBufferWithBytesNoCopy:buf
                                                         length:aligned_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        cache->layer_idx[i] = -1;
        cache->expert_idx[i] = -1;
        cache->last_used[i] = 0;
    }
    cache->num_entries = max_entries;

    printf("[malloc_cache] Pre-allocated %d entries in %.0f ms\n",
           max_entries, now_ms() - t_start);
    return cache;
}

// Lookup: returns Metal buffer wrapping cached data, or nil. Zero-copy dispatch.
static id<MTLBuffer> malloc_cache_lookup(MallocExpertCache *cache, int layer, int expert) {
    int idx = cache->entry_idx[layer][expert];
    if (idx >= 0) {
        cache->last_used[idx] = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer, expert);
        return cache->metal_bufs[idx];
    }
    cache->misses++;
    cache_telemetry_miss(layer, expert);
    return nil;
}

// Insert: evict LRU if needed, return entry index for pread target.
// Returns the Metal buffer for this entry (caller should pread into cache->data[idx]).
static id<MTLBuffer> malloc_cache_insert(MallocExpertCache *cache, int layer, int expert, int *out_idx) {
    int existing = cache->entry_idx[layer][expert];
    if (existing >= 0) {
        cache->last_used[existing] = ++cache->access_counter;
        if (out_idx) *out_idx = existing;
        return cache->metal_bufs[existing];
    }

    // Find a free slot (layer_idx == -1) or evict LRU
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }

    if (target < 0) {
        // Cache full: evict entry with smallest last_used
        target = 0;
        uint64_t min_used = cache->last_used[0];
        for (int i = 1; i < cache->num_entries; i++) {
            if (cache->last_used[i] < min_used) {
                min_used = cache->last_used[i];
                target = i;
            }
        }
        cache_telemetry_evict(cache->layer_idx[target], cache->expert_idx[target]);
        if (cache->layer_idx[target] >= 0 && cache->expert_idx[target] >= 0) {
            cache->entry_idx[cache->layer_idx[target]][cache->expert_idx[target]] = -1;
        }
    }

    cache->layer_idx[target] = layer;
    cache->expert_idx[target] = expert;
    cache->last_used[target] = ++cache->access_counter;
    cache->entry_idx[layer][expert] = target;
    if (out_idx) *out_idx = target;
    return cache->metal_bufs[target];
}

static void malloc_cache_free(MallocExpertCache *cache) {
    if (!cache) return;
    printf("[malloc_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    for (int i = 0; i < cache->num_entries; i++) {
        cache->metal_bufs[i] = nil;  // release Metal buffer wrapper
        free(cache->data[i]);
    }
    free(cache->data);
    free(cache->metal_bufs);
    free(cache->layer_idx);
    free(cache->expert_idx);
    free(cache->last_used);
    free(cache);
}

// ============================================================================
// Background prefetch thread for double-buffered expert I/O (from main.m).
// Runs pread on a background thread while main thread does GPU compute.
// Uses pure C I/O plan to avoid ARC issues across threads.
// ============================================================================

typedef struct {
    void *dst[MAX_K];       // raw pointers from [buf contents] (no ARC)
    off_t offset[MAX_K];    // file offsets per expert
    int K;                  // number of experts
    int fd;                 // file descriptor for this layer
    int valid[MAX_K];       // output: 1 if pread succeeded
    int loaded;             // output: count of successfully loaded experts
} InferIOPlan;

typedef struct {
    InferIOPlan plan;       // pre-built I/O plan (pure C, no ARC)
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int start;              // signal: set to 1 to start prefetch
    int done;               // signal: set to 1 when prefetch complete
    int shutdown;           // signal: set to 1 to exit thread
} InferPrefetchCtx;

static void *infer_prefetch_thread_fn(void *arg) {
    InferPrefetchCtx *pf = (InferPrefetchCtx *)arg;

    while (1) {
        pthread_mutex_lock(&pf->mutex);
        while (!pf->start && !pf->shutdown) {
            pthread_cond_wait(&pf->cond, &pf->mutex);
        }
        if (pf->shutdown) {
            pthread_mutex_unlock(&pf->mutex);
            break;
        }
        pf->start = 0;
        pthread_mutex_unlock(&pf->mutex);

        // Execute parallel pread (pure C, no ARC objects)
        size_t esz = active_expert_size();
        InferIOPlan *plan = &pf->plan;
        InferPreadTask tasks[MAX_K];
        for (int k = 0; k < plan->K; k++) {
            tasks[k].fd = plan->fd;
            tasks[k].dst = plan->dst[k];
            tasks[k].offset = plan->offset[k];
            tasks[k].size = esz;
            tasks[k].result = 0;
        }

        io_pool_dispatch(tasks, plan->K);

        plan->loaded = 0;
        for (int k = 0; k < plan->K; k++) {
            plan->valid[k] = (tasks[k].result == (ssize_t)esz);
            if (plan->valid[k]) plan->loaded++;
        }

        // Signal completion
        pthread_mutex_lock(&pf->mutex);
        pf->done = 1;
        pthread_cond_signal(&pf->cond);
        pthread_mutex_unlock(&pf->mutex);
    }

    return NULL;
}

// Build I/O plan on main thread (ARC-safe: extracts void* from id<MTLBuffer>),
// then signal background prefetch thread.
__attribute__((unused))
static void infer_prefetch_start(InferPrefetchCtx *pf, int packed_fd,
                                  int *expert_indices, int K,
                                  id<MTLBuffer> __strong *dst_bufs) {
    pthread_mutex_lock(&pf->mutex);
    size_t esz = active_expert_size();
    InferIOPlan *plan = &pf->plan;
    plan->fd = packed_fd;
    plan->K = K;
    for (int k = 0; k < K; k++) {
        plan->dst[k] = [dst_bufs[k] contents];
        plan->offset[k] = (off_t)expert_indices[k] * esz;
        plan->valid[k] = 0;
    }
    plan->loaded = 0;
    pf->done = 0;
    pf->start = 1;
    pthread_cond_signal(&pf->cond);
    pthread_mutex_unlock(&pf->mutex);
}

// Wait for background prefetch to complete. Returns number of loaded experts.
// Copies valid[] flags into caller's array.
__attribute__((unused))
static int infer_prefetch_wait(InferPrefetchCtx *pf, int *valid_out, int K) {
    pthread_mutex_lock(&pf->mutex);
    while (!pf->done) {
        pthread_cond_wait(&pf->cond, &pf->mutex);
    }
    int loaded = pf->plan.loaded;
    for (int k = 0; k < K; k++) {
        valid_out[k] = pf->plan.valid[k];
    }
    pthread_mutex_unlock(&pf->mutex);
    return loaded;
}

static InferPrefetchCtx *g_prefetch = NULL;
static pthread_t g_prefetch_tid;

__attribute__((unused))
static void infer_prefetch_init(void) {
    if (g_prefetch) return;
    g_prefetch = calloc(1, sizeof(InferPrefetchCtx));
    pthread_mutex_init(&g_prefetch->mutex, NULL);
    pthread_cond_init(&g_prefetch->cond, NULL);
    g_prefetch->shutdown = 0;
    pthread_create(&g_prefetch_tid, NULL, infer_prefetch_thread_fn, g_prefetch);
}

__attribute__((unused))
static void infer_prefetch_shutdown(void) {
    if (!g_prefetch) return;
    pthread_mutex_lock(&g_prefetch->mutex);
    g_prefetch->shutdown = 1;
    pthread_cond_signal(&g_prefetch->cond);
    pthread_mutex_unlock(&g_prefetch->mutex);
    pthread_join(g_prefetch_tid, NULL);
    pthread_mutex_destroy(&g_prefetch->mutex);
    pthread_cond_destroy(&g_prefetch->cond);
    free(g_prefetch);
    g_prefetch = NULL;
}

// ============================================================================
// Per-layer weight pointer cache — built once, eliminates 40+ snprintf+lookup
// per layer per token. With 60 layers and 15 tokens = 36,000 lookups saved.
// ============================================================================

typedef struct {
    // Input/post-attention layer norms
    uint16_t *input_norm_w;
    uint16_t *post_attn_norm_w;

    // Full attention weights (non-NULL only for full attention layers)
    uint32_t *q_w; uint16_t *q_s, *q_b;
    uint32_t *k_w; uint16_t *k_s, *k_b;
    uint32_t *v_w; uint16_t *v_s, *v_b;
    uint32_t *o_w; uint16_t *o_s, *o_b;
    uint16_t *q_norm_w, *k_norm_w;

    // Linear attention weights (non-NULL only for linear attention layers)
    uint32_t *qkv_w; uint16_t *qkv_s, *qkv_b;
    uint32_t *z_w;   uint16_t *z_s, *z_b;
    uint32_t *b_w;   uint16_t *b_s, *b_b;
    uint32_t *a_w;   uint16_t *a_s, *a_b;
    uint16_t *conv1d_w;
    float *A_log;
    uint16_t *dt_bias;
    uint16_t *gated_norm_w;
    uint32_t *out_proj_w; uint16_t *out_proj_s, *out_proj_b;

    // MoE routing + shared expert weights
    uint32_t *gate_w; uint16_t *gate_s, *gate_b;
    uint32_t *sg_w;   uint16_t *sg_s, *sg_b;   // shared gate_proj
    uint32_t *su_w;   uint16_t *su_s, *su_b;   // shared up_proj
    uint32_t *sd_w;   uint16_t *sd_s, *sd_b;   // shared down_proj
    uint32_t *seg_w;  uint16_t *seg_s, *seg_b; // shared_expert_gate
} LayerWeightCache;

static LayerWeightCache layer_cache[MAX_NUM_LAYERS];
static int layer_cache_built = 0;

static void build_layer_cache(WeightFile *wf) {
    if (layer_cache_built) return;
    char name[256];

    for (int i = 0; i < g_cfg.num_layers; i++) {
        LayerWeightCache *lc = &layer_cache[i];
        int is_full = ((i + 1) % g_cfg.full_attn_interval == 0);

        // Norms
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        lc->input_norm_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        lc->post_attn_norm_w = get_tensor_ptr(wf, name);

        if (is_full) {
            // Full attention
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            lc->q_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", i);
            lc->q_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", i);
            lc->q_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            lc->k_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", i);
            lc->k_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", i);
            lc->k_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            lc->v_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", i);
            lc->v_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", i);
            lc->v_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            lc->o_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", i);
            lc->o_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", i);
            lc->o_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
            lc->q_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
            lc->k_norm_w = get_tensor_ptr(wf, name);
        } else {
            // Linear attention
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            lc->qkv_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", i);
            lc->qkv_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", i);
            lc->qkv_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            lc->z_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", i);
            lc->z_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", i);
            lc->z_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            lc->b_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", i);
            lc->b_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", i);
            lc->b_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            lc->a_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", i);
            lc->a_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", i);
            lc->a_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", i);
            lc->conv1d_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", i);
            lc->A_log = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", i);
            lc->dt_bias = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", i);
            lc->gated_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", i);
            lc->out_proj_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", i);
            lc->out_proj_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", i);
            lc->out_proj_b = get_tensor_ptr(wf, name);
        }

        // MoE weights (same for all layers)
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", i);
        lc->gate_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", i);
        lc->gate_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", i);
        lc->gate_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
        lc->sg_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", i);
        lc->sg_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", i);
        lc->sg_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
        lc->su_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", i);
        lc->su_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", i);
        lc->su_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
        lc->sd_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", i);
        lc->sd_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", i);
        lc->sd_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", i);
        lc->seg_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", i);
        lc->seg_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", i);
        lc->seg_b = get_tensor_ptr(wf, name);
    }

    layer_cache_built = 1;
    if (g_timing_enabled) {
        printf("[cache] Pre-computed weight pointers for %d layers\n", g_cfg.num_layers);
    }
}

// ============================================================================
// Deferred expert state: holds state for async GPU expert compute.
// GPU experts are submitted async (commit without wait), and the wait+combine
// happens at the start of the NEXT layer. This overlaps ~1ms of GPU expert
// compute with the next layer's attention+routing CPU/GPU work.
// ============================================================================

typedef struct {
    int active;                         // 1 if there's a deferred GPU expert to wait for
    int gpu_combined;                   // 1 if CMD3 includes combine+residual+norm on GPU
                                        // (next layer can skip deferred_wait+finalize+input_norm
                                        //  and submit CMD1 immediately -- buf_input is ready)
    id<MTLCommandBuffer> cmd_experts;   // the async command buffer (committed but not waited)
    float expert_weights[MAX_K];        // routing weights for weighted accumulation
    int valid[MAX_K];                   // which experts loaded successfully
    int actual_K;                       // number of experts
    float h_mid[MAX_HIDDEN_DIM];            // saved h_mid for final combine
    float shared_gate_score;            // saved shared expert gate score
    float *hidden;                      // pointer to hidden state (for writing final result)
    int layer_idx;                      // which layer produced this deferred state
} DeferredExpertState;

static DeferredExpertState g_deferred = { .active = 0 };

// Wait for the deferred GPU expert command buffer to complete.
// Split from finalize so timing can be measured independently.
static void wait_deferred_experts_gpu(void) {
    if (!g_deferred.active) return;
    [g_deferred.cmd_experts waitUntilCompleted];
}

// CPU readback + accumulate + combine after GPU is done.
// Must be called after wait_deferred_experts_gpu().
// When gpu_combined=1, the GPU already computed the combine+residual+norm
// in CMD3, so we just need to read back the hidden state from buf_moe_hidden.
static void finalize_deferred_experts(void) {
    if (!g_deferred.active) return;

    if (g_deferred.gpu_combined) {
        // GPU-side combine: hidden state is already in buf_moe_hidden.
        // buf_input already has the normalized input for the next layer's CMD1.
        // Just read back hidden (needed for the residual connection in future layers).
        memcpy(g_deferred.hidden, [g_metal->buf_moe_hidden contents],
               g_cfg.hidden_dim * sizeof(float));
    } else {
        // CPU-side combine (original path)
        // Read back and accumulate routed expert outputs
        float moe_out[g_cfg.hidden_dim];
        memset(moe_out, 0, sizeof(moe_out));
        for (int k = 0; k < g_deferred.actual_K; k++) {
            if (!g_deferred.valid[k]) continue;
            float *expert_result = (float *)[g_metal->buf_multi_expert_out[k] contents];
            cpu_vec_madd(moe_out, expert_result, g_deferred.expert_weights[k], g_cfg.hidden_dim);
        }

        // Read shared expert result
        float shared_out[g_cfg.hidden_dim];
        memcpy(shared_out, [g_metal->buf_shared_out contents], g_cfg.hidden_dim * sizeof(float));

        // Apply shared expert gate
        float shared_weight = cpu_sigmoid(g_deferred.shared_gate_score);
        for (int i = 0; i < g_cfg.hidden_dim; i++) {
            shared_out[i] *= shared_weight;
        }

        // Final combine: hidden = h_mid + moe_out + shared_out
        for (int i = 0; i < g_cfg.hidden_dim; i++) {
            g_deferred.hidden[i] = g_deferred.h_mid[i] + moe_out[i] + shared_out[i];
        }
    }

    g_deferred.active = 0;
    g_deferred.gpu_combined = 0;
    g_deferred.cmd_experts = nil;
}

// Complete the deferred GPU expert compute: wait for GPU, read back, accumulate, combine.
// Must be called before the next layer modifies static scratch buffers.
static void complete_deferred_experts(void) {
    wait_deferred_experts_gpu();
    finalize_deferred_experts();
}

// Discard the deferred GPU expert result: wait for GPU to finish (for buffer safety)
// but skip the CPU readback/combine. Used during prefill for intermediate tokens
// where the hidden state will be immediately overwritten by the next token's embedding.
// This saves ~0.1-0.2ms per prefill token (avoids unnecessary memcpy + combine work).
static void discard_deferred_experts(void) {
    wait_deferred_experts_gpu();
    // Clear deferred state without reading back results
    if (g_deferred.active) {
        g_deferred.active = 0;
        g_deferred.gpu_combined = 0;
        g_deferred.cmd_experts = nil;
    }
}

// ============================================================================
// Fused layer forward: GPU/CPU overlap + deferred expert pipeline
//
// Pipeline per layer (3 cmd buffers, GPU-side combine in CMD3):
//
//   FAST PATH (when previous CMD3 did GPU-side combine):
//     CMD1: submit immediately (buf_input already populated by CMD3(N-1))
//     WAIT: CMD1 complete (implies CMD3(N-1) also done, queue is serial)
//     CPU:  finalize deferred (read back hidden from buf_moe_hidden)
//
//   SLOW PATH (first layer, or last layer's CMD3 without GPU combine):
//     [DEFERRED] Wait for PREVIOUS layer's CMD3 (if any) + CPU combine
//     CPU:  input_norm(hidden) -> normed -> buf_input
//     CMD1: attention projections (commit)
//     WAIT: CMD1 complete
//
//   Then (both paths):
//     CPU:  attention compute (RoPE/softmax/delta-net)
//     CMD2: o_proj + residual + norm + routing + shared expert projs (8 encoders, 1 commit)
//     WAIT: CMD2 complete
//     CPU:  softmax + top-K routing
//     I/O:  parallel pread K experts (4 pthreads)
//     CMD3: K expert forwards + shared SwiGLU + shared down
//           + moe_combine_residual + rms_norm -> buf_input (ASYNC commit, NO wait)
//     RETURN: GPU experts + combine running async
//
// GPU-side combine eliminates the 0.83ms deferred_wait + CPU combine + input_norm
// at the start of each layer, allowing CMD1 to be submitted immediately.
//
// Key optimizations:
//   1. Parallel pread (4 threads) instead of sequential: ~4x I/O speedup
//   2. o_proj fused into CMD2 with routing (saves 1 commit+wait)
//   3. Deferred CMD3 (expert GPU compute overlapped with next layer)
//   4. GPU-side combine in CMD3 (eliminates CPU deferred_wait + combine + norm)
// ============================================================================

// Static scratch buffers — allocated once, reused across all 60 layers per token.
// Eliminates ~20 malloc/free per layer = ~1200 alloc/free per token.
static float *s_normed    = NULL;   // [g_cfg.hidden_dim]
static float *s_residual  = NULL;   // [g_cfg.hidden_dim]
static float *s_attn_proj = NULL;   // [g_cfg.hidden_dim]
static float *s_h_post    = NULL;   // [g_cfg.hidden_dim]
static float *s_h_mid     = NULL;   // [g_cfg.hidden_dim]
static float *s_gate_scores = NULL; // [g_cfg.num_experts]
static float *s_spec_gate_scores = NULL; // [g_cfg.num_experts] speculative routing scratch
static int s_spec_indices[MAX_K];         // speculative routing predicted expert indices
static int s_spec_count = 0;              // number of speculative predictions this layer
static float *s_shared_gate = NULL; // [g_cfg.shared_intermediate]
static float *s_shared_up  = NULL;  // [g_cfg.shared_intermediate]
static float *s_moe_out   = NULL;   // [g_cfg.hidden_dim]
static float *s_shared_out = NULL;  // [g_cfg.hidden_dim]
// Full attention scratch
static float *s_q_proj_out = NULL;  // [g_cfg.num_attn_heads * g_cfg.head_dim * 2]
static float *s_k_proj_out = NULL;  // [g_cfg.num_kv_heads * g_cfg.head_dim]
static float *s_v_proj_out = NULL;  // [g_cfg.num_kv_heads * g_cfg.head_dim]
static float *s_q         = NULL;   // [g_cfg.num_attn_heads * g_cfg.head_dim]
static float *s_q_gate    = NULL;   // [g_cfg.num_attn_heads * g_cfg.head_dim]
static float *s_attn_out  = NULL;   // [g_cfg.num_attn_heads * g_cfg.head_dim]
// Linear attention scratch
static float *s_qkv_proj_out = NULL;   // [g_cfg.linear_conv_dim]
static float *s_z_proj_out   = NULL;   // [g_cfg.linear_total_value]
static float *s_beta_proj_out = NULL;  // [g_cfg.linear_num_v_heads]
static float *s_alpha_proj_out = NULL; // [g_cfg.linear_num_v_heads]
static float *s_conv_out  = NULL;   // [g_cfg.linear_conv_dim]
static float *s_out_vals  = NULL;   // [g_cfg.linear_total_value]
static float *s_gated_out = NULL;   // [g_cfg.linear_total_value]

static void init_layer_scratch(void) {
    if (s_normed) return;  // already initialized
    s_normed     = calloc(g_cfg.hidden_dim, sizeof(float));
    s_residual   = calloc(g_cfg.hidden_dim, sizeof(float));
    s_attn_proj  = calloc(g_cfg.hidden_dim, sizeof(float));
    s_h_post     = calloc(g_cfg.hidden_dim, sizeof(float));
    s_h_mid      = calloc(g_cfg.hidden_dim, sizeof(float));
    s_gate_scores = calloc(g_cfg.num_experts, sizeof(float));
    s_spec_gate_scores = calloc(g_cfg.num_experts, sizeof(float));
    s_shared_gate = calloc(g_cfg.shared_intermediate, sizeof(float));
    s_shared_up  = calloc(g_cfg.shared_intermediate, sizeof(float));
    s_moe_out    = calloc(g_cfg.hidden_dim, sizeof(float));
    s_shared_out = calloc(g_cfg.hidden_dim, sizeof(float));
    s_q_proj_out = calloc(g_cfg.num_attn_heads * g_cfg.head_dim * 2, sizeof(float));
    s_k_proj_out = calloc(g_cfg.num_kv_heads * g_cfg.head_dim, sizeof(float));
    s_v_proj_out = calloc(g_cfg.num_kv_heads * g_cfg.head_dim, sizeof(float));
    s_q          = calloc(g_cfg.num_attn_heads * g_cfg.head_dim, sizeof(float));
    s_q_gate     = calloc(g_cfg.num_attn_heads * g_cfg.head_dim, sizeof(float));
    s_attn_out   = calloc(g_cfg.num_attn_heads * g_cfg.head_dim, sizeof(float));
    s_qkv_proj_out = calloc(g_cfg.linear_conv_dim, sizeof(float));
    s_z_proj_out   = calloc(g_cfg.linear_total_value, sizeof(float));
    s_beta_proj_out = calloc(g_cfg.linear_num_v_heads, sizeof(float));
    s_alpha_proj_out = calloc(g_cfg.linear_num_v_heads, sizeof(float));
    s_conv_out   = calloc(g_cfg.linear_conv_dim, sizeof(float));
    s_out_vals   = calloc(g_cfg.linear_total_value, sizeof(float));
    s_gated_out  = calloc(g_cfg.linear_total_value, sizeof(float));
}

// Dense FFN (Qwen3.6-27B dense model, num_experts==0): no routing, no experts,
// no shared expert — just hidden = h_mid + down_proj(swiglu(gate_proj(h_post), up_proj(h_post))).
// All matvecs use matvec_fast (in_dim 5120/17408 > 4096 v3 limit). h_post is the
// post-attention-norm of h_mid; output written into hidden_out.
static void dense_mlp_forward(WeightFile *wf, int layer, const float *h_post,
                              const float *h_mid, float *hidden_out) {
    int H = g_cfg.hidden_dim, inter = g_cfg.dense_intermediate, gsz = g_cfg.group_size;
    char nm[256];
    #define DW(suf) (snprintf(nm,sizeof(nm),"model.layers.%d.mlp." suf,layer), get_tensor_ptr(wf,nm))
    uint32_t *gw=DW("gate_proj.weight"); uint16_t *gs=DW("gate_proj.scales"),*gb=DW("gate_proj.biases");
    uint32_t *uw=DW("up_proj.weight");   uint16_t *us=DW("up_proj.scales"),  *ub=DW("up_proj.biases");
    uint32_t *dw=DW("down_proj.weight"); uint16_t *ds=DW("down_proj.scales"),*db=DW("down_proj.biases");
    #undef DW
    float *g=malloc(inter*sizeof(float)),*u=malloc(inter*sizeof(float)),*a=malloc(inter*sizeof(float)),*o=malloc(H*sizeof(float));
    fast_dequant_matvec(gw,gs,gb,h_post,g,inter,H,gsz);
    fast_dequant_matvec(uw,us,ub,h_post,u,inter,H,gsz);
    cpu_swiglu(g,u,a,inter);
    fast_dequant_matvec(dw,ds,db,a,o,H,inter,gsz);
    for (int i=0;i<H;i++) hidden_out[i] = h_mid[i] + o[i];
    free(g); free(u); free(a); free(o);
}

static void fused_layer_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [g_cfg.hidden_dim] in/out
    KVCache *kv,             // non-NULL for full attention layers
    LinearAttnState *la_state, // non-NULL for linear attention layers
    int pos,                 // position for RoPE
    const void *mmap_base,   // mmap'd layer file (NULL if not available)
    int K,                   // number of active experts
    int packed_fd            // fd for packed expert file
) {
    double t_layer_start = 0, t0 = 0, t1 = 0;
    if (g_timing_enabled) { t_layer_start = now_ms(); }
    int pred_started = 0;  // set to 1 if we started prediction preads during CMD1_wait

    init_layer_scratch();
    if (!layer_cache_built) build_layer_cache(wf);
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int is_full = (kv != NULL);

    // =====================================================================
    // PHASE 1: Deferred completion + CMD1 (attention projections)
    // =====================================================================

    // ---- Prepare attention projection specs (doesn't depend on hidden) ----
    int num_attn_specs = 0;
    BatchMatvecSpec attn_specs[5];
    float *q_proj_out = NULL, *k_out = NULL, *v_out = NULL;
    float *qkv_out = NULL, *z_out = NULL, *beta_out = NULL, *alpha_out = NULL;

    if (is_full) {
        int q_proj_dim = g_cfg.num_attn_heads * g_cfg.head_dim * 2;
        int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;

        q_proj_out = s_q_proj_out;
        k_out = s_k_proj_out;
        v_out = s_v_proj_out;

        if (lc->q_w && lc->q_s && lc->q_b && lc->k_w && lc->k_s && lc->k_b &&
            lc->v_w && lc->v_s && lc->v_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->q_w, lc->q_s, lc->q_b, q_proj_out, (uint32_t)q_proj_dim, g_cfg.hidden_dim, g_cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->k_w, lc->k_s, lc->k_b, k_out,      (uint32_t)kv_dim,     g_cfg.hidden_dim, g_cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->v_w, lc->v_s, lc->v_b, v_out,      (uint32_t)kv_dim,     g_cfg.hidden_dim, g_cfg.group_size, 2 };
            num_attn_specs = 3;
        }
    } else {
        int qkv_dim = g_cfg.linear_conv_dim;
        int z_dim = g_cfg.linear_total_value;

        qkv_out = s_qkv_proj_out;
        z_out = s_z_proj_out;
        beta_out = s_beta_proj_out;
        alpha_out = s_alpha_proj_out;

        if (lc->qkv_w && lc->qkv_s && lc->qkv_b && lc->z_w && lc->z_s && lc->z_b &&
            lc->b_w && lc->b_s && lc->b_b && lc->a_w && lc->a_s && lc->a_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->qkv_w, lc->qkv_s, lc->qkv_b, qkv_out,   (uint32_t)qkv_dim,            g_cfg.hidden_dim, g_cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->z_w,   lc->z_s,   lc->z_b,   z_out,      (uint32_t)z_dim,              g_cfg.hidden_dim, g_cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->b_w,   lc->b_s,   lc->b_b,   beta_out,   (uint32_t)g_cfg.linear_num_v_heads, g_cfg.hidden_dim, g_cfg.group_size, 2 };
            attn_specs[3] = (BatchMatvecSpec){ lc->a_w,   lc->a_s,   lc->a_b,   alpha_out,  (uint32_t)g_cfg.linear_num_v_heads, g_cfg.hidden_dim, g_cfg.group_size, 3 };
            num_attn_specs = 4;
        }
    }

    // ---- Deferred completion + CMD1 (sequential) ----
    float *normed = s_normed;
    float *residual = s_residual;
    id<MTLCommandBuffer> cmd1 = nil;
    int gpu_linear_attn = 0;  // set to 1 if GPU handles entire linear attention pipeline

    // Pre-compute linear_layer_idx for GPU linear attention encoding in CMD1
    int linear_layer_idx = -1;
    if (!is_full) {
        linear_layer_idx = layer_idx - (layer_idx + 1) / g_cfg.full_attn_interval;
    }
    // Can we run the full linear attention pipeline on GPU in CMD1?
    int can_gpu_linear = (gpu_linear_attn_enabled &&
                          !is_full && g_metal && g_metal->delta_net_step &&
                          g_metal->conv1d_step && g_metal->rms_norm_qk &&
                          g_metal->compute_decay_beta && g_metal->gated_rms_norm &&
                          g_metal->wf_buf &&
                          linear_layer_idx >= 0 && linear_layer_idx < g_cfg.num_linear_layers &&
                          lc->conv1d_w && lc->A_log && lc->dt_bias && lc->gated_norm_w &&
                          !linear_attn_bypass);

    // Check if previous layer's CMD3 already computed combine+residual+norm on GPU.
    // If so, buf_input already contains the normalized input for this layer's CMD1.
    // We can submit CMD1 immediately — the GPU queue serializes CMD3(N-1) then CMD1(N).
    int prev_gpu_combined = (g_deferred.active && g_deferred.gpu_combined);

    if (prev_gpu_combined && g_metal && g_metal->wf_buf && num_attn_specs > 0) {
        // ---- FAST PATH: GPU-combined previous CMD3 ----
        // buf_input already has the normalized hidden state from CMD3(N-1).
        // Submit CMD1 immediately — GPU runs CMD3(N-1) then CMD1(N) back-to-back.
        if (g_timing_enabled) { t0 = now_ms(); }

        cmd1 = [g_metal->queue commandBuffer];
        gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

        // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
        if (can_gpu_linear && num_attn_specs == 4) {
            // batch_out[0]=qkv(12288), [1]=z(8192), [2]=beta(64), [3]=alpha(64)
            uint32_t conv_dim = g_cfg.linear_conv_dim;
            NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[g_metal->wf_buf contents]);

            // Enc L1: conv1d_step — input=batch_out[0], weights=conv1d_w, state=buf_conv_state, output=buf_conv_output
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->conv1d_step];
                [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1]; // qkv projection output
                [enc setBuffer:g_metal->wf_buf          offset:conv_w_off   atIndex:2]; // conv weights (bf16)
                [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3]; // conv output
                [enc setBytes:&conv_dim length:4 atIndex:4];
                uint32_t tgs = (conv_dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc L2: rms_norm_qk — normalize q and k in conv_output in-place
            {
                uint32_t key_dim = g_cfg.linear_key_dim;  // 128
                float inv_scale = 1.0f / sqrtf((float)g_cfg.linear_key_dim);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->rms_norm_qk];
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];  // q at offset 0
                [enc setBuffer:g_metal->buf_conv_output offset:g_cfg.linear_total_key * sizeof(float) atIndex:1];  // k at offset 2048 floats
                [enc setBytes:&key_dim   length:4 atIndex:2];
                [enc setBytes:&inv_scale length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(g_cfg.linear_num_k_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(g_cfg.linear_key_dim, 1, 1)];
                [enc endEncoding];
            }

            // Enc L3: compute_decay_beta — alpha=batch_out[3], beta=batch_out[2], A_log+dt_bias from wf_buf
            {
                NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[g_metal->wf_buf contents]);
                NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[g_metal->wf_buf contents]);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->compute_decay_beta];
                [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0]; // alpha
                [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1]; // beta
                [enc setBuffer:g_metal->wf_buf             offset:a_log_off  atIndex:2]; // A_log
                [enc setBuffer:g_metal->wf_buf             offset:dt_bias_off atIndex:3]; // dt_bias (bf16)
                [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4]; // g_decay output
                [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5]; // beta_gate output
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(g_cfg.linear_num_v_heads, 1, 1)];
                [enc endEncoding];
            }

            // Enc L4: gated_delta_net_step — the main recurrence
            {
                uint32_t khpv = g_cfg.linear_num_v_heads / g_cfg.linear_num_k_heads;  // 4
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0]; // persistent state
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1]; // q (first 2048 floats)
                [enc setBuffer:g_metal->buf_conv_output offset:g_cfg.linear_total_key * sizeof(float) atIndex:2]; // k (next 2048)
                [enc setBuffer:g_metal->buf_conv_output offset:2 * g_cfg.linear_total_key * sizeof(float) atIndex:3]; // v (next 8192)
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6]; // output [8192]
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(g_cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
            }

            // Enc L5: gated_rms_norm — normalize+gate delta-net output -> batch_out[6] for CMD2 o_proj
            {
                NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[g_metal->wf_buf contents]);
                uint32_t value_dim = g_cfg.linear_value_dim;  // 128
                float eps = g_cfg.rms_norm_eps;
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->gated_rms_norm];
                [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0]; // values [8192]
                [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1]; // z (z projection output) [8192]
                [enc setBuffer:g_metal->wf_buf           offset:gnorm_w_off atIndex:2]; // weight (bf16)
                [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3]; // output -> batch_out[6] for CMD2
                [enc setBytes:&value_dim length:4 atIndex:4];
                [enc setBytes:&eps       length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(g_cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(g_cfg.linear_value_dim, 1, 1)];
                [enc endEncoding];
            }

            gpu_linear_attn = 1;
        }

        [cmd1 commit];

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1 (implies CMD3(N-1) also done, since queue is serial)
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd1 waitUntilCompleted];
        if (!gpu_linear_attn) {
            gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }

        // Now CMD3(N-1) is done. Read back hidden state from GPU.
        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();  // reads buf_moe_hidden -> hidden

        // Start predicted expert preads AFTER CMD1_wait.
        // CMD3(N-1) is guaranteed done (serial queue), so buf_B is safe to overwrite.
        // Predictions overlap with CPU attn + CMD2 + routing (~0.6ms head start).
        // Predicted experts that hit page cache (same as previous token) complete in ~0.1ms.
        if (g_pred_enabled && g_pred_generating && g_pred_valid && packed_fd >= 0 &&
            g_metal->buf_multi_expert_data_B[0] && g_pred_count[layer_idx] > 0) {
            async_pread_start(packed_fd, g_pred_experts[layer_idx],
                              g_pred_count[layer_idx],
                              g_metal->buf_multi_expert_data_B, mmap_base);
            pred_started = 1;
        }
        // Set up residual for CMD2 (residual = hidden before this layer's attention)
        cpu_vec_copy(residual, hidden, g_cfg.hidden_dim);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // No input_norm needed — CMD3 already computed it into buf_input.
        // normed is only needed if speculative routing is enabled (currently disabled).
        // Skip the readback to avoid unnecessary overhead.
    } else {
        // ---- ORIGINAL PATH: CPU deferred completion + input norm ----
        // Complete deferred experts from previous layer
        if (g_timing_enabled) { t0 = now_ms(); }
        wait_deferred_experts_gpu();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_wait += t1 - t0; }

        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // Input norm
        if (g_timing_enabled) { t0 = now_ms(); }
        cpu_vec_copy(residual, hidden, g_cfg.hidden_dim);
        cpu_rms_norm(hidden, lc->input_norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.input_norm += t1 - t0; }

        // Submit CMD1: attention projections
        if (g_timing_enabled) { t0 = now_ms(); }
        if (g_metal && g_metal->wf_buf && num_attn_specs > 0) {
            memcpy([g_metal->buf_input contents], normed, g_cfg.hidden_dim * sizeof(float));
            cmd1 = [g_metal->queue commandBuffer];
            gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

            // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
            if (can_gpu_linear && num_attn_specs == 4) {
                uint32_t conv_dim = g_cfg.linear_conv_dim;
                NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[g_metal->wf_buf contents]);

                // Enc L1: conv1d_step
                {
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->conv1d_step];
                    [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1];
                    [enc setBuffer:g_metal->wf_buf          offset:conv_w_off   atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3];
                    [enc setBytes:&conv_dim length:4 atIndex:4];
                    uint32_t tgs = (conv_dim + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L2: rms_norm_qk
                {
                    uint32_t key_dim = g_cfg.linear_key_dim;
                    float inv_scale = 1.0f / sqrtf((float)g_cfg.linear_key_dim);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->rms_norm_qk];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:g_cfg.linear_total_key * sizeof(float) atIndex:1];
                    [enc setBytes:&key_dim   length:4 atIndex:2];
                    [enc setBytes:&inv_scale length:4 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(g_cfg.linear_num_k_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(g_cfg.linear_key_dim, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L3: compute_decay_beta
                {
                    NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[g_metal->wf_buf contents]);
                    NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[g_metal->wf_buf contents]);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->compute_decay_beta];
                    [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1];
                    [enc setBuffer:g_metal->wf_buf             offset:a_log_off  atIndex:2];
                    [enc setBuffer:g_metal->wf_buf             offset:dt_bias_off atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(g_cfg.linear_num_v_heads, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L4: gated_delta_net_step
                {
                    uint32_t khpv = g_cfg.linear_num_v_heads / g_cfg.linear_num_k_heads;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->delta_net_step];
                    [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_conv_output offset:g_cfg.linear_total_key * sizeof(float) atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:2 * g_cfg.linear_total_key * sizeof(float) atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                    [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                    [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(g_cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L5: gated_rms_norm -> batch_out[6]
                {
                    NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[g_metal->wf_buf contents]);
                    uint32_t value_dim = g_cfg.linear_value_dim;
                    float eps = g_cfg.rms_norm_eps;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->gated_rms_norm];
                    [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1];
                    [enc setBuffer:g_metal->wf_buf           offset:gnorm_w_off atIndex:2];
                    [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3];
                    [enc setBytes:&value_dim length:4 atIndex:4];
                    [enc setBytes:&eps       length:4 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(g_cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(g_cfg.linear_value_dim, 1, 1)];
                    [enc endEncoding];
                }

                gpu_linear_attn = 1;
            }

            [cmd1 commit];
        } else {
            for (int i = 0; i < num_attn_specs; i++) {
                BatchMatvecSpec *s = &attn_specs[i];
                cpu_dequant_matvec(s->W, s->scales, s->biases, normed, s->out_cpu,
                                   s->out_dim, s->in_dim, s->group_size);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1
        if (g_timing_enabled) { t0 = now_ms(); }
        if (cmd1) {
            [cmd1 waitUntilCompleted];
            if (!gpu_linear_attn) {
                gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }
    }

    // =====================================================================
    // SPECULATIVE EARLY ROUTING — overlap expert I/O with CPU attention
    // =====================================================================
    // Compute approximate routing using the PRE-attention normed hidden state.
    // The real routing (in CMD2/PHASE 3) uses the POST-attention state, so this
    // is an approximation. Fire off async pread for predicted cache misses via
    // dispatch_group so the I/O runs concurrently with CPU attention compute.
    // After CPU attention, we wait for the group to finish. When the real routing
    // happens later, predicted experts are already in the LRU cache as hits.

    dispatch_group_t spec_group = NULL;
    int spec_preload_count = 0;
    int spec_routing_enabled = 0;  // DISABLED: cache pollution + overhead makes it slower

    if (g_timing_enabled) { t0 = now_ms(); }
    s_spec_count = 0;

    if (spec_routing_enabled && (g_expert_cache || g_malloc_cache) && packed_fd >= 0 && lc->gate_w) {
        float *spec_scores = s_spec_gate_scores;
        memset(spec_scores, 0, g_cfg.num_experts * sizeof(float));

        // Gate projection matvec on pre-attention normed input (CPU, ~0.1ms for 512x4096)
        cpu_dequant_matvec(lc->gate_w, lc->gate_s, lc->gate_b,
                           normed, spec_scores,
                           g_cfg.num_experts, g_cfg.hidden_dim, g_cfg.group_size);
        cpu_softmax(spec_scores, g_cfg.num_experts);

        int spec_K = (K > MAX_K) ? MAX_K : K;
        float spec_weights[MAX_K];
        cpu_topk(spec_scores, g_cfg.num_experts, spec_K, s_spec_indices, spec_weights);
        s_spec_count = spec_K;

        g_spec_route_attempts += spec_K;

        // Initialize GCD queue if needed
        if (!g_io_gcd_queue)
            g_io_gcd_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // Check cache for each predicted expert, start async I/O for misses
        size_t spec_esz = active_expert_size();
        if (g_malloc_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, eidx);
                if (!cached) {
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, eidx, &cidx);
                    if (buf && cidx >= 0) {
                        int fd_copy = packed_fd;
                        void *dst = g_malloc_cache->data[cidx];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        } else if (g_expert_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, eidx);
                if (!cached) {
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, eidx);
                    if (buf) {
                        int fd_copy = packed_fd;
                        void *dst = [buf contents];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        }
    }
    (void)spec_preload_count;  // tracked via g_spec_route_preloads

    if (g_timing_enabled) { t1 = now_ms(); g_timing.spec_route += t1 - t0; }

    // =====================================================================
    // PHASE 2: CPU attention compute
    // =====================================================================

    if (g_timing_enabled) { t0 = now_ms(); }

    float *attn_projected = s_attn_proj;
    memset(attn_projected, 0, g_cfg.hidden_dim * sizeof(float));

    // Pre-lookup o_proj / out_proj weights (used after attention compute)
    // These are looked up NOW to avoid repeated snprintf later.
    uint32_t *oproj_w = NULL;
    uint16_t *oproj_s = NULL, *oproj_b = NULL;
    int oproj_in_dim = 0;

    if (is_full) {
        oproj_w = lc->o_w; oproj_s = lc->o_s; oproj_b = lc->o_b;
        oproj_in_dim = g_cfg.num_attn_heads * g_cfg.head_dim;
    } else if (!linear_attn_bypass) {
        oproj_w = lc->out_proj_w; oproj_s = lc->out_proj_s; oproj_b = lc->out_proj_b;
        oproj_in_dim = g_cfg.linear_total_value;
    }

    // All MoE weight pointers from cache (zero snprintf overhead)
    uint32_t *gate_w = lc->gate_w; uint16_t *gate_s = lc->gate_s, *gate_b = lc->gate_b;
    uint32_t *sgw = lc->sg_w;     uint16_t *sgs = lc->sg_s,       *sgb = lc->sg_b;
    uint32_t *suw = lc->su_w;     uint16_t *sus = lc->su_s,       *sub = lc->su_b;
    uint32_t *seg_w = lc->seg_w;  uint16_t *seg_s = lc->seg_s,   *seg_b = lc->seg_b;
    uint32_t *sdw = lc->sd_w;     uint16_t *sds = lc->sd_s,       *sdb = lc->sd_b;

    // ---- CPU attention compute (produces attn_out for o_proj) ----
    float *attn_out_for_oproj = NULL;

    if (is_full) {
        // ---- Full attention CPU compute ----
        int q_proj_dim = g_cfg.num_attn_heads * g_cfg.head_dim * 2;
        int q_dim = g_cfg.num_attn_heads * g_cfg.head_dim;
        int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
        (void)q_proj_dim;

        float *q = s_q;
        float *q_gate = s_q_gate;
        for (int h = 0; h < g_cfg.num_attn_heads; h++) {
            float *src = q_proj_out + h * (2 * g_cfg.head_dim);
            memcpy(q + h * g_cfg.head_dim, src, g_cfg.head_dim * sizeof(float));
            memcpy(q_gate + h * g_cfg.head_dim, src + g_cfg.head_dim, g_cfg.head_dim * sizeof(float));
        }

        // Q/K RMSNorm
        uint16_t *qnorm_w = lc->q_norm_w;
        uint16_t *knorm_w = lc->k_norm_w;
        if (qnorm_w) {
            for (int h = 0; h < g_cfg.num_attn_heads; h++) {
                float *qh = q + h * g_cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < g_cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / g_cfg.head_dim + g_cfg.rms_norm_eps);
                for (int i = 0; i < g_cfg.head_dim; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
        if (knorm_w) {
            for (int h = 0; h < g_cfg.num_kv_heads; h++) {
                float *kh = k_out + h * g_cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < g_cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / g_cfg.head_dim + g_cfg.rms_norm_eps);
                for (int i = 0; i < g_cfg.head_dim; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }

        // RoPE
        apply_rotary_emb(q, k_out, pos, g_cfg.num_attn_heads, g_cfg.num_kv_heads, g_cfg.head_dim, g_cfg.rotary_dim);

        // Update KV cache (CPU + GPU mirror)
        int cache_pos = kv->len;
        memcpy(kv->k_cache + cache_pos * kv_dim, k_out, kv_dim * sizeof(float));
        memcpy(kv->v_cache + cache_pos * kv_dim, v_out, kv_dim * sizeof(float));

        int fa_idx = (layer_idx + 1) / g_cfg.full_attn_interval - 1;
        if (g_metal && g_metal->attn_scores_pipe &&
            fa_idx >= 0 && fa_idx < g_cfg.num_full_attn_layers &&
            cache_pos < GPU_KV_SEQ) {
            memcpy((float *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                   k_out, kv_dim * sizeof(float));
            memcpy((float *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                   v_out, kv_dim * sizeof(float));
        }
        kv->len++;

        // Scaled dot-product attention (GQA) — GPU or CPU
        int heads_per_kv = g_cfg.num_attn_heads / g_cfg.num_kv_heads;
        float scale = 1.0f / sqrtf((float)g_cfg.head_dim);
        float *attn_out = s_attn_out;
        memset(attn_out, 0, q_dim * sizeof(float));

        // GPU attention: defer dispatches to CMD2 (fused into single cmd buffer).
        // Only enabled when seq_len >= 32 (below that, CPU is faster).
        // Dense (num_experts==0) never takes the fused CMD2 path that computes GPU
        // attention, so force CPU attention (else attn_out_for_oproj=NULL is never
        // consumed and the o_proj input is stale).
        int gpu_attn_ready = (g_metal && g_metal->attn_scores_pipe &&
                              fa_idx >= 0 && fa_idx < g_cfg.num_full_attn_layers &&
                              kv->len >= 32 && kv->len < GPU_KV_SEQ &&
                              g_cfg.num_experts > 0);

        if (gpu_attn_ready) {
            // Copy Q and gate to GPU; attention dispatches will be in CMD2
            memcpy([g_metal->buf_attn_q contents], q, q_dim * sizeof(float));
            memcpy([g_metal->buf_attn_gate contents], q_gate, q_dim * sizeof(float));
            // attn_out_for_oproj will be set to NULL below — CMD2 reads buf_attn_out
        } else {
            // CPU fallback
            for (int h = 0; h < g_cfg.num_attn_heads; h++) {
                int kv_h = h / heads_per_kv;
                float *qh = q + h * g_cfg.head_dim;
                float *scores = malloc(kv->len * sizeof(float));
                for (int p = 0; p < kv->len; p++) {
                    float *kp = kv->k_cache + p * kv_dim + kv_h * g_cfg.head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < g_cfg.head_dim; d++) dot += qh[d] * kp[d];
                    scores[p] = dot * scale;
                }
                cpu_softmax(scores, kv->len);
                float *oh = attn_out + h * g_cfg.head_dim;
                for (int p = 0; p < kv->len; p++) {
                    float *vp = kv->v_cache + p * kv_dim + kv_h * g_cfg.head_dim;
                    for (int d = 0; d < g_cfg.head_dim; d++) oh[d] += scores[p] * vp[d];
                }
                free(scores);
            }
            for (int i = 0; i < q_dim; i++) {
                float g = 1.0f / (1.0f + expf(-q_gate[i]));
                attn_out[i] *= g;
            }
        }

        if (gpu_attn_ready) {
            attn_out_for_oproj = NULL;  // signal CMD2 to use GPU buf_attn_out
        } else {
            attn_out_for_oproj = attn_out;
        }
        // q_proj_out, k_out, v_out, q, q_gate, attn_out are static scratch.
    } else if (gpu_linear_attn) {
        // ---- GPU linear attention: already computed in CMD1 ----
        // batch_out[6] already contains gated_rms_norm output (linear_total_value floats)
        if (g_cfg.num_experts == 0) {
            // Dense takes the non-fused fallback o_proj, which reads attn_out_for_oproj
            // directly — so copy the gated output into a real buffer (not a sentinel).
            memcpy(s_gated_out, [g_metal->batch_out[6] contents], g_cfg.linear_total_value * sizeof(float));
            attn_out_for_oproj = s_gated_out;
        } else {
            // MoE: fused CMD2 path reads batch_out[6] directly; sentinel just signals non-NULL.
            static float gpu_linear_sentinel;
            attn_out_for_oproj = &gpu_linear_sentinel;
        }
    } else {
        // ---- Linear attention CPU compute ----
        if (!linear_attn_bypass) {
            int qkv_dim = g_cfg.linear_conv_dim;

            // Conv1d step
            uint16_t *conv_w = lc->conv1d_w;
            float *conv_out = s_conv_out;
            memset(conv_out, 0, qkv_dim * sizeof(float));
            if (conv_w) {
                cpu_conv1d_step(la_state->conv_state, qkv_out, conv_w, conv_out,
                                qkv_dim, g_cfg.linear_conv_kernel_dim);
            }
            // Update conv state
            memmove(la_state->conv_state, la_state->conv_state + qkv_dim,
                    (g_cfg.linear_conv_kernel_dim - 2) * qkv_dim * sizeof(float));
            memcpy(la_state->conv_state + (g_cfg.linear_conv_kernel_dim - 2) * qkv_dim, qkv_out,
                   qkv_dim * sizeof(float));

            // Split into q, k, v
            float *lin_q = conv_out;
            float *lin_k = conv_out + g_cfg.linear_total_key;
            float *lin_v = conv_out + 2 * g_cfg.linear_total_key;

            // RMS normalize q and k
            float inv_scale = 1.0f / sqrtf((float)g_cfg.linear_key_dim);
            for (int h = 0; h < g_cfg.linear_num_k_heads; h++) {
                float *qh = lin_q + h * g_cfg.linear_key_dim;
                cpu_rms_norm_bare(qh, qh, g_cfg.linear_key_dim, 1e-6f);
                float q_scale = inv_scale * inv_scale;
                for (int d = 0; d < g_cfg.linear_key_dim; d++) qh[d] *= q_scale;
            }
            for (int h = 0; h < g_cfg.linear_num_k_heads; h++) {
                float *kh = lin_k + h * g_cfg.linear_key_dim;
                cpu_rms_norm_bare(kh, kh, g_cfg.linear_key_dim, 1e-6f);
                for (int d = 0; d < g_cfg.linear_key_dim; d++) kh[d] *= inv_scale;
            }

            // Gated delta net recurrence
            float *A_log = lc->A_log;
            uint16_t *dt_bias_bf16 = lc->dt_bias;

            float *out_values = s_out_vals;
            memset(out_values, 0, g_cfg.linear_total_value * sizeof(float));
            int k_heads_per_v = g_cfg.linear_num_v_heads / g_cfg.linear_num_k_heads;

            float g_decay[g_cfg.linear_num_v_heads];
            float beta_gate_arr[g_cfg.linear_num_v_heads];
            for (int vh = 0; vh < g_cfg.linear_num_v_heads; vh++) {
                float a_val = alpha_out[vh];
                float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
                float A_val = A_log ? expf(A_log[vh]) : 1.0f;
                float softplus_val = logf(1.0f + expf(a_val + dt_b));
                g_decay[vh] = expf(-A_val * softplus_val);
                beta_gate_arr[vh] = cpu_sigmoid(beta_out[vh]);
            }

            // Compute linear_layer_idx: count of non-full-attention layers before this one.
            // Full attention at (layer_idx+1) % 4 == 0, i.e. layers 3,7,11,...
            // linear_layer_idx = layer_idx - number_of_full_layers_at_or_before
            //                  = layer_idx - (layer_idx + 1) / g_cfg.full_attn_interval
            int linear_layer_idx = layer_idx - (layer_idx + 1) / g_cfg.full_attn_interval;

            // GPU delta-net path (falls back to CPU if pipeline unavailable)
            if (g_metal && g_metal->delta_net_step &&
                linear_layer_idx >= 0 && linear_layer_idx < g_cfg.num_linear_layers) {
                // Upload CPU-computed data to GPU scratch buffers
                memcpy([g_metal->buf_delta_q contents], lin_q, g_cfg.linear_total_key * sizeof(float));
                memcpy([g_metal->buf_delta_k contents], lin_k, g_cfg.linear_total_key * sizeof(float));
                memcpy([g_metal->buf_delta_v contents], lin_v, g_cfg.linear_total_value * sizeof(float));
                memcpy([g_metal->buf_delta_g_decay contents], g_decay, g_cfg.linear_num_v_heads * sizeof(float));
                memcpy([g_metal->buf_delta_beta contents], beta_gate_arr, g_cfg.linear_num_v_heads * sizeof(float));

                id<MTLCommandBuffer> cmd_dn = [g_metal->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd_dn computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_delta_q       offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_delta_k       offset:0 atIndex:2];
                [enc setBuffer:g_metal->buf_delta_v       offset:0 atIndex:3];
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                uint32_t khpv = (uint32_t)k_heads_per_v;
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(g_cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmd_dn commit];
                [cmd_dn waitUntilCompleted];

                // Read back GPU result
                memcpy(out_values, [g_metal->buf_delta_output contents], g_cfg.linear_total_value * sizeof(float));
            } else {
                // CPU delta-net with Accelerate BLAS
                for (int vh = 0; vh < g_cfg.linear_num_v_heads; vh++) {
                    int kh = vh / k_heads_per_v;
                    float g = g_decay[vh];
                    float b_gate = beta_gate_arr[vh];
                    float *S = la_state->ssm_state + vh * g_cfg.linear_value_dim * g_cfg.linear_key_dim;
                    float *v_h = lin_v + vh * g_cfg.linear_value_dim;
                    float *k_h = lin_k + kh * g_cfg.linear_key_dim;

                    // Step 1: Decay S *= g (BLAS sscal on entire state matrix)
                    cblas_sscal(g_cfg.linear_value_dim * g_cfg.linear_key_dim, g, S, 1);

                    // Step 2: kv_mem = S @ k (each row dot k)
                    // S is [VALUE_DIM x KEY_DIM] row-major, k is [KEY_DIM]
                    // kv_mem[vi] = sum_ki(S[vi,ki] * k[ki]) = matrix-vector: S @ k
                    float kv_mem_vec[g_cfg.linear_value_dim];
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                g_cfg.linear_value_dim, g_cfg.linear_key_dim,
                                1.0f, S, g_cfg.linear_key_dim, k_h, 1,
                                0.0f, kv_mem_vec, 1);

                    // Step 3: delta = (v - kv_mem) * beta, then rank-1 update S += k * delta^T
                    // delta[vi] = (v[vi] - kv_mem[vi]) * beta
                    float delta_vec[g_cfg.linear_value_dim];
                    for (int vi = 0; vi < g_cfg.linear_value_dim; vi++) {
                        delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
                    }
                    // S += delta @ k^T (rank-1 update: sger)
                    // S[vi,ki] += delta[vi] * k[ki]
                    cblas_sger(CblasRowMajor, g_cfg.linear_value_dim, g_cfg.linear_key_dim,
                               1.0f, delta_vec, 1, k_h, 1, S, g_cfg.linear_key_dim);

                    // Step 4: output = S @ q (matrix-vector multiply)
                    float *q_h = lin_q + kh * g_cfg.linear_key_dim;
                    float *o_h = out_values + vh * g_cfg.linear_value_dim;
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                g_cfg.linear_value_dim, g_cfg.linear_key_dim,
                                1.0f, S, g_cfg.linear_key_dim, q_h, 1,
                                0.0f, o_h, 1);
                }
            }

            // RMSNormGated
            uint16_t *gated_norm_w = lc->gated_norm_w;
            float *gated_out = s_gated_out;
            memset(gated_out, 0, g_cfg.linear_total_value * sizeof(float));
            for (int vh = 0; vh < g_cfg.linear_num_v_heads; vh++) {
                float *oh = out_values + vh * g_cfg.linear_value_dim;
                float *zh = z_out + vh * g_cfg.linear_value_dim;
                float *gh = gated_out + vh * g_cfg.linear_value_dim;
                if (gated_norm_w) {
                    cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, g_cfg.linear_value_dim, g_cfg.rms_norm_eps);
                } else {
                    memcpy(gh, oh, g_cfg.linear_value_dim * sizeof(float));
                }
            }

            attn_out_for_oproj = gated_out;

            // conv_out, out_values are static — no free needed
            // gated_out is static — freed/released after CMD2 submission below
        }
        // else: linear_attn_bypass — attn_projected stays zero
        // qkv_out, z_out, beta_out, alpha_out are static scratch.
    }

    // =====================================================================
    // PHASE 3: FULLY FUSED CMD2 — o_proj + residual + norm + routing (1 cmd buffer)
    //   Eliminates 1 GPU round-trip vs old 2-buffer approach.
    //   GPU handles residual_add + rms_norm between o_proj and routing,
    //   so no CPU intervention is needed. 8 encoders, 1 commit+wait.
    //   Buffer flow: batch_out[6]->buf_output->buf_h_mid->buf_input->batch_out[0-3]
    // =====================================================================

    if (g_timing_enabled) { t1 = now_ms(); g_timing.cpu_attn += t1 - t0; }

    // Wait for speculative expert I/O to complete (overlapped with CPU attention)
    if (spec_group) {
        dispatch_group_wait(spec_group, DISPATCH_TIME_FOREVER);
        spec_group = NULL;  // ARC releases the group
    }

    if (g_timing_enabled) { t0 = now_ms(); }

    float *h_post = s_h_post;
    float *h_mid = s_h_mid;
    float *gate_scores = s_gate_scores;
    memset(gate_scores, 0, g_cfg.num_experts * sizeof(float));
    float *shared_gate = s_shared_gate;
    memset(shared_gate, 0, g_cfg.shared_intermediate * sizeof(float));
    float *shared_up = s_shared_up;
    memset(shared_up, 0, g_cfg.shared_intermediate * sizeof(float));
    float shared_gate_score = 0.0f;

    int have_moe_weights = (gate_w && gate_s && gate_b && sgw && sgs && sgb &&
                            suw && sus && sub && seg_w && seg_s && seg_b);

    // gpu_attn_fuse: attention dispatches fused into CMD2 (full-attn layers only).
    // Only enabled when seq_len >= 32 — below that, CPU attention is faster
    // because GPU command encoder overhead dominates at short sequences.
    int gpu_attn_fuse = (is_full && !attn_out_for_oproj && g_metal && g_metal->attn_scores_pipe
                         && kv && kv->len >= 32 && kv->len < GPU_KV_SEQ
                         && g_cfg.num_experts > 0);  // dense: no MoE-fused CMD2, use CPU attn + fallback

    if ((attn_out_for_oproj || gpu_attn_fuse) && oproj_w && oproj_s && oproj_b &&
        g_metal && g_metal->wf_buf && have_moe_weights &&
        g_metal->residual_add && g_metal->rms_norm_sum &&
        g_metal->rms_norm_apply_bf16 && lc->post_attn_norm_w) {
        // ---- FULLY FUSED CMD2 ----
        // For GPU attention (full-attn layers): attention dispatches are prepended,
        //   o_proj reads from buf_attn_out instead of batch_out[6].
        // For CPU attention / linear attn: o_proj reads from batch_out[6] as before.
        //
        // GPU attn path (12 encoders):
        //   Enc 1-4: attn_scores + softmax + values + sigmoid -> buf_attn_out
        //   Enc 5:   o_proj (buf_attn_out -> buf_output)
        //   Enc 6-8: residual + norm -> buf_input
        //   Enc 9-12: routing + shared expert
        //
        // CPU attn path (8 encoders, unchanged):
        //   Enc 1:   o_proj (batch_out[6] -> buf_output)
        //   Enc 2-4: residual + norm -> buf_input
        //   Enc 5-8: routing + shared expert

        if (!gpu_attn_fuse && !gpu_linear_attn) {
            // CPU/linear attn: copy attention output to GPU input buffer
            memcpy([g_metal->batch_out[6] contents], attn_out_for_oproj,
                   oproj_in_dim * sizeof(float));
        }
        // gpu_linear_attn: batch_out[6] already has the result from CMD1 gated_rms_norm
        // Copy residual into GPU buffer for residual_add kernel
        memcpy([g_metal->buf_residual contents], residual, g_cfg.hidden_dim * sizeof(float));

        attn_out_for_oproj = NULL;

        id<MTLCommandBuffer> cmd_fused = [g_metal->queue commandBuffer];

        // ---- GPU attention dispatches (only for full-attn layers with GPU path) ----
        if (gpu_attn_fuse) {
            int fa_idx = (layer_idx + 1) / g_cfg.full_attn_interval - 1;
            int kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
            int heads_per_kv = g_cfg.num_attn_heads / g_cfg.num_kv_heads;
            float scale = 1.0f / sqrtf((float)g_cfg.head_dim);
            uint32_t hd = g_cfg.head_dim;
            uint32_t kvd = (uint32_t)kv_dim;
            uint32_t sl = (uint32_t)kv->len;
            uint32_t seq_stride = GPU_KV_SEQ;
            uint32_t hpkv = (uint32_t)heads_per_kv;

            // Enc A1: attn_scores_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_scores_pipe];
                [enc setBuffer:g_metal->buf_attn_q          offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_scores     offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&scale     length:4 atIndex:7];
                [enc setBytes:&hpkv      length:4 atIndex:8];
                [enc setBytes:&sl        length:4 atIndex:9];
                uint32_t total_tgs = sl * g_cfg.num_attn_heads;
                [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A2: attn_softmax_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_softmax_pipe];
                [enc setBuffer:g_metal->buf_attn_scores offset:0 atIndex:0];
                [enc setBytes:&sl         length:4 atIndex:1];
                [enc setBytes:&seq_stride  length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(g_cfg.num_attn_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A3: attn_values_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_values_pipe];
                [enc setBuffer:g_metal->buf_attn_scores   offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_out      offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&hpkv      length:4 atIndex:7];
                uint32_t total_threads = g_cfg.head_dim * g_cfg.num_attn_heads;
                uint32_t tgs = (total_threads + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A4: sigmoid_gate
            {
                uint32_t qdim = g_cfg.num_attn_heads * g_cfg.head_dim;
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->sigmoid_gate_pipe];
                [enc setBuffer:g_metal->buf_attn_out  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_attn_gate offset:0 atIndex:1];
                [enc setBytes:&qdim length:4 atIndex:2];
                uint32_t tgs = (qdim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // ---- o_proj matvec ----
        {
            NSUInteger w_off = (NSUInteger)((const char *)oproj_w - (const char *)[g_metal->wf_buf contents]);
            NSUInteger s_off = (NSUInteger)((const char *)oproj_s - (const char *)[g_metal->wf_buf contents]);
            NSUInteger b_off = (NSUInteger)((const char *)oproj_b - (const char *)[g_metal->wf_buf contents]);

            // For GPU attention: o_proj reads from buf_attn_out
            // For CPU attention: o_proj reads from batch_out[6]
            id<MTLBuffer> oproj_input = gpu_attn_fuse ? g_metal->buf_attn_out : g_metal->batch_out[6];

            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t o_out_dim = g_cfg.hidden_dim;
            uint32_t o_in_dim = (uint32_t)oproj_in_dim;
            uint32_t o_gs = g_cfg.group_size;
            [enc setComputePipelineState:g_metal->matvec_fast];
            [enc setBuffer:g_metal->wf_buf  offset:w_off atIndex:0];
            [enc setBuffer:g_metal->wf_buf  offset:s_off atIndex:1];
            [enc setBuffer:g_metal->wf_buf  offset:b_off atIndex:2];
            [enc setBuffer:oproj_input      offset:0    atIndex:3];
            [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
            [enc setBytes:&o_out_dim  length:4 atIndex:5];
            [enc setBytes:&o_in_dim   length:4 atIndex:6];
            [enc setBytes:&o_gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 2: residual_add (buf_output + buf_residual -> buf_h_mid) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = g_cfg.hidden_dim;
            [enc setComputePipelineState:g_metal->residual_add];
            [enc setBuffer:g_metal->buf_residual offset:0 atIndex:0];  // a = residual
            [enc setBuffer:g_metal->buf_output   offset:0 atIndex:1];  // b = o_proj result
            [enc setBuffer:g_metal->buf_h_mid    offset:0 atIndex:2];  // out = h_mid
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 3: rms_norm_sum_sq (buf_h_mid -> buf_sum_sq) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = g_cfg.hidden_dim;
            [enc setComputePipelineState:g_metal->rms_norm_sum];
            [enc setBuffer:g_metal->buf_h_mid  offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_sum_sq offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 4: rms_norm_apply_bf16 (buf_h_mid + norm_w -> buf_input) ----
        {
            NSUInteger norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                               (const char *)[g_metal->wf_buf contents]);
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = g_cfg.hidden_dim;
            float eps = g_cfg.rms_norm_eps;
            [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
            [enc setBuffer:g_metal->buf_h_mid  offset:0       atIndex:0];  // x
            [enc setBuffer:g_metal->wf_buf     offset:norm_off atIndex:1]; // weight (bf16)
            [enc setBuffer:g_metal->buf_sum_sq offset:0       atIndex:2];  // sum_sq
            [enc setBuffer:g_metal->buf_input  offset:0       atIndex:3];  // out = h_post
            [enc setBytes:&dim length:4 atIndex:4];
            [enc setBytes:&eps length:4 atIndex:5];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 5-8: routing + shared expert projections (read buf_input) ----
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)g_cfg.num_experts,        g_cfg.hidden_dim, g_cfg.group_size, 0 },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size, 1 },
            { suw,    sus,    sub,    shared_up,           (uint32_t)g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size, 2 },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            g_cfg.hidden_dim, g_cfg.group_size, 3 },
        };
        // buf_input already contains h_post from Enc 4 output -- no memcpy needed
        gpu_encode_batch_matvec(g_metal, cmd_fused, moe_specs, 4);

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }

        // ---- Single commit+wait for all 8 encoders ----
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd_fused commit];
        [cmd_fused waitUntilCompleted];

        // Read back results
        gpu_flush_batch_results(g_metal, moe_specs, 4);
        // Read h_mid from GPU buffer (needed for final combine)
        memcpy(h_mid, [g_metal->buf_h_mid contents], g_cfg.hidden_dim * sizeof(float));
        // Read h_post from buf_input (needed for expert input)
        memcpy(h_post, [g_metal->buf_input contents], g_cfg.hidden_dim * sizeof(float));
        // Update hidden state to h_mid (= residual + o_proj)
        memcpy(hidden, h_mid, g_cfg.hidden_dim * sizeof(float));
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_wait += t1 - t0; }

    } else {
        // ---- Non-fused fallback path ----
        // O projection
        if (attn_out_for_oproj && oproj_w && oproj_s && oproj_b) {
            fast_dequant_matvec(oproj_w, oproj_s, oproj_b, attn_out_for_oproj,
                                attn_projected, g_cfg.hidden_dim, oproj_in_dim, g_cfg.group_size);
        }
        // attn_out_for_oproj is static — no free needed
        attn_out_for_oproj = NULL;

        // Residual connection
        for (int i = 0; i < g_cfg.hidden_dim; i++) {
            hidden[i] = residual[i] + attn_projected[i];
        }
        // attn_projected, normed, residual are static — no free needed

        cpu_vec_copy(h_mid, hidden, g_cfg.hidden_dim);

        // Post-attention norm
        cpu_rms_norm(hidden, lc->post_attn_norm_w, h_post, g_cfg.hidden_dim, g_cfg.rms_norm_eps);

        // Routing + shared expert batch
        if (have_moe_weights) {
            BatchMatvecSpec moe_specs[4] = {
                { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)g_cfg.num_experts,        g_cfg.hidden_dim, g_cfg.group_size, 0 },
                { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size, 1 },
                { suw,    sus,    sub,    shared_up,           (uint32_t)g_cfg.shared_intermediate, g_cfg.hidden_dim, g_cfg.group_size, 2 },
                { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            g_cfg.hidden_dim, g_cfg.group_size, 3 },
            };
            fast_batch_matvec(h_post, g_cfg.hidden_dim, moe_specs, 4);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }
    }

    // ---- DENSE FFN branch (num_experts==0): no routing/experts/CMD3 ----
    // h_mid (=residual+o_proj) and h_post (=post_attn_norm) are computed above.
    if (g_cfg.num_experts == 0) {
        dense_mlp_forward(wf, layer_idx, h_post, h_mid, hidden);
        if (g_timing_enabled) { g_timing.total += now_ms() - t_layer_start; g_timing.count++; }
        return;
    }

    // ---- Softmax + top-K (CPU) ----
    if (g_timing_enabled) { t0 = now_ms(); }
    cpu_softmax(gate_scores, g_cfg.num_experts);
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, g_cfg.num_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    if (g_freq_tracking) {
        for (int k = 0; k < K; k++) {
            g_expert_freq[layer_idx][expert_indices[k]]++;
        }
        if (layer_idx == 0) g_freq_total_tokens++;
    }

    // Track speculative routing prediction accuracy
    if (s_spec_count > 0) {
        int cmp_K = (K > MAX_K) ? MAX_K : K;
        for (int s = 0; s < s_spec_count; s++) {
            for (int r = 0; r < cmp_K; r++) {
                if (s_spec_indices[s] == expert_indices[r]) {
                    g_spec_route_hits++;
                    break;
                }
            }
        }
    }

    if (g_timing_enabled) { t1 = now_ms(); g_timing.routing_cpu += t1 - t0; }

    // Log routing data for predictor training
    if (g_routing_log) {
        int32_t li = layer_idx;
        int32_t ki = (K > MAX_K) ? MAX_K : K;
        fwrite(&li, sizeof(int32_t), 1, g_routing_log);
        fwrite(&ki, sizeof(int32_t), 1, g_routing_log);
        fwrite(hidden, sizeof(float), g_cfg.hidden_dim, g_routing_log);
        fwrite(expert_indices, sizeof(int32_t), ki, g_routing_log);
        g_routing_log_samples++;
    }

    // ---- Parallel pread + GPU experts ----
    if (g_timing_enabled) { t0 = now_ms(); }
    float *moe_out = s_moe_out;
    memset(moe_out, 0, g_cfg.hidden_dim * sizeof(float));
    float *shared_out = s_shared_out;
    memset(shared_out, 0, g_cfg.hidden_dim * sizeof(float));

    int actual_K = (K > MAX_K) ? MAX_K : K;

    // MTP overlap: compare this layer's active experts to the same layer's set
    // from the previous decode step, then store the current set for the next.
    if (g_mtp_overlap_enabled && g_overlap_in_decode && layer_idx < MAX_NUM_LAYERS) {
        int ck = actual_K < 16 ? actual_K : 16;
        int pk = g_overlap_prev_k[layer_idx];
        if (pk > 0) {
            int inter = 0;
            for (int i = 0; i < ck; i++) {
                for (int j = 0; j < pk; j++) {
                    if (expert_indices[i] == g_overlap_prev[layer_idx][j]) { inter++; break; }
                }
            }
            g_overlap_intersect_sum += inter;
            g_overlap_single_sum   += ck;
            g_overlap_separate_sum += ck + pk;
            g_overlap_union_sum    += ck + pk - inter;
            g_overlap_layer_pairs++;
        }
        for (int i = 0; i < ck; i++) g_overlap_prev[layer_idx][i] = expert_indices[i];
        g_overlap_prev_k[layer_idx] = ck;
    }

    if (packed_fd >= 0 && g_metal && g_metal->buf_multi_expert_data[0]) {
        // GPU multi-expert path with LRU cache + parallel I/O:
        // For each expert:
        //   - Cache HIT:  dispatch directly from cached Metal buffer (skip pread)
        //   - Cache MISS: pread into cache buffer, then dispatch from it
        // Falls back to original parallel_pread_experts when cache is disabled.

        int valid[MAX_K];
        id<MTLBuffer> expert_bufs[MAX_K];  // buffer to dispatch from per expert

        if (g_malloc_cache) {
            // ---- Malloc cache path (zero-copy Metal buffer wrappers) ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];
            int miss_cache_idx[MAX_K];  // cache entry index for each miss
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: zero-copy dispatch directly from cache buffer
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert entry (get buffer to pread into)
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, expert_indices[k], &cidx);
                    expert_bufs[k] = buf;
                    miss_indices[num_misses] = k;
                    miss_cache_idx[num_misses] = cidx;
                    num_misses++;
                    valid[k] = 0;
                }
            }

            // Phase 2: parallel pread misses directly into cache buffers (zero-copy)
            if (num_misses > 0) {
                size_t esz = active_expert_size();
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    int cidx = miss_cache_idx[m];
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = g_malloc_cache->data[cidx];
                    tasks[m].offset = (off_t)expert_indices[k] * esz;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = NULL;  // always pread for cache population
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, esz);
                    }
                }
            }
        } else if (g_expert_cache) {
            // ---- Metal buffer LRU cache path ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];       // indices into expert_indices[] for misses
            id<MTLBuffer> miss_bufs[MAX_K]; // cache buffers to pread into
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: use this buffer directly for GPU dispatch
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert into cache (allocates or evicts), will pread below
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, expert_indices[k]);
                    if (buf) {
                        expert_bufs[k] = buf;
                        miss_indices[num_misses] = k;
                        miss_bufs[num_misses] = buf;
                        num_misses++;
                        valid[k] = 0;  // not yet loaded
                    } else {
                        expert_bufs[k] = nil;
                        valid[k] = 0;
                    }
                }
            }

            // Phase 2: parallel pread all cache misses
            if (num_misses > 0) {
                size_t esz = active_expert_size();
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = [miss_bufs[m] contents];
                    tasks[m].offset = (off_t)expert_indices[k] * esz;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = mmap_base;
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark successfully loaded misses as valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, esz);
                    }
                }
            }
        } else if (pred_started) {
            // ---- Prediction path: predicted experts already loading into buf_B ----
            // Wait for predicted preads (they've had ~1.6ms: CMD1_wait + attn + CMD2 + routing)
            async_pread_wait();
            g_pred_layers++;

            // Match predictions against actual routing
            int miss_ei[MAX_K];       // actual expert indices for misses
            int miss_k_slots[MAX_K];  // which k-slot each miss maps to
            int miss_count = 0;
            int hit_count = 0;

            for (int k = 0; k < actual_K; k++) {
                int found = 0;
                for (int p = 0; p < g_pred_count[layer_idx]; p++) {
                    if (expert_indices[k] == g_pred_experts[layer_idx][p] &&
                        g_async_pread.valid[p]) {
                        // Hit! This expert was pre-loaded into buf_B[p]
                        expert_bufs[k] = g_metal->buf_multi_expert_data_B[p];
                        valid[k] = 1;
                        found = 1;
                        hit_count++;
                        break;
                    }
                }
                if (!found) {
                    miss_ei[miss_count] = expert_indices[k];
                    miss_k_slots[miss_count] = k;
                    expert_bufs[k] = g_metal->buf_multi_expert_data[k];
                    miss_count++;
                }
            }
            g_pred_hits += hit_count;
            g_pred_misses += miss_count;

            // Parallel sync-pread misses into buf_A
            if (miss_count > 0) {
                InferPreadTask tasks[MAX_K];
                size_t esz = active_expert_size();
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    tasks[m].fd = packed_fd;
                    tasks[m].dst = [g_metal->buf_multi_expert_data[k] contents];
                    tasks[m].offset = (off_t)miss_ei[m] * esz;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                }
                io_pool_dispatch(tasks, miss_count);
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    valid[k] = (tasks[m].result == (ssize_t)active_expert_size());
                }
            }
        } else if (g_use_lz4 && g_lz4_index[layer_idx]) {
            // ---- LZ4 compressed path: read compressed + decompress via io_pool ----
            size_t esz = active_expert_size();
            InferPreadTask tasks[MAX_K];
            for (int k = 0; k < actual_K; k++) {
                LZ4IndexEntry *ie = &g_lz4_index[layer_idx][expert_indices[k]];
                tasks[k].fd = packed_fd;
                tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
                tasks[k].offset = ie->offset;
                tasks[k].size = esz;
                tasks[k].result = 0;
                tasks[k].mmap_base = NULL;
                tasks[k].lz4_comp_buf = g_lz4_comp_bufs[k];
                tasks[k].lz4_comp_size = ie->comp_size;
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
            io_pool_dispatch(tasks, actual_K);
            for (int k = 0; k < actual_K; k++) {
                valid[k] = (tasks[k].result == (ssize_t)esz);
            }
        } else {
            // ---- No cache, no prediction, no LZ4: ASYNC parallel pread ----
            async_pread_start(packed_fd, expert_indices, actual_K,
                              g_metal->buf_multi_expert_data, mmap_base);
            for (int k = 0; k < actual_K; k++) {
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
        }

        // Shared expert prep (doesn't need expert data — can overlap with async pread)
        memcpy([g_metal->buf_multi_expert_input contents], h_post, g_cfg.hidden_dim * sizeof(float));
        memcpy([g_metal->buf_shared_gate contents], shared_gate,
               g_cfg.shared_intermediate * sizeof(float));
        memcpy([g_metal->buf_shared_up contents], shared_up,
               g_cfg.shared_intermediate * sizeof(float));

        // Wait for non-prediction async pread to complete
        if (!pred_started && g_async_pread.active) {
            async_pread_wait();
            for (int k = 0; k < actual_K; k++) {
                valid[k] = g_async_pread.valid[k];
            }
        }

        if (g_timing_enabled) { t1 = now_ms(); g_timing.expert_io += t1 - t0; }

        // Store this layer's routing for next token's temporal prediction.
        // MUST happen AFTER the prediction hit check above (which reads g_pred_experts).
        if (g_pred_enabled && g_pred_generating) {
            for (int k = 0; k < actual_K; k++) {
                g_pred_experts[layer_idx][k] = expert_indices[k];
            }
            g_pred_count[layer_idx] = actual_K;
            if (layer_idx == g_cfg.num_layers - 1) {
                g_pred_valid = 1;
            }
        }

        if (g_timing_enabled) { t0 = now_ms(); }

        // Step 3: encode ALL experts + shared expert into ONE command buffer.
        // Batched encoding: 4 encoders for K experts + 2 for shared = 6 total
        // (vs. 4*K + 2 = 18 with old per-expert encoding).
        id<MTLCommandBuffer> cmd_experts = [g_metal->queue commandBuffer];

        gpu_encode_experts_batched(g_metal, cmd_experts, actual_K, valid, expert_bufs);

        // Shared expert SwiGLU + down_proj (2 more encoders)
        // Note: shared_gate/up already copied to GPU buffers above (before async pread wait)

        // SwiGLU dispatch
        {
            id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
            [enc setComputePipelineState:g_metal->swiglu];
            [enc setBuffer:g_metal->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_shared_up   offset:0 atIndex:1];
            [enc setBuffer:g_metal->buf_shared_act  offset:0 atIndex:2];
            uint32_t dim = g_cfg.shared_intermediate;
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t swiglu_tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Shared down_proj dispatch
        if (sdw && sds && sdb) {
            gpu_encode_dequant_matvec_with_io_bufs(
                g_metal, cmd_experts, sdw, sds, sdb,
                g_metal->buf_shared_act, g_metal->buf_shared_out,
                g_cfg.hidden_dim, g_cfg.shared_intermediate, g_cfg.group_size);
        }

        // Step 4: GPU-side combine + residual + norm (if not last layer)
        // Appends dispatches to CMD3 so the next layer's CMD1 can submit immediately
        // without waiting for CMD3 to complete + CPU readback.
        //
        // For non-last layers with the combine pipeline available:
        //   Enc C1: moe_combine_residual (expert_outs + h_mid + shared_out -> buf_moe_hidden)
        //   Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
        //   Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_layer_norm_w -> buf_input)
        //
        // This makes CMD3 self-contained: it produces buf_input for the next layer's CMD1.
        // The next layer skips deferred_wait + finalize + input_norm entirely at layer start.

        int gpu_combine = (g_metal->moe_combine_residual &&
                           g_metal->rms_norm_sum &&
                           g_metal->rms_norm_apply_bf16 &&
                           g_metal->wf_buf &&
                           layer_idx < g_cfg.num_layers - 1 &&
                           layer_cache[layer_idx + 1].input_norm_w != NULL);

        if (gpu_combine) {
            // Copy h_mid from buf_h_mid (populated by CMD2) — it's still valid on GPU.
            // h_mid is already in buf_h_mid from CMD2's residual_add dispatch.

            // Prepare combine params: expert_weights[0..K-1] + shared_gate_score
            {
                float *params = (float *)[g_metal->buf_combine_params contents];
                // Zero all 10 slots first (unused experts get weight=0)
                memset(params, 0, 10 * sizeof(float));
                for (int k = 0; k < actual_K; k++) {
                    params[k] = valid[k] ? expert_weights[k] : 0.0f;
                }
                params[8] = shared_gate_score;
            }

            // Enc C1: moe_combine_residual
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                [enc setComputePipelineState:g_metal->moe_combine_residual];
                [enc setBuffer:g_metal->buf_h_mid         offset:0 atIndex:0];   // h_mid
                [enc setBuffer:g_metal->buf_shared_out    offset:0 atIndex:1];   // shared_out
                [enc setBuffer:g_metal->buf_moe_hidden    offset:0 atIndex:2];   // output: hidden
                // Bind all 8 expert output buffers (unused ones have weight=0 in params)
                for (int k = 0; k < MAX_K; k++) {
                    [enc setBuffer:g_metal->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
                }
                [enc setBuffer:g_metal->buf_combine_params offset:0 atIndex:11]; // params
                uint32_t dim = g_cfg.hidden_dim;
                uint32_t k_val = (uint32_t)actual_K;
                [enc setBytes:&dim   length:4 atIndex:12];
                [enc setBytes:&k_val length:4 atIndex:13];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = g_cfg.hidden_dim;
                [enc setComputePipelineState:g_metal->rms_norm_sum];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_norm_w -> buf_input)
            {
                uint16_t *next_norm_w = layer_cache[layer_idx + 1].input_norm_w;
                NSUInteger norm_off = (NSUInteger)((const char *)next_norm_w -
                                                   (const char *)[g_metal->wf_buf contents]);
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = g_cfg.hidden_dim;
                float eps = g_cfg.rms_norm_eps;
                [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0       atIndex:0]; // x
                [enc setBuffer:g_metal->wf_buf          offset:norm_off atIndex:1]; // weight (bf16)
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0       atIndex:2]; // sum_sq
                [enc setBuffer:g_metal->buf_input       offset:0       atIndex:3]; // out = normed
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // DEFERRED commit — submit async, don't wait.
        [cmd_experts commit];
        if (g_timing_enabled) {
            t1 = now_ms();
            g_timing.cmd3_encode += t1 - t0;
            g_timing.count++;
            g_timing.total += t1 - t_layer_start;
        }

        // Save state for deferred completion
        g_deferred.active = 1;
        g_deferred.gpu_combined = gpu_combine;
        g_deferred.cmd_experts = cmd_experts;
        g_deferred.actual_K = actual_K;
        g_deferred.shared_gate_score = shared_gate_score;
        g_deferred.hidden = hidden;
        g_deferred.layer_idx = layer_idx;
        if (!gpu_combine) {
            // Only need to save h_mid for CPU-side combine path
            memcpy(g_deferred.h_mid, h_mid, g_cfg.hidden_dim * sizeof(float));
        }
        for (int k = 0; k < actual_K; k++) {
            g_deferred.expert_weights[k] = expert_weights[k];
            g_deferred.valid[k] = valid[k];
        }

        // Return immediately — GPU experts are running async.
        // The next call to fused_layer_forward() or complete_deferred_experts()
        // will wait for the GPU and apply the final combine.
        return;

    } else if (packed_fd >= 0) {
        // CPU fallback for experts
        size_t esz = active_expert_size();
        float *expert_out_cpu = malloc(g_cfg.hidden_dim * sizeof(float));
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;
            void *expert_data = malloc(esz);
            ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
            if (nread != (ssize_t)esz) {
                fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                        layer_idx, eidx, nread, esz);
                free(expert_data);
                continue;
            }

            // CPU fallback offsets
            uint32_t *gw = (uint32_t *)expert_data;
            uint16_t *gs_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size);
            uint16_t *gb_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size);
            uint32_t *uw = (uint32_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size);
            uint16_t *us_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size);
            uint16_t *ub_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size);
            uint32_t *dw = (uint32_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size);
            uint16_t *ds_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size + g_cfg.down_w_size);
            uint16_t *db_p = (uint16_t *)((char *)expert_data + g_cfg.gate_w_size + g_cfg.gate_s_size + g_cfg.gate_b_size + g_cfg.up_w_size + g_cfg.up_s_size + g_cfg.up_b_size + g_cfg.down_w_size + g_cfg.down_s_size);

            float *gate_proj_out = malloc(g_cfg.moe_intermediate * sizeof(float));
            float *up_proj_out = malloc(g_cfg.moe_intermediate * sizeof(float));
            float *act_out = malloc(g_cfg.moe_intermediate * sizeof(float));

            cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                               g_cfg.moe_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
            cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                               g_cfg.moe_intermediate, g_cfg.hidden_dim, g_cfg.group_size);
            cpu_swiglu(gate_proj_out, up_proj_out, act_out, g_cfg.moe_intermediate);
            cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out_cpu,
                               g_cfg.hidden_dim, g_cfg.moe_intermediate, g_cfg.group_size);

            free(gate_proj_out);
            free(up_proj_out);
            free(act_out);
            free(expert_data);

            cpu_vec_madd(moe_out, expert_out_cpu, expert_weights[k], g_cfg.hidden_dim);
        }
        free(expert_out_cpu);

        // CPU shared expert
        float *shared_act = calloc(g_cfg.shared_intermediate, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, g_cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            cpu_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                               g_cfg.hidden_dim, g_cfg.shared_intermediate, g_cfg.group_size);
        }
        free(shared_act);
    } else {
        // No experts available -- still need shared expert
        float *shared_act = calloc(g_cfg.shared_intermediate, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, g_cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                                g_cfg.hidden_dim, g_cfg.shared_intermediate, g_cfg.group_size);
        }
        free(shared_act);
    }

    // ---- Shared expert gate ----
    float shared_weight = cpu_sigmoid(shared_gate_score);
    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Final combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < g_cfg.hidden_dim; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (g_timing_enabled) {
        t1 = now_ms();
        g_timing.cmd3_encode += t1 - t0;  // includes CPU expert compute for non-GPU paths
        g_timing.count++;
        g_timing.total += t1 - t_layer_start;
    }

    // h_post, h_mid, gate_scores, moe_out, shared_out, shared_gate, shared_up
    // are all static scratch buffers — no free needed.
}

// ============================================================================
// Main inference loop
// ============================================================================

// ============================================================================
// Expert frequency analysis (--freq)
// ============================================================================

static int freq_cmp_desc(const void *a, const void *b) {
    return *(const int *)b - *(const int *)a;
}

static void freq_print_analysis(int K) {
    if (!g_freq_tracking || g_freq_total_tokens == 0) return;

    int total_activations_per_layer = g_freq_total_tokens * K;

    fprintf(stderr, "\n=== Expert Frequency Analysis ===\n");
    fprintf(stderr, "Tokens tracked: %d, K=%d, activations/layer=%d\n\n",
            g_freq_total_tokens, K, total_activations_per_layer);

    // Per-layer analysis
    int experts_for_80_total = 0;  // sum across layers for overall estimate

    for (int l = 0; l < g_cfg.num_layers; l++) {
        // Count unique experts and sort frequencies descending
        int sorted[g_cfg.num_experts];
        memcpy(sorted, g_expert_freq[l], g_cfg.num_experts * sizeof(int));
        qsort(sorted, g_cfg.num_experts, sizeof(int), freq_cmp_desc);

        int unique = 0;
        for (int e = 0; e < g_cfg.num_experts; e++) {
            if (sorted[e] > 0) unique++;
        }

        // Compute cumulative coverage thresholds
        int cum = 0;
        int top10_cov = 0, top30_cov = 0, top60_cov = 0;
        int n_for_50 = 0, n_for_80 = 0, n_for_90 = 0;
        for (int e = 0; e < g_cfg.num_experts; e++) {
            cum += sorted[e];
            if (e == 9)  top10_cov = cum;
            if (e == 29) top30_cov = cum;
            if (e == 59) top60_cov = cum;
            if (n_for_50 == 0 && cum * 100 >= total_activations_per_layer * 50)
                n_for_50 = e + 1;
            if (n_for_80 == 0 && cum * 100 >= total_activations_per_layer * 80)
                n_for_80 = e + 1;
            if (n_for_90 == 0 && cum * 100 >= total_activations_per_layer * 90)
                n_for_90 = e + 1;
        }

        double pct10 = 100.0 * top10_cov / total_activations_per_layer;
        double pct30 = 100.0 * top30_cov / total_activations_per_layer;
        double pct60 = 100.0 * top60_cov / total_activations_per_layer;

        fprintf(stderr, "Layer %2d: %3d unique experts, "
                "top-10 cover %.0f%%, top-30 cover %.0f%%, top-60 cover %.0f%% "
                "(50%%@%d, 80%%@%d, 90%%@%d)\n",
                l, unique, pct10, pct30, pct60, n_for_50, n_for_80, n_for_90);

        experts_for_80_total += n_for_80;
    }

    // Overall summary: average experts needed for 80% across all layers
    double avg_experts_80 = (double)experts_for_80_total / g_cfg.num_layers;
    // Expert size in GB: each expert is active_expert_size() bytes
    double expert_gb = (double)active_expert_size() / (1024.0 * 1024.0 * 1024.0);
    double total_pin_gb = avg_experts_80 * g_cfg.num_layers * expert_gb;

    fprintf(stderr, "\n--- Overall Summary ---\n");
    fprintf(stderr, "To achieve 80%% hit rate across all layers, need %d experts pinned "
            "(avg %.0f/layer, %.2f GB)\n",
            experts_for_80_total, avg_experts_80, total_pin_gb);
    fprintf(stderr, "Expert size: %zu bytes (%.3f MB), %d layers x %d experts = %d total\n",
            active_expert_size(), (double)active_expert_size() / (1024.0 * 1024.0),
            g_cfg.num_layers, g_cfg.num_experts, g_cfg.num_layers * g_cfg.num_experts);
}

#ifndef CHAT_MODE

// ============================================================================
// HTTP Serve Mode — OpenAI-compatible /v1/chat/completions (SSE streaming)
// ============================================================================

// Read exactly n bytes from fd, returns 0 on success, -1 on error/EOF
static int read_exact(int fd, char *buf, int n) {
    int got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += (int)r;
    }
    return 0;
}

// Read HTTP request into buf (up to bufsz-1). Returns total bytes read, or -1.
// Reads headers, then Content-Length body if present.
static int read_http_request(int fd, char *buf, int bufsz) {
    int total = 0;
    // Read until we find \r\n\r\n (end of headers)
    while (total < bufsz - 1) {
        ssize_t r = read(fd, buf + total, 1);
        if (r <= 0) return -1;
        total++;
        if (total >= 4 &&
            buf[total-4] == '\r' && buf[total-3] == '\n' &&
            buf[total-2] == '\r' && buf[total-1] == '\n') {
            break;
        }
    }
    buf[total] = '\0';

    // Find Content-Length
    const char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        int content_len = atoi(cl + 15);
        if (content_len > 0 && total + content_len < bufsz - 1) {
            if (read_exact(fd, buf + total, content_len) < 0) return -1;
            total += content_len;
            buf[total] = '\0';
        }
    }
    return total;
}

static char *load_system_prompt(void);
static char *json_escape_alloc(const char *src);

// Tool definitions for OpenAI function calling
#define MAX_TOOLS 64
#define MAX_TOOL_NAME 64
#define MAX_TOOL_DESC 16000

typedef enum {
    API_KIND_CHAT = 1,
    API_KIND_RESPONSES = 2,
} ApiKind;

typedef enum {
    TOOL_CHOICE_AUTO = 0,
    TOOL_CHOICE_NONE = 1,
    TOOL_CHOICE_FORCED = 2,
} ToolChoiceMode;

typedef struct {
    char name[MAX_TOOL_NAME];
    char description[MAX_TOOL_DESC];
    char *parameters;
    int has_parameters;
} ToolDef;

static char *build_tool_instructions(ToolDef *tools, int tool_count);

typedef struct {
    size_t upstream_system_chars;
    size_t default_system_chars;
    size_t tool_instruction_chars;
    size_t final_system_chars;
} PromptBuildInfo;

typedef struct {
    ApiKind api_kind;
    char *system_prompt;
    char *conversation_text;
    char session_id[64];
    char model[128];
    int stream;
    int max_tokens;
    float temperature;
    float top_p;
    int top_k;
    float min_p;
    float presence_penalty;
    float repetition_penalty;
    int reasoning_enabled;
    int reasoning_explicit;
    ToolChoiceMode tool_choice_mode;
    char forced_tool_name[MAX_TOOL_NAME];
    ToolDef tools[MAX_TOOLS];
    int tool_count;
    int used_snapshot;
} ApiRequest;

typedef struct {
    int is_tool_call;
    char id[64];
    char name[MAX_TOOL_NAME];
    char *arguments;
} ParsedToolCall;

static NSString *json_stringify_obj(id obj) {
    if (!obj || obj == [NSNull null]) return @"";
    if (![NSJSONSerialization isValidJSONObject:obj]) {
        if ([obj isKindOfClass:[NSString class]]) return obj;
        return [obj description];
    }
    NSData *data = [NSJSONSerialization dataWithJSONObject:obj options:0 error:nil];
    if (!data) return @"";
    return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
}

static char *dup_nsstring(NSString *s) {
    if (!s) return strdup("");
    return strdup([s UTF8String] ?: "");
}

static NSString *flatten_content_value(id content) {
    if (!content || content == [NSNull null]) return @"";
    if ([content isKindOfClass:[NSString class]]) return content;
    if ([content isKindOfClass:[NSArray class]]) {
        NSMutableString *out = [NSMutableString string];
        for (id item in (NSArray *)content) {
            if ([item isKindOfClass:[NSString class]]) {
                [out appendString:item];
                continue;
            }
            if (![item isKindOfClass:[NSDictionary class]]) continue;
            NSString *type = item[@"type"] ?: @"";
            NSString *text = item[@"text"];
            if (!text) text = item[@"content"];
            if (!text && [type isEqualToString:@"input_text"]) text = item[@"text"];
            if (!text && [type isEqualToString:@"output_text"]) text = item[@"text"];
            if (!text && item[@"input"]) text = item[@"input"];
            if (text) {
                if ([out length] > 0) [out appendString:@"\n"];
                [out appendString:text];
            }
        }
        return out;
    }
    if ([content isKindOfClass:[NSDictionary class]]) {
        NSString *text = content[@"text"];
        if (text) return text;
    }
    return [content description];
}

static void append_chat_turn(NSMutableString *dst, NSString *role, NSString *content) {
    if (!dst || !role) return;
    [dst appendFormat:@"<|im_start|>%@\n", role];
    if (content) [dst appendString:content];
    [dst appendString:@"<|im_end|>\n"];
}

// Render assistant tool_calls in the native Qwen3 XML form, matching the
// model's chat_template:
//   <tool_call>
//   <function=NAME>
//   <parameter=KEY>
//   {value}
//   </parameter>
//   </function>
//   </tool_call>
// Parameter values are emitted as raw text for strings and JSON-encoded for
// anything else (matching the template's `string if string else tojson|safe`).
static void append_assistant_tool_calls(NSMutableString *dst, NSArray *tool_calls) {
    if (!tool_calls || ![tool_calls isKindOfClass:[NSArray class]]) return;
    BOOL prefix_with_separator = ([dst length] > 0);
    for (NSDictionary *tool_call in tool_calls) {
        if (![tool_call isKindOfClass:[NSDictionary class]]) continue;
        NSDictionary *function = tool_call[@"function"] ?: tool_call;
        NSString *name = function[@"name"] ?: tool_call[@"name"] ?: @"bash";
        id arguments_obj = function[@"arguments"] ?: tool_call[@"arguments"];

        // arguments may arrive as a JSON-encoded string (OpenAI form) or a dict.
        NSDictionary *args_dict = nil;
        if ([arguments_obj isKindOfClass:[NSString class]]) {
            NSData *data = [(NSString *)arguments_obj dataUsingEncoding:NSUTF8StringEncoding];
            id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
            if ([parsed isKindOfClass:[NSDictionary class]]) args_dict = parsed;
        } else if ([arguments_obj isKindOfClass:[NSDictionary class]]) {
            args_dict = arguments_obj;
        }
        if (!args_dict) args_dict = @{};

        if (prefix_with_separator) [dst appendString:@"\n\n"];
        prefix_with_separator = NO;
        [dst appendFormat:@"<tool_call>\n<function=%@>\n", name];
        for (NSString *key in args_dict) {
            id value = args_dict[key];
            NSString *value_str;
            if ([value isKindOfClass:[NSString class]]) {
                value_str = (NSString *)value;
            } else {
                NSData *data = [NSJSONSerialization dataWithJSONObject:@[value] options:0 error:NULL];
                NSString *wrapped = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
                if ([wrapped length] >= 2) {
                    value_str = [wrapped substringWithRange:NSMakeRange(1, [wrapped length] - 2)];
                } else {
                    value_str = @"";
                }
            }
            [dst appendFormat:@"<parameter=%@>\n%@\n</parameter>\n", key, value_str];
        }
        [dst appendString:@"</function>\n</tool_call>"];
        prefix_with_separator = YES;
    }
}

// Render a `role=tool` message body for the native template. Combined with
// `append_chat_turn`'s "<|im_start|>user\n" header and "<|im_end|>\n" footer,
// this produces:
//   <|im_start|>user
//   <tool_response>
//   {content}
//   </tool_response><|im_end|>
static NSString *normalized_tool_response(NSDictionary *msg) {
    NSString *content = flatten_content_value(msg[@"content"]);
    if (!content) content = @"";
    return [NSString stringWithFormat:@"<tool_response>\n%@\n</tool_response>", content];
}

static void api_request_init(ApiRequest *req, ApiKind kind) {
    memset(req, 0, sizeof(*req));
    req->api_kind = kind;
    req->stream = 1;
    req->max_tokens = 8192;
    req->temperature = g_default_temperature;
    req->top_p = g_default_top_p;
    req->top_k = g_default_top_k;
    req->min_p = g_default_min_p;
    req->presence_penalty = g_default_presence_penalty;
    req->repetition_penalty = g_default_repetition_penalty;
    req->reasoning_enabled = g_default_reasoning_enabled;
    req->tool_choice_mode = TOOL_CHOICE_AUTO;
    strncpy(req->model, kApiModelId, sizeof(req->model) - 1);
}

static void api_request_free(ApiRequest *req) {
    if (!req) return;
    free(req->system_prompt);
    free(req->conversation_text);
    for (int i = 0; i < req->tool_count && i < MAX_TOOLS; i++) {
        free(req->tools[i].parameters);
        req->tools[i].parameters = NULL;
    }
    req->system_prompt = NULL;
    req->conversation_text = NULL;
}

static void parsed_tool_call_free(ParsedToolCall *tool_call) {
    if (!tool_call) return;
    free(tool_call->arguments);
    tool_call->arguments = NULL;
}

static int append_bytes(char **buf, size_t *len, size_t *cap, const char *src, size_t src_len) {
    if (!buf || !len || !cap || !src) return -1;
    if (!*buf) {
        *cap = src_len + 1;
        if (*cap < 256) *cap = 256;
        *buf = calloc(1, *cap);
        if (!*buf) return -1;
    }
    if (*len + src_len + 1 > *cap) {
        size_t new_cap = *cap;
        while (*len + src_len + 1 > new_cap) new_cap *= 2;
        char *grown = realloc(*buf, new_cap);
        if (!grown) return -1;
        *buf = grown;
        *cap = new_cap;
    }
    memcpy(*buf + *len, src, src_len);
    *len += src_len;
    (*buf)[*len] = 0;
    return 0;
}

static int parse_tool_defs(NSArray *tools, ApiRequest *req) {
    if (!tools || ![tools isKindOfClass:[NSArray class]]) return 0;
    int count = 0;
    for (NSDictionary *tool in tools) {
        if (count >= MAX_TOOLS || ![tool isKindOfClass:[NSDictionary class]]) break;
        NSString *type = tool[@"type"] ?: @"function";
        if (![type isEqualToString:@"function"]) continue;
        NSDictionary *function = tool[@"function"] ?: tool;
        NSString *name = function[@"name"];
        if (![name length]) continue;
        ToolDef *dst = &req->tools[count++];
        memset(dst, 0, sizeof(*dst));
        strncpy(dst->name, [name UTF8String], sizeof(dst->name) - 1);
        NSString *desc = function[@"description"] ?: @"";
        strncpy(dst->description, [desc UTF8String], sizeof(dst->description) - 1);
        id params = function[@"parameters"];
        if (params && params != [NSNull null]) {
            NSString *params_json = json_stringify_obj(params);
            dst->parameters = dup_nsstring(params_json);
            dst->has_parameters = (dst->parameters && dst->parameters[0] != '\0');
        }
    }
    req->tool_count = count;
    return count;
}

static void parse_tool_choice(id tool_choice, ApiRequest *req) {
    if (!tool_choice || tool_choice == [NSNull null]) return;
    if ([tool_choice isKindOfClass:[NSString class]]) {
        NSString *choice = tool_choice;
        if ([choice isEqualToString:@"none"]) req->tool_choice_mode = TOOL_CHOICE_NONE;
        else req->tool_choice_mode = TOOL_CHOICE_AUTO;
        return;
    }
    if (![tool_choice isKindOfClass:[NSDictionary class]]) return;
    NSString *type = tool_choice[@"type"] ?: @"";
    NSDictionary *function = tool_choice[@"function"];
    NSString *name = function[@"name"] ?: tool_choice[@"name"];
    if ([type isEqualToString:@"function"] && [name length] > 0) {
        req->tool_choice_mode = TOOL_CHOICE_FORCED;
        strncpy(req->forced_tool_name, [name UTF8String], sizeof(req->forced_tool_name) - 1);
    }
}

static int request_has_tool_named(ApiRequest *req, const char *name) {
    if (!req || !name || !name[0]) return 0;
    for (int i = 0; i < req->tool_count && i < MAX_TOOLS; i++) {
        if (strcmp(req->tools[i].name, name) == 0) return 1;
    }
    return 0;
}

static int parse_reasoning_value(id value, int fallback) {
    if (!value || value == [NSNull null]) return fallback;
    if ([value isKindOfClass:[NSNumber class]]) return [value boolValue] ? 1 : 0;
    if ([value isKindOfClass:[NSString class]]) {
        NSString *s = [(NSString *)value lowercaseString];
        if ([s isEqualToString:@"true"] || [s isEqualToString:@"enabled"]) return 1;
        if ([s isEqualToString:@"false"] || [s isEqualToString:@"disabled"]) return 0;
        return fallback;
    }
    if ([value isKindOfClass:[NSDictionary class]]) {
        id enabled = value[@"enabled"];
        if (enabled) return parse_reasoning_value(enabled, fallback);
    }
    return fallback;
}

static NSString *strip_think_directive(NSString *prompt) {
    if (!prompt) return @"You are a helpful assistant.";
    NSString *stripped = [prompt stringByReplacingOccurrencesOfString:@"/think" withString:@""];
    stripped = [stripped stringByReplacingOccurrencesOfString:@"<think>" withString:@""];
    stripped = [stripped stringByReplacingOccurrencesOfString:@"</think>" withString:@""];
    return stripped;
}

// froggeric v18 "Smart False-Positive Error Detection": substring matching on
// the word "error" is dangerous because successful API/JSON returns routinely
// contain it as a key prefix (`error_rate`, `errors_logged`). We use strict
// structural guards instead, plus a length gate so long well-formed responses
// don't trip on a stray keyword, and exclusion guards for shell-echo lines
// (commands starting with `$`) and search-tool timing footers.
static BOOL looks_like_tool_error(NSString *content) {
    if (!content) return NO;
    NSString *trimmed = [content stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if ([trimmed length] == 0) return NO;
    // Length gate: only short responses are scanned. A 50KB stdout buffer that
    // happens to contain the word "error" is almost certainly normal output.
    if ([trimmed length] >= 500) return NO;
    // Exclusion: a line beginning with `$ ` is a shell echo of the command we
    // ran, not an error from running it.
    if ([trimmed hasPrefix:@"$ "]) return NO;
    // Exclusion: search-result footers like "Took 0.42s" or "found 3 results".
    NSString *lower = [trimmed lowercaseString];
    if ([lower hasPrefix:@"took "] || [lower hasPrefix:@"found "]) return NO;
    // Positive guards: JSON error key, Python exception, stack trace, missing
    // command. These are the structural signatures of actual failures.
    if ([content rangeOfString:@"\"error\":"].location != NSNotFound) return YES;
    if ([content rangeOfString:@"Exception:"].location != NSNotFound) return YES;
    if ([content rangeOfString:@"Traceback"].location != NSNotFound) return YES;
    if ([content rangeOfString:@"command not found"].location != NSNotFound) return YES;
    return NO;
}

// Scan a message's content for the froggeric think-toggle tags. Returns:
//   1  if `<|think_on|>` was found (caller flips reasoning on)
//  -1  if `<|think_off|>` was found (caller flips reasoning off)
//   0  if neither tag is present
// The matching tag is stripped from `*content_io` in place. If both tags are
// present, the LAST one wins (matches froggeric's interception order).
static int extract_think_toggle(NSMutableString *content_io) {
    if (!content_io) return 0;
    int result = 0;
    NSRange on_range = [content_io rangeOfString:@"<|think_on|>" options:NSBackwardsSearch];
    NSRange off_range = [content_io rangeOfString:@"<|think_off|>" options:NSBackwardsSearch];
    if (on_range.location != NSNotFound && (off_range.location == NSNotFound || on_range.location > off_range.location)) {
        result = 1;
    } else if (off_range.location != NSNotFound) {
        result = -1;
    }
    [content_io replaceOccurrencesOfString:@"<|think_on|>"
                                withString:@""
                                   options:0
                                     range:NSMakeRange(0, [content_io length])];
    [content_io replaceOccurrencesOfString:@"<|think_off|>"
                                withString:@""
                                   options:0
                                     range:NSMakeRange(0, [content_io length])];
    return result;
}

// Assemble the system block in native Qwen3 order, matching chat_template.jinja:
// the tool-instruction block (`# Tools` ... `</IMPORTANT>`) comes FIRST, then
// `\n\n` + user system content. The model was post-trained with the tool
// grammar at the start of the system block as the primer that activates
// tool-calling mode; placing user content before it shifts the grammar far
// out of distribution and degrades tool-calling reliability vs lmstudio.
//
// The returned string is the body of the <|im_start|>system ... <|im_end|>
// block; callers wrap it. Trailing newline is intentionally NOT included so
// the wrapping is symmetric with the template.
static char *build_system_prompt_for_request(ApiRequest *req, PromptBuildInfo *info) {
    if (info) memset(info, 0, sizeof(*info));
    char *base_c = load_system_prompt();
    NSString *base = [NSString stringWithUTF8String:base_c ?: "You are a helpful assistant. /think"];
    free(base_c);
    NSString *user_sys = req->system_prompt && req->system_prompt[0]
        ? [NSString stringWithUTF8String:req->system_prompt]
        : base;
    if (info) {
        info->default_system_chars = strlen([base UTF8String] ?: "");
        info->upstream_system_chars = req->system_prompt ? strlen(req->system_prompt) : 0;
    }
    NSString *user_sys_norm = req->reasoning_enabled ? user_sys : strip_think_directive(user_sys);

    int has_tools = (req->tool_count > 0 && req->tool_choice_mode != TOOL_CHOICE_NONE);
    NSMutableString *final = [NSMutableString string];

    if (has_tools) {
        char *tool_block = build_tool_instructions(req->tools, req->tool_count);
        if (info) info->tool_instruction_chars = tool_block ? strlen(tool_block) : 0;
        [final appendString:[NSString stringWithUTF8String:tool_block ?: ""]];
        free(tool_block);
        NSString *trimmed = [user_sys_norm stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if ([trimmed length] > 0) {
            [final appendString:@"\n\n"];
            [final appendString:trimmed];
        }
        // tool_choice=forced: do NOT inject prose. The forcing happens at the
        // assistant-turn prefix (see fill_request_from_chat_json), which
        // prefills <tool_call><function=NAME> so the model has no degree of
        // freedom about whether to call. Imperative prose (we tried "You
        // must call the function named X") drives the 35B-A3B model out of
        // distribution and produces malformed XML — verified empirically.
    } else {
        [final appendString:user_sys_norm];
        if (req->tool_choice_mode == TOOL_CHOICE_NONE) {
            [final appendString:@"\n\nDo not call tools. Respond directly to the user."];
        }
    }
    if (info) info->final_system_chars = strlen([final UTF8String] ?: "");
    return dup_nsstring(final);
}

static int count_sys_prompt_tokens(const char *sys_prompt) {
    size_t len = strlen(sys_prompt);
    size_t total = len + 64;
    char *prefix = malloc(total);
    if (!prefix) return 0;
    snprintf(prefix, total, "<|im_start|>system\n%s<|im_end|>\n", sys_prompt);
    PromptTokens *pt = encode_prompt_text_to_tokens(prefix);
    free(prefix);
    int count = pt ? pt->count : 0;
    if (pt) { free(pt->ids); free(pt); }
    return count;
}

static PromptTokens *tokenize_request_prompt(ApiRequest *req, const char *request_id) {
    if (!req || !req->conversation_text) return NULL;
    size_t conv_len = strlen(req->conversation_text);
    if (req->used_snapshot) {
        server_log_errorf("[serve] %s tokenizing snapshot continuation chars=%zu\n",
                          request_id ? request_id : "request", conv_len);
        PromptTokens *pt = encode_prompt_text_to_tokens(req->conversation_text);
        server_log_errorf("[serve] %s tokenized snapshot continuation tokens=%d\n",
                          request_id ? request_id : "request", pt ? pt->count : -1);
        return pt;
    }
    PromptBuildInfo build_info;
    char *sys_prompt = build_system_prompt_for_request(req, &build_info);
    size_t total = strlen(sys_prompt) + conv_len + 128;
    server_log_errorf("[serve] %s prompt_parts upstream_system_chars=%zu default_system_chars=%zu tool_instruction_chars=%zu final_system_chars=%zu conversation_chars=%zu total_chars=%zu\n",
                      request_id ? request_id : "request",
                      build_info.upstream_system_chars,
                      build_info.default_system_chars,
                      build_info.tool_instruction_chars,
                      build_info.final_system_chars,
                      conv_len, total);
    if (g_server_debug_enabled) {
        server_debug_write_text(request_id ? request_id : "request", "system_prompt.txt", sys_prompt);
    }
    char *prompt = malloc(total);
    snprintf(prompt, total, "<|im_start|>system\n%s<|im_end|>\n%s", sys_prompt, req->conversation_text);
    server_log_errorf("[serve] %s tokenizing prompt chars=%zu\n",
                      request_id ? request_id : "request", strlen(prompt));
    if (g_server_debug_enabled) {
        server_debug_write_text(request_id ? request_id : "request", "assembled_prompt.txt", prompt);
    }
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    server_log_errorf("[serve] %s tokenized prompt tokens=%d\n",
                      request_id ? request_id : "request", pt ? pt->count : -1);
    free(sys_prompt);
    free(prompt);
    return pt;
}

static NSDictionary *parse_json_body(const char *body, NSError **error) {
    NSData *data = [NSData dataWithBytes:body length:strlen(body)];
    id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:error];
    if (![obj isKindOfClass:[NSDictionary class]]) return nil;
    return obj;
}

static int fill_request_from_chat_json(NSDictionary *root, ApiRequest *req, char **err_msg) {
    NSArray *messages = root[@"messages"];
    if (![messages isKindOfClass:[NSArray class]] || [messages count] == 0) {
        *err_msg = strdup("messages array is required");
        return -1;
    }
    NSString *model = root[@"model"];
    if ([model length] > 0) strncpy(req->model, [model UTF8String], sizeof(req->model) - 1);
    id stream = root[@"stream"];
    if (stream) req->stream = [stream boolValue] ? 1 : 0;
    NSNumber *max_completion = root[@"max_completion_tokens"];
    NSNumber *max_tokens = root[@"max_tokens"];
    if (max_completion) req->max_tokens = [max_completion intValue];
    else if (max_tokens) req->max_tokens = [max_tokens intValue];
    if (req->max_tokens <= 0) req->max_tokens = 8192;
    if (req->max_tokens > 32768) req->max_tokens = 32768;
    NSNumber *temperature = root[@"temperature"];
    if (temperature) req->temperature = [temperature floatValue];
    NSNumber *top_p = root[@"top_p"];
    if (top_p) req->top_p = [top_p floatValue];
    NSNumber *top_k = root[@"top_k"];
    if (top_k) req->top_k = [top_k intValue];
    NSNumber *min_p = root[@"min_p"];
    if (min_p) req->min_p = [min_p floatValue];
    NSNumber *presence_penalty = root[@"presence_penalty"];
    if (presence_penalty) req->presence_penalty = [presence_penalty floatValue];
    NSNumber *repetition_penalty = root[@"repetition_penalty"];
    if (repetition_penalty) req->repetition_penalty = [repetition_penalty floatValue];
    id reasoning = root[@"reasoning"];
    if (reasoning && reasoning != [NSNull null]) {
        req->reasoning_explicit = 1;
        req->reasoning_enabled = parse_reasoning_value(reasoning, req->reasoning_enabled);
    }
    parse_tool_defs(root[@"tools"], req);
    parse_tool_choice(root[@"tool_choice"], req);
    // Native Qwen3 tool calls are far more format-compliant when the assistant
    // turn opens with `<think>\n` (matching chat_template's enable_thinking=true).
    // Without thinking, a long client system prompt (e.g. nanocoder's 11kB prose)
    // dominates the format primer and the model emits Python-style or partial
    // XML. With thinking, the model commits to tool-calling mode during the
    // think block and reliably produces correct `<tool_call>` XML. Enable by
    // default when tools are active and the client did not pin reasoning.
    if (req->tool_count > 0 && !req->reasoning_explicit) {
        req->reasoning_enabled = 1;
    }
    NSString *session_id = root[@"session_id"];
    if ([session_id length] > 0) strncpy(req->session_id, [session_id UTF8String], sizeof(req->session_id) - 1);

    NSMutableString *system_prompt = [NSMutableString string];
    NSMutableString *conversation = [NSMutableString string];

    // froggeric v18/v19 "preserve_thinking" + "abolish empty think". Past
    // assistant turns are rendered with whatever `<think>...</think>` content
    // they were originally generated with — we never strip non-empty thinking
    // and we never inject empty `<think>\n</think>` as a placeholder. The
    // older behavior (strip on old turns, wrap empty on recent turns) created
    // KV-cache invalidation every turn and trained the model in-context to
    // associate empty think blocks with imminent tool calls, producing 80%+
    // premature `<|im_end|>` aborts.
    //
    // We still walk the message tail to detect the most recent
    // `<tool_response>` and count how many consecutive responses look like
    // tool errors. That feeds the two-tier error-escalation injection below:
    //   * 1 consecutive error  → seed the assistant `<think>` with a
    //                            correction directive at generation time
    //   * 2+ consecutive errors → bypass thinking and inject an out-of-band
    //                             `<IMPORTANT>` directive after the last
    //                             `</tool_response>` in the user turn
    NSInteger last_tool_response_index = -1;
    int consecutive_tool_errors = 0;
    {
        BOOL still_in_tail = YES;
        for (NSInteger i = (NSInteger)[messages count] - 1; i >= 0 && still_in_tail; i--) {
            NSDictionary *m = messages[i];
            if (![m isKindOfClass:[NSDictionary class]]) continue;
            NSString *r = m[@"role"] ?: @"user";
            BOOL is_tool_role = [r isEqualToString:@"tool"];
            NSString *c = flatten_content_value(m[@"content"]) ?: @"";
            NSString *trimmed = [c stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            BOOL is_wrapped_response = ([r isEqualToString:@"user"]
                && [trimmed hasPrefix:@"<tool_response>"]
                && [trimmed hasSuffix:@"</tool_response>"]);
            if (is_tool_role || is_wrapped_response) {
                if (last_tool_response_index < 0) last_tool_response_index = i;
                NSString *inner = c;
                if (is_wrapped_response) {
                    NSString *body = trimmed;
                    if ([body hasPrefix:@"<tool_response>"]) {
                        body = [body substringFromIndex:[@"<tool_response>" length]];
                    }
                    if ([body hasSuffix:@"</tool_response>"]) {
                        body = [body substringToIndex:[body length] - [@"</tool_response>" length]];
                    }
                    inner = body;
                }
                if (looks_like_tool_error(inner)) {
                    consecutive_tool_errors++;
                } else {
                    still_in_tail = NO;
                }
                continue;
            }
            if ([r isEqualToString:@"assistant"]) continue;
            still_in_tail = NO;
        }
    }

    NSCharacterSet *ws = [NSCharacterSet whitespaceAndNewlineCharacterSet];
    NSInteger msg_index = -1;
    for (NSDictionary *msg in messages) {
        msg_index++;
        if (![msg isKindOfClass:[NSDictionary class]]) continue;
        NSString *role = msg[@"role"] ?: @"user";
        // chat_template.jinja line 82: `set content = render_content(...)|trim`.
        // Apply the same trim per message so structural newlines around tags
        // come from the template, not from incidental client formatting.
        NSMutableString *content_buf = [NSMutableString stringWithString:
            [flatten_content_value(msg[@"content"]) stringByTrimmingCharactersInSet:ws]];
        // froggeric thinking-toggle interception: if the message embeds
        // `<|think_on|>` or `<|think_off|>`, flip the reasoning flag and
        // strip the tag before the model ever sees it.
        int toggle = extract_think_toggle(content_buf);
        if (toggle != 0) {
            req->reasoning_explicit = 1;
            req->reasoning_enabled = (toggle > 0) ? 1 : 0;
        }
        NSString *content = [content_buf stringByTrimmingCharactersInSet:ws];
        // froggeric "developer" role support: modern OpenAI-compatible APIs
        // use `developer` for system-level instructions on reasoning models.
        // Fold it into the system prompt block.
        if ([role isEqualToString:@"system"] || [role isEqualToString:@"developer"]) {
            if ([system_prompt length] > 0) [system_prompt appendString:@"\n\n"];
            [system_prompt appendString:content ?: @""];
            continue;
        }
        if ([role isEqualToString:@"assistant"]) {
            // Extract reasoning either from an explicit `reasoning_content`
            // field (OpenAI separated form) or by splitting `content` on
            // <think>...</think>. Past assistant turns are always rendered
            // with whatever they originally produced — non-empty reasoning
            // is preserved verbatim, empty reasoning emits NO think markers
            // at all (v19 "abolish empty think").
            NSString *reasoning = nil;
            NSString *body = content ?: @"";
            id raw_reasoning = msg[@"reasoning_content"];
            if ([raw_reasoning isKindOfClass:[NSString class]]) {
                reasoning = raw_reasoning;
            } else {
                NSRange close_think = [body rangeOfString:@"</think>"];
                if (close_think.location != NSNotFound) {
                    NSString *before = [body substringToIndex:close_think.location];
                    NSString *after = [body substringFromIndex:NSMaxRange(close_think)];
                    NSRange open_think = [before rangeOfString:@"<think>" options:NSBackwardsSearch];
                    if (open_think.location != NSNotFound) {
                        reasoning = [before substringFromIndex:NSMaxRange(open_think)];
                    } else {
                        reasoning = before;
                    }
                    while ([after hasPrefix:@"\n"]) after = [after substringFromIndex:1];
                    body = after;
                }
            }
            if (!reasoning) reasoning = @"";
            reasoning = [reasoning stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];

            NSMutableString *assistant = [NSMutableString string];
            if ([reasoning length] > 0) {
                [assistant appendFormat:@"<think>\n%@\n</think>\n\n", reasoning];
            }
            [assistant appendString:body];
            append_assistant_tool_calls(assistant, msg[@"tool_calls"]);
            append_chat_turn(conversation, @"assistant", assistant);
            continue;
        }
        if ([role isEqualToString:@"tool"]) {
            NSString *body = normalized_tool_response(msg);
            // Tier-2 escalation: when the same call has failed twice in a
            // row, inject an out-of-band <IMPORTANT> directive after the
            // </tool_response> close. The model is wrapped inside the user
            // turn so the directive cannot be confused with assistant
            // output. We respect `reasoning_enabled=0`: the directive does
            // NOT open a <think> block in that mode (froggeric v18 fix).
            if (msg_index == last_tool_response_index && consecutive_tool_errors >= 2) {
                body = [body stringByAppendingString:
                    @"\n<IMPORTANT>\nThe previous tool call has failed twice in a row with the same kind of error. Do not repeat the identical call. Either change the parameters, choose a different function, or stop calling tools and explain the problem to the user.\n</IMPORTANT>"];
            }
            append_chat_turn(conversation, @"user", body);
            continue;
        }
        NSString *body = content ?: @"";
        if (msg_index == last_tool_response_index && consecutive_tool_errors >= 2) {
            body = [body stringByAppendingString:
                @"\n<IMPORTANT>\nThe previous tool call has failed twice in a row with the same kind of error. Do not repeat the identical call. Either change the parameters, choose a different function, or stop calling tools and explain the problem to the user.\n</IMPORTANT>"];
        }
        append_chat_turn(conversation, @"user", body);
    }
    // Native Qwen3 generation prompt: open the assistant turn with <think>\n
    // for reasoning-on (model generates content INSIDE the think block first),
    // or inject an empty <think>...</think> block for reasoning-off. The
    // empty-think-here is at GENERATION time only — it's required because the
    // model was trained to always start with a think marker. v19's "abolish
    // empty think" applies to past-turn rendering, not the generation prefix.
    if (req->reasoning_enabled) {
        [conversation appendString:@"<|im_start|>assistant\n<think>\n"];
        // Tier-1 escalation: exactly one consecutive tool error and reasoning
        // is enabled. Seed the <think> block with a concrete correction
        // directive at a different token position than the previous attempt's
        // think opener — this breaks the cached attractor state that was
        // driving the model to re-emit the identical failing call.
        if (consecutive_tool_errors == 1) {
            [conversation appendString:
                @"The previous tool call returned an error. I need to reconsider: check parameter names and types against the schema, fix whatever was wrong, and call the function again with corrected arguments.\n"];
        }
    } else {
        [conversation appendString:@"<|im_start|>assistant\n<think>\n\n</think>\n\n"];
    }
    // When tool_choice forces a specific function, prefill the assistant turn
    // with the canonical XML opener. This bypasses thinking entirely (empty
    // <think></think>) and starts the model inside <tool_call><function=NAME>
    // so it can only emit parameter blocks and the closing tags. This is far
    // more reliable than imperative prose, which drives the model out of
    // distribution.
    if (req->tool_choice_mode == TOOL_CHOICE_FORCED && req->forced_tool_name[0]) {
        if (req->reasoning_enabled) {
            // Replace the open <think>\n we just appended with an empty think
            // block so the model jumps straight into the tool call.
            NSRange last_open_think = [conversation rangeOfString:@"<think>\n" options:NSBackwardsSearch];
            if (last_open_think.location != NSNotFound) {
                [conversation replaceCharactersInRange:last_open_think withString:@"<think>\n\n</think>\n\n"];
            }
        }
        [conversation appendFormat:@"<tool_call>\n<function=%s>\n", req->forced_tool_name];
    }
    req->system_prompt = dup_nsstring(system_prompt);
    req->conversation_text = dup_nsstring(conversation);
    req->used_snapshot = 0;
    return 0;
}

static int fill_request_from_responses_json(NSDictionary *root, ApiRequest *req, char **err_msg) {
    NSString *model = root[@"model"];
    if ([model length] > 0) strncpy(req->model, [model UTF8String], sizeof(req->model) - 1);
    id stream = root[@"stream"];
    if (stream) req->stream = [stream boolValue] ? 1 : 0;
    NSNumber *max_output_tokens = root[@"max_output_tokens"];
    if (max_output_tokens) req->max_tokens = [max_output_tokens intValue];
    if (req->max_tokens <= 0) req->max_tokens = 8192;
    if (req->max_tokens > 32768) req->max_tokens = 32768;
    NSNumber *temperature = root[@"temperature"];
    if (temperature) req->temperature = [temperature floatValue];
    NSNumber *top_p = root[@"top_p"];
    if (top_p) req->top_p = [top_p floatValue];
    NSNumber *top_k = root[@"top_k"];
    if (top_k) req->top_k = [top_k intValue];
    NSNumber *min_p = root[@"min_p"];
    if (min_p) req->min_p = [min_p floatValue];
    NSNumber *presence_penalty = root[@"presence_penalty"];
    if (presence_penalty) req->presence_penalty = [presence_penalty floatValue];
    NSNumber *repetition_penalty = root[@"repetition_penalty"];
    if (repetition_penalty) req->repetition_penalty = [repetition_penalty floatValue];
    id reasoning = root[@"reasoning"];
    if (reasoning && reasoning != [NSNull null]) {
        req->reasoning_explicit = 1;
        req->reasoning_enabled = parse_reasoning_value(reasoning, req->reasoning_enabled);
    }
    parse_tool_defs(root[@"tools"], req);
    parse_tool_choice(root[@"tool_choice"], req);
    // Mirror chat-completions: tools active without explicit reasoning enables
    // thinking, which dramatically improves native XML format adherence.
    if (req->tool_count > 0 && !req->reasoning_explicit) {
        req->reasoning_enabled = 1;
    }

    NSMutableArray *messages = [NSMutableArray array];
    id input = root[@"input"];
    if ([input isKindOfClass:[NSString class]]) {
        [messages addObject:@{@"role": @"user", @"content": input}];
    } else if ([input isKindOfClass:[NSArray class]]) {
        for (id item in (NSArray *)input) {
            if ([item isKindOfClass:[NSDictionary class]]) {
                NSString *type = item[@"type"] ?: @"";
                if ([type isEqualToString:@"message"] || item[@"role"]) {
                    [messages addObject:item];
                } else if ([type isEqualToString:@"input_text"]) {
                    [messages addObject:@{@"role": @"user", @"content": item[@"text"] ?: @""}];
                } else if ([type isEqualToString:@"function_call_output"]) {
                    [messages addObject:@{
                        @"role": @"tool",
                        @"tool_call_id": item[@"call_id"] ?: @"",
                        @"name": item[@"name"] ?: @"",
                        @"content": item[@"output"] ?: @""
                    }];
                } else if ([type isEqualToString:@"function_call"]) {
                    NSDictionary *function = @{
                        @"id": item[@"call_id"] ?: @"",
                        @"function": @{
                            @"name": item[@"name"] ?: @"bash",
                            @"arguments": item[@"arguments"] ?: @"{}"
                        }
                    };
                    [messages addObject:@{
                        @"role": @"assistant",
                        @"content": @"",
                        @"tool_calls": @[function]
                    }];
                }
            }
        }
    }
    if ([messages count] == 0) {
        *err_msg = strdup("responses input is required");
        return -1;
    }
    return fill_request_from_chat_json(@{
        @"model": root[@"model"] ?: @"",
        @"messages": messages,
        @"stream": @(req->stream),
        @"max_tokens": @(req->max_tokens),
        @"temperature": @(req->temperature),
        @"top_p": @(req->top_p),
        @"top_k": @(req->top_k),
        @"min_p": @(req->min_p),
        @"presence_penalty": @(req->presence_penalty),
        @"repetition_penalty": @(req->repetition_penalty),
        @"reasoning": @(req->reasoning_enabled),
        @"tools": root[@"tools"] ?: @[],
        @"tool_choice": root[@"tool_choice"] ?: @"auto"
    }, req, err_msg);
}

// Save a conversation turn to ~/.config/flashchat/sessions/<session_id>.jsonl
// Shared data store with the chat client.
__attribute__((unused))
static void server_save_turn(const char *session_id, const char *role, const char *content) {
    if (!session_id || !session_id[0] || !content) return;
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    char dir[1024], path[1024];
    const char *sessions_env = getenv("FLASHCHAT_SESSIONS_DIR");
    if (sessions_env && sessions_env[0]) {
        snprintf(dir, sizeof(dir), "%s", sessions_env);
    } else {
        snprintf(dir, sizeof(dir), "%s/.config/flashchat/sessions", home);
    }
    char config_parent[1024];
    mkdir(home, 0755);
    snprintf(config_parent, sizeof(config_parent), "%s/.config", home);
    mkdir(config_parent, 0755);
    char app_parent[1024];
    snprintf(app_parent, sizeof(app_parent), "%s/.config/flashchat", home);
    mkdir(app_parent, 0755);
    mkdir(dir, 0755);
    snprintf(path, sizeof(path), "%s/%s.jsonl", dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;
    // JSON-escape content
    size_t clen = strlen(content);
    char *escaped = malloc(clen * 2 + 1);
    int j = 0;
    for (size_t i = 0; i < clen; i++) {
        switch (content[i]) {
            case '"': escaped[j++]='\\'; escaped[j++]='"'; break;
            case '\\': escaped[j++]='\\'; escaped[j++]='\\'; break;
            case '\n': escaped[j++]='\\'; escaped[j++]='n'; break;
            case '\r': escaped[j++]='\\'; escaped[j++]='r'; break;
            case '\t': escaped[j++]='\\'; escaped[j++]='t'; break;
            default: escaped[j++]=content[i]; break;
        }
    }
    escaped[j] = 0;
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    free(escaped);
    fclose(f);
}

// Extract "session_id" string from JSON body. Copies into out_buf (max out_size).
// Returns 1 if found, 0 if missing.
__attribute__((unused))
static int extract_session_id(const char *buf, char *out_buf, int out_size) {
    const char *p = strstr(buf, "\"session_id\"");
    if (!p) return 0;
    p += 12; // skip "session_id"
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return 0;
    p++; // skip opening quote
    int i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        out_buf[i++] = *p++;
    }
    out_buf[i] = '\0';
    return i > 0 ? 1 : 0;
}

// Extract tool results from messages for continuing after tool call.
// Returns allocated string with tool results formatted as <tool_response>, or NULL if none.
// The returned string should be freed by caller.
static char *extract_tool_results(const char *buf) {
    // Look for role: "tool" with tool_call_id and content
    // Format: {"role": "tool", "tool_call_id": "call_xxx", "content": "..."}
    // We need to find all such entries and format them as tool responses
    
    char *results = malloc(131072);  // 128KB should be enough
    if (!results) return NULL;
    results[0] = '\0';
    int results_len = 0;
    
    const char *p = buf;
    while (*p) {
        // Find next role field
        const char *role_start = strstr(p, "\"role\"");
        if (!role_start) break;
        
        // Check if it's "tool"
        const char *colon = strchr(role_start, ':');
        if (!colon) break;
        colon++;
        while (*colon == ' ' || *colon == '\t') colon++;
        if (*colon != '"') { p = colon; continue; }
        colon++;
        
        // Check for "tool"
        if (strncmp(colon, "tool", 4) != 0 || colon[4] != '"') {
            p = colon;
            continue;
        }
        
        // Found tool role, now find tool_call_id
        const char *id_start = strstr(colon, "\"tool_call_id\"");
        if (!id_start) { p = colon; continue; }
        
        const char *id_colon = strchr(id_start, ':');
        if (!id_colon) { p = id_start; continue; }
        id_colon++;
        while (*id_colon == ' ' || *id_colon == '\t') id_colon++;
        if (*id_colon != '"') { p = id_colon; continue; }
        id_colon++;
        
        // Extract tool_call_id
        char tool_call_id[64] = {0};
        int id_len = 0;
        while (*id_colon && *id_colon != '"' && id_len < 63) {
            tool_call_id[id_len++] = *id_colon++;
        }
        
        // Find content after tool_call_id
        const char *content_start = strstr(id_colon, "\"content\"");
        if (!content_start) { p = id_colon; continue; }
        
        const char *content_colon = strchr(content_start, ':');
        if (!content_colon) { p = content_start; continue; }
        content_colon++;
        while (*content_colon == ' ' || *content_colon == '\t') content_colon++;
        if (*content_colon != '"') { p = content_colon; continue; }
        content_colon++;
        
        // Extract content
        char content[65536] = {0};
        int content_len = 0;
        while (*content_colon && content_len < 65535) {
            if (*content_colon == '\\' && *(content_colon+1)) {
                content_colon++;
                switch (*content_colon) {
                    case 'n': content[content_len++] = '\n'; break;
                    case '"': content[content_len++] = '"'; break;
                    case '\\': content[content_len++] = '\\'; break;
                    default: content[content_len++] = *content_colon; break;
                }
                content_colon++;
            } else if (*content_colon == '"') {
                break;
            } else {
                content[content_len++] = *content_colon++;
            }
        }
        
        // Append as tool response
        int written = snprintf(results + results_len, 131072 - results_len,
            "<tool_response>\n%s\n</tool_response>", content);
        if (written > 0) {
            results_len += written;
        }
        
        p = content_colon;
    }
    
    if (results_len == 0) {
        free(results);
        return NULL;
    }
    
    return results;
}

// Check if request contains tool results (continuing after tool call)
__attribute__((unused))
static int has_tool_results(const char *buf) {
    return extract_tool_results(buf) != NULL;
}

// Write a full HTTP response string to fd
static void http_write(int fd, const char *data, int len) {
    int sent = 0;
    while (sent < len) {
        ssize_t w = write(fd, data + sent, len - sent);
        if (w <= 0) break;
        sent += (int)w;
    }
}

static void http_write_str(int fd, const char *s) {
    http_write(fd, s, (int)strlen(s));
}

static long sse_created_timestamp(void) {
    return (long)time(NULL);
}

// Send an SSE chunk with a token delta
// Returns 0 on success, -1 if client disconnected
// Inline JSON-escape helper used by sse_send_delta and sse_send_reasoning_delta.
// Escapes the four characters that are illegal in JSON strings (", \, control
// chars rendered as \n/\r/\t). Truncates if escaped > out_sz - 8 to leave
// room for trailing nul. Returns the escaped string in `out`.
static void sse_escape_token(const char *token_text, char *out, size_t out_sz) {
    char *w = out;
    char *end = out + out_sz - 8;
    for (const char *r = token_text; *r && w < end; r++) {
        switch (*r) {
            case '"':  *w++ = '\\'; *w++ = '"';  break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n';  break;
            case '\r': *w++ = '\\'; *w++ = 'r';  break;
            case '\t': *w++ = '\\'; *w++ = 't';  break;
            default:   *w++ = *r; break;
        }
    }
    *w = '\0';
}

static int sse_send_delta(int fd, const char *request_id, const char *token_text) {
    char chunk[4096];
    char escaped[2048];
    sse_escape_token(token_text, escaped, sizeof(escaped));
    long created = sse_created_timestamp();
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        request_id, created, kApiModelId, escaped);
    server_http_log_block(request_id, "response", "sse chat.completion.chunk", chunk);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

// DeepSeek-style streaming for thinking tokens. Routes the model's `<think>...
// </think>` content into `delta.reasoning_content` instead of `delta.content`,
// so OpenAI-extension-aware clients can render reasoning separately from the
// final assistant response. Used while in_think is true; once `</think>` fires
// the loop reverts to sse_send_delta for the actual response body.
static int sse_send_reasoning_delta(int fd, const char *request_id, const char *token_text) {
    char chunk[4096];
    char escaped[2048];
    sse_escape_token(token_text, escaped, sizeof(escaped));
    long created = sse_created_timestamp();
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        request_id, created, kApiModelId, escaped);
    server_http_log_block(request_id, "response", "sse chat.completion.reasoning_chunk", chunk);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void sse_send_done(int fd, const char *request_id) {
    char chunk[1024];
    long created = sse_created_timestamp();
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n",
        request_id, created, kApiModelId);
    server_http_log_block(request_id, "response", "sse chat.completion.done", chunk);
    http_write(fd, chunk, n);
}

// Send SSE chunk with tool_calls (for OpenAI-compatible tool calling)
// Returns 0 on success, -1 if client disconnected
static int sse_send_tool_calls(int fd, const char *request_id, const char *tool_call_id, 
                                const char *function_name, const char *arguments) {
    char escaped_name[256];
    char *escaped_args = json_escape_alloc(arguments ?: "{}");
    if (!escaped_args) return -1;

    char *w = escaped_name;
    for (const char *r = function_name; *r && w < escaped_name + sizeof(escaped_name) - 2; r++) {
        if (*r == '"' || *r == '\\') { *w++ = '\\'; }
        *w++ = *r;
    }
    *w = '\0';

    size_t chunk_cap = strlen(escaped_args) + strlen(request_id) + strlen(kApiModelId ?: "") + strlen(tool_call_id) + strlen(escaped_name) + 1024;
    char *chunk = malloc(chunk_cap);
    if (!chunk) {
        free(escaped_args);
        return -1;
    }
    long created = sse_created_timestamp();
    int n = snprintf(chunk, chunk_cap,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"%s\",\"type\":\"function\","
        "\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        request_id, created, kApiModelId, tool_call_id, escaped_name, escaped_args);
    server_http_log_block(request_id, "response", "sse chat.completion.tool_calls", chunk);
    
    ssize_t wr = write(fd, chunk, n);
    if (wr <= 0) {
        free(chunk);
        free(escaped_args);
        return -1;
    }
    
    // Send done after tool_calls
    created = sse_created_timestamp();
    n = snprintf(chunk, chunk_cap,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
        "data: [DONE]\n\n",
        request_id, created, kApiModelId);
    server_http_log_block(request_id, "response", "sse chat.completion.tool_calls.done", chunk);
    wr = write(fd, chunk, n);
    int rc = (wr <= 0) ? -1 : 0;
    free(chunk);
    free(escaped_args);
    return rc;
}

static const char *SSE_HEADERS =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "\r\n";

static int sse_send_initial_role_chunk(int fd, const char *request_id) {
    char chunk[1024];
    long created = sse_created_timestamp();
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        request_id, created, kApiModelId);
    server_http_log_block(request_id, "response", "sse chat.completion.initial_role", chunk);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static const char *CORS_RESPONSE =
    "HTTP/1.1 204 No Content\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
    "Access-Control-Max-Age: 86400\r\n"
    "\r\n";

static int json_escape_cstr(const char *src, char *buf, int bufsize) {
    int j = 0;
    for (int i = 0; src && src[i] && j < bufsize - 8; i++) {
        switch (src[i]) {
            case '"':  buf[j++]='\\'; buf[j++]='"'; break;
            case '\\': buf[j++]='\\'; buf[j++]='\\'; break;
            case '\n': buf[j++]='\\'; buf[j++]='n'; break;
            case '\r': buf[j++]='\\'; buf[j++]='r'; break;
            case '\t': buf[j++]='\\'; buf[j++]='t'; break;
            default:   buf[j++]=src[i]; break;
        }
    }
    buf[j] = 0;
    return j;
}

static char *json_escape_alloc(const char *src) {
    if (!src) return strdup("");
    size_t len = strlen(src);
    size_t cap = len * 2 + 1;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    size_t j = 0;
    for (size_t i = 0; src[i]; i++) {
        if (j + 3 >= cap) {
            cap *= 2;
            char *grown = realloc(buf, cap);
            if (!grown) {
                free(buf);
                return NULL;
            }
            buf = grown;
        }
        switch (src[i]) {
            case '"':  buf[j++]='\\'; buf[j++]='"'; break;
            case '\\': buf[j++]='\\'; buf[j++]='\\'; break;
            case '\n': buf[j++]='\\'; buf[j++]='n'; break;
            case '\r': buf[j++]='\\'; buf[j++]='r'; break;
            case '\t': buf[j++]='\\'; buf[j++]='t'; break;
            default:   buf[j++]=src[i]; break;
        }
    }
    buf[j] = 0;
    return buf;
}

static void send_json_error(int fd, int status_code, const char *type, const char *message) {
    const char *status_text = "Bad Request";
    if (status_code == 500) status_text = "Internal Server Error";
    else if (status_code == 404) status_text = "Not Found";
    char escaped[4096];
    json_escape_cstr(message ?: "unknown error", escaped, sizeof(escaped));
    char body[4608];
    snprintf(body, sizeof(body),
             "{\"error\":{\"message\":\"%s\",\"type\":\"%s\"}}\n",
             escaped, type ? type : "invalid_request_error");
    char header[512];
    snprintf(header, sizeof(header),
             "HTTP/1.1 %d %s\r\n"
             "Content-Type: application/json\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Connection: close\r\n"
             "Content-Length: %zu\r\n"
             "\r\n",
             status_code, status_text, strlen(body));
    char full[6144];
    snprintf(full, sizeof(full), "%s%s", header, body);
    server_http_log_block(NULL, "response", "json error", full);
    http_write_str(fd, header);
    http_write_str(fd, body);
}

static void send_json_ok(int fd, const char *body) {
    char header[512];
    snprintf(header, sizeof(header),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: application/json\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Connection: close\r\n"
             "Content-Length: %zu\r\n"
             "\r\n",
             strlen(body));
    char *full = malloc(strlen(header) + strlen(body) + 1);
    if (full) {
        strcpy(full, header);
        strcat(full, body);
        server_http_log_block(NULL, "response", "json ok", full);
        free(full);
    }
    http_write_str(fd, header);
    http_write_str(fd, body);
}

// Convert a parameter value emitted by the model into the right JSON type.
// Native Qwen3 tool calls render values as raw text for strings and as
// JSON literals for everything else (`tojson | safe`). On the way back, we
// try JSON.parse first; if it succeeds and yields a non-string, we trust
// it. Otherwise we treat the value as a string.
static id native_param_value_to_json_type(NSString *raw) {
    NSString *trimmed = [raw stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if ([trimmed length] == 0) return @"";
    NSData *data = [trimmed dataUsingEncoding:NSUTF8StringEncoding];
    NSError *err = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data
                                                options:NSJSONReadingFragmentsAllowed
                                                  error:&err];
    if (parsed && ![parsed isKindOfClass:[NSString class]]) {
        // Numbers, booleans, null, objects, arrays — trust the model's typing.
        return parsed;
    }
    // Strings and parse failures: keep raw (un-trimmed for fidelity).
    return raw;
}

// Parse the native Qwen3 XML tool-call form from the streaming output buffer:
//
//   <tool_call>
//   <function=NAME>
//   <parameter=KEY1>
//   value...
//   </parameter>
//   <parameter=KEY2>
//   value...
//   </parameter>
//   </function>
//   </tool_call>
//
// Strings come through raw, non-strings come through JSON-encoded — see
// `native_param_value_to_json_type`. Returns 1 only when the FULL block has
// arrived (closing `</tool_call>` present) so the caller can break out of
// the generation loop.
static int parse_tool_call_from_buffer(const char *tool_call_buf, ParsedToolCall *parsed) {
    memset(parsed, 0, sizeof(*parsed));
    const char *tc_start = strstr(tool_call_buf, "<tool_call>");
    if (!tc_start) return 0;
    const char *tc_end = strstr(tc_start, "</tool_call>");
    if (!tc_end) return 0;

    const char *body = tc_start + strlen("<tool_call>");
    const char *fn_open = strstr(body, "<function=");
    if (!fn_open || fn_open >= tc_end) return 0;
    fn_open += strlen("<function=");
    const char *fn_name_end = strchr(fn_open, '>');
    if (!fn_name_end || fn_name_end >= tc_end) return 0;
    size_t name_len = (size_t)(fn_name_end - fn_open);
    if (name_len == 0 || name_len >= sizeof(parsed->name)) return 0;
    char name_buf[sizeof(parsed->name)];
    memcpy(name_buf, fn_open, name_len);
    name_buf[name_len] = '\0';

    NSMutableDictionary *args = [NSMutableDictionary dictionary];
    const char *cursor = fn_name_end + 1;
    const char *fn_close = strstr(cursor, "</function>");
    if (!fn_close || fn_close > tc_end) fn_close = tc_end;

    while (cursor < fn_close) {
        const char *p_open = strstr(cursor, "<parameter=");
        if (!p_open || p_open >= fn_close) break;
        p_open += strlen("<parameter=");
        const char *p_key_end = strchr(p_open, '>');
        if (!p_key_end || p_key_end >= fn_close) break;
        size_t key_len = (size_t)(p_key_end - p_open);
        if (key_len == 0 || key_len > 128) { cursor = p_key_end + 1; continue; }

        // Value begins right after the '>'; the template emits a leading
        // newline before the value, so skip exactly one if present.
        const char *val_start = p_key_end + 1;
        if (*val_start == '\n') val_start++;

        // The closing tag is `\n</parameter>` (template puts \n before it),
        // but be lenient and accept either form.
        const char *p_close = strstr(val_start, "</parameter>");
        if (!p_close || p_close > fn_close) break;
        const char *val_end = p_close;
        if (val_end > val_start && val_end[-1] == '\n') val_end--;

        NSString *key = [[NSString alloc] initWithBytes:p_open length:key_len encoding:NSUTF8StringEncoding];
        NSString *raw_value = [[NSString alloc] initWithBytes:val_start
                                                        length:(NSUInteger)(val_end - val_start)
                                                      encoding:NSUTF8StringEncoding];
        if (key) {
            args[key] = native_param_value_to_json_type(raw_value ?: @"");
        }
        cursor = p_close + strlen("</parameter>");
    }

    strncpy(parsed->name, name_buf, sizeof(parsed->name) - 1);
    NSString *args_json = json_stringify_obj(args);
    parsed->arguments = dup_nsstring(args_json);
    parsed->is_tool_call = 1;
    return 1;
}

static char *read_text_file_alloc(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t n = fread(buf, 1, (size_t)sz, f);
    buf[n] = 0;
    fclose(f);
    return buf;
}

static int write_text_file(const char *dir, const char *name, const char *text) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (text) fwrite(text, 1, strlen(text), f);
    fclose(f);
    return 0;
}

static int render_request_debug(const char *request_path, const char *output_dir, const char *kind_name) {
    char *body = read_text_file_alloc(request_path);
    if (!body) {
        fprintf(stderr, "ERROR: cannot read request JSON: %s\n", request_path);
        return 1;
    }

    NSError *json_error = nil;
    NSDictionary *root = parse_json_body(body, &json_error);
    if (!root) {
        fprintf(stderr, "ERROR: invalid request JSON: %s\n",
                json_error ? [[json_error localizedDescription] UTF8String] : "unknown error");
        free(body);
        return 1;
    }

    ApiKind kind = API_KIND_CHAT;
    if (kind_name && strcmp(kind_name, "responses") == 0) {
        kind = API_KIND_RESPONSES;
    } else if ((!kind_name || strcmp(kind_name, "auto") == 0) && root[@"input"]) {
        kind = API_KIND_RESPONSES;
    }

    ApiRequest req;
    api_request_init(&req, kind);
    char *err_msg = NULL;
    int fill_rc = (kind == API_KIND_RESPONSES)
        ? fill_request_from_responses_json(root, &req, &err_msg)
        : fill_request_from_chat_json(root, &req, &err_msg);
    if (fill_rc != 0) {
        fprintf(stderr, "ERROR: %s\n", err_msg ? err_msg : "request parse failed");
        free(err_msg);
        api_request_free(&req);
        free(body);
        return 1;
    }

    PromptBuildInfo info;
    char *sys_prompt = build_system_prompt_for_request(&req, &info);
    size_t total = strlen(sys_prompt) + strlen(req.conversation_text ?: "") + 128;
    char *assembled = malloc(total);
    if (!assembled) {
        fprintf(stderr, "ERROR: render allocation failed\n");
        free(sys_prompt);
        api_request_free(&req);
        free(body);
        return 1;
    }
    snprintf(assembled, total, "<|im_start|>system\n%s<|im_end|>\n%s",
             sys_prompt, req.conversation_text ?: "");

    int sys_tokens = count_sys_prompt_tokens(sys_prompt);
    PromptTokens *full_tokens = encode_prompt_text_to_tokens(assembled);
    int full_token_count = full_tokens ? full_tokens->count : -1;
    if (full_tokens) {
        free(full_tokens->ids);
        free(full_tokens);
    }

    printf("kind=%s\n", kind == API_KIND_RESPONSES ? "responses" : "chat");
    printf("tools=%d\n", req.tool_count);
    printf("tool_choice=%d\n", req.tool_choice_mode);
    printf("system_chars=%zu\n", strlen(sys_prompt));
    printf("conversation_chars=%zu\n", strlen(req.conversation_text ?: ""));
    printf("assembled_chars=%zu\n", strlen(assembled));
    printf("system_tokens=%d\n", sys_tokens);
    printf("assembled_tokens=%d\n", full_token_count);
    printf("temperature=%.3f\n", req.temperature);
    printf("top_p=%.3f\n", req.top_p);
    printf("top_k=%d\n", req.top_k);
    printf("min_p=%.3f\n", req.min_p);
    printf("presence_penalty=%.3f\n", req.presence_penalty);
    printf("repetition_penalty=%.3f\n", req.repetition_penalty);

    if (output_dir && output_dir[0]) {
        mkdir(output_dir, 0755);
        write_text_file(output_dir, "request.json", body);
        write_text_file(output_dir, "system_prompt.txt", sys_prompt);
        write_text_file(output_dir, "conversation.txt", req.conversation_text ?: "");
        write_text_file(output_dir, "assembled_prompt.txt", assembled);
        char summary[2048];
        snprintf(summary, sizeof(summary),
                 "{\n"
                 "  \"kind\": \"%s\",\n"
                 "  \"tools\": %d,\n"
                 "  \"tool_choice\": %d,\n"
                 "  \"system_chars\": %zu,\n"
                 "  \"conversation_chars\": %zu,\n"
                 "  \"assembled_chars\": %zu,\n"
                 "  \"system_tokens\": %d,\n"
                 "  \"assembled_tokens\": %d,\n"
                 "  \"temperature\": %.3f,\n"
                 "  \"top_p\": %.3f,\n"
                 "  \"top_k\": %d,\n"
                 "  \"min_p\": %.3f,\n"
                 "  \"presence_penalty\": %.3f,\n"
                 "  \"repetition_penalty\": %.3f\n"
                 "}\n",
                 kind == API_KIND_RESPONSES ? "responses" : "chat",
                 req.tool_count, req.tool_choice_mode,
                 strlen(sys_prompt), strlen(req.conversation_text ?: ""),
                 strlen(assembled), sys_tokens, full_token_count,
                 req.temperature, req.top_p, req.top_k, req.min_p,
                 req.presence_penalty, req.repetition_penalty);
        write_text_file(output_dir, "summary.json", summary);
    }

    free(assembled);
    free(sys_prompt);
    api_request_free(&req);
    free(body);
    return 0;
}

static int parse_tool_call_debug(const char *path) {
    char *buf = read_text_file_alloc(path);
    if (!buf) {
        fprintf(stderr, "ERROR: cannot read tool-call text: %s\n", path);
        return 1;
    }
    ParsedToolCall parsed;
    memset(&parsed, 0, sizeof(parsed));
    if (!parse_tool_call_from_buffer(buf, &parsed)) {
        fprintf(stderr, "ERROR: no complete native XML tool call found\n");
        free(buf);
        return 1;
    }
    char escaped_name[256];
    char *escaped_args = json_escape_alloc(parsed.arguments ?: "{}");
    if (!escaped_args) {
        parsed_tool_call_free(&parsed);
        free(buf);
        return 1;
    }
    json_escape_cstr(parsed.name, escaped_name, sizeof(escaped_name));
    printf("{\"name\":\"%s\",\"arguments\":\"%s\"}\n", escaped_name, escaped_args);
    free(escaped_args);
    parsed_tool_call_free(&parsed);
    free(buf);
    return 0;
}

static int sse_send_response_text_delta(int fd, const char *response_id, const char *text) {
    char escaped[2048];
    json_escape_cstr(text, escaped, sizeof(escaped));
    char chunk[4096];
    int n = snprintf(chunk, sizeof(chunk),
                     "event: response.output_text.delta\n"
                     "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"%s\",\"delta\":\"%s\"}\n\n",
                     response_id, escaped);
    server_http_log_block(response_id, "response", "sse response.output_text.delta", chunk);
    return write(fd, chunk, n) <= 0 ? -1 : 0;
}

static int sse_send_response_tool_call(int fd, const char *response_id, const ParsedToolCall *tool_call) {
    char escaped_name[256];
    char *escaped_args = json_escape_alloc(tool_call->arguments ?: "{}");
    if (!escaped_args) return -1;
    json_escape_cstr(tool_call->name, escaped_name, sizeof(escaped_name));
    size_t chunk_cap = strlen(escaped_args) + strlen(response_id) + strlen(tool_call->id) + strlen(escaped_name) + 1024;
    char *chunk = malloc(chunk_cap);
    if (!chunk) {
        free(escaped_args);
        return -1;
    }
    int n = snprintf(chunk, chunk_cap,
                     "event: response.function_call_arguments.delta\n"
                     "data: {\"type\":\"response.function_call_arguments.delta\",\"response_id\":\"%s\",\"call_id\":\"%s\",\"name\":\"%s\",\"delta\":\"%s\"}\n\n",
                     response_id, tool_call->id, escaped_name, escaped_args);
    server_http_log_block(response_id, "response", "sse response.function_call_arguments.delta", chunk);
    if (write(fd, chunk, n) <= 0) {
        free(chunk);
        free(escaped_args);
        return -1;
    }
    n = snprintf(chunk, chunk_cap,
                 "event: response.output_item.done\n"
                 "data: {\"type\":\"response.output_item.done\",\"response_id\":\"%s\",\"item\":{\"type\":\"function_call\",\"call_id\":\"%s\",\"name\":\"%s\",\"arguments\":\"%s\"}}\n\n",
                 response_id, tool_call->id, escaped_name, escaped_args);
    server_http_log_block(response_id, "response", "sse response.output_item.done", chunk);
    int rc = write(fd, chunk, n) <= 0 ? -1 : 0;
    free(chunk);
    free(escaped_args);
    return rc;
}

static void sse_send_response_done(int fd, const char *response_id, const char *final_json) {
    char chunk[8192];
    int n = snprintf(chunk, sizeof(chunk),
                     "event: response.completed\n"
                     "data: %s\n\n"
                     "data: [DONE]\n\n",
                     final_json);
    (void)response_id;
    server_http_log_block(response_id, "response", "sse response.completed", chunk);
    http_write(fd, chunk, n);
}

// Build the non-streaming OpenAI-compatible chat completion response.
// `text` is the post-`</think>` assistant content; `reasoning_text` is the
// in-`<think>` content (DeepSeek-style separated reasoning). Pass NULL or ""
// for reasoning_text to omit the field. When a tool_call fires, content is
// "" by convention and tool_calls carries the structured call; reasoning is
// still included if present so clients can show "the model thought X then
// decided to call Y".
static char *build_chat_completion_json(const char *request_id, const char *model,
                                        const char *text, const char *reasoning_text,
                                        const ParsedToolCall *tool_call) {
    char *escaped_text = json_escape_alloc(text ?: "");
    if (!escaped_text) return NULL;
    char *escaped_reasoning = NULL;
    if (reasoning_text && reasoning_text[0]) {
        escaped_reasoning = json_escape_alloc(reasoning_text);
        if (!escaped_reasoning) {
            free(escaped_text);
            return NULL;
        }
    }
    // Optional `,"reasoning_content":"<escaped>"` fragment, empty when absent.
    char *reasoning_field = NULL;
    if (escaped_reasoning) {
        size_t rf_cap = strlen(escaped_reasoning) + 32;
        reasoning_field = malloc(rf_cap);
        if (!reasoning_field) {
            free(escaped_reasoning);
            free(escaped_text);
            return NULL;
        }
        snprintf(reasoning_field, rf_cap, ",\"reasoning_content\":\"%s\"", escaped_reasoning);
    }
    const char *rf = reasoning_field ? reasoning_field : "";

    size_t body_cap = strlen(escaped_text) + strlen(rf) + 4096;
    char *body = calloc(1, body_cap);
    if (!body) {
        free(reasoning_field);
        free(escaped_reasoning);
        free(escaped_text);
        return NULL;
    }
    if (tool_call && tool_call->is_tool_call) {
        char escaped_name[256];
        char *escaped_args = json_escape_alloc(tool_call->arguments ?: "{}");
        if (!escaped_args) {
            free(body);
            free(reasoning_field);
            free(escaped_reasoning);
            free(escaped_text);
            return NULL;
        }
        json_escape_cstr(tool_call->name, escaped_name, sizeof(escaped_name));
        body_cap = strlen(escaped_args) + strlen(rf) + strlen(request_id) + strlen(model) + strlen(escaped_name) + 4096;
        char *grown = realloc(body, body_cap);
        if (!grown) {
            free(escaped_args);
            free(body);
            free(reasoning_field);
            free(escaped_reasoning);
            free(escaped_text);
            return NULL;
        }
        body = grown;
        snprintf(body, body_cap,
                 "{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\","
                 "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\"%s,"
                 "\"tool_calls\":[{\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}]},"
                 "\"finish_reason\":\"tool_calls\"}]}\n",
                 request_id, model, rf, tool_call->id, escaped_name, escaped_args);
        free(escaped_args);
    } else {
        snprintf(body, body_cap,
                 "{\"id\":\"%s\",\"object\":\"chat.completion\",\"model\":\"%s\","
                 "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"%s},"
                 "\"finish_reason\":\"stop\"}]}\n",
                 request_id, model, escaped_text, rf);
    }
    free(reasoning_field);
    free(escaped_reasoning);
    free(escaped_text);
    return body;
}

static char *build_responses_json(const char *response_id, const char *model,
                                  const char *text, const ParsedToolCall *tool_call) {
    char *escaped_text = json_escape_alloc(text ?: "");
    if (!escaped_text) return NULL;
    size_t body_cap = strlen(escaped_text) * 2 + 4096;
    char *body = calloc(1, body_cap);
    if (!body) {
        free(escaped_text);
        return NULL;
    }
    if (tool_call && tool_call->is_tool_call) {
        char escaped_name[256];
        char *escaped_args = json_escape_alloc(tool_call->arguments ?: "{}");
        if (!escaped_args) {
            free(body);
            free(escaped_text);
            return NULL;
        }
        json_escape_cstr(tool_call->name, escaped_name, sizeof(escaped_name));
        body_cap = strlen(escaped_args) + strlen(response_id) + strlen(model) + strlen(escaped_name) + 4096;
        char *grown = realloc(body, body_cap);
        if (!grown) {
            free(escaped_args);
            free(body);
            free(escaped_text);
            return NULL;
        }
        body = grown;
        snprintf(body, body_cap,
                 "{\"id\":\"%s\",\"object\":\"response\",\"model\":\"%s\",\"status\":\"completed\","
                 "\"output\":[{\"type\":\"function_call\",\"call_id\":\"%s\",\"name\":\"%s\",\"arguments\":\"%s\"}]}\n",
                 response_id, model, tool_call->id, escaped_name, escaped_args);
        free(escaped_args);
    } else {
        snprintf(body, body_cap,
                 "{\"id\":\"%s\",\"object\":\"response\",\"model\":\"%s\",\"status\":\"completed\","
                 "\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"%s\"}]}],"
                 "\"output_text\":\"%s\"}\n",
                 response_id, model, escaped_text, escaped_text);
    }
    free(escaped_text);
    return body;
}

// Tokenize a user turn (system prompt already cached in KV).
// Only encodes: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
__attribute__((unused))
static PromptTokens *tokenize_user_turn(const char *user_content) {
    const char *prefix = "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Tokenize a continuation turn for session caching.
// Prefixes with <|im_end|>\n to close the previous assistant turn, then the new user turn.
// Used when the KV cache already contains the prior conversation state.
__attribute__((unused))
static PromptTokens *tokenize_continuation_turn(const char *user_content) {
    // EOS/<|im_end|> is already in the state (fed through model at end of generation)
    // Just need the newline + new user turn + assistant prompt
    const char *prefix = "\n<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Load custom system prompt from ~/.config/flashchat/system.md, or use default
static char *load_system_prompt(void) {
    const char *home = getenv("HOME");
    char path[1024] = {0};
    const char *prompt_env = getenv("FLASHCHAT_SYSTEM_PROMPT");
    if (prompt_env && prompt_env[0]) {
        snprintf(path, sizeof(path), "%s", prompt_env);
    } else if (home) {
        snprintf(path, sizeof(path), "%s/.config/flashchat/system.md", home);
    }
    if (path[0]) {
        FILE *f = fopen(path, "r");
        if (!f) return strdup("You are a helpful assistant. /think");
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = malloc(sz + 1);
        size_t n = fread(buf, 1, sz, f);
        buf[n] = 0;
        fclose(f);
        fprintf(stderr, "[serve] Loaded custom system prompt from %s (%ld bytes)\n", path, sz);
        return buf;
    }
    return strdup("You are a helpful assistant. /think");
}

// Build the native Qwen3 tool-block exactly as the model's chat_template
// emits it: a `# Tools` header, a `<tools>...</tools>` block containing one
// full OpenAI tool JSON object per line, the canonical `<tool_call>` example,
// and the `<IMPORTANT>` reminder paragraph. The model was post-trained on
// these exact strings; do not paraphrase.
//
// Emitted text starts on its own line and DOES NOT have a trailing newline,
// so the caller can append "\n\n{user_system_content}" or close the
// <|im_start|>system block directly after.
static char *build_tool_instructions(ToolDef *tools, int tool_count) {
    if (tool_count == 0) return strdup("");

    NSMutableString *out = [NSMutableString string];
    [out appendString:@"# Tools\n\nYou have access to the following functions:\n\n<tools>"];
    for (int i = 0; i < tool_count; i++) {
        char name_esc[MAX_TOOL_NAME * 2];
        char desc_esc[MAX_TOOL_DESC * 2];
        json_escape_cstr(tools[i].name, name_esc, sizeof(name_esc));
        json_escape_cstr(tools[i].description, desc_esc, sizeof(desc_esc));
        const char *params = (tools[i].has_parameters && tools[i].parameters && tools[i].parameters[0])
            ? tools[i].parameters
            : "{}";
        // Match the template's tojson key order (function before type) — not
        // load-bearing for the model but easier to diff against LMStudio.
        [out appendFormat:@"\n{\"function\": {\"description\": \"%s\", \"name\": \"%s\", \"parameters\": %s}, \"type\": \"function\"}",
                          desc_esc, name_esc, params];
    }
    [out appendString:@"\n</tools>\n\n"];
    [out appendString:
        @"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        @"<tool_call>\n"
        @"<function=example_function_name>\n"
        @"<parameter=example_parameter_1>\n"
        @"value_1\n"
        @"</parameter>\n"
        @"<parameter=example_parameter_2>\n"
        @"This is the value for the second parameter\n"
        @"that can span\n"
        @"multiple lines\n"
        @"</parameter>\n"
        @"</function>\n"
        @"</tool_call>\n\n"
        @"<IMPORTANT>\n"
        @"Reminder:\n"
        @"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        @"- Required parameters MUST be specified\n"
        @"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        @"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        @"</IMPORTANT>"];
    return dup_nsstring(out);
}

// Tokenize a full chat message (system prompt + user turn) for first-time use.
__attribute__((unused))
static PromptTokens *tokenize_chat_message(const char *user_content) {
    static char *sys_prompt_text = NULL;
    if (!sys_prompt_text) sys_prompt_text = load_system_prompt();

    // Build: <|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
    size_t sys_len = strlen(sys_prompt_text);
    size_t user_len = strlen(user_content);
    size_t total = 30 + sys_len + 30 + user_len + 40;  // generous padding for tags
    char *prompt = malloc(total);
    if (!prompt) return NULL;
    snprintf(prompt, total, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             sys_prompt_text, user_content);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Keep old signature for backward compat (unused but prevents compiler warning)
__attribute__((unused))
static PromptTokens *tokenize_chat_message_old(const char *user_content) {
    const char *prefix =
        "<|im_start|>system\nYou are a helpful assistant. /think<|im_end|>\n"
        "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;

    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// The main serve loop. Model state must already be initialized.
// Sync CPU linear attention state → GPU buffers
__attribute__((unused))
static void sync_cpu_to_gpu_delta_state_serve(void **layer_states) {
    if (!g_metal || !g_metal->delta_net_step || !layer_states) return;
    int li = 0;
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if ((i + 1) % g_cfg.full_attn_interval == 0) continue;
        if (!layer_states[i]) { li++; continue; }
        LinearAttnState *la = (LinearAttnState *)layer_states[i];
        if (li < g_cfg.num_linear_layers) {
            if (g_metal->buf_delta_state[li] && la->ssm_state)
                memcpy([g_metal->buf_delta_state[li] contents], la->ssm_state,
                       g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float));
            if (g_metal->buf_conv_state[li] && la->conv_state)
                memcpy([g_metal->buf_conv_state[li] contents], la->conv_state,
                       (g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float));
        }
        li++;
    }
}

__attribute__((unused))
static void clear_runtime_state_serve(void **layer_states, KVCache **kv_caches) {
    size_t kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    size_t conv_state_size = (g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float);
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (kv_caches[i]) {
            kv_caches[i]->len = 0;
            if (kv_caches[i]->k_cache) memset(kv_caches[i]->k_cache, 0, (size_t)GPU_KV_SEQ * kv_dim * sizeof(float));
            if (kv_caches[i]->v_cache) memset(kv_caches[i]->v_cache, 0, (size_t)GPU_KV_SEQ * kv_dim * sizeof(float));
        }
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            memset(s->conv_state, 0, conv_state_size);
            memset(s->ssm_state, 0, ssm_state_size);
        }
    }
    reset_delta_net_state();
}

typedef struct {
    float *k_snapshot;
    float *v_snapshot;
    int len;
} KVSnapshot;

enum {
    SYSPROMPT_CACHE_VERSION = 1,
    SYSPROMPT_CACHE_KIND_KV_K = 1,
    SYSPROMPT_CACHE_KIND_KV_V = 2,
    SYSPROMPT_CACHE_KIND_LA_CONV = 3,
    SYSPROMPT_CACHE_KIND_LA_SSM = 4,
    SYSPROMPT_CACHE_KIND_GPU_DELTA = 5,
    SYSPROMPT_CACHE_KIND_GPU_CONV = 6,
    SYSPROMPT_CACHE_ALG_RAW = 0,
    SYSPROMPT_CACHE_ALG_LZFSE = 1,
};

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    char model_id[64];
    uint64_t prompt_hash;
    uint32_t token_count;
    uint32_t num_layers;
    uint32_t num_full_attn_layers;
    uint32_t num_linear_layers;
    uint32_t kv_dim;
    uint32_t compression;
    uint64_t conv_state_size;
    uint64_t ssm_state_size;
    uint64_t gpu_delta_size;
    uint64_t gpu_conv_size;
    uint32_t chunk_count;
    uint32_t reserved;
    uint64_t raw_bytes;
    uint64_t stored_bytes;
} SysPromptCacheHeader;

typedef struct {
    uint32_t kind;
    int32_t layer;
    uint32_t algorithm;
    uint32_t reserved;
    uint64_t raw_size;
    uint64_t stored_size;
    uint64_t checksum;
} SysPromptCacheChunk;

static uint64_t fnv1a_bytes(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int mkdir_p_cstr(const char *path) {
    if (!path || !path[0]) return -1;
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0) return -1;
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

static int system_prompt_cache_dir(const char *model_path, char *out, size_t out_sz) {
    if (!model_path || !model_path[0] || !out || out_sz == 0) return -1;
    snprintf(out, out_sz, "%s/flashchat/system_prompt_cache", model_path);
    return 0;
}

static int system_prompt_cache_path(const char *model_path, uint64_t hash, char *out, size_t out_sz) {
    char dir[PATH_MAX];
    if (system_prompt_cache_dir(model_path, dir, sizeof(dir)) != 0) return -1;
    snprintf(out, out_sz, "%s/%016llx-v%d.fcache", dir, (unsigned long long)hash, SYSPROMPT_CACHE_VERSION);
    return 0;
}

static size_t serve_gpu_delta_snapshot_size(void) {
    return 64 * 128 * 128 * sizeof(float);
}

static size_t serve_gpu_conv_snapshot_size(void) {
    return 3 * 12288 * sizeof(float);
}

static int write_cache_chunk(FILE *f, uint32_t kind, int32_t layer,
                             const void *src, size_t src_size,
                             uint64_t *raw_total, uint64_t *stored_total) {
    if (!f || !src || src_size == 0) return -1;

    SysPromptCacheChunk chunk;
    memset(&chunk, 0, sizeof(chunk));
    chunk.kind = kind;
    chunk.layer = layer;
    chunk.raw_size = src_size;
    chunk.checksum = fnv1a_bytes(src, src_size);

    void *stored = (void *)src;
    size_t stored_size = src_size;
    void *compressed = NULL;

    if (src_size > 1024) {
        compressed = malloc(src_size);
        if (compressed) {
            size_t n = compression_encode_buffer((uint8_t *)compressed, src_size,
                                                 (const uint8_t *)src, src_size,
                                                 NULL, COMPRESSION_LZFSE);
            if (n > 0 && n + 4096 < src_size) {
                stored = compressed;
                stored_size = n;
                chunk.algorithm = SYSPROMPT_CACHE_ALG_LZFSE;
            }
        }
    }

    chunk.stored_size = stored_size;
    if (fwrite(&chunk, 1, sizeof(chunk), f) != sizeof(chunk) ||
        fwrite(stored, 1, stored_size, f) != stored_size) {
        free(compressed);
        return -1;
    }

    if (raw_total) *raw_total += src_size;
    if (stored_total) *stored_total += stored_size;
    free(compressed);
    return 0;
}

static void free_system_prompt_snapshots(KVSnapshot *kv_snapshots,
                                         float **la_conv_snapshots,
                                         float **la_ssm_snapshots,
                                         void **gpu_delta_snapshots,
                                         void **gpu_conv_snapshots) {
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (kv_snapshots) {
            free(kv_snapshots[i].k_snapshot);
            free(kv_snapshots[i].v_snapshot);
            kv_snapshots[i].k_snapshot = NULL;
            kv_snapshots[i].v_snapshot = NULL;
            kv_snapshots[i].len = 0;
        }
        if (la_conv_snapshots) {
            free(la_conv_snapshots[i]);
            la_conv_snapshots[i] = NULL;
        }
        if (la_ssm_snapshots) {
            free(la_ssm_snapshots[i]);
            la_ssm_snapshots[i] = NULL;
        }
    }
    for (int i = 0; i < MAX_LINEAR_LAYERS; i++) {
        if (gpu_delta_snapshots) {
            free(gpu_delta_snapshots[i]);
            gpu_delta_snapshots[i] = NULL;
        }
        if (gpu_conv_snapshots) {
            free(gpu_conv_snapshots[i]);
            gpu_conv_snapshots[i] = NULL;
        }
    }
}

static int restore_system_prompt_snapshots(int token_count,
                                           KVSnapshot *kv_snapshots,
                                           float **la_conv_snapshots,
                                           float **la_ssm_snapshots,
                                           void **gpu_delta_snapshots,
                                           void **gpu_conv_snapshots,
                                           void **layer_states,
                                           KVCache **kv_caches) {
    size_t kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    size_t conv_state_size = (g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float);

    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (kv_caches[i] && kv_snapshots[i].k_snapshot && kv_snapshots[i].v_snapshot) {
            size_t sz = (size_t)token_count * kv_dim * sizeof(float);
            memcpy(kv_caches[i]->k_cache, kv_snapshots[i].k_snapshot, sz);
            memcpy(kv_caches[i]->v_cache, kv_snapshots[i].v_snapshot, sz);
            kv_caches[i]->len = kv_snapshots[i].len;
            if (g_metal) {
                int fa_idx = (i + 1) / g_cfg.full_attn_interval - 1;
                if (fa_idx >= 0 && fa_idx < g_cfg.num_full_attn_layers) {
                    size_t gpu_tokens = token_count < GPU_KV_SEQ ? (size_t)token_count : (size_t)GPU_KV_SEQ;
                    size_t gpu_sz = gpu_tokens * kv_dim * sizeof(float);
                    memcpy([g_metal->buf_kv_k[fa_idx] contents], kv_snapshots[i].k_snapshot, gpu_sz);
                    memcpy([g_metal->buf_kv_v[fa_idx] contents], kv_snapshots[i].v_snapshot, gpu_sz);
                }
            }
        }
        if (layer_states[i] && la_conv_snapshots[i] && la_ssm_snapshots[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            memcpy(s->conv_state, la_conv_snapshots[i], conv_state_size);
            memcpy(s->ssm_state, la_ssm_snapshots[i], ssm_state_size);
        }
    }
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < g_cfg.num_linear_layers; i++) {
            if (gpu_delta_snapshots[i] && g_metal->buf_delta_state[i]) {
                memcpy([g_metal->buf_delta_state[i] contents], gpu_delta_snapshots[i], serve_gpu_delta_snapshot_size());
            }
            if (gpu_conv_snapshots[i] && g_metal->buf_conv_state[i]) {
                memcpy([g_metal->buf_conv_state[i] contents], gpu_conv_snapshots[i], serve_gpu_conv_snapshot_size());
            }
        }
    } else {
        reset_delta_net_state();
    }
    return 0;
}

static int capture_system_prompt_snapshots(int token_count,
                                           KVSnapshot *kv_snapshots,
                                           float **la_conv_snapshots,
                                           float **la_ssm_snapshots,
                                           void **gpu_delta_snapshots,
                                           void **gpu_conv_snapshots,
                                           void **layer_states,
                                           KVCache **kv_caches) {
    size_t kv_dim = g_cfg.num_kv_heads * g_cfg.head_dim;
    size_t conv_state_size = (g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float);

    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (kv_caches[i]) {
            size_t sz = (size_t)token_count * kv_dim * sizeof(float);
            free(kv_snapshots[i].k_snapshot);
            free(kv_snapshots[i].v_snapshot);
            kv_snapshots[i].k_snapshot = malloc(sz);
            kv_snapshots[i].v_snapshot = malloc(sz);
            if (!kv_snapshots[i].k_snapshot || !kv_snapshots[i].v_snapshot) return -1;
            memcpy(kv_snapshots[i].k_snapshot, kv_caches[i]->k_cache, sz);
            memcpy(kv_snapshots[i].v_snapshot, kv_caches[i]->v_cache, sz);
            kv_snapshots[i].len = kv_caches[i]->len;
        }
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            free(la_conv_snapshots[i]);
            free(la_ssm_snapshots[i]);
            la_conv_snapshots[i] = malloc(conv_state_size);
            la_ssm_snapshots[i] = malloc(ssm_state_size);
            if (!la_conv_snapshots[i] || !la_ssm_snapshots[i]) return -1;
            memcpy(la_conv_snapshots[i], s->conv_state, conv_state_size);
            memcpy(la_ssm_snapshots[i], s->ssm_state, ssm_state_size);
        }
    }
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < g_cfg.num_linear_layers; i++) {
            if (g_metal->buf_delta_state[i]) {
                size_t sz = serve_gpu_delta_snapshot_size();
                free(gpu_delta_snapshots[i]);
                gpu_delta_snapshots[i] = malloc(sz);
                if (!gpu_delta_snapshots[i]) return -1;
                memcpy(gpu_delta_snapshots[i], [g_metal->buf_delta_state[i] contents], sz);
            }
            if (g_metal->buf_conv_state[i]) {
                size_t sz = serve_gpu_conv_snapshot_size();
                free(gpu_conv_snapshots[i]);
                gpu_conv_snapshots[i] = malloc(sz);
                if (!gpu_conv_snapshots[i]) return -1;
                memcpy(gpu_conv_snapshots[i], [g_metal->buf_conv_state[i] contents], sz);
            }
        }
    }
    return 0;
}

static size_t first_byte_diff(const void *a, const void *b, size_t n);
static int load_system_prompt_disk_cache(const char *model_path, uint64_t prompt_hash, int expected_tokens,
                                         KVSnapshot *kv_snapshots,
                                         float **la_conv_snapshots, float **la_ssm_snapshots,
                                         void **gpu_delta_snapshots, void **gpu_conv_snapshots,
                                         int *loaded_tokens);

// Fingerprint every persistent runtime buffer that affects inference. Used
// to localize "snapshot does not faithfully represent runtime state" bugs:
// fingerprint live state immediately after a cold prefill, then again
// immediately after restore-from-snapshot — any per-buffer mismatch points
// to either incomplete capture, missing-from-snapshot state, or buggy
// restore. Gated by FLASHCHAT_CACHE_FINGERPRINT=1.
typedef struct {
    int valid;
    int token_count;
    int kv_len[MAX_NUM_LAYERS];
    uint64_t fp_kv_k_cpu[MAX_NUM_LAYERS];
    uint64_t fp_kv_v_cpu[MAX_NUM_LAYERS];
    uint64_t fp_conv_cpu[MAX_NUM_LAYERS];
    uint64_t fp_ssm_cpu[MAX_NUM_LAYERS];
    uint64_t fp_kv_k_gpu[MAX_FULL_ATTN_LAYERS];
    uint64_t fp_kv_v_gpu[MAX_FULL_ATTN_LAYERS];
    uint64_t fp_buf_delta[MAX_LINEAR_LAYERS];
    uint64_t fp_buf_conv[MAX_LINEAR_LAYERS];
} RuntimeFingerprint;

static RuntimeFingerprint g_fp_post_cold = {0};

static uint64_t fnv1a64(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int cache_fingerprint_enabled(void) {
    const char *e = getenv("FLASHCHAT_CACHE_FINGERPRINT");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

static void fingerprint_runtime_state(int token_count,
                                      KVCache **kv_caches,
                                      void **layer_states,
                                      RuntimeFingerprint *fp) {
    memset(fp, 0, sizeof(*fp));
    fp->token_count = token_count;
    size_t kv_dim = (size_t)g_cfg.num_kv_heads * g_cfg.head_dim;
    size_t kv_sz = (size_t)token_count * kv_dim * sizeof(float);
    size_t conv_state_size = (size_t)(g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = (size_t)g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float);
    size_t gpu_tokens = token_count < GPU_KV_SEQ ? (size_t)token_count : (size_t)GPU_KV_SEQ;
    size_t gpu_kv_sz = gpu_tokens * kv_dim * sizeof(float);

    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (kv_caches[i]) {
            fp->fp_kv_k_cpu[i] = fnv1a64(kv_caches[i]->k_cache, kv_sz);
            fp->fp_kv_v_cpu[i] = fnv1a64(kv_caches[i]->v_cache, kv_sz);
            fp->kv_len[i] = kv_caches[i]->len;
            if (g_metal) {
                int fa_idx = (i + 1) / g_cfg.full_attn_interval - 1;
                if (fa_idx >= 0 && fa_idx < g_cfg.num_full_attn_layers) {
                    if (g_metal->buf_kv_k[fa_idx])
                        fp->fp_kv_k_gpu[fa_idx] = fnv1a64([g_metal->buf_kv_k[fa_idx] contents], gpu_kv_sz);
                    if (g_metal->buf_kv_v[fa_idx])
                        fp->fp_kv_v_gpu[fa_idx] = fnv1a64([g_metal->buf_kv_v[fa_idx] contents], gpu_kv_sz);
                }
            }
        }
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            if (s->conv_state) fp->fp_conv_cpu[i] = fnv1a64(s->conv_state, conv_state_size);
            if (s->ssm_state) fp->fp_ssm_cpu[i] = fnv1a64(s->ssm_state, ssm_state_size);
        }
    }
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < g_cfg.num_linear_layers; i++) {
            if (g_metal->buf_delta_state[i])
                fp->fp_buf_delta[i] = fnv1a64([g_metal->buf_delta_state[i] contents], serve_gpu_delta_snapshot_size());
            if (g_metal->buf_conv_state[i])
                fp->fp_buf_conv[i] = fnv1a64([g_metal->buf_conv_state[i] contents], serve_gpu_conv_snapshot_size());
        }
    }
    fp->valid = 1;
}

static void diff_runtime_fingerprints(const RuntimeFingerprint *cold,
                                      const RuntimeFingerprint *restored) {
    if (!cold->valid) {
        server_log_errorf("[cache-fp] no baseline captured; first cold prefill must run with FLASHCHAT_CACHE_FINGERPRINT=1\n");
        return;
    }
    if (cold->token_count != restored->token_count) {
        server_log_errorf("[cache-fp] token_count mismatch cold=%d restored=%d (skipping diff)\n",
                          cold->token_count, restored->token_count);
        return;
    }
    int total = 0, kv_cpu = 0, kv_gpu = 0, ls = 0, gpu_lin = 0;
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (cold->fp_kv_k_cpu[i] != restored->fp_kv_k_cpu[i]) {
            server_log_errorf("[cache-fp] DIFF layer=%d kv_k_cpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_kv_k_cpu[i], restored->fp_kv_k_cpu[i]);
            kv_cpu++; total++;
        }
        if (cold->fp_kv_v_cpu[i] != restored->fp_kv_v_cpu[i]) {
            server_log_errorf("[cache-fp] DIFF layer=%d kv_v_cpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_kv_v_cpu[i], restored->fp_kv_v_cpu[i]);
            kv_cpu++; total++;
        }
        if (cold->kv_len[i] != restored->kv_len[i]) {
            server_log_errorf("[cache-fp] DIFF layer=%d kv_len cold=%d restored=%d\n",
                              i, cold->kv_len[i], restored->kv_len[i]);
            kv_cpu++; total++;
        }
        if (cold->fp_conv_cpu[i] != restored->fp_conv_cpu[i]) {
            server_log_errorf("[cache-fp] DIFF layer=%d la_conv_cpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_conv_cpu[i], restored->fp_conv_cpu[i]);
            ls++; total++;
        }
        if (cold->fp_ssm_cpu[i] != restored->fp_ssm_cpu[i]) {
            server_log_errorf("[cache-fp] DIFF layer=%d la_ssm_cpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_ssm_cpu[i], restored->fp_ssm_cpu[i]);
            ls++; total++;
        }
    }
    for (int i = 0; i < g_cfg.num_full_attn_layers; i++) {
        if (cold->fp_kv_k_gpu[i] != restored->fp_kv_k_gpu[i]) {
            server_log_errorf("[cache-fp] DIFF fa_idx=%d kv_k_gpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_kv_k_gpu[i], restored->fp_kv_k_gpu[i]);
            kv_gpu++; total++;
        }
        if (cold->fp_kv_v_gpu[i] != restored->fp_kv_v_gpu[i]) {
            server_log_errorf("[cache-fp] DIFF fa_idx=%d kv_v_gpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_kv_v_gpu[i], restored->fp_kv_v_gpu[i]);
            kv_gpu++; total++;
        }
    }
    for (int i = 0; i < g_cfg.num_linear_layers; i++) {
        if (cold->fp_buf_delta[i] != restored->fp_buf_delta[i]) {
            server_log_errorf("[cache-fp] DIFF linear_idx=%d buf_delta_gpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_buf_delta[i], restored->fp_buf_delta[i]);
            gpu_lin++; total++;
        }
        if (cold->fp_buf_conv[i] != restored->fp_buf_conv[i]) {
            server_log_errorf("[cache-fp] DIFF linear_idx=%d buf_conv_gpu cold=%016llx restored=%016llx\n",
                              i, cold->fp_buf_conv[i], restored->fp_buf_conv[i]);
            gpu_lin++; total++;
        }
    }
    if (total == 0) {
        server_log_errorf("[cache-fp] PASS post-restore == post-cold-prefill (all buffers identical)\n");
    } else {
        server_log_errorf("[cache-fp] FAILED %d divergence(s) — kv_cpu=%d, kv_gpu=%d, layer_state_cpu=%d, gpu_linear=%d\n",
                          total, kv_cpu, kv_gpu, ls, gpu_lin);
    }
}

static int save_system_prompt_disk_cache(const char *model_path,
                                         uint64_t prompt_hash,
                                         int token_count,
                                         KVSnapshot *kv_snapshots,
                                         float **la_conv_snapshots,
                                         float **la_ssm_snapshots,
                                         void **gpu_delta_snapshots,
                                         void **gpu_conv_snapshots) {
    if (!g_system_prompt_cache_enabled || g_system_prompt_cache_max_entries <= 0 ||
        !model_path || token_count <= 0) return -1;

    char dir[PATH_MAX], path[PATH_MAX], tmp_path[PATH_MAX];
    if (system_prompt_cache_dir(model_path, dir, sizeof(dir)) != 0 ||
        system_prompt_cache_path(model_path, prompt_hash, path, sizeof(path)) != 0 ||
        mkdir_p_cstr(dir) != 0) {
        return -1;
    }
    if (access(path, R_OK) == 0) return 0;

    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld", path, (long)getpid());
    FILE *f = fopen(tmp_path, "wb");
    if (!f) return -1;

    SysPromptCacheHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "FCSCACH", 7);
    hdr.version = SYSPROMPT_CACHE_VERSION;
    hdr.header_size = sizeof(hdr);
    snprintf(hdr.model_id, sizeof(hdr.model_id), "%s", g_cfg.model_id);
    hdr.prompt_hash = prompt_hash;
    hdr.token_count = (uint32_t)token_count;
    hdr.num_layers = (uint32_t)g_cfg.num_layers;
    hdr.num_full_attn_layers = (uint32_t)g_cfg.num_full_attn_layers;
    hdr.num_linear_layers = (uint32_t)g_cfg.num_linear_layers;
    hdr.kv_dim = (uint32_t)(g_cfg.num_kv_heads * g_cfg.head_dim);
    hdr.compression = SYSPROMPT_CACHE_ALG_LZFSE;
    hdr.conv_state_size = (uint64_t)((g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float));
    hdr.ssm_state_size = (uint64_t)(g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float));
    hdr.gpu_delta_size = (uint64_t)serve_gpu_delta_snapshot_size();
    hdr.gpu_conv_size = (uint64_t)serve_gpu_conv_snapshot_size();

    if (fwrite(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f);
        unlink(tmp_path);
        return -1;
    }

    uint64_t raw_total = 0, stored_total = 0;
    uint32_t chunk_count = 0;
    size_t kv_sz = (size_t)token_count * hdr.kv_dim * sizeof(float);
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (kv_snapshots[i].k_snapshot && kv_snapshots[i].v_snapshot) {
            if (write_cache_chunk(f, SYSPROMPT_CACHE_KIND_KV_K, i, kv_snapshots[i].k_snapshot, kv_sz, &raw_total, &stored_total) != 0 ||
                write_cache_chunk(f, SYSPROMPT_CACHE_KIND_KV_V, i, kv_snapshots[i].v_snapshot, kv_sz, &raw_total, &stored_total) != 0) {
                fclose(f);
                unlink(tmp_path);
                return -1;
            }
            chunk_count += 2;
        }
        if (la_conv_snapshots[i] && la_ssm_snapshots[i]) {
            if (write_cache_chunk(f, SYSPROMPT_CACHE_KIND_LA_CONV, i, la_conv_snapshots[i], (size_t)hdr.conv_state_size, &raw_total, &stored_total) != 0 ||
                write_cache_chunk(f, SYSPROMPT_CACHE_KIND_LA_SSM, i, la_ssm_snapshots[i], (size_t)hdr.ssm_state_size, &raw_total, &stored_total) != 0) {
                fclose(f);
                unlink(tmp_path);
                return -1;
            }
            chunk_count += 2;
        }
    }
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < g_cfg.num_linear_layers; i++) {
            if (gpu_delta_snapshots[i]) {
                if (write_cache_chunk(f, SYSPROMPT_CACHE_KIND_GPU_DELTA, i, gpu_delta_snapshots[i], (size_t)hdr.gpu_delta_size, &raw_total, &stored_total) != 0) {
                    fclose(f);
                    unlink(tmp_path);
                    return -1;
                }
                chunk_count++;
            }
            if (gpu_conv_snapshots[i]) {
                if (write_cache_chunk(f, SYSPROMPT_CACHE_KIND_GPU_CONV, i, gpu_conv_snapshots[i], (size_t)hdr.gpu_conv_size, &raw_total, &stored_total) != 0) {
                    fclose(f);
                    unlink(tmp_path);
                    return -1;
                }
                chunk_count++;
            }
        }
    }

    hdr.chunk_count = chunk_count;
    hdr.raw_bytes = raw_total;
    hdr.stored_bytes = stored_total;
    if (fseek(f, 0, SEEK_SET) != 0 ||
        fwrite(&hdr, 1, sizeof(hdr), f) != sizeof(hdr) ||
        fflush(f) != 0 ||
        fsync(fileno(f)) != 0) {
        fclose(f);
        unlink(tmp_path);
        return -1;
    }
    fclose(f);
    if (rename(tmp_path, path) != 0) {
        unlink(tmp_path);
        return -1;
    }
    server_log_errorf("[serve] sys_prompt_disk_cache saved %s raw=%.2fGiB stored=%.2fGiB chunks=%u\n",
                      path, raw_total / 1073741824.0, stored_total / 1073741824.0, chunk_count);

    // Optional runtime self-check: reload the file we just wrote and diff it
    // against the in-memory snapshot we just persisted. Catches GPU MTLBuffer
    // coherency at capture time, wrong size constants for live data, and any
    // path the synthetic unit test can't simulate. Gated behind env var so it
    // only runs when investigating; it allocates ~0.5GiB of shadow buffers
    // and adds a few seconds to the save path.
    const char *validate_env = getenv("FLASHCHAT_CACHE_VALIDATE");
    if (validate_env && validate_env[0] && validate_env[0] != '0') {
        size_t kv_dim = (size_t)g_cfg.num_kv_heads * g_cfg.head_dim;
        size_t kv_sz = (size_t)token_count * kv_dim * sizeof(float);
        size_t conv_state_size = (size_t)hdr.conv_state_size;
        size_t ssm_state_size = (size_t)hdr.ssm_state_size;
        size_t gpu_delta_size = (size_t)hdr.gpu_delta_size;
        size_t gpu_conv_size = (size_t)hdr.gpu_conv_size;

        KVSnapshot *kv_load = calloc((size_t)g_cfg.num_layers, sizeof(KVSnapshot));
        float **conv_load = calloc((size_t)g_cfg.num_layers, sizeof(float *));
        float **ssm_load = calloc((size_t)g_cfg.num_layers, sizeof(float *));
        void *gpu_delta_load[MAX_LINEAR_LAYERS] = {0};
        void *gpu_conv_load[MAX_LINEAR_LAYERS] = {0};
        int loaded_tokens = 0;
        int diffs = 0;

        if (load_system_prompt_disk_cache(model_path, prompt_hash, token_count,
                                          kv_load, conv_load, ssm_load,
                                          gpu_delta_load, gpu_conv_load,
                                          &loaded_tokens) != 0) {
            server_log_errorf("[cache-validate] FAIL load returned error for %s\n", path);
            diffs++;
        } else {
            for (int i = 0; i < g_cfg.num_layers; i++) {
                int is_full = ((i + 1) % g_cfg.full_attn_interval == 0);
                if (is_full && kv_snapshots[i].k_snapshot && kv_load[i].k_snapshot) {
                    if (memcmp(kv_snapshots[i].k_snapshot, kv_load[i].k_snapshot, kv_sz) != 0) {
                        size_t off = first_byte_diff(kv_snapshots[i].k_snapshot, kv_load[i].k_snapshot, kv_sz);
                        server_log_errorf("[cache-validate] FAIL layer=%d kind=KV_K diverges at byte %zu of %zu\n", i, off, kv_sz);
                        diffs++;
                    }
                    if (memcmp(kv_snapshots[i].v_snapshot, kv_load[i].v_snapshot, kv_sz) != 0) {
                        size_t off = first_byte_diff(kv_snapshots[i].v_snapshot, kv_load[i].v_snapshot, kv_sz);
                        server_log_errorf("[cache-validate] FAIL layer=%d kind=KV_V diverges at byte %zu of %zu\n", i, off, kv_sz);
                        diffs++;
                    }
                }
                if (!is_full && la_conv_snapshots[i] && conv_load[i]) {
                    if (memcmp(la_conv_snapshots[i], conv_load[i], conv_state_size) != 0) {
                        size_t off = first_byte_diff(la_conv_snapshots[i], conv_load[i], conv_state_size);
                        server_log_errorf("[cache-validate] FAIL layer=%d kind=LA_CONV diverges at byte %zu of %zu\n", i, off, conv_state_size);
                        diffs++;
                    }
                    if (memcmp(la_ssm_snapshots[i], ssm_load[i], ssm_state_size) != 0) {
                        size_t off = first_byte_diff(la_ssm_snapshots[i], ssm_load[i], ssm_state_size);
                        server_log_errorf("[cache-validate] FAIL layer=%d kind=LA_SSM diverges at byte %zu of %zu\n", i, off, ssm_state_size);
                        diffs++;
                    }
                }
            }
            if (g_metal && g_metal->delta_net_step) {
                for (int i = 0; i < g_cfg.num_linear_layers; i++) {
                    if (gpu_delta_snapshots[i] && gpu_delta_load[i] &&
                        memcmp(gpu_delta_snapshots[i], gpu_delta_load[i], gpu_delta_size) != 0) {
                        size_t off = first_byte_diff(gpu_delta_snapshots[i], gpu_delta_load[i], gpu_delta_size);
                        server_log_errorf("[cache-validate] FAIL linear_layer=%d kind=GPU_DELTA diverges at byte %zu of %zu\n", i, off, gpu_delta_size);
                        diffs++;
                    }
                    if (gpu_conv_snapshots[i] && gpu_conv_load[i] &&
                        memcmp(gpu_conv_snapshots[i], gpu_conv_load[i], gpu_conv_size) != 0) {
                        size_t off = first_byte_diff(gpu_conv_snapshots[i], gpu_conv_load[i], gpu_conv_size);
                        server_log_errorf("[cache-validate] FAIL linear_layer=%d kind=GPU_CONV diverges at byte %zu of %zu\n", i, off, gpu_conv_size);
                        diffs++;
                    }
                }
            }
            free_system_prompt_snapshots(kv_load, conv_load, ssm_load, gpu_delta_load, gpu_conv_load);
        }
        free(kv_load); free(conv_load); free(ssm_load);

        if (diffs == 0) {
            server_log_errorf("[cache-validate] PASS in-memory == disk-roundtrip for hash=%llx\n", (unsigned long long)prompt_hash);
        } else {
            server_log_errorf("[cache-validate] FAILED %d divergence(s) for hash=%llx\n", diffs, (unsigned long long)prompt_hash);
        }
    }
    return 0;
}

static void *read_cache_chunk_payload(FILE *f, const SysPromptCacheChunk *chunk) {
    if (!f || !chunk || chunk->raw_size == 0 || chunk->stored_size == 0) return NULL;
    void *stored = malloc((size_t)chunk->stored_size);
    if (!stored) return NULL;
    if (fread(stored, 1, (size_t)chunk->stored_size, f) != (size_t)chunk->stored_size) {
        free(stored);
        return NULL;
    }
    void *raw = stored;
    if (chunk->algorithm == SYSPROMPT_CACHE_ALG_LZFSE) {
        raw = malloc((size_t)chunk->raw_size);
        if (!raw) {
            free(stored);
            return NULL;
        }
        size_t n = compression_decode_buffer((uint8_t *)raw, (size_t)chunk->raw_size,
                                             (const uint8_t *)stored, (size_t)chunk->stored_size,
                                             NULL, COMPRESSION_LZFSE);
        free(stored);
        if (n != (size_t)chunk->raw_size) {
            free(raw);
            return NULL;
        }
    } else if (chunk->algorithm != SYSPROMPT_CACHE_ALG_RAW) {
        free(stored);
        return NULL;
    }
    if (fnv1a_bytes(raw, (size_t)chunk->raw_size) != chunk->checksum) {
        free(raw);
        return NULL;
    }
    return raw;
}

static int load_system_prompt_disk_cache(const char *model_path,
                                         uint64_t prompt_hash,
                                         int expected_tokens,
                                         KVSnapshot *kv_snapshots,
                                         float **la_conv_snapshots,
                                         float **la_ssm_snapshots,
                                         void **gpu_delta_snapshots,
                                         void **gpu_conv_snapshots,
                                         int *loaded_tokens) {
    if (!g_system_prompt_cache_enabled || !model_path || expected_tokens <= 0) return -1;

    char path[PATH_MAX];
    if (system_prompt_cache_path(model_path, prompt_hash, path, sizeof(path)) != 0) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    SysPromptCacheHeader hdr;
    if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr) ||
        memcmp(hdr.magic, "FCSCACH", 7) != 0 ||
        hdr.version != SYSPROMPT_CACHE_VERSION ||
        hdr.header_size != sizeof(hdr) ||
        hdr.prompt_hash != prompt_hash ||
        hdr.token_count != (uint32_t)expected_tokens ||
        strcmp(hdr.model_id, g_cfg.model_id) != 0 ||
        hdr.num_layers != (uint32_t)g_cfg.num_layers ||
        hdr.num_full_attn_layers != (uint32_t)g_cfg.num_full_attn_layers ||
        hdr.num_linear_layers != (uint32_t)g_cfg.num_linear_layers ||
        hdr.kv_dim != (uint32_t)(g_cfg.num_kv_heads * g_cfg.head_dim) ||
        hdr.conv_state_size != (uint64_t)((g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float)) ||
        hdr.ssm_state_size != (uint64_t)(g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float))) {
        fclose(f);
        return -1;
    }

    KVSnapshot *tmp_kv = calloc((size_t)g_cfg.num_layers, sizeof(KVSnapshot));
    float **tmp_conv = calloc((size_t)g_cfg.num_layers, sizeof(float *));
    float **tmp_ssm = calloc((size_t)g_cfg.num_layers, sizeof(float *));
    void **tmp_gpu_delta = calloc(MAX_LINEAR_LAYERS, sizeof(void *));
    void **tmp_gpu_conv = calloc(MAX_LINEAR_LAYERS, sizeof(void *));
    if (!tmp_kv || !tmp_conv || !tmp_ssm || !tmp_gpu_delta || !tmp_gpu_conv) {
        free(tmp_kv); free(tmp_conv); free(tmp_ssm); free(tmp_gpu_delta); free(tmp_gpu_conv);
        fclose(f);
        return -1;
    }

    int kv_k_count = 0, kv_v_count = 0, conv_count = 0, ssm_count = 0, gpu_delta_count = 0, gpu_conv_count = 0;
    size_t kv_sz = (size_t)hdr.token_count * hdr.kv_dim * sizeof(float);
    int ok = 1;
    for (uint32_t c = 0; c < hdr.chunk_count; c++) {
        SysPromptCacheChunk chunk;
        if (fread(&chunk, 1, sizeof(chunk), f) != sizeof(chunk)) {
            ok = 0;
            break;
        }
        void *payload = read_cache_chunk_payload(f, &chunk);
        if (!payload) {
            ok = 0;
            break;
        }
        int layer = chunk.layer;
        switch (chunk.kind) {
            case SYSPROMPT_CACHE_KIND_KV_K:
                if (layer < 0 || layer >= g_cfg.num_layers || chunk.raw_size != kv_sz) { ok = 0; free(payload); break; }
                free(tmp_kv[layer].k_snapshot);
                tmp_kv[layer].k_snapshot = payload;
                tmp_kv[layer].len = hdr.token_count;
                kv_k_count++;
                break;
            case SYSPROMPT_CACHE_KIND_KV_V:
                if (layer < 0 || layer >= g_cfg.num_layers || chunk.raw_size != kv_sz) { ok = 0; free(payload); break; }
                free(tmp_kv[layer].v_snapshot);
                tmp_kv[layer].v_snapshot = payload;
                tmp_kv[layer].len = hdr.token_count;
                kv_v_count++;
                break;
            case SYSPROMPT_CACHE_KIND_LA_CONV:
                if (layer < 0 || layer >= g_cfg.num_layers || chunk.raw_size != hdr.conv_state_size) { ok = 0; free(payload); break; }
                free(tmp_conv[layer]);
                tmp_conv[layer] = payload;
                conv_count++;
                break;
            case SYSPROMPT_CACHE_KIND_LA_SSM:
                if (layer < 0 || layer >= g_cfg.num_layers || chunk.raw_size != hdr.ssm_state_size) { ok = 0; free(payload); break; }
                free(tmp_ssm[layer]);
                tmp_ssm[layer] = payload;
                ssm_count++;
                break;
            case SYSPROMPT_CACHE_KIND_GPU_DELTA:
                if (layer < 0 || layer >= g_cfg.num_linear_layers || chunk.raw_size != hdr.gpu_delta_size) { ok = 0; free(payload); break; }
                free(tmp_gpu_delta[layer]);
                tmp_gpu_delta[layer] = payload;
                gpu_delta_count++;
                break;
            case SYSPROMPT_CACHE_KIND_GPU_CONV:
                if (layer < 0 || layer >= g_cfg.num_linear_layers || chunk.raw_size != hdr.gpu_conv_size) { ok = 0; free(payload); break; }
                free(tmp_gpu_conv[layer]);
                tmp_gpu_conv[layer] = payload;
                gpu_conv_count++;
                break;
            default:
                free(payload);
                ok = 0;
                break;
        }
        if (!ok) break;
    }
    fclose(f);

    int expect_gpu = (g_metal && g_metal->delta_net_step) ? g_cfg.num_linear_layers : 0;
    if (ok) {
        for (int i = 0; i < g_cfg.num_layers; i++) {
            int is_full = ((i + 1) % g_cfg.full_attn_interval == 0);
            if (is_full) {
                if (!tmp_kv[i].k_snapshot || !tmp_kv[i].v_snapshot) ok = 0;
            } else {
                if (!tmp_conv[i] || !tmp_ssm[i]) ok = 0;
            }
        }
        if (expect_gpu > 0) {
            for (int i = 0; i < g_cfg.num_linear_layers; i++) {
                if (!tmp_gpu_delta[i] || !tmp_gpu_conv[i]) ok = 0;
            }
        }
    }
    if (!ok ||
        kv_k_count != g_cfg.num_full_attn_layers ||
        kv_v_count != g_cfg.num_full_attn_layers ||
        conv_count != g_cfg.num_linear_layers ||
        ssm_count != g_cfg.num_linear_layers ||
        (expect_gpu > 0 && (gpu_delta_count != expect_gpu || gpu_conv_count != expect_gpu))) {
        free_system_prompt_snapshots(tmp_kv, tmp_conv, tmp_ssm, tmp_gpu_delta, tmp_gpu_conv);
        free(tmp_kv); free(tmp_conv); free(tmp_ssm); free(tmp_gpu_delta); free(tmp_gpu_conv);
        return -1;
    }

    free_system_prompt_snapshots(kv_snapshots, la_conv_snapshots, la_ssm_snapshots, gpu_delta_snapshots, gpu_conv_snapshots);
    for (int i = 0; i < g_cfg.num_layers; i++) {
        kv_snapshots[i] = tmp_kv[i];
        la_conv_snapshots[i] = tmp_conv[i];
        la_ssm_snapshots[i] = tmp_ssm[i];
    }
    for (int i = 0; i < MAX_LINEAR_LAYERS; i++) {
        gpu_delta_snapshots[i] = tmp_gpu_delta[i];
        gpu_conv_snapshots[i] = tmp_gpu_conv[i];
    }
    free(tmp_kv); free(tmp_conv); free(tmp_ssm); free(tmp_gpu_delta); free(tmp_gpu_conv);
    if (loaded_tokens) *loaded_tokens = (int)hdr.token_count;
    server_log_errorf("[serve] sys_prompt_disk_cache loaded %s raw=%.2fGiB stored=%.2fGiB chunks=%u\n",
                      path, hdr.raw_bytes / 1073741824.0, hdr.stored_bytes / 1073741824.0, hdr.chunk_count);
    return 0;
}

static void prune_system_prompt_disk_cache(const char *model_path) {
    if (!model_path || g_system_prompt_cache_max_entries <= 0) return;
    char dir[PATH_MAX];
    if (system_prompt_cache_dir(model_path, dir, sizeof(dir)) != 0) return;
    DIR *d = opendir(dir);
    if (!d) return;

    typedef struct {
        char path[PATH_MAX];
        time_t mtime;
    } CacheEntry;
    CacheEntry entries[128];
    int count = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!strstr(ent->d_name, ".fcache")) continue;
        if (count >= (int)(sizeof(entries) / sizeof(entries[0]))) break;
        snprintf(entries[count].path, sizeof(entries[count].path), "%s/%s", dir, ent->d_name);
        struct stat st;
        if (stat(entries[count].path, &st) == 0) {
            entries[count].mtime = st.st_mtime;
            count++;
        }
    }
    closedir(d);
    while (count > g_system_prompt_cache_max_entries) {
        int oldest = 0;
        for (int i = 1; i < count; i++) {
            if (entries[i].mtime < entries[oldest].mtime) oldest = i;
        }
        unlink(entries[oldest].path);
        entries[oldest] = entries[count - 1];
        count--;
    }
}

// Round-trip self-test for the on-disk system prompt cache. Synthesizes
// snapshot data with deterministic patterns, runs it through the real
// save+load codepath into a temp directory, and memcmp's loaded vs original
// for every (kind, layer) tuple. Catches serializer regressions: header
// rewrites, LZFSE roundtrip, FNV1a checksum, chunk ordering, struct layout.
//
// Caller must have g_cfg loaded; weights and Metal are NOT required (the
// GPU snapshot path is exercised separately by the runtime validator since
// it depends on live MTLBuffer state). Returns 0 on PASS, nonzero on FAIL.
static size_t first_byte_diff(const void *a, const void *b, size_t n) {
    const uint8_t *pa = (const uint8_t *)a, *pb = (const uint8_t *)b;
    for (size_t i = 0; i < n; i++) if (pa[i] != pb[i]) return i;
    return n;
}

static int cache_roundtrip_test(void) {
    int prev_enabled = g_system_prompt_cache_enabled;
    int prev_max = g_system_prompt_cache_max_entries;
    g_system_prompt_cache_enabled = 1;
    if (g_system_prompt_cache_max_entries <= 0) g_system_prompt_cache_max_entries = 6;

    int failures = 0;
    uint64_t hash = 0xCAFEBABE12345678ULL;
    fprintf(stderr, "[cache-roundtrip] model=%s layers=%d full=%d linear=%d\n",
            g_cfg.model_id, g_cfg.num_layers, g_cfg.num_full_attn_layers, g_cfg.num_linear_layers);

    char tmp_dir[] = "/tmp/flashchat-cache-roundtrip-XXXXXX";
    if (!mkdtemp(tmp_dir)) {
        fprintf(stderr, "[cache-roundtrip] FAIL mkdtemp: %s\n", strerror(errno));
        return 1;
    }

    // Use a small token count so the test runs in milliseconds even though
    // realistic prefills are 7000+. Compression and roundtrip semantics are
    // unaffected by token count.
    int token_count = 128;
    size_t kv_dim = (size_t)g_cfg.num_kv_heads * g_cfg.head_dim;
    size_t kv_sz = (size_t)token_count * kv_dim * sizeof(float);
    size_t conv_state_size = (size_t)(g_cfg.linear_conv_kernel_dim - 1) * g_cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = (size_t)g_cfg.linear_num_v_heads * g_cfg.linear_value_dim * g_cfg.linear_key_dim * sizeof(float);

    KVSnapshot *kv_orig = calloc((size_t)g_cfg.num_layers, sizeof(KVSnapshot));
    float **conv_orig = calloc((size_t)g_cfg.num_layers, sizeof(float *));
    float **ssm_orig = calloc((size_t)g_cfg.num_layers, sizeof(float *));
    void *gpu_delta_dummy[MAX_LINEAR_LAYERS] = {0};
    void *gpu_conv_dummy[MAX_LINEAR_LAYERS] = {0};

    if (!kv_orig || !conv_orig || !ssm_orig) {
        fprintf(stderr, "[cache-roundtrip] FAIL alloc originals\n");
        failures++;
        goto cleanup;
    }

    // Deterministic, non-trivially-compressible patterns so LZFSE actually
    // round-trips real-looking content (not all zeros).
    for (int i = 0; i < g_cfg.num_layers; i++) {
        int is_full = ((i + 1) % g_cfg.full_attn_interval == 0);
        if (is_full) {
            kv_orig[i].k_snapshot = malloc(kv_sz);
            kv_orig[i].v_snapshot = malloc(kv_sz);
            if (!kv_orig[i].k_snapshot || !kv_orig[i].v_snapshot) { failures++; goto cleanup; }
            kv_orig[i].len = token_count;
            float *k = kv_orig[i].k_snapshot;
            float *v = kv_orig[i].v_snapshot;
            size_t n = kv_sz / sizeof(float);
            for (size_t j = 0; j < n; j++) {
                k[j] = sinf((float)(i * 7919u + j * 31u) * 1e-4f);
                v[j] = cosf((float)(i * 7919u + j * 31u) * 1e-4f) + 0.5f;
            }
        } else {
            conv_orig[i] = malloc(conv_state_size);
            ssm_orig[i] = malloc(ssm_state_size);
            if (!conv_orig[i] || !ssm_orig[i]) { failures++; goto cleanup; }
            float *c = conv_orig[i];
            float *s = ssm_orig[i];
            for (size_t j = 0; j < conv_state_size / sizeof(float); j++) c[j] = (float)((i + 1) * 13 + j) * 1e-3f;
            for (size_t j = 0; j < ssm_state_size / sizeof(float); j++) s[j] = (float)((i + 1) * 17 + j) * 1e-4f;
        }
    }

    int save_rc = save_system_prompt_disk_cache(tmp_dir, hash, token_count,
                                                kv_orig, conv_orig, ssm_orig,
                                                gpu_delta_dummy, gpu_conv_dummy);
    if (save_rc != 0) {
        fprintf(stderr, "[cache-roundtrip] FAIL save rc=%d\n", save_rc);
        failures++;
        goto cleanup;
    }

    KVSnapshot *kv_load = calloc((size_t)g_cfg.num_layers, sizeof(KVSnapshot));
    float **conv_load = calloc((size_t)g_cfg.num_layers, sizeof(float *));
    float **ssm_load = calloc((size_t)g_cfg.num_layers, sizeof(float *));
    void *gpu_delta_load[MAX_LINEAR_LAYERS] = {0};
    void *gpu_conv_load[MAX_LINEAR_LAYERS] = {0};
    int loaded_tokens = 0;
    int load_rc = load_system_prompt_disk_cache(tmp_dir, hash, token_count,
                                                kv_load, conv_load, ssm_load,
                                                gpu_delta_load, gpu_conv_load,
                                                &loaded_tokens);
    if (load_rc != 0) {
        fprintf(stderr, "[cache-roundtrip] FAIL load rc=%d\n", load_rc);
        failures++;
        if (kv_load) { free(kv_load); free(conv_load); free(ssm_load); }
        goto cleanup;
    }
    if (loaded_tokens != token_count) {
        fprintf(stderr, "[cache-roundtrip] FAIL token count saved=%d loaded=%d\n", token_count, loaded_tokens);
        failures++;
    }

    int kv_diffs = 0, conv_diffs = 0, ssm_diffs = 0;
    for (int i = 0; i < g_cfg.num_layers; i++) {
        int is_full = ((i + 1) % g_cfg.full_attn_interval == 0);
        if (is_full) {
            if (memcmp(kv_orig[i].k_snapshot, kv_load[i].k_snapshot, kv_sz) != 0) {
                size_t off = first_byte_diff(kv_orig[i].k_snapshot, kv_load[i].k_snapshot, kv_sz);
                fprintf(stderr, "[cache-roundtrip] FAIL layer=%d kind=KV_K diverges at byte %zu of %zu\n", i, off, kv_sz);
                kv_diffs++;
            }
            if (memcmp(kv_orig[i].v_snapshot, kv_load[i].v_snapshot, kv_sz) != 0) {
                size_t off = first_byte_diff(kv_orig[i].v_snapshot, kv_load[i].v_snapshot, kv_sz);
                fprintf(stderr, "[cache-roundtrip] FAIL layer=%d kind=KV_V diverges at byte %zu of %zu\n", i, off, kv_sz);
                kv_diffs++;
            }
            if (kv_load[i].len != kv_orig[i].len) {
                fprintf(stderr, "[cache-roundtrip] FAIL layer=%d KVSnapshot.len orig=%d load=%d\n", i, kv_orig[i].len, kv_load[i].len);
                kv_diffs++;
            }
        } else {
            if (memcmp(conv_orig[i], conv_load[i], conv_state_size) != 0) {
                size_t off = first_byte_diff(conv_orig[i], conv_load[i], conv_state_size);
                fprintf(stderr, "[cache-roundtrip] FAIL layer=%d kind=LA_CONV diverges at byte %zu of %zu\n", i, off, conv_state_size);
                conv_diffs++;
            }
            if (memcmp(ssm_orig[i], ssm_load[i], ssm_state_size) != 0) {
                size_t off = first_byte_diff(ssm_orig[i], ssm_load[i], ssm_state_size);
                fprintf(stderr, "[cache-roundtrip] FAIL layer=%d kind=LA_SSM diverges at byte %zu of %zu\n", i, off, ssm_state_size);
                ssm_diffs++;
            }
        }
    }
    failures += kv_diffs + conv_diffs + ssm_diffs;

    free_system_prompt_snapshots(kv_load, conv_load, ssm_load, gpu_delta_load, gpu_conv_load);
    free(kv_load); free(conv_load); free(ssm_load);

cleanup:
    if (kv_orig) free_system_prompt_snapshots(kv_orig, conv_orig, ssm_orig, gpu_delta_dummy, gpu_conv_dummy);
    free(kv_orig); free(conv_orig); free(ssm_orig);

    char path[PATH_MAX];
    if (system_prompt_cache_path(tmp_dir, hash, path, sizeof(path)) == 0) unlink(path);
    char dir[PATH_MAX];
    if (system_prompt_cache_dir(tmp_dir, dir, sizeof(dir)) == 0) rmdir(dir);
    char fc_dir[PATH_MAX];
    snprintf(fc_dir, sizeof(fc_dir), "%s/flashchat", tmp_dir);
    rmdir(fc_dir);
    rmdir(tmp_dir);

    g_system_prompt_cache_enabled = prev_enabled;
    g_system_prompt_cache_max_entries = prev_max;

    if (failures == 0) {
        fprintf(stderr, "[cache-roundtrip] PASS all chunks identical (token_count=%d, full=%d, linear=%d)\n",
                token_count, g_cfg.num_full_attn_layers, g_cfg.num_linear_layers);
        return 0;
    }
    fprintf(stderr, "[cache-roundtrip] FAILED %d divergence(s)\n", failures);
    return 1;
}

static void serve_loop(
    int port,
    const char *model_path,
    WeightFile *wf, Vocabulary *vocab,
    void **layer_states, KVCache **kv_caches,
    void **layer_mmaps, int *layer_fds,
    float *hidden, float *logits,
    uint16_t *final_norm_w, int K)
{
    g_server_shutdown_signal = 0;
    g_server_listen_fd = -1;
    install_serve_signal_handlers();
    server_log_open();

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); server_log_errorf("[serve] socket failed: %s\n", strerror(errno)); return; }
    g_server_listen_fd = server_fd;

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); server_log_errorf("[serve] bind failed on port %d: %s\n", port, strerror(errno)); close(server_fd); g_server_listen_fd = -1; return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("listen"); server_log_errorf("[serve] listen failed on port %d: %s\n", port, strerror(errno)); close(server_fd); g_server_listen_fd = -1; return;
    }

    server_logf("[serve] Listening on http://0.0.0.0:%d\n", port);
    server_logf("[serve] Endpoints: POST /v1/chat/completions, POST /v1/responses, GET /v1/models, GET /health\n");
    if (g_server_log_path[0]) {
        server_logf("[serve] Persistent log: %s\n", g_server_log_path);
    }
    server_logf("[serve] Persistent system prompt cache: %s (max entries: %d)\n",
                g_system_prompt_cache_enabled ? "enabled" : "disabled",
                g_system_prompt_cache_max_entries);

    static uint64_t req_counter = 0;

    // ---- Lazy system prompt cache: snapshot keyed by system prompt hash ----
    // On first request with a given system prompt, prefill it and save state.
    // On subsequent requests with the same hash, restore the snapshot.
    uint64_t cached_sys_hash = 0;
    int cached_sys_token_count = 0;

    KVSnapshot kv_snapshots[g_cfg.num_layers];
    memset(kv_snapshots, 0, sizeof(kv_snapshots));

    float *la_conv_snapshots[g_cfg.num_layers];
    float *la_ssm_snapshots[g_cfg.num_layers];
    memset(la_conv_snapshots, 0, sizeof(la_conv_snapshots));
    memset(la_ssm_snapshots, 0, sizeof(la_ssm_snapshots));

    void *gpu_delta_snapshots[MAX_LINEAR_LAYERS];
    void *gpu_conv_snapshots[MAX_LINEAR_LAYERS];
    memset(gpu_delta_snapshots, 0, sizeof(gpu_delta_snapshots));
    memset(gpu_conv_snapshots, 0, sizeof(gpu_conv_snapshots));

    for (;;) {
        if (g_server_shutdown_signal) {
            server_log_errorf("[serve] Shutdown requested by signal %d\n", g_server_shutdown_signal);
            break;
        }
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (g_server_shutdown_signal) {
                server_log_errorf("[serve] Shutdown requested by signal %d\n", g_server_shutdown_signal);
                break;
            }
            perror("accept"); server_log_errorf("[serve] accept failed: %s\n", strerror(errno)); continue;
        }

        char *reqbuf = malloc(1024 * 1024);
        int reqlen = read_http_request(client_fd, reqbuf, 1024 * 1024);
        if (reqlen <= 0) { free(reqbuf); close(client_fd); continue; }

        char method[16] = {0}, path[256] = {0};
        sscanf(reqbuf, "%15s %255s", method, path);
        uint64_t req_num = ++req_counter;
        char early_request_id[64];
        snprintf(early_request_id, sizeof(early_request_id), "conn-%llu", req_num);
        server_http_log_block(early_request_id, "request", "raw http request", reqbuf);

        if (strcmp(method, "OPTIONS") == 0) {
            server_http_log_block(early_request_id, "response", "cors preflight", CORS_RESPONSE);
            http_write_str(client_fd, CORS_RESPONSE);
            free(reqbuf); close(client_fd);
            continue;
        }

        if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
            char health_json[256];
            snprintf(health_json, sizeof(health_json),
                     "{\"status\":\"ok\",\"model\":\"%s\",\"ready\":true}\n", g_cfg.model_id);
            send_json_ok(client_fd, health_json);
            free(reqbuf); close(client_fd);
            continue;
        }

        if (strcmp(method, "GET") == 0 && strcmp(path, "/v1") == 0) {
            send_json_ok(client_fd,
                         "{\"object\":\"service\",\"id\":\"flashchat\",\"api\":\"openai-compatible\","
                         "\"endpoints\":[\"/v1/chat/completions\",\"/v1/responses\",\"/v1/models\",\"/health\"]}\n");
            free(reqbuf); close(client_fd);
            continue;
        }

        if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
            char models_json[256];
            snprintf(models_json, sizeof(models_json),
                     "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"local\"}]}\n", g_cfg.model_id);
            send_json_ok(client_fd, models_json);
            free(reqbuf); close(client_fd);
            continue;
        }

        int is_chat = (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0);
        int is_responses = (strcmp(method, "POST") == 0 && strcmp(path, "/v1/responses") == 0);
        if (!is_chat && !is_responses) {
            send_json_error(client_fd, 404, "invalid_request_error", "not found");
            free(reqbuf); close(client_fd);
            continue;
        }

        char *body = strstr(reqbuf, "\r\n\r\n");
        if (!body) {
            send_json_error(client_fd, 400, "invalid_request_error", "request body is required");
            free(reqbuf); close(client_fd);
            continue;
        }
        body += 4;

        char request_id[64];
        snprintf(request_id, sizeof(request_id),
                 "%s-%llu", is_chat ? "chatcmpl" : "resp", req_num);
        if (g_server_debug_enabled) {
            server_debug_write_text(request_id, "request.json", body);
        }
        server_http_log_block(request_id, "request", is_chat ? "chat body" : "responses body", body);

        NSError *json_error = nil;
        NSDictionary *root = parse_json_body(body, &json_error);
        if (!root) {
            send_json_error(client_fd, 400, "invalid_request_error",
                            json_error ? [[json_error localizedDescription] UTF8String] : "invalid json");
            free(reqbuf); close(client_fd);
            continue;
        }

        ApiRequest req;
        api_request_init(&req, is_chat ? API_KIND_CHAT : API_KIND_RESPONSES);
        char *req_error = NULL;
        int parse_rc = is_chat
            ? fill_request_from_chat_json(root, &req, &req_error)
            : fill_request_from_responses_json(root, &req, &req_error);
        if (parse_rc < 0) {
            send_json_error(client_fd, 400, "invalid_request_error", req_error ? req_error : "invalid request");
            free(req_error);
            api_request_free(&req);
            free(reqbuf);
            close(client_fd);
            continue;
        }
        if (req.stream) {
            server_http_log_block(request_id, "response", "sse headers", SSE_HEADERS);
            http_write_str(client_fd, SSE_HEADERS);
            if (is_chat && sse_send_initial_role_chunk(client_fd, request_id) < 0) {
                server_log_errorf("[serve] %s client disconnected before streaming began\n", request_id);
                api_request_free(&req);
                free(reqbuf);
                close(client_fd);
                continue;
            }
        }
        server_log_errorf("[serve] %s endpoint=%s prompt_chars=%zu tools=%d stream=%d active_experts=%d/%d temp=%.2f top_p=%.2f top_k=%d min_p=%.3f presence=%.2f repetition=%.2f reasoning=%d snapshot=%d\n",
                          request_id, is_chat ? "chat" : "responses",
                          req.conversation_text ? strlen(req.conversation_text) : 0,
                          req.tool_count, req.stream,
                          K, g_cfg.num_experts_per_tok,
                          req.temperature, req.top_p,
                          req.top_k, req.min_p, req.presence_penalty, req.repetition_penalty,
                          req.reasoning_enabled, req.used_snapshot);

        // Workaround: opencode fires a "title generator" request before every chat.
        // The 35B model loops on this prompt (degenerate \n / token spam) and blocks
        // the real request behind it for minutes. Until the EOS / chat-template bug
        // is diagnosed, short-circuit any request whose system prompt begins with
        // the unmistakable opencode title-gen signature and return a fixed string.
        if (is_chat && req.system_prompt &&
            strncmp(req.system_prompt, "You are a title generator", 25) == 0) {
            const char *fixed_title = "New conversation";
            server_log_errorf("[serve] %s title_gen_shortcut active; returning fixed string \"%s\"\n",
                              request_id, fixed_title);
            if (req.stream) {
                sse_send_delta(client_fd, request_id, fixed_title);
                sse_send_done(client_fd, request_id);
            } else {
                char *final_json = build_chat_completion_json(request_id, req.model, fixed_title, NULL, NULL);
                if (final_json) {
                    send_json_ok(client_fd, final_json);
                    free(final_json);
                } else {
                    send_json_error(client_fd, 500, "server_error", "title shortcut alloc failed");
                }
            }
            api_request_free(&req);
            free(reqbuf);
            close(client_fd);
            if (g_server_shutdown_signal) {
                server_log_errorf("[serve] Shutdown requested by signal %d after request drain\n", g_server_shutdown_signal);
                break;
            }
            continue;
        }

        // Build system prompt and hash it for cache lookup
        char *req_sys_prompt = build_system_prompt_for_request(&req, NULL);
        uint64_t req_sys_hash = hash_string_djb2(req_sys_prompt);
        int sys_prompt_token_count = count_sys_prompt_tokens(req_sys_prompt);
        int snapshot_restored = 0;
        int pos = 0;

        if (cached_sys_hash == req_sys_hash && cached_sys_token_count > 0) {
            // Cache hit: restore snapshot
            server_log_errorf("[serve] %s sys_prompt_cache hit hash=%llu tokens=%d\n",
                              request_id, req_sys_hash, cached_sys_token_count);
            restore_system_prompt_snapshots(cached_sys_token_count,
                                            kv_snapshots,
                                            la_conv_snapshots,
                                            la_ssm_snapshots,
                                            gpu_delta_snapshots,
                                            gpu_conv_snapshots,
                                            layer_states,
                                            kv_caches);
            pos = cached_sys_token_count;
            snapshot_restored = 1;
            req.used_snapshot = 1;
            if (cache_fingerprint_enabled() && g_fp_post_cold.valid) {
                RuntimeFingerprint fp_post_restore;
                fingerprint_runtime_state(cached_sys_token_count, kv_caches, layer_states, &fp_post_restore);
                server_log_errorf("[cache-fp] diff post-in-memory-restore vs post-cold-prefill (request=%s)\n", request_id);
                diff_runtime_fingerprints(&g_fp_post_cold, &fp_post_restore);
            }
        } else if (g_system_prompt_cache_enabled &&
                   load_system_prompt_disk_cache(model_path, req_sys_hash, sys_prompt_token_count,
                                                 kv_snapshots,
                                                 la_conv_snapshots,
                                                 la_ssm_snapshots,
                                                 gpu_delta_snapshots,
                                                 gpu_conv_snapshots,
                                                 &cached_sys_token_count) == 0) {
            cached_sys_hash = req_sys_hash;
            restore_system_prompt_snapshots(cached_sys_token_count,
                                            kv_snapshots,
                                            la_conv_snapshots,
                                            la_ssm_snapshots,
                                            gpu_delta_snapshots,
                                            gpu_conv_snapshots,
                                            layer_states,
                                            kv_caches);
            server_log_errorf("[serve] %s sys_prompt_cache disk hit hash=%llu tokens=%d\n",
                              request_id, cached_sys_hash, cached_sys_token_count);
            pos = cached_sys_token_count;
            snapshot_restored = 1;
            req.used_snapshot = 1;
            if (cache_fingerprint_enabled() && g_fp_post_cold.valid) {
                RuntimeFingerprint fp_post_restore;
                fingerprint_runtime_state(cached_sys_token_count, kv_caches, layer_states, &fp_post_restore);
                server_log_errorf("[cache-fp] diff post-disk-restore vs post-cold-prefill (request=%s)\n", request_id);
                diff_runtime_fingerprints(&g_fp_post_cold, &fp_post_restore);
            }
        } else {
            // Cache miss: clear state and run full prefill
            if (cached_sys_hash != 0) {
                server_log_errorf("[serve] %s sys_prompt_cache miss old_hash=%llu new_hash=%llu\n",
                                  request_id, cached_sys_hash, req_sys_hash);
            } else {
                server_log_errorf("[serve] %s sys_prompt_cache miss new_hash=%llu tokens=%d\n",
                                  request_id, req_sys_hash, sys_prompt_token_count);
            }
            clear_runtime_state_serve(layer_states, kv_caches);
            req.used_snapshot = 0;
        }
        if (g_cache_telemetry_enabled) cache_telemetry_reset();

        server_log_errorf("[serve] %s request parsed; beginning prompt tokenization\n", request_id);
        PromptTokens *pt = tokenize_request_prompt(&req, request_id);
        if (!pt) {
            send_json_error(client_fd, 500, "server_error", "tokenization failed");
            free(req_sys_prompt);
            api_request_free(&req);
            free(reqbuf);
            close(client_fd);
            continue;
        }
        int active_tools = (req.tool_count > 0 && req.tool_choice_mode != TOOL_CHOICE_NONE);
        float effective_presence_penalty = active_tools ? 0.0f : req.presence_penalty;
        float effective_repetition_penalty = active_tools ? 1.0f : req.repetition_penalty;
        int *token_counts = NULL;
        if (fabsf(effective_presence_penalty) > 0.000001f ||
            fabsf(effective_repetition_penalty - 1.0f) > 0.000001f) {
            token_counts = calloc((size_t)g_cfg.vocab_size, sizeof(int));
            if (token_counts) {
                if (!active_tools) {
                    int history_start = req.used_snapshot ? 0 : sys_prompt_token_count;
                    seed_token_counts_from_prompt(token_counts, g_cfg.vocab_size, pt, history_start);
                }
            } else {
                server_log_errorf("[serve] %s sampler penalties disabled: token count allocation failed\n", request_id);
            }
        }

        double t_prefill = now_ms();
        float *serve_embed_batch = NULL;
        if (pt->count > 1) {
            serve_embed_batch = malloc((size_t)pt->count * g_cfg.hidden_dim * sizeof(float));
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(wf, pt->ids[i], serve_embed_batch + (size_t)i * g_cfg.hidden_dim);
            }
        }

        // Split prefill: system prompt first, then conversation
        int sys_token_end = req.used_snapshot ? 0 : sys_prompt_token_count;
        if (!snapshot_restored && sys_token_end > 0) {
            // Prefill system prompt tokens
            for (int i = 0; i < sys_token_end; i++) {
                cache_telemetry_note_token();
                if (serve_embed_batch) {
                    memcpy(hidden, serve_embed_batch + (size_t)i * g_cfg.hidden_dim, g_cfg.hidden_dim * sizeof(float));
                } else {
                    embed_lookup(wf, pt->ids[i], hidden);
                }
                for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                    int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                if (i == pt->count - 1) complete_deferred_experts();
                else discard_deferred_experts();
                pos++;
            }
            // Save snapshot at system prompt boundary
            int snapshot_saved = 0;
            if (capture_system_prompt_snapshots(sys_token_end,
                                                kv_snapshots,
                                                la_conv_snapshots,
                                                la_ssm_snapshots,
                                                gpu_delta_snapshots,
                                                gpu_conv_snapshots,
                                                layer_states,
                                                kv_caches) != 0) {
                server_log_errorf("[serve] %s sys_prompt_cache snapshot allocation failed\n", request_id);
            } else {
                snapshot_saved = 1;
                if (g_system_prompt_cache_enabled) {
                    if (save_system_prompt_disk_cache(model_path,
                                                      req_sys_hash,
                                                      sys_token_end,
                                                      kv_snapshots,
                                                      la_conv_snapshots,
                                                      la_ssm_snapshots,
                                                      gpu_delta_snapshots,
                                                      gpu_conv_snapshots) == 0) {
                        prune_system_prompt_disk_cache(model_path);
                    } else {
                        server_log_errorf("[serve] %s sys_prompt_disk_cache save skipped/failed\n", request_id);
                    }
                }
            }
            if (snapshot_saved) {
                cached_sys_hash = req_sys_hash;
                cached_sys_token_count = sys_token_end;
                server_log_errorf("[serve] %s sys_prompt_cache saved hash=%llu tokens=%d\n",
                                  request_id, cached_sys_hash, cached_sys_token_count);
                if (cache_fingerprint_enabled()) {
                    fingerprint_runtime_state(sys_token_end, kv_caches, layer_states, &g_fp_post_cold);
                    server_log_errorf("[cache-fp] captured live runtime fingerprint after cold prefill (request=%s tokens=%d)\n",
                                      request_id, sys_token_end);
                }
            } else {
                cached_sys_hash = 0;
                cached_sys_token_count = 0;
            }
        }

        // Prefill remaining tokens (conversation, or all tokens if snapshot restored)
        int prefill_start = sys_token_end;
        for (int i = prefill_start; i < pt->count; i++) {
            cache_telemetry_note_token();
            if (serve_embed_batch) {
                memcpy(hidden, serve_embed_batch + (size_t)i * g_cfg.hidden_dim, g_cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, pt->ids[i], hidden);
            }
            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            if (i == pt->count - 1) complete_deferred_experts();
            else discard_deferred_experts();
            pos++;
        }
        free(serve_embed_batch);
        free(req_sys_prompt);
        server_log_errorf("[serve] %s prefill=%d tokens in %.0fms\n", request_id, pt->count, now_ms() - t_prefill);

        // Generation-side state. Declared up here (instead of further down where
        // it used to live) so the very first sampler call can honor the
        // tool-call-greedy override for forced tool_choice (where the prompt
        // already places us inside <tool_call><function=NAME> and the first
        // generated token should likewise be format-deterministic).
        char *tool_call_buf = NULL;
        size_t tool_call_len = 0;
        size_t tool_call_cap = 0;
        int saw_tool_call_start = 0;
        // When tool_choice forces a specific function, the prompt was prefilled
        // with `<tool_call>\n<function=NAME>\n`. Mirror that prefix into the
        // parser's buffer so it can match `<tool_call>` and extract NAME the
        // moment the model emits a closing `</tool_call>`.
        if (req.tool_choice_mode == TOOL_CHOICE_FORCED && req.forced_tool_name[0]) {
            char prefix[256];
            int prefix_len = snprintf(prefix, sizeof(prefix),
                                      "<tool_call>\n<function=%s>\n", req.forced_tool_name);
            append_bytes(&tool_call_buf, &tool_call_len, &tool_call_cap, prefix, (size_t)prefix_len);
            saw_tool_call_start = 1;
        }

        float mtp_backbone_hidden[g_cfg.hidden_dim];
        int mtp_shadow_available = mtp_can_shadow_draft(wf);
        int mtp_pending_draft = -1;
        int mtp_shadow_hits = 0;
        int mtp_shadow_checks = 0;
        g_mtp_kv_len = 0;  // fresh MTP attention context per generation
        if (mtp_shadow_available) {
            memcpy(mtp_backbone_hidden, hidden, g_cfg.hidden_dim * sizeof(float));
        }

        if (final_norm_w) {
            float *normed = malloc(g_cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(hidden, final_norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
            memcpy(hidden, normed, g_cfg.hidden_dim * sizeof(float));
            free(normed);
        }
        lm_head_forward(wf, hidden, logits);
        // Tool-call format-stability override: when we're inside a <tool_call>
        // block, force greedy decoding on every token. The model has high
        // confidence on the right format tokens; sampling diversity here only
        // produces malformed XML / dropped function-name tokens. Disable via
        // FLASHCHAT_TOOL_CALL_GREEDY=0.
        float effective_temperature = (g_tool_call_greedy_enabled && saw_tool_call_start)
                                      ? 0.0f
                                      : req.temperature;
        int next_token = pick_next_token(logits, g_cfg.vocab_size, effective_temperature, req.top_p,
                                         req.top_k, req.min_p, effective_presence_penalty,
                                         effective_repetition_penalty, token_counts,
                                         req.reasoning_enabled);
        if (mtp_shadow_available) {
            int draft_token = -1;
            if (mtp_shadow_draft_token(wf, model_path, mtp_backbone_hidden, next_token, &draft_token) == 0) {
                server_log_errorf("[mtp] %s shadow draft after token=%d -> %d\n",
                                  request_id, next_token, draft_token);
                mtp_pending_draft = draft_token;
            }
        }
        server_log_errorf("[serve] %s first_token=%d%s\n", request_id, next_token,
                          (g_tool_call_greedy_enabled && saw_tool_call_start) ? " [tool_call greedy]" : "");

        if (g_pred_enabled) {
            g_pred_generating = 1;
            g_pred_valid = 0;
        }
        g_overlap_in_decode = 1;

        char *gen_response = calloc(1, 262144);
        int gen_resp_len = 0;
        // DeepSeek-style separated reasoning. While in_think is true the
        // generation tokens accumulate here (and stream as
        // delta.reasoning_content); after </think> they go to gen_response
        // (and stream as delta.content). think_budget caps at 2048 tokens, so
        // 65536 bytes is a generous bound.
        char *gen_reasoning = calloc(1, 65536);
        int gen_reasoning_len = 0;
        ParsedToolCall parsed_tool_call;
        memset(&parsed_tool_call, 0, sizeof(parsed_tool_call));
        int gen_count = 0;
        double t_gen = now_ms();
        // Native Qwen3 generation prompt opens with <think>\n when reasoning
        // is on, so the model starts INSIDE the think block — generation will
        // emit content tokens, then </think>, then the actual response.
        // For reasoning off, the prompt closed the think block already.
        // Forced tool_choice also prepares the prompt with an empty <think></think>
        // block, so generation begins post-think.
        int in_think = (req.reasoning_enabled &&
                        !(req.tool_choice_mode == TOOL_CHOICE_FORCED && req.forced_tool_name[0]))
                       ? 1 : 0;
        int think_tokens = 0;

        for (int gen = 0; gen < req.max_tokens; gen++) {
            if (next_token == g_cfg.eos_token_1 || next_token == g_cfg.eos_token_2) break;
            if (gen > 0 && mtp_pending_draft >= 0) {
                mtp_shadow_checks++;
                if (mtp_pending_draft == next_token) {
                    mtp_shadow_hits++;
                    server_log_errorf("[mtp] %s shadow hit token=%d\n", request_id, next_token);
                } else {
                    server_log_errorf("[mtp] %s shadow miss draft=%d actual=%d\n",
                                      request_id, mtp_pending_draft, next_token);
                }
                mtp_pending_draft = -1;
            }
            if (next_token == g_cfg.think_start_token) in_think = 1;
            if (next_token == g_cfg.think_end_token) in_think = 0;
            if (in_think) think_tokens++;
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = g_cfg.think_end_token;
                in_think = 0;
            }

            const char *tok_str = decode_token(vocab, next_token);
            int send_as_delta = 1;
            if (!req.reasoning_enabled && (next_token == g_cfg.think_start_token || next_token == g_cfg.think_end_token)) {
                send_as_delta = 0;
            }
            if (req.reasoning_enabled && (next_token == g_cfg.think_start_token || next_token == g_cfg.think_end_token)) {
                send_as_delta = 0;
            }
            if (in_think && !g_show_thinking_enabled) {
                send_as_delta = 0;
            }

            int is_think_marker = (next_token == g_cfg.think_start_token || next_token == g_cfg.think_end_token);
            if (!is_think_marker && tok_str) {
                int tlen = (int)strlen(tok_str);
                if (in_think) {
                    if (gen_reasoning_len + tlen < 65535) {
                        memcpy(gen_reasoning + gen_reasoning_len, tok_str, tlen);
                        gen_reasoning_len += tlen;
                        gen_reasoning[gen_reasoning_len] = 0;
                    }
                } else if (gen_resp_len + tlen < 262143) {
                    memcpy(gen_response + gen_resp_len, tok_str, tlen);
                    gen_resp_len += tlen;
                    gen_response[gen_resp_len] = 0;
                }
                if (req.tool_count > 0 && req.tool_choice_mode != TOOL_CHOICE_NONE) {
                    if (append_bytes(&tool_call_buf, &tool_call_len, &tool_call_cap, tok_str, (size_t)tlen) == 0 &&
                        tool_call_buf && strstr(tool_call_buf, "<tool_call")) {
                        if (!saw_tool_call_start) {
                            saw_tool_call_start = 1;
                            server_log_errorf("[serve] %s detected native tool_call start at generated=%d\n",
                                              request_id, gen_count);
                        }
                        send_as_delta = 0;
                    }
                    if (tool_call_buf && parse_tool_call_from_buffer(tool_call_buf, &parsed_tool_call)) {
                        if (!request_has_tool_named(&req, parsed_tool_call.name)) {
                            server_log_errorf("[serve] %s rejected unavailable tool_call name=%s\n",
                                              request_id, parsed_tool_call.name);
                            if (g_server_debug_enabled) {
                                server_debug_write_text(request_id, "tool_call_rejected.txt", tool_call_buf);
                            }
                            parsed_tool_call_free(&parsed_tool_call);
                            memset(&parsed_tool_call, 0, sizeof(parsed_tool_call));
                            tool_call_len = 0;
                            if (tool_call_buf) tool_call_buf[0] = '\0';
                            saw_tool_call_start = 0;
                            goto tool_call_checked;
                        }
                        static int tool_call_counter = 0;
                        snprintf(parsed_tool_call.id, sizeof(parsed_tool_call.id), "call_%d", ++tool_call_counter);
                        if (g_server_debug_enabled) {
                            server_debug_write_text(request_id, "tool_call_buf.txt", tool_call_buf);
                        }
                        break;
                    }
tool_call_checked:
                    ;
                }
            }

            if (send_as_delta && req.stream && tok_str) {
                int rc;
                if (is_chat) {
                    // DeepSeek-style routing: in-think tokens → delta.reasoning_content,
                    // post-think tokens → delta.content. OpenAI-extension-aware clients
                    // (and our chat template renderer on the next turn) can then keep
                    // reasoning structurally separated from the assistant response.
                    rc = in_think
                        ? sse_send_reasoning_delta(client_fd, request_id, tok_str)
                        : sse_send_delta(client_fd, request_id, tok_str);
                } else {
                    rc = sse_send_response_text_delta(client_fd, request_id, tok_str);
                }
                if (rc < 0) {
                    server_log_errorf("[serve] %s client disconnected during stream\n", request_id);
                    break;
                }
            }
            gen_count++;
            if ((gen_count % 32) == 0) {
                server_log_errorf("[serve] %s progress generated=%d pos=%d next_token=%d\n",
                                  request_id, gen_count, pos, next_token);
            }

            cache_telemetry_note_token();
            if (token_counts && next_token >= 0 && next_token < g_cfg.vocab_size) {
                token_counts[next_token]++;
            }
            embed_lookup(wf, next_token, hidden);
            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            complete_deferred_experts();
            pos++;

            if (mtp_shadow_available) {
                memcpy(mtp_backbone_hidden, hidden, g_cfg.hidden_dim * sizeof(float));
            }

            if (final_norm_w) {
                float *normed = malloc(g_cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
                memcpy(hidden, normed, g_cfg.hidden_dim * sizeof(float));
                free(normed);
            }
            lm_head_forward(wf, hidden, logits);
            // Mirror the first-token override: inside a <tool_call> block we
            // sample greedily so the rigid XML format doesn't get sampled
            // into something malformed (e.g. <function=\n instead of
            // <function=name).
            float iter_temperature = (g_tool_call_greedy_enabled && saw_tool_call_start)
                                     ? 0.0f
                                     : req.temperature;
            next_token = pick_next_token(logits, g_cfg.vocab_size, iter_temperature, req.top_p,
                                         req.top_k, req.min_p, effective_presence_penalty,
                                         effective_repetition_penalty, token_counts,
                                         req.reasoning_enabled);
            if (mtp_shadow_available) {
                int draft_token = -1;
                if (mtp_shadow_draft_token(wf, model_path, mtp_backbone_hidden, next_token, &draft_token) == 0) {
                    server_log_errorf("[mtp] %s shadow draft after token=%d -> %d\n",
                                      request_id, next_token, draft_token);
                    mtp_pending_draft = draft_token;
                }
            }
        }

        if (mtp_shadow_checks > 0) {
            double hit_rate = (100.0 * (double)mtp_shadow_hits) / (double)mtp_shadow_checks;
            server_log_errorf("[mtp] %s shadow summary hits=%d checks=%d rate=%.1f%%\n",
                              request_id, mtp_shadow_hits, mtp_shadow_checks, hit_rate);
        }
        mtp_overlap_report(mtp_shadow_hits, mtp_shadow_checks);
        g_overlap_in_decode = 0;

        if (!parsed_tool_call.is_tool_call && saw_tool_call_start) {
            server_log_errorf("[serve] %s native tool_call started but was not parsed before completion\n", request_id);
            if (g_server_debug_enabled && tool_call_buf) {
                server_debug_write_text(request_id, "tool_call_unparsed.txt", tool_call_buf);
            }
        } else if (!parsed_tool_call.is_tool_call && req.tool_count > 0 && g_server_debug_enabled && tool_call_buf) {
            server_debug_write_text(request_id, "tool_probe_buf.txt", tool_call_buf);
        }

        char *final_json = is_chat
            ? build_chat_completion_json(request_id, req.model, gen_response, gen_reasoning, parsed_tool_call.is_tool_call ? &parsed_tool_call : NULL)
            : build_responses_json(request_id, req.model, gen_response, parsed_tool_call.is_tool_call ? &parsed_tool_call : NULL);

        if (req.stream) {
            if (parsed_tool_call.is_tool_call) {
                if (is_chat) sse_send_tool_calls(client_fd, request_id, parsed_tool_call.id, parsed_tool_call.name, parsed_tool_call.arguments);
                else sse_send_response_tool_call(client_fd, request_id, &parsed_tool_call);
            }
            if (is_chat && !parsed_tool_call.is_tool_call) {
                sse_send_done(client_fd, request_id);
            } else if (!is_chat) {
                sse_send_response_done(client_fd, request_id, final_json);
            }
        } else {
            send_json_ok(client_fd, final_json);
        }

        server_log_errorf("[serve] %s generated=%d tokens in %.0fms (%.2f tok/s)%s\n",
                          request_id, gen_count, now_ms() - t_gen,
                          gen_count > 0 ? gen_count * 1000.0 / (now_ms() - t_gen) : 0.0,
                          parsed_tool_call.is_tool_call ? " [tool_call]" : "");
        if (g_expert_cache) cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
        else if (g_malloc_cache) cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);

        free(final_json);
        free(gen_response);
        free(gen_reasoning);
        free(token_counts);
        parsed_tool_call_free(&parsed_tool_call);
        free(tool_call_buf);
        free(pt->ids);
        free(pt);
        api_request_free(&req);
        free(reqbuf);
        close(client_fd);

        if (g_server_shutdown_signal) {
            server_log_errorf("[serve] Shutdown requested by signal %d after request drain\n", g_server_shutdown_signal);
            break;
        }
    }

    if (server_fd >= 0 && g_server_listen_fd == server_fd) close(server_fd);
    g_server_listen_fd = -1;
}

// ============================================================================

enum {
    OPT_RENDER_REQUEST = 1000,
    OPT_RENDER_OUTPUT,
    OPT_RENDER_KIND,
    OPT_PARSE_TOOL_CALL,
    OPT_CACHE_ROUNDTRIP_TEST,
    OPT_MTP,
    OPT_NO_MTP,
    OPT_MTP_PREFLIGHT,
    OPT_MTP_BENCH_MATMUL,
    OPT_MTP_BENCH_ATTN,
    OPT_MTP_BENCH_MOE,
    OPT_MTP_VERIFY_LAYER,
    OPT_MTP_VERIFY_FORWARD,
    OPT_MTP_VERIFY_ROLLBACK,
    OPT_MTP_VERIFY_DEEP,
    OPT_MTP_VERIFY_LINEAR,
    OPT_MTP_VERIFY_FORWARD2,
    OPT_MTP_GENERATE,
    OPT_MTP_GENERATE_GPU,
    OPT_MTP_GENERATE_DENSE,
};

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model-id ID        Model ID from assets/model_configs.json (default from registry)\n");
    printf("  --model PATH         Model path\n");
    printf("  --weights PATH       model_weights.bin path\n");
    printf("  --manifest PATH      model_weights.json path\n");
    printf("  --vocab PATH         vocab.bin path\n");
    printf("  --prompt-tokens PATH prompt_tokens.bin path\n");
    printf("  --prompt TEXT         Prompt text (requires encode_prompt.py)\n");
    printf("  --tokens N           Max tokens to generate (default: 20)\n");
    printf("  --k N                Active experts per layer (default: 4)\n");
    printf("  --cache-entries N    Expert LRU cache size (default: 2500, 0 = disabled)\n");
    printf("  --malloc-cache N     Malloc expert cache entries (e.g., 2581 = 17GB for 80%% hit)\n");
    printf("  --cpu-linear         Disable fused GPU delta-net and use the older CPU/hybrid linear path\n");
    printf("  --timing             Enable per-layer timing breakdown\n");
    printf("  --freq               Enable expert frequency tracking + analysis\n");
    printf("  --cache-telemetry    Report cold vs eviction misses and reuse distance\n");
    printf("  --gpu-linear         Alias for the fused GPU delta-net path (default)\n");
    printf("  --predict            Enable temporal expert prediction (prefetch during CMD1_wait)\n");
    printf("  --collect-routing F  Log routing data to binary file F (for predictor training)\n");
    printf("  --think-budget N     Max thinking tokens before force </think> (default: 2048, 0=unlimited)\n");
    printf("  --serve PORT         Run HTTP server (OpenAI-compatible API)\n");
    printf("  --render-request F   Render an API request JSON to prompt/debug files and exit\n");
    printf("  --render-output DIR  Directory for --render-request debug files\n");
    printf("  --render-kind KIND   Render kind: auto, chat, responses (default: auto)\n");
    printf("  --parse-tool-call F  Parse native XML tool-call text and exit\n");
    printf("  --cache-roundtrip-test  Run synthetic disk-cache save/load roundtrip self-test and exit\n");
    printf("  --mtp                Enable experimental multi-token prediction artifact path\n");
    printf("  --no-mtp             Disable experimental multi-token prediction (default)\n");
    printf("  --mtp-preflight      Load MTP artifacts, run pre-FC smoke computation, and exit\n");
    printf("  --help               This message\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        // Support FLASHCHAT_MODEL_PATH environment variable
        const char *env_model_path = getenv("FLASHCHAT_MODEL_PATH");
        const char *model_path = env_model_path ? env_model_path : MODEL_PATH_DEFAULT;
        const char *weights_path = NULL;
        const char *manifest_path = NULL;
        const char *vocab_path = NULL;
        const char *prompt_tokens_path = NULL;
        const char *prompt_text = NULL;
        const char *env_model_id = getenv("FLASHCHAT_MODEL");
        const char *model_id = (env_model_id && env_model_id[0]) ? env_model_id : NULL;
        int max_tokens = 20;
        int K = 0;  // 0 = use g_cfg.num_experts_per_tok after config load; --k overrides
        int cache_entries = 0;  // default 0: trust OS page cache (38% faster than Metal LRU)
        int malloc_cache_entries = 0;  // 0 = disabled (override with --malloc-cache)
        const char *render_request_path = NULL;
        const char *render_output_dir = NULL;
        const char *render_kind = "auto";
        const char *parse_tool_call_path = NULL;
        int cache_roundtrip_test_requested = 0;
        int mtp_preflight_requested = 0;
        int mtp_bench_matmul_requested = 0;
        int mtp_bench_attn_requested = 0;
        int mtp_bench_moe_requested = 0;
        int mtp_verify_layer_requested = 0;
        int mtp_verify_forward_requested = 0;
        int mtp_verify_rollback_requested = 0;
        int mtp_verify_deep_requested = 0;
        int mtp_verify_linear_requested = 0;
        int mtp_verify_forward2_requested = 0;
        int mtp_generate_requested = 0;
        int mtp_generate_gpu_requested = 0;
        int mtp_generate_dense_requested = 0;
        
        // Support FLASHCHAT_SERVER_PORT environment variable
        const char *env_serve_port = getenv("FLASHCHAT_SERVER_PORT");
        int serve_port = env_serve_port ? atoi(env_serve_port) : 0;
        const char *env_temperature = getenv("FLASHCHAT_TEMPERATURE");
        const char *env_top_p = getenv("FLASHCHAT_TOP_P");
        const char *env_top_k = getenv("FLASHCHAT_TOP_K");
        const char *env_min_p = getenv("FLASHCHAT_MIN_P");
        const char *env_presence_penalty = getenv("FLASHCHAT_PRESENCE_PENALTY");
        const char *env_repetition_penalty = getenv("FLASHCHAT_REPETITION_PENALTY");
        const char *env_active_experts = getenv("FLASHCHAT_ACTIVE_EXPERTS");
        const char *env_tool_call_greedy = getenv("FLASHCHAT_TOOL_CALL_GREEDY");
        const char *env_reasoning = getenv("FLASHCHAT_REASONING");
        const char *env_show_thinking = getenv("FLASHCHAT_SHOW_THINKING");
        const char *env_system_prompt_cache = getenv("FLASHCHAT_SYSTEM_PROMPT_CACHE");
        const char *env_system_prompt_cache_max_entries = getenv("FLASHCHAT_SYSTEM_PROMPT_CACHE_MAX_ENTRIES");
        const char *env_mtp = getenv("FLASHCHAT_MTP");
        if (env_temperature && env_temperature[0]) g_default_temperature = strtof(env_temperature, NULL);
        if (env_top_p && env_top_p[0]) g_default_top_p = strtof(env_top_p, NULL);
        if (env_top_k && env_top_k[0]) g_default_top_k = atoi(env_top_k);
        if (env_min_p && env_min_p[0]) g_default_min_p = strtof(env_min_p, NULL);
        if (env_presence_penalty && env_presence_penalty[0]) g_default_presence_penalty = strtof(env_presence_penalty, NULL);
        if (env_repetition_penalty && env_repetition_penalty[0]) g_default_repetition_penalty = strtof(env_repetition_penalty, NULL);
        // FLASHCHAT_ACTIVE_EXPERTS: ad-hoc override for K (active experts per token).
        // Useful for K-sweep benchmarks. Has the same precedence as --k (whichever
        // is parsed last wins; --k is parsed via getopt below, env is read here, so
        // a CLI --k passed simultaneously will override the env). Empty/0 = no
        // override, use g_cfg.num_experts_per_tok from registry.
        if (env_active_experts && env_active_experts[0]) {
            int v = atoi(env_active_experts);
            if (v > 0) {
                K = v;
                fprintf(stderr, "[init] FLASHCHAT_ACTIVE_EXPERTS=%d overrides config K\n", v);
            }
        }
        if (env_tool_call_greedy && env_tool_call_greedy[0]) {
            g_tool_call_greedy_enabled = server_flag_enabled(env_tool_call_greedy);
            fprintf(stderr, "[init] FLASHCHAT_TOOL_CALL_GREEDY=%d (sampling inside <tool_call>...</tool_call> %s)\n",
                    g_tool_call_greedy_enabled,
                    g_tool_call_greedy_enabled ? "forced greedy" : "uses request sampling");
        }
        if (env_reasoning && env_reasoning[0]) {
            g_default_reasoning_enabled = server_flag_enabled(env_reasoning);
        }
        if (env_show_thinking && env_show_thinking[0]) {
            g_show_thinking_enabled = server_flag_enabled(env_show_thinking);
        }
        if (env_system_prompt_cache && env_system_prompt_cache[0]) {
            g_system_prompt_cache_enabled = server_flag_enabled(env_system_prompt_cache);
        }
        if (env_system_prompt_cache_max_entries && env_system_prompt_cache_max_entries[0]) {
            g_system_prompt_cache_max_entries = atoi(env_system_prompt_cache_max_entries);
            if (g_system_prompt_cache_max_entries < 0) g_system_prompt_cache_max_entries = 0;
        }
        if (env_mtp && env_mtp[0]) {
            g_mtp_predictions = parse_mtp_predictions(env_mtp);
        }

        // Also try to read config from ~/.config/flashchat/config if env vars not set
        const char *home = getenv("HOME");
        if (home) {
            char config_path[1024];
            snprintf(config_path, sizeof(config_path), "%s/.config/flashchat/config", home);
            
            FILE *cfg = fopen(config_path, "r");
            if (cfg) {
                char line[512];
                while (fgets(line, sizeof(line), cfg)) {
                    if (!env_model_id && strncmp(line, "MODEL=", 6) == 0) {
                        char *val = line + 6;
                        char *quote = strchr(val, '"');
                        if (quote) {
                            char *end_quote = strchr(quote + 1, '"');
                            if (end_quote) *end_quote = '\0';
                            model_id = strdup(quote + 1);
                        }
                    }
                    // Parse MODEL_PATH from config file (for when env var not set)
                    if (strncmp(line, "MODEL_PATH=", 11) == 0) {
                        if (!env_model_path) {
                            char *val = line + 11;
                            char *quote = strchr(val, '"');
                            if (quote) {
                                char *end_quote = strchr(quote + 1, '"');
                                if (end_quote) *end_quote = '\0';
                                model_path = strdup(quote + 1);
                            }
                        }
                    }
                    if (!env_temperature && strncmp(line, "TEMPERATURE=", 12) == 0) {
                        char *val = line + 12;
                        char *quote = strchr(val, '"');
                        if (quote) g_default_temperature = strtof(quote + 1, NULL);
                    }
                    if (!env_top_p && strncmp(line, "TOP_P=", 6) == 0) {
                        char *val = line + 6;
                        char *quote = strchr(val, '"');
                        if (quote) g_default_top_p = strtof(quote + 1, NULL);
                    }
                    if (!env_top_k && strncmp(line, "TOP_K=", 6) == 0) {
                        char *val = line + 6;
                        char *quote = strchr(val, '"');
                        if (quote) g_default_top_k = atoi(quote + 1);
                    }
                    if (!env_min_p && strncmp(line, "MIN_P=", 6) == 0) {
                        char *val = line + 6;
                        char *quote = strchr(val, '"');
                        if (quote) g_default_min_p = strtof(quote + 1, NULL);
                    }
                    if (!env_presence_penalty && strncmp(line, "PRESENCE_PENALTY=", 17) == 0) {
                        char *val = line + 17;
                        char *quote = strchr(val, '"');
                        if (quote) g_default_presence_penalty = strtof(quote + 1, NULL);
                    }
                    if (!env_repetition_penalty && strncmp(line, "REPETITION_PENALTY=", 19) == 0) {
                        char *val = line + 19;
                        char *quote = strchr(val, '"');
                        if (quote) g_default_repetition_penalty = strtof(quote + 1, NULL);
                    }
                    if (!env_show_thinking && strncmp(line, "SHOW_THINKING=", 14) == 0) {
                        char *val = line + 14;
                        char *quote = strchr(val, '"');
                        if (quote) {
                            char *end_quote = strchr(quote + 1, '"');
                            if (end_quote) *end_quote = '\0';
                            g_show_thinking_enabled = server_flag_enabled(quote + 1);
                        }
                    }
                    if (!env_system_prompt_cache && strncmp(line, "SYSTEM_PROMPT_CACHE=", 20) == 0) {
                        char *val = line + 20;
                        char *quote = strchr(val, '"');
                        if (quote) {
                            char *end_quote = strchr(quote + 1, '"');
                            if (end_quote) *end_quote = '\0';
                            g_system_prompt_cache_enabled = server_flag_enabled(quote + 1);
                        }
                    }
                    if (!env_system_prompt_cache_max_entries && strncmp(line, "SYSTEM_PROMPT_CACHE_MAX_ENTRIES=", 32) == 0) {
                        char *val = line + 32;
                        char *quote = strchr(val, '"');
                        if (quote) {
                            g_system_prompt_cache_max_entries = atoi(quote + 1);
                            if (g_system_prompt_cache_max_entries < 0) g_system_prompt_cache_max_entries = 0;
                        }
                    }
                    if (!env_mtp && strncmp(line, "MTP=", 4) == 0) {
                        char *val = line + 4;
                        char *quote = strchr(val, '"');
                        if (quote) {
                            char *end_quote = strchr(quote + 1, '"');
                            if (end_quote) *end_quote = '\0';
                            g_mtp_predictions = parse_mtp_predictions(quote + 1);
                        }
                    }
                }
                fclose(cfg);
            }
        }

        static struct option long_options[] = {
            {"model-id",      required_argument, 0, 'I'},
            {"model",         required_argument, 0, 'm'},
            {"weights",       required_argument, 0, 'w'},
            {"manifest",      required_argument, 0, 'j'},
            {"vocab",         required_argument, 0, 'v'},
            {"prompt-tokens", required_argument, 0, 'p'},
            {"prompt",        required_argument, 0, 'P'},
            {"tokens",        required_argument, 0, 't'},
            {"k",             required_argument, 0, 'k'},
            {"cache-entries",  required_argument, 0, 'C'},
            {"malloc-cache",   required_argument, 0, 'M'},
            {"cpu-linear",    no_argument,       0, 'L'},
            {"skip-linear",   no_argument,       0, 'S'},
            {"timing",        no_argument,       0, 'T'},
            {"freq",          no_argument,       0, 'F'},
            {"cache-telemetry", no_argument,     0, 'E'},
            {"gpu-linear",    no_argument,       0, 'G'},
            {"think-budget",  required_argument, 0, 'B'},
            {"serve",         required_argument, 0, 'R'},
            {"predict",       no_argument,       0, 'D'},
            {"collect-routing", required_argument, 0, 'Z'},
            {"render-request", required_argument, 0, OPT_RENDER_REQUEST},
            {"render-output",  required_argument, 0, OPT_RENDER_OUTPUT},
            {"render-kind",    required_argument, 0, OPT_RENDER_KIND},
            {"parse-tool-call", required_argument, 0, OPT_PARSE_TOOL_CALL},
            {"cache-roundtrip-test", no_argument,  0, OPT_CACHE_ROUNDTRIP_TEST},
            {"mtp",           no_argument,       0, OPT_MTP},
            {"no-mtp",        no_argument,       0, OPT_NO_MTP},
            {"mtp-preflight",  no_argument,       0, OPT_MTP_PREFLIGHT},
            {"mtp-bench-matmul", no_argument,     0, OPT_MTP_BENCH_MATMUL},
            {"mtp-bench-attn", no_argument,       0, OPT_MTP_BENCH_ATTN},
            {"mtp-bench-moe", no_argument,        0, OPT_MTP_BENCH_MOE},
            {"mtp-verify-layer", no_argument,     0, OPT_MTP_VERIFY_LAYER},
            {"mtp-verify-forward", no_argument,   0, OPT_MTP_VERIFY_FORWARD},
            {"mtp-verify-rollback", no_argument,  0, OPT_MTP_VERIFY_ROLLBACK},
            {"mtp-verify-deep", no_argument,      0, OPT_MTP_VERIFY_DEEP},
            {"mtp-verify-linear", no_argument,    0, OPT_MTP_VERIFY_LINEAR},
            {"mtp-verify-forward2", no_argument,  0, OPT_MTP_VERIFY_FORWARD2},
            {"mtp-generate", no_argument,         0, OPT_MTP_GENERATE},
            {"mtp-generate-gpu", no_argument,     0, OPT_MTP_GENERATE_GPU},
            {"mtp-generate-dense", no_argument,   0, OPT_MTP_GENERATE_DENSE},
            {"help",          no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "I:m:w:j:v:p:P:t:k:C:M:R:B:LSTFEGh", long_options, NULL)) != -1) {
            switch (c) {
                case 'I': model_id = optarg; break;
                case 'm': model_path = optarg; break;
                case 'w': weights_path = optarg; break;
                case 'j': manifest_path = optarg; break;
                case 'v': vocab_path = optarg; break;
                case 'p': prompt_tokens_path = optarg; break;
                case 'P': prompt_text = optarg; break;
                case 't': max_tokens = atoi(optarg); break;
                case 'k': K = atoi(optarg); break;
                case 'C': cache_entries = atoi(optarg); break;
                case 'M': malloc_cache_entries = atoi(optarg); break;
                case 'L': gpu_linear_attn_enabled = 0; break;
                case 'S': linear_attn_bypass = 1; break;
                case 'T': g_timing_enabled = 1; break;
                case 'F': g_freq_tracking = 1; break;
                case 'E': g_cache_telemetry_enabled = 1; break;
                case 'G': gpu_linear_attn_enabled = 1; break;
                case 'D': g_pred_enabled = 1; break;
                case 'Z':
                    g_routing_log = fopen(optarg, "wb");
                    if (!g_routing_log) {
                        fprintf(stderr, "ERROR: cannot open routing log: %s\n", optarg);
                        return 1;
                    }
                    break;
                case 'B': g_think_budget = atoi(optarg); break;
                case 'R': serve_port = atoi(optarg); break;
                case OPT_RENDER_REQUEST: render_request_path = optarg; break;
                case OPT_RENDER_OUTPUT: render_output_dir = optarg; break;
                case OPT_RENDER_KIND: render_kind = optarg; break;
                case OPT_PARSE_TOOL_CALL: parse_tool_call_path = optarg; break;
                case OPT_CACHE_ROUNDTRIP_TEST: cache_roundtrip_test_requested = 1; break;
                case OPT_MTP: g_mtp_predictions = 1; break;
                case OPT_NO_MTP: g_mtp_predictions = 0; break;
                case OPT_MTP_PREFLIGHT: mtp_preflight_requested = 1; g_mtp_predictions = 1; break;
                case OPT_MTP_BENCH_MATMUL: mtp_bench_matmul_requested = 1; break;
                case OPT_MTP_BENCH_ATTN: mtp_bench_attn_requested = 1; break;
                case OPT_MTP_BENCH_MOE: mtp_bench_moe_requested = 1; break;
                case OPT_MTP_VERIFY_LAYER: mtp_verify_layer_requested = 1; break;
                case OPT_MTP_VERIFY_FORWARD: mtp_verify_forward_requested = 1; break;
                case OPT_MTP_VERIFY_ROLLBACK: mtp_verify_rollback_requested = 1; break;
                case OPT_MTP_VERIFY_DEEP: mtp_verify_deep_requested = 1; break;
                case OPT_MTP_VERIFY_LINEAR: mtp_verify_linear_requested = 1; break;
                case OPT_MTP_VERIFY_FORWARD2: mtp_verify_forward2_requested = 1; break;
                case OPT_MTP_GENERATE: mtp_generate_requested = 1; g_mtp_predictions = 1; break;
                case OPT_MTP_GENERATE_GPU: mtp_generate_gpu_requested = 1; g_mtp_predictions = 1; break;
                case OPT_MTP_GENERATE_DENSE: mtp_generate_dense_requested = 1; g_mtp_predictions = 1; break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        const char *config_json_path = resolve_model_config_path();

        // Load model configuration from registry
        if (!model_id) {
            static char default_model_id[64];
            if (load_default_model_id(config_json_path, default_model_id, sizeof(default_model_id)) == 0
                && default_model_id[0] != '\0') {
                model_id = default_model_id;
            } else {
                fprintf(stderr,
                    "ERROR: No model ID could be determined. Checked:\n"
                    "  - --model-id CLI flag: not provided\n"
                    "  - FLASHCHAT_MODEL environment variable: not set\n"
                    "  - %s default_model field: could not be read (file missing or malformed)\n"
                    "Run 'flashchat config' to set a default model, or pass --model-id explicitly.\n",
                    config_json_path ? config_json_path : "model_configs.json");
                return 1;
            }
        }
        if (load_model_config(config_json_path, model_id, &g_cfg) != 0) {
            fprintf(stderr, "ERROR: Failed to load model config for '%s'\n", model_id);
            return 1;
        }
        kApiModelId = g_cfg.model_id;
        if (g_mtp_predictions < 0) {
            g_mtp_predictions = g_cfg.mtp_default_predictions > 0 ? g_cfg.mtp_default_predictions : 0;
        }
        if (g_cfg.mtp_max_predictions > 0 && g_mtp_predictions > g_cfg.mtp_max_predictions) {
            fprintf(stderr, "[mtp] requested draft budget %d exceeds model max %d; clamping\n",
                    g_mtp_predictions, g_cfg.mtp_max_predictions);
            g_mtp_predictions = g_cfg.mtp_max_predictions;
        }

        // K (active experts per token) defaults to the model's num_experts_per_tok
        // from config — i.e. exactly what the model was trained with. Previously
        // hardcoded to 4 across all models, which silently halved active experts
        // for Qwen3.5-A17B (trained with K=10) and Qwen3.6-A3B (trained with K=8).
        // --k still overrides for the streaming-bound 397B model where the SSD
        // I/O cost of more experts may justify the quality tradeoff.
        //
        // Hard cap at MAX_K=8 (Metal multi-expert buffer slots). For models trained
        // with K>8 (e.g. 397B-A17B's K=10) this is a known under-experting that
        // requires expanding MAX_K to fix; report it loudly so it is not silent.
        if (K <= 0) {
            K = g_cfg.num_experts_per_tok > 0 ? g_cfg.num_experts_per_tok : 4;
            fprintf(stderr, "[init] K=%d active experts (from config num_experts_per_tok)\n", K);
        } else {
            fprintf(stderr, "[init] K=%d active experts (--k override; config wants %d)\n",
                    K, g_cfg.num_experts_per_tok);
        }
        if (K > MAX_K) {
            fprintf(stderr, "[init] WARNING: requested K=%d exceeds MAX_K=%d (Metal buffer slots); "
                            "clamping to %d. Model will run UNDER-EXPERTED. Increase MAX_K to fix.\n",
                    K, MAX_K, MAX_K);
            K = MAX_K;
        }

        if (parse_tool_call_path) {
            return parse_tool_call_debug(parse_tool_call_path);
        }
        if (render_request_path) {
            return render_request_debug(render_request_path, render_output_dir, render_kind);
        }
        if (cache_roundtrip_test_requested) {
            return cache_roundtrip_test();
        }

        // Build default paths under the per-model Flashchat artifact directory.
        char default_weights[1024], default_manifest[1024], default_vocab[1024];

        if (!weights_path) {
            snprintf(default_weights, sizeof(default_weights),
                     "%s/flashchat/model_weights.bin", model_path);
            weights_path = default_weights;
        }
        if (!manifest_path) {
            snprintf(default_manifest, sizeof(default_manifest),
                     "%s/flashchat/model_weights.json", model_path);
            manifest_path = default_manifest;
        }
        if (!vocab_path) {
            snprintf(default_vocab, sizeof(default_vocab),
                     "%s/flashchat/vocab.bin", model_path);
            vocab_path = default_vocab;
        }

        // ---- Initialize Metal ----
        g_metal = metal_setup();
        if (!g_metal) {
            fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
        }

        // ---- Initialize persistent I/O thread pool ----
        io_pool_init();

        // ---- Initialize malloc expert cache (if requested) ----
        if (malloc_cache_entries > 0) {
            g_malloc_cache = malloc_cache_init(malloc_cache_entries, g_metal ? g_metal->device : MTLCreateSystemDefaultDevice());
            cache_entries = 0;  // disable Metal LRU cache when malloc cache is active
        }

        // ---- Initialize expert LRU cache ----
        if (cache_entries > 0 && g_metal) {
            g_expert_cache = expert_cache_new(g_metal->device, cache_entries);
        }

        printf("=== %s Metal Inference Engine ===\n", g_cfg.model_name[0] ? g_cfg.model_name : "Qwen3.5-397B-A17B");
        printf("Model:    %s\n", model_path);
        printf("Weights:  %s\n", weights_path);
        printf("Manifest: %s\n", manifest_path);
        printf("Vocab:    %s\n", vocab_path);
        printf("K:        %d experts/layer\n", K);
        printf("Experts:  %zu bytes each\n", active_expert_size());
        printf("Linear:   %s\n", gpu_linear_attn_enabled ? "fused GPU delta-net" : "CPU/hybrid fallback");
        printf("Tokens:   %d\n", max_tokens);
        if (g_malloc_cache) {
            printf("Cache:    malloc %d entries (%.1f GB)\n",
                   malloc_cache_entries, (double)malloc_cache_entries * active_expert_size() / 1e9);
        } else {
            printf("Cache:    %d entries%s\n", cache_entries,
                   cache_entries > 0 ? "" : " (disabled)");
        }

        double t0 = now_ms();

        // ---- Load weights ----
        WeightFile *wf = open_weights(weights_path, manifest_path);
        if (!wf) {
            fprintf(stderr, "ERROR: Failed to load weights\n");
            return 1;
        }

        MTPArtifacts mtp = detect_mtp_artifacts(wf, model_path);
        if (g_mtp_predictions > 0) {
            if (mtp.tensors_present && mtp.packed_experts_present) {
                fprintf(stderr, "[mtp] enabled; draft budget=%d; artifacts detected (%s)\n",
                        g_mtp_predictions, mtp.packed_dir);
                fprintf(stderr, "[mtp] draft/verify decode is not active yet; using shadow path\n");
                if (g_mtp_predictions > 1) {
                    fprintf(stderr, "[mtp] shadow path currently evaluates one draft token; extra budget is reserved for verify decode\n");
                }
            } else {
                fprintf(stderr, "[mtp] requested but artifacts are incomplete; using existing decode path\n");
                fprintf(stderr, "[mtp] tensors=%s packed_experts=%s\n",
                        mtp.tensors_present ? "yes" : "no",
                        mtp.packed_experts_present ? "yes" : "no");
            }
        } else if (mtp.tensors_present || mtp.packed_experts_present) {
            fprintf(stderr, "[mtp] artifacts detected but disabled; using existing decode path\n");
        }
        if (g_mtp_predictions > 0) {
            build_mtp_cache(wf);
        }
        {
            const char *env_overlap = getenv("FLASHCHAT_MTP_OVERLAP");
            g_mtp_overlap_enabled = (g_mtp_predictions > 0) ||
                                    (env_overlap && server_flag_enabled(env_overlap));
            if (g_mtp_overlap_enabled) {
                fprintf(stderr, "[mtp] expert-overlap instrumentation active (decode steps only)\n");
            }
        }
        if (mtp_preflight_requested) {
            int rc_fc = mtp_preflight_forward(wf);
            int rc_layer = mtp_preflight_decoder_layer(wf, model_path);
            return rc_fc == 0 && rc_layer == 0 ? 0 : 1;
        }

        // Wrap weight file for Metal GPU access
        if (g_metal) {
            metal_set_weights(g_metal, wf->data, wf->size);
        }

        if (mtp_bench_matmul_requested) {
            return mtp_bench_matmul(wf);
        }
        if (mtp_bench_attn_requested) {
            return mtp_bench_attn(wf);
        }
        if (mtp_bench_moe_requested) {
            return mtp_bench_moe(wf, model_path);
        }
        if (mtp_verify_layer_requested) {
            return mtp_verify_layer(wf, model_path);
        }
        if (mtp_verify_forward_requested) {
            return mtp_verify_forward(wf, model_path);
        }
        if (mtp_verify_rollback_requested) {
            return mtp_verify_rollback(wf, model_path);
        }
        if (mtp_verify_deep_requested) {
            return mtp_verify_deep(wf, model_path);
        }
        if (mtp_verify_linear_requested) {
            return mtp_verify_linear(wf, model_path);
        }
        if (mtp_verify_forward2_requested) {
            return mtp_verify_forward2(wf, model_path);
        }
        if (mtp_generate_requested) {
            return mtp_generate(wf, model_path, 48);
        }
        if (mtp_generate_gpu_requested) {
            return mtp_generate_gpu(wf, model_path, 48);
        }
        if (mtp_generate_dense_requested) {
            return mtp_generate_dense(wf, model_path, 64);
        }

        // ---- Load vocabulary ----
        Vocabulary *vocab = load_vocab(vocab_path);
        if (!vocab) {
            fprintf(stderr, "ERROR: Failed to load vocabulary\n");
            return 1;
        }

        // ---- Get prompt tokens (skip in serve mode) ----
        PromptTokens *pt = NULL;
        if (serve_port == 0) {
            if (prompt_text) {
                pt = encode_prompt_text_to_tokens(prompt_text);
                if (!pt) {
                    fprintf(stderr, "ERROR: Failed to encode prompt. Make sure encode_prompt.py exists.\n");
                    return 1;
                }
            } else if (!prompt_tokens_path) {
                pt = encode_prompt_text_to_tokens("Hello, what is");
                if (!pt) {
                    fprintf(stderr, "ERROR: No prompt tokens and encode_prompt.py not found\n");
                    return 1;
                }
            } else {
                pt = load_prompt_tokens(prompt_tokens_path);
            }

            if (!pt) {
                fprintf(stderr, "ERROR: Failed to load prompt tokens from %s\n", prompt_tokens_path);
                return 1;
            }
            printf("[prompt] %d tokens:", pt->count);
            for (int i = 0; i < pt->count && i < 20; i++) {
                printf(" %d", pt->ids[i]);
            }
            printf("\n");
        }

        // ---- Open + mmap packed expert files ----
        // Tiered I/O: two fds per layer file.
        //   layer_fds[i]      = warm fd (page cached) — for experts seen before
        //   layer_fds_cold[i] = cold fd (F_NOCACHE)   — for first-time expert reads
        // Seen-expert bitset tracks which (layer, expert) pairs have been read before.
        // First read goes through cold fd (no page cache pollution).
        // Subsequent reads go through warm fd (page cache hit = 32 GB/s vs 5.5 GB/s).
        int layer_fds[g_cfg.num_layers];
        int layer_fds_cold[g_cfg.num_layers];
        void *layer_mmaps[g_cfg.num_layers];
        size_t layer_mmap_sizes[g_cfg.num_layers];
        int expert_layers_available = 0;

        // Reset the global seen-expert bitset
        memset(g_expert_seen, 0, sizeof(g_expert_seen));

        for (int i = 0; i < g_cfg.num_layers; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/flashchat/packed_experts/layer_%02d.bin", model_path, i);
            layer_fds[i] = open(path, O_RDONLY);
            layer_fds_cold[i] = -1;  // no longer used (trust OS page cache)
            layer_mmaps[i] = MAP_FAILED;
            layer_mmap_sizes[i] = 0;
            if (layer_fds[i] >= 0) {
                expert_layers_available++;
                // Disable readahead: expert reads are random (different offsets per token).
                // Read-ahead prefetches adjacent data we won't use, wasting SSD bandwidth.
                fcntl(layer_fds[i], F_RDAHEAD, 0);
                struct stat st;
                if (fstat(layer_fds[i], &st) == 0 && st.st_size > 0) {
                    layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[i], 0);
                    if (layer_mmaps[i] != MAP_FAILED) {
                        layer_mmap_sizes[i] = st.st_size;
                        // No madvise: kernel default is best.
                        // MADV_RANDOM disables readahead (tested: hurts).
                        // MADV_SEQUENTIAL doesn't reduce I/O fragmentation (tested: no effect).
                        // The kernel fragments 3.9MB preads into ~5.7 disk ops regardless
                        // of hints — this is inherent to the page cache's physical page layout.
                    }
                }
            }
        }
        printf("[experts] %d/%d packed layer files available (mmap'd)\n", expert_layers_available, g_cfg.num_layers);

        // ---- LZ4 compressed experts: auto-detect and load ----
        {
            char lz4_probe[1024];
            snprintf(lz4_probe, sizeof(lz4_probe), "%s/flashchat/packed_experts_lz4/layer_00.bin", model_path);
            if (access(lz4_probe, R_OK) == 0) {
                int lz4_layers = 0;
                for (int i = 0; i < g_cfg.num_layers; i++) {
                    char lz4_path[1024];
                    snprintf(lz4_path, sizeof(lz4_path), "%s/flashchat/packed_experts_lz4/layer_%02d.bin", model_path, i);
                    int lz4_fd = open(lz4_path, O_RDONLY);
                    if (lz4_fd >= 0) {
                        // Load index header (512 entries × 16 bytes = 8KB)
                        g_lz4_index[i] = malloc(g_cfg.num_experts * sizeof(LZ4IndexEntry));
                        ssize_t nr = pread(lz4_fd, g_lz4_index[i],
                                           g_cfg.num_experts * sizeof(LZ4IndexEntry), 0);
                        if (nr == g_cfg.num_experts * (ssize_t)sizeof(LZ4IndexEntry)) {
                            // Replace the raw fd with the LZ4 fd
                            close(layer_fds[i]);
                            layer_fds[i] = lz4_fd;
                            fcntl(lz4_fd, F_RDAHEAD, 1);
                            lz4_layers++;
                        } else {
                            free(g_lz4_index[i]);
                            g_lz4_index[i] = NULL;
                            close(lz4_fd);
                        }
                    }
                }
                if (lz4_layers > 0) {
                    g_use_lz4 = 1;
                    // Allocate compressed read buffers (one per expert slot)
                    for (int k = 0; k < MAX_K; k++) {
                        g_lz4_comp_bufs[k] = malloc(g_cfg.expert_size + 4096);
                    }
                    printf("[lz4] %d/%d layers using LZ4 compressed experts\n",
                           lz4_layers, g_cfg.num_layers);
                }
            }
        }

        // Wire up tiered I/O globals
        g_layer_fds_cold = layer_fds_cold;
        if (!g_use_lz4)
            printf("[tiered-io] Cold fds (F_NOCACHE) + warm fds (page cached) active\n");

        // Warm page cache hint
        if (expert_layers_available > 0) {
            double t_warm = now_ms();
            for (int i = 0; i < g_cfg.num_layers; i++) {
                if (layer_fds[i] >= 0) {
                    char dummy[4096];
                    pread(layer_fds[i], dummy, sizeof(dummy), 0);
                }
            }
            if (g_timing_enabled) {
                printf("[warmup] Page cache hint: %.1f ms\n", now_ms() - t_warm);
            }
        }

        // ---- Allocate per-layer state ----
        void **layer_states = calloc(g_cfg.num_layers, sizeof(void *));
        KVCache **kv_caches = calloc(g_cfg.num_layers, sizeof(KVCache *));

        for (int i = 0; i < g_cfg.num_layers; i++) {
            int is_full = ((i + 1) % g_cfg.full_attn_interval == 0);
            if (is_full) {
                kv_caches[i] = kv_cache_new();
            } else {
                layer_states[i] = linear_attn_state_new();
            }
        }

        double t_init = now_ms();
        printf("[init] Setup: %.1f ms\n\n", t_init - t0);

        // ---- Allocate working buffers ----
        float *hidden = calloc(g_cfg.hidden_dim, sizeof(float));
        float *logits = calloc(g_cfg.vocab_size, sizeof(float));
        uint16_t *final_norm_w = get_tensor_ptr(wf, "model.norm.weight");

        // ---- Serve mode: enter HTTP server loop (never returns) ----
        if (serve_port > 0) {
            reset_delta_net_state();
            serve_loop(serve_port, model_path, wf, vocab,
                       layer_states, kv_caches,
                       (void **)layer_mmaps, layer_fds,
                       hidden, logits, final_norm_w, K);
            // serve_loop never returns, but cleanup just in case
            free(hidden); free(logits);
            return 0;
        }

        // ---- Generate tokens ----
        reset_delta_net_state();  // zero GPU delta-net state before generation
        if (g_cache_telemetry_enabled) cache_telemetry_reset();
        if (g_timing_enabled) {
            printf("--- Generating %d tokens ---\n", max_tokens);
        }
        int pos = 0;  // position counter for RoPE

        // ---- Batch prefill: pre-embed all prompt tokens ----
        // Embedding all tokens upfront into a batch buffer avoids interleaving
        // embed_lookup with GPU work, and enables the optimized prefill loop below.
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * g_cfg.hidden_dim * sizeof(float));
            double t_embed = now_ms();
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(wf, pt->ids[i], embed_batch + (size_t)i * g_cfg.hidden_dim);
            }
            double embed_ms = now_ms() - t_embed;
            if (g_timing_enabled) {
                printf("  [prefill] batch embed %d tokens: %.1f ms\n", pt->count, embed_ms);
            }
        }

        // ---- Batch prefill loop ----
        // Process all prompt tokens through the model. For intermediate tokens
        // (not the last), we use discard_deferred_experts() which waits for the GPU
        // but skips the CPU readback/combine of the last layer's expert outputs.
        // This is safe because the hidden state from intermediate prefill tokens
        // is immediately overwritten by the next token's embedding — the recurrent
        // state (KV cache, delta-net state) is already updated inside fused_layer_forward.
        if (pt->count > 1) {
            double t_prefill_batch = now_ms();
            double first_tok_ms = 0;

            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                double t_tok = now_ms();

                // Load pre-embedded token from batch buffer
                cache_telemetry_note_token();
                memcpy(hidden, embed_batch + (size_t)token_idx * g_cfg.hidden_dim,
                       g_cfg.hidden_dim * sizeof(float));

                // Run through all 60 transformer layers
                for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                    int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }

                // Discard last layer's expert output — hidden will be overwritten
                // by the next token's embedding. Only wait for GPU (buffer safety).
                discard_deferred_experts();
                pos++;

                if (token_idx == 0) {
                    first_tok_ms = now_ms() - t_tok;
                }
            }

            double prefill_batch_ms = now_ms() - t_prefill_batch;
            double avg_ms = (pt->count > 2) ?
                (prefill_batch_ms - first_tok_ms) / (pt->count - 2) : first_tok_ms;
            if (g_timing_enabled) {
                printf("  [prefill] %d/%d tokens: %.0f ms (first: %.0f ms, rest avg: %.0f ms)\n",
                       pt->count - 1, pt->count, prefill_batch_ms, first_tok_ms, avg_ms);
            }
        }

        // ---- Last prefill token (or single-token prompt) ----
        // This one needs full completion since we need hidden state for logits.
        {
            cache_telemetry_note_token();
            if (embed_batch) {
                memcpy(hidden, embed_batch + (size_t)(pt->count - 1) * g_cfg.hidden_dim,
                       g_cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, pt->ids[0], hidden);
            }

            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Full completion — need hidden state for final norm + lm_head
            complete_deferred_experts();
            pos++;
        }

        if (embed_batch) { free(embed_batch); embed_batch = NULL; }

        float mtp_backbone_hidden[g_cfg.hidden_dim];
        int mtp_shadow_available = mtp_can_shadow_draft(wf);
        int mtp_pending_draft = -1;
        int mtp_shadow_hits = 0;
        int mtp_shadow_checks = 0;
        g_mtp_kv_len = 0;  // fresh MTP attention context per generation
        if (mtp_shadow_available) {
            memcpy(mtp_backbone_hidden, hidden, g_cfg.hidden_dim * sizeof(float));
        }

        // ---- Final norm ----
        if (final_norm_w) {
            float *normed = malloc(g_cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(hidden, final_norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
            memcpy(hidden, normed, g_cfg.hidden_dim * sizeof(float));
            free(normed);
        }

        // ---- LM head ----
        double t_lm = now_ms();
        lm_head_forward(wf, hidden, logits);
        double lm_ms = now_ms() - t_lm;

        // ---- Sample first token ----
        int next_token = cpu_argmax(logits, g_cfg.vocab_size);
        if (mtp_shadow_available) {
            int draft_token = -1;
            if (mtp_shadow_draft_token(wf, model_path, mtp_backbone_hidden, next_token, &draft_token) == 0) {
                fprintf(stderr, "[mtp] shadow draft after token=%d -> %d\n", next_token, draft_token);
                mtp_pending_draft = draft_token;
            }
        }
        double ttft_ms = now_ms() - t0;

        if (g_timing_enabled) {
            int top5[5] = {0,0,0,0,0};
            float topv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int i = 0; i < g_cfg.vocab_size; i++) {
                int min_k = 0;
                for (int k = 1; k < 5; k++) if (topv[k] < topv[min_k]) min_k = k;
                if (logits[i] > topv[min_k]) { topv[min_k] = logits[i]; top5[min_k] = i; }
            }
            fprintf(stderr, "[debug] Top 5 logits (next_token=%d):\n", next_token);
            for (int i = 0; i < 5; i++) {
                fprintf(stderr, "  token %d (\"%s\") logit=%.4f\n",
                        top5[i], decode_token(vocab, top5[i]), topv[i]);
            }
            fprintf(stderr, "[debug] hidden rms after final_norm=%.4f, logits rms=%.4f\n",
                    vec_rms(hidden, g_cfg.hidden_dim), vec_rms(logits, g_cfg.vocab_size));
        }
        if (g_timing_enabled) {
            printf("[ttft] %.0f ms (prefill %d tokens + lm_head %.0f ms)\n",
                   ttft_ms, pt->count, lm_ms);
        }

        printf("\n--- Output ---\n");
        printf("%s", decode_token(vocab, next_token));
        fflush(stdout);

        int total_generated = 1;
        int in_think = (next_token == g_cfg.think_start_token) ? 1 : 0;
        int think_tokens = 0;

        // ---- Auto-regressive generation ----
        if (g_timing_enabled) timing_reset();
        if (g_pred_enabled) {
            g_pred_generating = 1;  // enable prediction storage/use during generation
            g_pred_valid = 0;       // reset — first gen token builds predictions
        }
        g_overlap_in_decode = 1;
        for (int gen = 1; gen < max_tokens; gen++) {
            double t_gen_start = now_ms();

            // Check EOS
            if (next_token == g_cfg.eos_token_1 || next_token == g_cfg.eos_token_2) {
                fprintf(stderr, "\n[eos] Token %d at position %d\n", next_token, gen);
                break;
            }

            // Think budget enforcement
            if (next_token == g_cfg.think_start_token) in_think = 1;
            if (next_token == g_cfg.think_end_token) in_think = 0;
            if (in_think) think_tokens++;

            // Embed the just-generated token (next iteration)
            cache_telemetry_note_token();
            embed_lookup(wf, next_token, hidden);

            // Run 60 layers (fused: 1+K cmd buffers per layer)
            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % g_cfg.full_attn_interval == 0);
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Complete last layer's deferred GPU experts before final norm
            complete_deferred_experts();
            pos++;

            if (mtp_shadow_available) {
                memcpy(mtp_backbone_hidden, hidden, g_cfg.hidden_dim * sizeof(float));
            }

            // Final norm
            if (final_norm_w) {
                float *normed = malloc(g_cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, g_cfg.hidden_dim, g_cfg.rms_norm_eps);
                memcpy(hidden, normed, g_cfg.hidden_dim * sizeof(float));
                free(normed);
            }

            // LM head
            lm_head_forward(wf, hidden, logits);

            // Greedy sample
            next_token = cpu_argmax(logits, g_cfg.vocab_size);
            if (mtp_pending_draft >= 0) {
                mtp_shadow_checks++;
                if (mtp_pending_draft == next_token) {
                    mtp_shadow_hits++;
                    fprintf(stderr, "[mtp] shadow hit token=%d\n", next_token);
                } else {
                    fprintf(stderr, "[mtp] shadow miss draft=%d actual=%d\n", mtp_pending_draft, next_token);
                }
                mtp_pending_draft = -1;
            }
            if (mtp_shadow_available) {
                int draft_token = -1;
                if (mtp_shadow_draft_token(wf, model_path, mtp_backbone_hidden, next_token, &draft_token) == 0) {
                    fprintf(stderr, "[mtp] shadow draft after token=%d -> %d\n", next_token, draft_token);
                    mtp_pending_draft = draft_token;
                }
            }

            // Think budget: force end thinking if over budget
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = g_cfg.think_end_token;
                in_think = 0;
            }
            total_generated++;

            // Print decoded token
            printf("%s", decode_token(vocab, next_token));
            fflush(stdout);

            double t_gen_end = now_ms();
            double tok_time = t_gen_end - t_gen_start;

            // Print progress to stderr
            if (g_timing_enabled) {
                fprintf(stderr, "  [gen %d/%d] token_id=%d (%.0f ms, %.2f tok/s)\n",
                        gen, max_tokens, next_token, tok_time, 1000.0 / tok_time);
            }
        }

        if (mtp_shadow_checks > 0) {
            double hit_rate = (100.0 * (double)mtp_shadow_hits) / (double)mtp_shadow_checks;
            fprintf(stderr, "[mtp] shadow summary hits=%d checks=%d rate=%.1f%%\n",
                    mtp_shadow_hits, mtp_shadow_checks, hit_rate);
        }
        mtp_overlap_report(mtp_shadow_hits, mtp_shadow_checks);

        if (g_timing_enabled) timing_print();
        printf("\n\n--- Statistics ---\n");
        double total_time = now_ms() - t0;
        printf("Total time:     %.1f s\n", total_time / 1000.0);
        if (g_timing_enabled) {
            printf("TTFT:           %.0f ms\n", ttft_ms);
        }
        printf("Tokens:         %d generated\n", total_generated);
        if (total_generated > 1) {
            double gen_time = total_time - ttft_ms;
            printf("Generation:     %.1f s (%.2f tok/s)\n",
                   gen_time / 1000.0, (total_generated - 1) * 1000.0 / gen_time);
        }
        printf("Config:         K=%d experts, %d layers\n", K, g_cfg.num_layers);
        if (g_expert_cache) {
            uint64_t total = g_expert_cache->hits + g_expert_cache->misses;
            printf("Expert cache:   %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_expert_cache->hits, g_expert_cache->misses,
                   total > 0 ? 100.0 * g_expert_cache->hits / total : 0.0,
                   g_expert_cache->num_entries, g_expert_cache->max_entries);
            cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
        } else if (g_malloc_cache) {
            uint64_t total = g_malloc_cache->hits + g_malloc_cache->misses;
            printf("Expert cache:   malloc %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_malloc_cache->hits, g_malloc_cache->misses,
                   total > 0 ? 100.0 * g_malloc_cache->hits / total : 0.0,
                   g_malloc_cache->num_entries, g_malloc_cache->max_entries);
            cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
        }

        if (g_spec_route_attempts > 0) {
            printf("Spec routing:   %llu attempts, %llu preloads, %llu hits (%.1f%% prediction accuracy)\n",
                   g_spec_route_attempts, g_spec_route_preloads, g_spec_route_hits,
                   g_spec_route_attempts > 0
                       ? 100.0 * g_spec_route_hits / g_spec_route_attempts : 0.0);
        }

        if (g_freq_tracking) freq_print_analysis(K);
        if (g_routing_log) {
            fclose(g_routing_log);
            fprintf(stderr, "[routing] Logged %d samples to routing data file\n",
                    g_routing_log_samples);
            g_routing_log = NULL;
        }

        // ---- Cleanup ----
        io_pool_shutdown();
        if (g_malloc_cache) {
            malloc_cache_free(g_malloc_cache);
            g_malloc_cache = NULL;
        }
        if (g_expert_cache) {
            expert_cache_free(g_expert_cache);
            g_expert_cache = NULL;
        }
        for (int i = 0; i < g_cfg.num_layers; i++) {
            if (kv_caches[i]) kv_cache_free(kv_caches[i]);
            if (layer_states[i]) linear_attn_state_free(layer_states[i]);
            if (layer_mmaps[i] != MAP_FAILED) munmap(layer_mmaps[i], layer_mmap_sizes[i]);
            if (layer_fds[i] >= 0) close(layer_fds[i]);
            if (layer_fds_cold[i] >= 0) close(layer_fds_cold[i]);
        }
        free(layer_states);
        free(kv_caches);
        free(hidden);
        free(logits);

        return 0;
    }
}
#endif // CHAT_MODE
