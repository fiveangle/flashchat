/*
 * test_lzfse.c — Measure LZFSE compression ratio and decompression throughput
 * on 4-bit quantized expert weight data.
 *
 * Tests the hypothesis: if 4-bit weights only carry ~1.5 bits of entropy,
 * LZFSE compresses them aggressively, giving low-bit I/O savings with
 * 4-bit quality. Apple Silicon has HW-accelerated LZFSE decompression.
 *
 * Build: clang -O2 -o test_lzfse test_lzfse.c -lcompression
 * Usage: ./test_lzfse [path_to_layer_bin] [num_experts]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <compression.h>

#define EXPERT_SIZE 7077888  /* 4-bit expert size in bytes */

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Compute Shannon entropy of byte values */
static double byte_entropy(const uint8_t *data, size_t len) {
    size_t freq[256] = {0};
    for (size_t i = 0; i < len; i++) freq[data[i]]++;
    double entropy = 0;
    for (int i = 0; i < 256; i++) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / len;
        entropy -= p * log2(p);
    }
    return entropy;
}

/* Compute entropy of 4-bit nibbles */
static double nibble_entropy(const uint8_t *data, size_t len) {
    size_t freq[16] = {0};
    for (size_t i = 0; i < len; i++) {
        freq[data[i] & 0x0F]++;
        freq[(data[i] >> 4) & 0x0F]++;
    }
    double entropy = 0;
    size_t total = len * 2;
    for (int i = 0; i < 16; i++) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / total;
        entropy -= p * log2(p);
    }
    return entropy;
}

static void print_nibble_histogram(const uint8_t *data, size_t len) {
    size_t freq[16] = {0};
    for (size_t i = 0; i < len; i++) {
        freq[data[i] & 0x0F]++;
        freq[(data[i] >> 4) & 0x0F]++;
    }
    size_t total = len * 2;
    printf("  Nibble distribution:\n");
    for (int i = 0; i < 16; i++) {
        double pct = (double)freq[i] / total * 100.0;
        int bar = (int)(pct * 2);  /* 2 chars per percent */
        printf("    %2d: %5.1f%% ", i, pct);
        for (int b = 0; b < bar && b < 60; b++) printf("#");
        printf("\n");
    }
}

int main(int argc, char **argv) {
    const char *layer_path = "packed_experts/layer_00.bin";
    int num_experts = 4;

    if (argc > 1) layer_path = argv[1];
    if (argc > 2) num_experts = atoi(argv[2]);

    printf("=== LZFSE Compression Test for 4-bit Expert Weights ===\n");
    printf("File: %s\n", layer_path);
    printf("Expert size: %.2f MB (%d bytes)\n",
           EXPERT_SIZE / (1024.0 * 1024.0), EXPERT_SIZE);
    printf("Testing %d experts\n\n", num_experts);

    int fd = open(layer_path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    uint8_t *raw = malloc(EXPERT_SIZE);
    /* Worst case: compressed might be larger */
    size_t comp_buf_size = EXPERT_SIZE + 4096;
    uint8_t *compressed = malloc(comp_buf_size);
    uint8_t *decompressed = malloc(EXPERT_SIZE);

    if (!raw || !compressed || !decompressed) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Also test different algorithms */
    typedef struct {
        const char *name;
        compression_algorithm algo;
    } AlgoInfo;

    AlgoInfo algos[] = {
        {"LZFSE",  COMPRESSION_LZFSE},
        {"LZ4",    COMPRESSION_LZ4},
        {"ZLIB",   COMPRESSION_ZLIB},
        {"LZMA",   COMPRESSION_LZMA},
    };
    int num_algos = sizeof(algos) / sizeof(algos[0]);

    double total_raw = 0, total_comp[4] = {0}, total_decomp_ms[4] = {0};
    int valid_experts = 0;

    for (int e = 0; e < num_experts; e++) {
        off_t offset = (off_t)e * EXPERT_SIZE;
        ssize_t nread = pread(fd, raw, EXPERT_SIZE, offset);
        if (nread != EXPERT_SIZE) {
            fprintf(stderr, "Short read for expert %d: %zd\n", e, nread);
            continue;
        }
        valid_experts++;

        /* Entropy analysis (first expert only for detail) */
        if (e == 0) {
            printf("--- Expert 0 Entropy Analysis ---\n");
            printf("  Byte entropy:   %.3f bits/byte (max 8.0)\n",
                   byte_entropy(raw, EXPERT_SIZE));
            printf("  Nibble entropy: %.3f bits/nibble (max 4.0)\n",
                   nibble_entropy(raw, EXPERT_SIZE));
            printf("  Theoretical minimum: %.2f MB (at nibble entropy)\n",
                   nibble_entropy(raw, EXPERT_SIZE) * EXPERT_SIZE * 2 / 8.0 / (1024*1024));

            /* Show distribution for weight data only (skip scales/biases) */
            /* gate_w is the first 2097152 bytes */
            printf("\n  gate_proj weights (first 2MB):\n");
            printf("    Nibble entropy: %.3f bits/nibble\n",
                   nibble_entropy(raw, 2097152));
            print_nibble_histogram(raw, 2097152);
        }

        /* Test each compression algorithm */
        total_raw += EXPERT_SIZE;

        for (int a = 0; a < num_algos; a++) {
            /* Compress */
            size_t comp_size = compression_encode_buffer(
                compressed, comp_buf_size, raw, EXPERT_SIZE,
                NULL, algos[a].algo);

            if (comp_size == 0) {
                if (e == 0) printf("  %s: compression failed\n", algos[a].name);
                continue;
            }

            total_comp[a] += comp_size;

            /* Decompress (measure throughput) */
            int decomp_iters = (a == 3) ? 1 : 5;  /* LZMA is slow, fewer iters */
            double t0 = now_ms();
            for (int iter = 0; iter < decomp_iters; iter++) {
                size_t dec_size = compression_decode_buffer(
                    decompressed, EXPERT_SIZE, compressed, comp_size,
                    NULL, algos[a].algo);
                if (dec_size != EXPERT_SIZE) {
                    fprintf(stderr, "Decompress mismatch: %zu != %d\n",
                            dec_size, EXPERT_SIZE);
                    break;
                }
            }
            double elapsed = (now_ms() - t0) / decomp_iters;
            total_decomp_ms[a] += elapsed;

            /* Verify first expert */
            if (e == 0) {
                int ok = (memcmp(raw, decompressed, EXPERT_SIZE) == 0);
                if (!ok) {
                    fprintf(stderr, "  %s: VERIFICATION FAILED!\n", algos[a].name);
                }
            }
        }

        if (e == 0 || e == num_experts - 1) {
            printf("\n--- Expert %d Compression ---\n", e);
            for (int a = 0; a < num_algos; a++) {
                if (total_comp[a] == 0 && e == 0) continue;
                double ratio = (e == 0)
                    ? total_comp[a] / EXPERT_SIZE
                    : total_comp[a] / total_raw;
                double decomp_ms = (e == 0)
                    ? total_decomp_ms[a]
                    : total_decomp_ms[a] / valid_experts;
                double decomp_gbps = (EXPERT_SIZE / (1024.0*1024*1024)) / (decomp_ms / 1000.0);
                printf("  %-6s: %.2f MB → %.2f MB (%.1f%% of original, %.2f bits/weight)\n",
                       algos[a].name,
                       EXPERT_SIZE / (1024.0*1024),
                       (e == 0 ? total_comp[a] : total_comp[a]/valid_experts) / (1024.0*1024),
                       ratio * 100,
                       ratio * 4.0);  /* 4 bits/weight * compression ratio */
                printf("         Decompress: %.2f ms (%.1f GB/s)\n",
                       decomp_ms, decomp_gbps);
            }
        }
    }

    /* Summary */
    printf("\n=== Summary (avg of %d experts) ===\n", valid_experts);
    printf("Raw size: %.2f MB per expert\n", EXPERT_SIZE / (1024.0 * 1024));

    double current_pread_ms = 2.4 / 4.0;  /* ~0.6ms per expert at K=4 */

    for (int a = 0; a < num_algos; a++) {
        if (total_comp[a] == 0) continue;
        double avg_comp = total_comp[a] / valid_experts;
        double avg_decomp = total_decomp_ms[a] / valid_experts;
        double ratio = avg_comp / EXPERT_SIZE;
        double bits = ratio * 4.0;
        double comp_read_ms = current_pread_ms * ratio;
        double total_ms = comp_read_ms + avg_decomp;

        printf("\n  %-6s: %.2f MB compressed (%.1f%%, %.2f bits/weight)\n",
               algos[a].name, avg_comp / (1024*1024), ratio * 100, bits);
        printf("         Decompress: %.2f ms (%.1f GB/s)\n",
               avg_decomp, (EXPERT_SIZE / (1024.0*1024*1024)) / (avg_decomp / 1000.0));
        printf("         Read compressed + decompress: %.2f ms vs %.2f ms raw read\n",
               total_ms, current_pread_ms);
        printf("         %s (%.1f%%)\n",
               total_ms < current_pread_ms ? ">>> FASTER <<<" : "slower",
               (1.0 - total_ms / current_pread_ms) * 100);

        /* K=4 estimate */
        double k4_raw = 2.4;  /* current expert_io for K=4 */
        double k4_comp = total_ms * 4;  /* 4 experts compressed+decompressed */
        printf("         K=4 estimate: %.1f ms vs %.1f ms → %.1f tok/s vs %.1f tok/s\n",
               k4_comp, k4_raw,
               1000.0 / ((4.0 - k4_raw + k4_comp) * 60),
               1000.0 / (4.0 * 60));
    }

    close(fd);
    free(raw);
    free(compressed);
    free(decompressed);
    return 0;
}
