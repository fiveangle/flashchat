/*
 * repack_experts_lz4.c — Repack 4-bit expert files with LZ4 compression.
 *
 * Reads packed_experts/layer_XX.bin (512 experts × 7MB raw each),
 * writes packed_experts_lz4/layer_XX.bin with index header + LZ4 blobs.
 *
 * File format:
 *   [LZ4IndexEntry × 512]  (header: 8KB)
 *   [compressed expert 0]
 *   [compressed expert 1]
 *   ...
 *   [compressed expert 511]
 *
 * LZ4IndexEntry: { uint64_t offset; uint32_t comp_size; uint32_t raw_size; }
 *
 * Build: clang -O2 -o repack_experts_lz4 repack_experts_lz4.c -lcompression
 * Usage: ./repack_experts_lz4 <model_path>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <compression.h>

#define EXPERT_SIZE     7077888
#define NUM_EXPERTS     512
#define NUM_LAYERS      60

typedef struct {
    uint64_t offset;
    uint32_t comp_size;
    uint32_t raw_size;
} LZ4IndexEntry;

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
    fprintf(stderr, "  Reads:  <model_path>/flashchat/packed_experts/layer_XX.bin\n");
    fprintf(stderr, "  Writes: <model_path>/flashchat/packed_experts_lz4/layer_XX.bin\n");
    return 1;
}

    const char *model_path = argv[1];
    char src_dir[1024], dst_dir[1024];
    snprintf(src_dir, sizeof(src_dir), "%s/flashchat/packed_experts", model_path);
    snprintf(dst_dir, sizeof(dst_dir), "%s/flashchat/packed_experts_lz4", model_path);

    /* Create output directory */
    mkdir(dst_dir, 0755);

    void *raw_buf = malloc(EXPERT_SIZE);
    void *comp_buf = malloc(EXPERT_SIZE + 4096);
    if (!raw_buf || !comp_buf) { fprintf(stderr, "malloc failed\n"); return 1; }

    size_t grand_raw = 0, grand_comp = 0;
    double t_start = now_ms();

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        char src_path[1024], dst_path[1024];
        snprintf(src_path, sizeof(src_path), "%s/layer_%02d.bin", src_dir, layer);
        snprintf(dst_path, sizeof(dst_path), "%s/layer_%02d.bin", dst_dir, layer);

        int fd_in = open(src_path, O_RDONLY);
        if (fd_in < 0) {
            fprintf(stderr, "SKIP layer %d: %s not found\n", layer, src_path);
            continue;
        }

        int fd_out = open(dst_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd_out < 0) {
            perror("open output");
            close(fd_in);
            continue;
        }

        LZ4IndexEntry index[NUM_EXPERTS];
        memset(index, 0, sizeof(index));

        /* Write placeholder index */
        write(fd_out, index, sizeof(index));
        uint64_t data_offset = sizeof(index);
        size_t layer_raw = 0, layer_comp = 0;

        double t_layer = now_ms();

        for (int e = 0; e < NUM_EXPERTS; e++) {
            ssize_t nr = pread(fd_in, raw_buf, EXPERT_SIZE, (off_t)e * EXPERT_SIZE);
            if (nr != EXPERT_SIZE) {
                fprintf(stderr, "Short read layer %d expert %d: %zd\n", layer, e, nr);
                index[e].offset = 0;
                index[e].comp_size = 0;
                index[e].raw_size = 0;
                continue;
            }

            size_t cs = compression_encode_buffer(comp_buf, EXPERT_SIZE + 4096,
                                                  raw_buf, EXPERT_SIZE,
                                                  NULL, COMPRESSION_LZ4);
            if (cs == 0 || cs >= (size_t)EXPERT_SIZE) {
                /* Compression failed or didn't help — store raw */
                index[e].offset = data_offset;
                index[e].comp_size = EXPERT_SIZE;
                index[e].raw_size = EXPERT_SIZE;
                write(fd_out, raw_buf, EXPERT_SIZE);
                data_offset += EXPERT_SIZE;
                layer_comp += EXPERT_SIZE;
            } else {
                index[e].offset = data_offset;
                index[e].comp_size = (uint32_t)cs;
                index[e].raw_size = EXPERT_SIZE;
                write(fd_out, comp_buf, cs);
                data_offset += cs;
                layer_comp += cs;
            }
            layer_raw += EXPERT_SIZE;
        }

        /* Rewrite index */
        lseek(fd_out, 0, SEEK_SET);
        write(fd_out, index, sizeof(index));

        close(fd_in);
        close(fd_out);

        double layer_ms = now_ms() - t_layer;
        double ratio = (double)layer_comp / layer_raw;
        grand_raw += layer_raw;
        grand_comp += layer_comp;

        printf("Layer %2d: %4zu MB → %4zu MB (%.1f%%, %.2f bits/w) [%.1f s]\n",
               layer, layer_raw >> 20, layer_comp >> 20,
               ratio * 100, ratio * 4.0, layer_ms / 1000);
    }

    double total_s = (now_ms() - t_start) / 1000;
    double ratio = (double)grand_comp / grand_raw;
    printf("\n=== Done ===\n");
    printf("Total: %zu MB → %zu MB (%.1f%%, %.2f bits/weight)\n",
           grand_raw >> 20, grand_comp >> 20, ratio * 100, ratio * 4.0);
    printf("Time: %.1f s (%.1f MB/s)\n", total_s, (grand_raw >> 20) / total_s);
    printf("Output: %s/\n", dst_dir);

    free(raw_buf);
    free(comp_buf);
    return 0;
}
