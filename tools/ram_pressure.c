#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int signo) {
    (void)signo;
    g_stop = 1;
}

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s GIB [touch_ms]\n"
            "\n"
            "Allocates GIB GiB of anonymous memory, touches every page, then\n"
            "keeps dirtying pages until interrupted. Use Ctrl-C to release it.\n"
            "\n"
            "Examples:\n"
            "  %s 8\n"
            "  %s 16 500\n",
            argv0, argv0, argv0);
}

static int parse_double(const char *s, double *out) {
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || !end || *end || v <= 0.0) return -1;
    *out = v;
    return 0;
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v <= 0 || v > 60000) return -1;
    *out = (int)v;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        usage(argv[0]);
        return 2;
    }

    double gib = 0.0;
    if (parse_double(argv[1], &gib) != 0 || gib > 1024.0) {
        usage(argv[0]);
        return 2;
    }

    int touch_ms = 1000;
    if (argc == 3 && parse_int(argv[2], &touch_ms) != 0) {
        usage(argv[0]);
        return 2;
    }

    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t bytes = (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
    bytes = (bytes / page) * page;
    if (bytes == 0) {
        fprintf(stderr, "requested size rounds to zero\n");
        return 2;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    uint8_t *mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mem == MAP_FAILED) {
        fprintf(stderr, "mmap %.2f GiB failed: %s\n", gib, strerror(errno));
        return 1;
    }

    size_t pages = bytes / page;
    printf("ram_pressure: reserving %.2f GiB (%zu pages, page=%zu bytes)\n",
           (double)bytes / 1024.0 / 1024.0 / 1024.0, pages, page);
    fflush(stdout);

    double start = now_s();
    for (size_t i = 0; i < pages; i++) {
        mem[i * page] = (uint8_t)i;
        if ((i & 0x3fff) == 0 && i > 0) {
            printf("\rinitial touch: %.1f%%", 100.0 * (double)i / (double)pages);
            fflush(stdout);
        }
        if (g_stop) break;
    }
    printf("\rinitial touch: 100.0%% in %.1fs\n", now_s() - start);
    fflush(stdout);

    unsigned pass = 0;
    while (!g_stop) {
        pass++;
        for (size_t i = 0; i < pages; i++) {
            mem[i * page] = (uint8_t)(mem[i * page] + 1);
            if (g_stop) break;
        }
        printf("dirty pass %u complete; holding %.2f GiB. Ctrl-C to release.\n",
               pass, (double)bytes / 1024.0 / 1024.0 / 1024.0);
        fflush(stdout);
        usleep((useconds_t)touch_ms * 1000);
    }

    printf("releasing memory\n");
    munmap(mem, bytes);
    return 0;
}
