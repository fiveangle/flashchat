#include "../metal_infer/server_http.h"
#include <netinet/in.h>

static volatile sig_atomic_t stopping;
static void stop(int sig) { (void)sig; stopping = 1; }
static void status(char *out, size_t size) {
    snprintf(out, size, "{\"status\":\"ok\",\"context_used\":1234}");
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGTERM, stop);
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    const char *address = argc > 1 ? argv[1] : "127.0.0.1";
    struct sockaddr_in addr = {0};
    if (server_http_bind(listener, address, 0) || listen(listener, 16)) return 1;
    socklen_t size = sizeof(addr);
    getsockname(listener, (void *)&addr, &size);
    if (argc > 1) {
        char actual[INET_ADDRSTRLEN];
        puts(inet_ntop(AF_INET, &addr.sin_addr, actual, sizeof(actual)));
        close(listener);
        return 0;
    }
    server_http_t server = {.listener = listener, .shutdown = &stopping,
        .status_json = status, .model_id = "fixture"};
    if (server_http_start(&server)) return 2;
    printf("%d\n", ntohs(addr.sin_port));
    fflush(stdout);
    while (!stopping) {
        char *request;
        int fd = server_http_take(&server, &request);
        if (fd < 0) break;
        const char *head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
        write(fd, head, strlen(head));
        if (strstr(request, "quiet") || strstr(request, "prefill")) {
            // Simulate long compute with no token output. Prefill has heartbeats;
            // quiet represents a buffered response interrupted with TCP reset.
            for (int i = 0; i < 200; i++) {
                if (server_http_cancelled(fd)) break;
                usleep(50000);
                if (strstr(request, "prefill") && i % 4 == 0)
                    if (write(fd, ": keepalive\n\n", 13) <= 0) break;
            }
        } else if (strstr(request, "bulk")) {
            char buffer[16384];
            memset(buffer, 'x', sizeof(buffer));
            for (int i = 0; i < 4096; i++) {
                if (write(fd, buffer, sizeof(buffer)) <= 0) break;
            }
        } else {
            for (int i = 0; i < 15; i++) {
                usleep(100000);
                if (write(fd, "data: token\n\n", 13) <= 0) break;
            }
        }
        free(request);
        close(fd);
    }
    server_http_join(&server);
    close(listener);
    return 0;
}
