#include "../metal_infer/server_http.h"
#include <netinet/in.h>

static volatile sig_atomic_t stopping;
static void stop(int sig) { (void)sig; stopping = 1; }
static void status(char *out, size_t size) {
    snprintf(out, size, "{\"status\":\"ok\",\"context_used\":1234}");
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGTERM, stop);
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    if (bind(listener, (void *)&addr, sizeof(addr)) || listen(listener, 16)) return 1;
    socklen_t size = sizeof(addr);
    getsockname(listener, (void *)&addr, &size);
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
        if (strstr(request, "bulk")) {
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
