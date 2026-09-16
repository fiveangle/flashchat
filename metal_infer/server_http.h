// HTTP transport with one inference owner. The event thread owns network
// sockets; the inference thread receives a request and a private output pipe.
#ifndef FLASHCHAT_SERVER_HTTP_H
#define FLASHCHAT_SERVER_HTTP_H
#include <pthread.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <time.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>

static int server_http_bind(int fd, const char *address, unsigned short port) {
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(port)};
    if (inet_pton(AF_INET, address, &addr.sin_addr) != 1) {
        errno = EINVAL;
        return -1;
    }
    return bind(fd, (struct sockaddr *)&addr, sizeof(addr));
}

#define HTTP_CLIENTS 16
#define HTTP_REQUEST_LIMIT (1024 * 1024)
#define HTTP_OUTPUT_LIMIT 65536

typedef struct {
    int fd, output;
    char *data;
    size_t len, sent;
    double deadline;
    void (*trace)(const char *, const char *);
} http_client_t;

typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    int listener, stopped, busy, pending_fd, worker_started, worker_done;
    char *pending_request;
    volatile sig_atomic_t *shutdown;
    // Called only on the event thread; copies published state, never model buffers.
    void (*status_json)(char *, size_t);
    const char *model_id;
    void (*trace)(const char *, const char *);
} server_http_t;

static double http_clock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void http_nonblocking(int fd) {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
}

// The event thread closes the private output socket when the client is gone.
// Check without writing so hidden reasoning/tool output and prefill can stop too.
static int server_http_cancelled(int fd) {
    if (fd < 0) return 0;
    struct pollfd p = {fd, POLLIN, 0};
    if (poll(&p, 1, 0) <= 0) return 0;
    if (p.revents & (POLLHUP | POLLERR | POLLNVAL)) return 1;
    char unused;
    return (p.revents & POLLIN) && recv(fd, &unused, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
}

static void http_client_close(http_client_t *c) {
    if (c->fd >= 0) close(c->fd);
    free(c->data);
    memset(c, 0, sizeof(*c));
    c->fd = -1;
}

static void http_reply(http_client_t *c, int code, const char *body) {
    free(c->data);
    c->data = malloc(strlen(body) + 512);
    if (!c->data) { http_client_close(c); return; }
    c->len = snprintf(c->data, strlen(body) + 512,
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\nContent-Length: %zu\r\n\r\n%s",
        code, code == 200 ? "OK" : code == 204 ? "No Content" :
        code == 503 ? "Service Unavailable" : code == 404 ? "Not Found" : "Bad Request", strlen(body), body);
    c->sent = 0;
    c->output = 1;
    c->deadline = http_clock() + 5;
    if (c->trace) c->trace("response", c->data);
}

// Return the complete message length, 0 for incomplete, -1 for invalid framing.
// Transfer-Encoding and ambiguous lengths are rejected, never guessed.
static int http_request_length(char *data, size_t len) {
    char *end = strstr(data, "\r\n\r\n");
    if (!end) return len >= 16384 ? -1 : 0;
    size_t header = end + 4 - data, body = 0;
    int has_length = 0;
    char *line = strstr(data, "\r\n");
    if (!line) return -1;
    while ((line += 2) < end) {
        char *next = strstr(line, "\r\n");
        if (!next) return -1;
        if (!strncasecmp(line, "Transfer-Encoding:", 18)) return -1;
        if (!strncasecmp(line, "Content-Length:", 15)) {
            if (has_length++) return -1;
            char *p = line + 15;
            while (*p == ' ' || *p == '\t') p++;
            if (*p < '0' || *p > '9') return -1;
            for (; p < next && *p >= '0' && *p <= '9'; p++) {
                body = body * 10 + (*p - '0');
                if (body >= HTTP_REQUEST_LIMIT) return -1;
            }
            while (p < next && (*p == ' ' || *p == '\t')) p++;
            if (p != next) return -1;
        }
        line = next;
    }
    if (header + body >= HTTP_REQUEST_LIMIT) return -1;
    return len >= header + body ? (int)(header + body) : 0;
}

static void *http_event_loop(void *arg) {
    server_http_t *s = arg;
    http_client_t clients[HTTP_CLIENTS];
    memset(clients, 0, sizeof(clients));
    for (int i = 0; i < HTTP_CLIENTS; i++) clients[i].fd = -1;
    int active = -1, pipe_fd = -1, pipe_eof = 0, read_closed = 0;
    char output[HTTP_OUTPUT_LIMIT];
    size_t buffered = 0;
    double output_deadline = 0;
    http_nonblocking(s->listener);
    while (!*s->shutdown) {
        struct pollfd fds[HTTP_CLIENTS + 3];
        fds[0] = (struct pollfd){s->listener, POLLIN, 0};
        for (int i = 0; i < HTTP_CLIENTS; i++)
            fds[i + 1] = (struct pollfd){clients[i].fd, clients[i].output ? POLLOUT : POLLIN, 0};
        fds[HTTP_CLIENTS + 1] = (struct pollfd){active,
            (read_closed ? 0 : POLLIN) | (buffered ? POLLOUT : 0), 0};
        fds[HTTP_CLIENTS + 2] = (struct pollfd){pipe_eof ? -1 : pipe_fd,
            !pipe_eof && buffered < sizeof(output) ? POLLIN : 0, 0};
        if (poll(fds, HTTP_CLIENTS + 3, 100) < 0 && errno != EINTR) break;
        if (*s->shutdown) break;
        double now = http_clock();
        if (fds[0].revents & POLLIN) {
            int fd = accept(s->listener, NULL, NULL);
            if (fd >= 0) {
                http_nonblocking(fd);
                int slot = 0;
                while (slot < HTTP_CLIENTS && clients[slot].fd >= 0) slot++;
                if (slot == HTTP_CLIENTS) close(fd);
                else {
                    http_client_t *c = &clients[slot];
                    c->fd = fd;
                    c->trace = s->trace;
                    c->data = malloc(HTTP_REQUEST_LIMIT);
                    c->deadline = now + 10;
                    if (!c->data) http_client_close(c);
                }
            }
        }
        for (int i = 0; i < HTTP_CLIENTS; i++) {
            http_client_t *c = &clients[i];
            if (c->fd < 0) continue;
            short ev = fds[i + 1].revents;
            if (now > c->deadline || (ev & (POLLERR | POLLNVAL))) {
                http_client_close(c); continue;
            }
            if (c->output) {
                if (!(ev & POLLOUT)) continue;
                ssize_t n = write(c->fd, c->data + c->sent, c->len - c->sent);
                if (n > 0) c->sent += n;
                if (c->sent == c->len || (n < 0 && errno != EAGAIN && errno != EINTR))
                    http_client_close(c);
                continue;
            }
            if (!(ev & (POLLIN | POLLHUP))) continue;
            ssize_t n = read(c->fd, c->data + c->len, HTTP_REQUEST_LIMIT - 1 - c->len);
            if (n <= 0) {
                if (!n || (errno != EAGAIN && errno != EINTR)) http_client_close(c);
                continue;
            }
            if (memchr(c->data + c->len, 0, n)) {
                http_reply(c, 400, "{\"error\":{\"message\":\"NUL in request\"}}"); continue;
            }
            c->len += n;
            c->data[c->len] = 0;
            int length = http_request_length(c->data, c->len);
            if (length < 0 || c->len == HTTP_REQUEST_LIMIT - 1) {
                http_reply(c, 400, "{\"error\":{\"message\":\"Invalid or oversized HTTP request\"}}"); continue;
            }
            if (!length) continue;
            c->data[length] = 0;
            if (c->trace) c->trace("request", c->data);
            char method[16], path[256];
            if (sscanf(c->data, "%15s %255s", method, path) != 2) {
                http_reply(c, 400, "{\"error\":{\"message\":\"Invalid request line\"}}"); continue;
            }
            char body[2048];
            if (!strcmp(method, "OPTIONS")) http_reply(c, 204, "");
            else if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
                s->status_json(body, sizeof(body));
                http_reply(c, 200, body);
            } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
                snprintf(body, sizeof(body), "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"local\"}]}", s->model_id);
                http_reply(c, 200, body);
            } else if (!strcmp(method, "GET") && !strcmp(path, "/v1")) {
                http_reply(c, 200, "{\"object\":\"service\",\"id\":\"flashchat\",\"api\":\"openai-compatible\",\"endpoints\":[\"/v1/chat/completions\",\"/v1/responses\",\"/v1/models\",\"/health\"]}");
            } else if (strcmp(method, "POST") || (strcmp(path, "/v1/chat/completions") && strcmp(path, "/v1/responses"))) {
                http_reply(c, 404, "{\"error\":{\"message\":\"Not found\"}}");
            } else if (s->busy) {
                http_reply(c, 503, "{\"error\":{\"type\":\"server_busy\",\"message\":\"Flashchat is processing another request. Try again when it finishes.\"}}");
            } else {
                int pair[2];
                if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) {
                    http_reply(c, 503, "{\"error\":{\"message\":\"Unable to accept generation\"}}"); continue;
                }
                http_nonblocking(pair[0]);
                active = c->fd;
                pipe_fd = pair[0];
                pipe_eof = 0;
                read_closed = 0;
                buffered = 0;
                s->busy = 1;
                pthread_mutex_lock(&s->mutex);
                s->worker_done = 0;
                s->pending_fd = pair[1];
                s->pending_request = c->data;
                pthread_cond_signal(&s->ready);
                pthread_mutex_unlock(&s->mutex);
                memset(c, 0, sizeof(*c));
                c->fd = -1;
            }
        }
        if (pipe_fd >= 0) {
            if (active >= 0 && !read_closed && (fds[HTTP_CLIENTS + 1].revents & (POLLIN | POLLHUP))) {
                char extra;
                ssize_t n = recv(active, &extra, 1, 0);
                // A client may half-close its request side and still read output.
                if (n == 0) read_closed = 1;
                if (n < 0 && errno != EAGAIN && errno != EINTR) {
                    close(active); active = -1;
                }
            }
            if (buffered < sizeof(output) && (fds[HTTP_CLIENTS + 2].revents & (POLLIN | POLLHUP))) {
                ssize_t n = read(pipe_fd, output + buffered, sizeof(output) - buffered);
                if (n > 0) {
                    if (!buffered) output_deadline = now + 10;
                    buffered += n;
                } else if (!n || (errno != EAGAIN && errno != EINTR)) pipe_eof = 1;
            }
            if (active >= 0 && buffered && (fds[HTTP_CLIENTS + 1].revents & (POLLOUT | POLLHUP))) {
                ssize_t n = write(active, output, buffered);
                if (n > 0) {
                    buffered -= n;
                    memmove(output, output + n, buffered);
                    output_deadline = now + 10;
                } else if (n < 0 && errno != EAGAIN && errno != EINTR) {
                    close(active); active = -1;
                }
            }
            if (active >= 0 && ((fds[HTTP_CLIENTS + 1].revents & (POLLERR | POLLNVAL)) ||
                               (buffered && now > output_deadline))) {
                close(active); active = -1;
            }
            if (active < 0) {
                // Wake failed writes on the inference side, but retain ownership
                // until that worker closes its pipe; no concurrent generation.
                close(pipe_fd);
                pipe_fd = -1;
                pipe_eof = 1;
                buffered = 0;
            }
        }
        if (s->busy) {
            pthread_mutex_lock(&s->mutex);
            int done = s->worker_done;
            pthread_mutex_unlock(&s->mutex);
            if (done && pipe_eof && !buffered) {
                if (active >= 0) close(active);
                if (pipe_fd >= 0) close(pipe_fd);
                active = pipe_fd = -1;
                s->busy = 0;
            }
        }
    }
    for (int i = 0; i < HTTP_CLIENTS; i++) http_client_close(&clients[i]);
    if (active >= 0) close(active);
    if (pipe_fd >= 0) close(pipe_fd);
    pthread_mutex_lock(&s->mutex);
    s->stopped = 1;
    pthread_cond_broadcast(&s->ready);
    pthread_mutex_unlock(&s->mutex);
    return NULL;
}

static int server_http_start(server_http_t *s) {
    int rc = pthread_mutex_init(&s->mutex, NULL);
    if (rc) return rc;
    rc = pthread_cond_init(&s->ready, NULL);
    if (rc) { pthread_mutex_destroy(&s->mutex); return rc; }
    s->pending_fd = -1;
    rc = pthread_create(&s->thread, NULL, http_event_loop, s);
    if (rc) {
        pthread_cond_destroy(&s->ready);
        pthread_mutex_destroy(&s->mutex);
    }
    return rc;
}

static int server_http_take(server_http_t *s, char **request) {
    pthread_mutex_lock(&s->mutex);
    if (s->worker_started) s->worker_done = 1;
    while (!s->pending_request && !s->stopped) pthread_cond_wait(&s->ready, &s->mutex);
    if (s->stopped) {
        *request = NULL;
        pthread_mutex_unlock(&s->mutex);
        return -1;
    }
    int fd = s->pending_fd;
    *request = s->pending_request;
    s->pending_request = NULL;
    s->pending_fd = -1;
    s->worker_started = fd >= 0;
    pthread_mutex_unlock(&s->mutex);
    return fd;
}

static void server_http_join(server_http_t *s) {
    pthread_join(s->thread, NULL);
    if (s->pending_fd >= 0) close(s->pending_fd);
    free(s->pending_request);
    pthread_cond_destroy(&s->ready);
    pthread_mutex_destroy(&s->mutex);
}
#endif
