/*
 * chat.m — Interactive TUI chat client for Flash-MoE inference server
 *
 * Thin wrapper that connects to ./infer --serve on localhost.
 * No model loading — just readline + HTTP + SSE streaming.
 *
 * Build:  make chat
 * Run:    ./chat [--port 8000] [--no-think]
 *
 * Requires: ./infer --serve 8000 running separately
 */

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <getopt.h>

#define MAX_INPUT_LINE 4096
#define MAX_RESPONSE (1024 * 1024)  // 1MB max response buffer

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// JSON-escape a string into buf. Returns bytes written.
static int json_escape(const char *src, char *buf, int bufsize) {
    int j = 0;
    for (int i = 0; src[i] && j < bufsize - 2; i++) {
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

// Connect to server, send POST, return socket fd (caller reads response)
static int send_chat_request(int port, const char *user_message, int max_tokens) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return -1; }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "\n[error] Cannot connect to server on port %d.\n", port);
        fprintf(stderr, "Start the server first: ./infer --serve %d\n\n", port);
        close(sock);
        return -1;
    }

    // Build JSON body
    char escaped[MAX_INPUT_LINE * 2];
    json_escape(user_message, escaped, sizeof(escaped));

    char body[MAX_INPUT_LINE * 3];
    int body_len = snprintf(body, sizeof(body),
        "{\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
        "\"max_tokens\":%d,\"stream\":true}",
        escaped, max_tokens);

    // Build HTTP request
    char request[MAX_INPUT_LINE * 4];
    int req_len = snprintf(request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        port, body_len, body);

    if (write(sock, request, req_len) != req_len) {
        perror("write");
        close(sock);
        return -1;
    }

    return sock;
}

// Read SSE stream from socket, print tokens as they arrive.
// Returns total tokens received.
static int stream_response(int sock, int show_thinking) {
    // Skip HTTP headers
    char buf[4096];
    int header_done = 0;
    int buf_pos = 0;
    int tokens = 0;
    int in_think = 0;
    double t_first = 0, t_start = now_ms();

    FILE *stream = fdopen(sock, "r");
    if (!stream) { close(sock); return 0; }

    char line[MAX_RESPONSE];
    while (fgets(line, sizeof(line), stream)) {
        // Skip until we pass the HTTP headers
        if (!header_done) {
            if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) {
                header_done = 1;
            }
            continue;
        }

        // Parse SSE: lines starting with "data: "
        if (strncmp(line, "data: ", 6) != 0) continue;
        char *data = line + 6;

        // Check for [DONE]
        if (strncmp(data, "[DONE]", 6) == 0) break;

        // Extract content from JSON: find "content":"..."
        char *content_key = strstr(data, "\"content\":\"");
        if (!content_key) continue;
        content_key += 11; // skip past "content":"

        // Find closing quote (handle escapes)
        char decoded[4096];
        int di = 0;
        for (int i = 0; content_key[i] && content_key[i] != '"' && di < 4095; i++) {
            if (content_key[i] == '\\' && content_key[i+1]) {
                i++;
                switch (content_key[i]) {
                    case 'n': decoded[di++] = '\n'; break;
                    case 't': decoded[di++] = '\t'; break;
                    case 'r': decoded[di++] = '\r'; break;
                    case '"': decoded[di++] = '"'; break;
                    case '\\': decoded[di++] = '\\'; break;
                    default: decoded[di++] = content_key[i]; break;
                }
            } else {
                decoded[di++] = content_key[i];
            }
        }
        decoded[di] = 0;

        if (strlen(decoded) == 0) continue;

        // Track thinking tokens
        if (strstr(decoded, "<think>")) { in_think = 1; }
        if (strstr(decoded, "</think>")) { in_think = 0; tokens++; continue; }

        tokens++;
        if (t_first == 0) t_first = now_ms();

        // Print (skip thinking unless requested)
        if (in_think && !show_thinking) continue;
        if (in_think) {
            // Dim thinking output
            printf("\033[2m%s\033[0m", decoded);
        } else {
            printf("%s", decoded);
        }
        fflush(stdout);
    }

    fclose(stream);

    double elapsed = now_ms() - t_start;
    double gen_time = t_first > 0 ? now_ms() - t_first : 0;
    int gen_tokens = tokens > 0 ? tokens - 1 : 0;

    printf("\n\n");
    if (gen_tokens > 0 && gen_time > 0) {
        printf("[%d tokens, %.1f tok/s, TTFT %.1fs]\n\n",
               tokens, gen_tokens * 1000.0 / gen_time,
               t_first > 0 ? (t_first - t_start) / 1000.0 : 0);
    }

    return tokens;
}

int main(int argc, char **argv) {
    int port = 8000;
    int max_tokens = 2048;
    int show_thinking = 0;

    static struct option long_options[] = {
        {"port",        required_argument, 0, 'p'},
        {"max-tokens",  required_argument, 0, 't'},
        {"show-think",  no_argument,       0, 's'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "p:t:sh", long_options, NULL)) != -1) {
        switch (c) {
            case 'p': port = atoi(optarg); break;
            case 't': max_tokens = atoi(optarg); break;
            case 's': show_thinking = 1; break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  --port N         Server port (default: 8000)\n");
                printf("  --max-tokens N   Max response tokens (default: 2048)\n");
                printf("  --show-think     Show <think> blocks (dimmed)\n");
                printf("  --help           This message\n");
                printf("\nRequires: ./infer --serve %d running\n", port);
                return 0;
            default: return 1;
        }
    }

    printf("==================================================\n");
    printf("  Qwen3.5-397B-A17B Chat (Flash-MoE Client)\n");
    printf("==================================================\n");
    printf("  Server:  http://localhost:%d\n", port);
    printf("  Tokens:  %d max per response\n", max_tokens);
    printf("  Think:   %s\n", show_thinking ? "visible (dimmed)" : "hidden");
    printf("\n  Commands: /quit /exit\n");
    printf("==================================================\n\n");

    // Check server health
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Server not running on port %d.\n", port);
        fprintf(stderr, "Start it: ./infer --serve %d\n\n", port);
        close(sock);
        return 1;
    }
    close(sock);
    printf("Connected to server. Ready to chat.\n\n");

    char input_line[MAX_INPUT_LINE];

    for (;;) {
        printf("> ");
        fflush(stdout);

        if (!fgets(input_line, sizeof(input_line), stdin)) {
            printf("\n");
            break;
        }

        // Strip trailing newline
        size_t len = strlen(input_line);
        while (len > 0 && (input_line[len-1] == '\n' || input_line[len-1] == '\r'))
            input_line[--len] = 0;

        if (len == 0) continue;
        if (strcmp(input_line, "/quit") == 0 || strcmp(input_line, "/exit") == 0) {
            printf("Goodbye.\n");
            break;
        }

        sock = send_chat_request(port, input_line, max_tokens);
        if (sock < 0) continue;

        printf("\n");
        stream_response(sock, show_thinking);
    }

    return 0;
}
