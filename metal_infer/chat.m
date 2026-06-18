/*
 * chat.m — Interactive TUI chat client for Flashchat inference server
 *
 * Thin HTTP/SSE client with session persistence.
 * Conversations saved to ~/.config/flashchat/sessions/<session_id>.jsonl
 * Resume with: ./chat --resume <session_id>
 *
 * Build:  make chat (from repo root)
 * Run:    ./chat [--port 8000] [--show-think] [--resume <id>]
 */

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <getopt.h>
#include <dirent.h>
#include "linenoise.h"

#define MAX_INPUT_LINE 4096
#define MAX_RESPONSE (1024 * 1024)
#define FLASHCHAT_CONFIG_BASE ".config/flashchat"

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int json_escape(const char *src, char *buf, int bufsize) {
    int j = 0;
    for (int i = 0; src[i] && j < bufsize - 6; i++) {
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

static int flag_enabled(const char *value) {
    if (!value || !value[0]) return 0;
    if (strcmp(value, "0") == 0) return 0;
    if (strcasecmp(value, "false") == 0) return 0;
    if (strcasecmp(value, "off") == 0) return 0;
    if (strcasecmp(value, "no") == 0) return 0;
    if (strcasecmp(value, "disabled") == 0) return 0;
    return 1;
}

// ============================================================================
// Session persistence
// ============================================================================

static char g_sessions_dir[1024];
static char g_history_path[1024];

static void init_app_state_paths(void) {
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";

    const char *sessions_env = getenv("FLASHCHAT_SESSIONS_DIR");
    if (sessions_env && sessions_env[0]) {
        snprintf(g_sessions_dir, sizeof(g_sessions_dir), "%s", sessions_env);
    } else {
        snprintf(g_sessions_dir, sizeof(g_sessions_dir), "%s/%s/sessions", home, FLASHCHAT_CONFIG_BASE);
    }

    const char *history_env = getenv("FLASHCHAT_HISTORY_FILE");
    if (history_env && history_env[0]) {
        snprintf(g_history_path, sizeof(g_history_path), "%s", history_env);
    } else {
        snprintf(g_history_path, sizeof(g_history_path), "%s/%s/history", home, FLASHCHAT_CONFIG_BASE);
    }
}

static void init_sessions_dir(void) {
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    init_app_state_paths();
    mkdir(home, 0755);
    char config_parent[1024];
    snprintf(config_parent, sizeof(config_parent), "%s/.config", home);
    mkdir(config_parent, 0755);
    char app_parent[1024];
    snprintf(app_parent, sizeof(app_parent), "%s/%s", home, FLASHCHAT_CONFIG_BASE);
    mkdir(app_parent, 0755);
    mkdir(g_sessions_dir, 0755);
}

static void session_path(const char *session_id, char *path, size_t pathsize) {
    snprintf(path, pathsize, "%s/%s.jsonl", g_sessions_dir, session_id);
}

// Append a turn to the session JSONL file
static void session_save_turn(const char *session_id, const char *role, const char *content) {
    char path[1024];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "a");
    if (!f) return;
    char escaped[MAX_RESPONSE * 2];
    json_escape(content, escaped, sizeof(escaped));
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    fclose(f);
}

// Load session history and replay to screen
static int session_load(const char *session_id) {
    char path[1024];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    printf("[resuming session %s]\n\n", session_id);
    int turns = 0;
    char line[MAX_RESPONSE];
    while (fgets(line, sizeof(line), f)) {
        // Simple parsing: find role and content
        char *role_start = strstr(line, "\"role\":\"");
        char *content_start = strstr(line, "\"content\":\"");
        if (!role_start || !content_start) continue;

        role_start += 8;
        char role[32]; int ri = 0;
        while (*role_start && *role_start != '"' && ri < 31) role[ri++] = *role_start++;
        role[ri] = 0;

        content_start += 11;
        // Decode the content (unescape)
        char content[MAX_RESPONSE]; int ci = 0;
        for (int i = 0; content_start[i] && ci < MAX_RESPONSE - 1; i++) {
            // Stop at closing quote (not escaped)
            if (content_start[i] == '"' && (i == 0 || content_start[i-1] != '\\')) break;
            if (content_start[i] == '\\' && content_start[i+1]) {
                i++;
                switch (content_start[i]) {
                    case 'n': content[ci++] = '\n'; break;
                    case 't': content[ci++] = '\t'; break;
                    case '"': content[ci++] = '"'; break;
                    case '\\': content[ci++] = '\\'; break;
                    default: content[ci++] = content_start[i]; break;
                }
            } else {
                content[ci++] = content_start[i];
            }
        }
        content[ci] = 0;

        if (strcmp(role, "user") == 0) {
            printf("\033[1m> %s\033[0m\n\n", content);
        } else if (strcmp(role, "assistant") == 0) {
            printf("%s\n\n", content);
        }
        turns++;
    }
    fclose(f);
    if (turns > 0) printf("[%d turns loaded]\n\n", turns);
    return turns;
}

// List recent sessions
static void session_list(void) {
    DIR *dir = opendir(g_sessions_dir);
    if (!dir) { printf("No sessions found.\n\n"); return; }

    printf("Recent sessions:\n");
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        char *dot = strrchr(entry->d_name, '.');
        if (!dot || strcmp(dot, ".jsonl") != 0) continue;
        *dot = 0; // strip .jsonl

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.jsonl", g_sessions_dir, entry->d_name);
        struct stat st;
        stat(path, &st);

        // Count lines (turns)
        FILE *f = fopen(path, "r");
        int lines = 0;
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) lines++;
            fclose(f);
        }

        printf("  %s  (%d turns)\n", entry->d_name, lines);
        count++;
    }
    closedir(dir);
    if (count == 0) printf("  (none)\n");
    printf("\n");
}

// ============================================================================
// HTTP / SSE
// ============================================================================

static void generate_session_id(char *buf, size_t bufsize) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    snprintf(buf, bufsize, "chat-%d-%ld%06d",
             (int)getpid(), (long)tv.tv_sec, (int)tv.tv_usec);
}

static int connect_to_server(const char *host, int port) {
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo *result = NULL;
    int rc = getaddrinfo(host, port_str, &hints, &result);
    if (rc != 0) {
        fprintf(stderr, "\n[error] Cannot resolve server host %s: %s\n", host, gai_strerror(rc));
        return -1;
    }

    int sock = -1;
    for (struct addrinfo *rp = result; rp != NULL; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock < 0) continue;
        if (connect(sock, rp->ai_addr, rp->ai_addrlen) == 0) {
            freeaddrinfo(result);
            return sock;
        }
        close(sock);
        sock = -1;
    }

    freeaddrinfo(result);
    fprintf(stderr, "\n[error] Cannot connect to server at %s:%d.\n", host, port);
    return -1;
}

static int send_chat_request(const char *host, int port, const char *user_message, int max_tokens,
                             const char *session_id, int reasoning_enabled) {
    int sock = connect_to_server(host, port);
    if (sock < 0) return -1;

    char escaped[MAX_INPUT_LINE * 2];
    json_escape(user_message, escaped, sizeof(escaped));

    char body[MAX_INPUT_LINE * 3];
    int body_len = snprintf(body, sizeof(body),
        "{\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
        "\"max_tokens\":%d,\"stream\":true,\"session_id\":\"%s\",\"reasoning\":%s}",
        escaped, max_tokens, session_id, reasoning_enabled ? "true" : "false");

    char request[MAX_INPUT_LINE * 4];
    int req_len = snprintf(request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        host, port, body_len, body);

    write(sock, request, req_len);
    return sock;
}

// ============================================================================
// Streaming markdown renderer — stateful ANSI escape code emitter
// ============================================================================
// Handles: **bold**, *italic*, `inline code`, ```code blocks```, # headers
// State persists across token boundaries (e.g. "**" in one token, text in next)

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_ITALIC  "\033[3m"
#define ANSI_CODE    "\033[36m"      // cyan for inline code
#define ANSI_CODEBLK "\033[48;5;236m\033[38;5;252m"  // dark bg + light fg
#define ANSI_CODEBLK_LINE "\033[48;5;236m\033[K"     // extend bg to end of line
#define ANSI_HEADER  "\033[1;34m"    // bold blue for headers
#define ANSI_DIM     "\033[2m"

typedef struct {
    int bold;        // inside **...**
    int italic;      // inside *...*
    int strike;      // inside ~~...~~
    int code_inline; // inside `...`
    int code_block;  // inside ```...```
    int skip_lang;   // eating language tag after opening ```
    int line_start;  // at start of a new line
    int quote;       // inside a > blockquote line
    int in_table;          // buffering a markdown table (rendered on flush)
    int table_line_start;  // at the start of a line while buffering a table
    int table_len;         // bytes used in table_buf
    char table_buf[8192];  // raw table rows, accumulated until the table ends
    char pending[64]; // bytes held for cross-chunk lookahead (split ``` ### --- **)
    int pending_len;
} MdState;

static MdState g_md = { .line_start = 1 };

static void md_reset(void) {
    memset(&g_md, 0, sizeof(g_md));
    g_md.line_start = 1;
}

static int md_lang_char(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '_' || c == '-' || c == '+' || c == '#' || c == '.';
}

// --- Markdown tables -------------------------------------------------------
// Tables can't be rendered while streaming (column widths need every row), so
// md_print buffers the raw rows and md_flush_table() renders an aligned box
// once the table ends (a non-"|" line, or end of response via md_flush()).

static void md_tbl_append(char c) {
    if (g_md.table_len < (int)sizeof(g_md.table_buf) - 1)
        g_md.table_buf[g_md.table_len++] = c;
}

// Terminal column width of a Unicode codepoint: 0 (combining / variation
// selector / ZWJ), 2 (emoji, CJK, fullwidth), or 1. Covers the common cases —
// notably ✅/❌ and CJK — so table columns line up.
static int md_wcwidth(unsigned int cp) {
    if (cp == 0x200D ||                       // zero-width joiner
        (cp >= 0x0300 && cp <= 0x036F) ||     // combining diacritics
        (cp >= 0xFE00 && cp <= 0xFE0F)) return 0;  // variation selectors
    if ((cp >= 0x1100 && cp <= 0x115F) ||     // Hangul Jamo
        (cp >= 0x2300 && cp <= 0x23FF) ||     // misc technical (⌚ ⏰ …)
        (cp >= 0x2600 && cp <= 0x27BF) ||     // misc symbols + dingbats (✅ ❌ ⭐ …)
        (cp >= 0x2B00 && cp <= 0x2BFF) ||     // misc symbols & arrows
        (cp >= 0x2E80 && cp <= 0xA4CF) ||     // CJK radicals … Yi
        (cp >= 0xAC00 && cp <= 0xD7A3) ||     // Hangul syllables
        (cp >= 0xF900 && cp <= 0xFAFF) ||     // CJK compatibility ideographs
        (cp >= 0xFE30 && cp <= 0xFE4F) ||     // CJK compatibility forms
        (cp >= 0xFF00 && cp <= 0xFF60) ||     // fullwidth forms
        (cp >= 0xFFE0 && cp <= 0xFFE6) ||
        (cp >= 0x1F000 && cp <= 0x1FAFF) ||   // emoji
        (cp >= 0x20000 && cp <= 0x3FFFD)) return 2;  // CJK ext B+
    return 1;
}

// Visible terminal width of a cell: sum of codepoint widths with inline markers
// (**, *, `, ~~) removed, so columns align even with emoji/CJK content.
static int md_cell_width(const char *s) {
    int w = 0;
    for (int i = 0; s[i]; ) {
        if ((s[i] == '*' && s[i+1] == '*') || (s[i] == '~' && s[i+1] == '~')) { i += 2; continue; }
        if (s[i] == '*' || s[i] == '`') { i += 1; continue; }
        unsigned char c = (unsigned char)s[i];
        unsigned int cp; int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { i += 1; continue; }  // stray continuation/invalid byte
        for (int k = 1; k < len; k++) {
            if ((s[i+k] & 0xC0) != 0x80) { len = k; break; }  // truncated sequence
            cp = (cp << 6) | (s[i+k] & 0x3F);
        }
        w += md_wcwidth(cp);
        i += len;
    }
    return w;
}

// Emit a cell's text honoring inline bold/italic/code/strikethrough.
static void md_emit_cell(const char *s) {
    int bold = 0, ital = 0, code = 0, strike = 0;
    for (int i = 0; s[i]; ) {
        if (s[i] == '*' && s[i+1] == '*') { printf(bold ? ANSI_RESET : ANSI_BOLD); bold = !bold; i += 2; continue; }
        if (s[i] == '~' && s[i+1] == '~') { printf(strike ? ANSI_RESET : "\033[9m"); strike = !strike; i += 2; continue; }
        if (s[i] == '*') { printf(ital ? ANSI_RESET : ANSI_ITALIC); ital = !ital; i++; continue; }
        if (s[i] == '`') { printf(code ? ANSI_RESET : ANSI_CODE); code = !code; i++; continue; }
        putchar(s[i]); i++;
    }
    if (bold || ital || code || strike) printf(ANSI_RESET);
}

static int md_is_separator_cell(const char *s) {
    int sawdash = 0;
    for (const char *q = s; *q; q++) {
        if (*q == '-') sawdash = 1;
        else if (*q != ':' && *q != ' ') return 0;
    }
    return sawdash;
}

static void md_flush_table(void) {
    g_md.in_table = 0;
    g_md.table_line_start = 0;
    int len = g_md.table_len;
    g_md.table_len = 0;
    if (len <= 0) return;
    g_md.table_buf[len] = '\0';

    enum { MD_MAXR = 128, MD_MAXC = 24 };
    static char *cells[MD_MAXR][MD_MAXC];
    static int ncol[MD_MAXR];
    int nrows = 0, sep_row = -1;

    char *line = g_md.table_buf;
    while (line && *line && nrows < MD_MAXR) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '|') p++;
        int nc = 0;
        char *cs = p;
        for (;; p++) {
            if (*p == '|' || *p == '\0') {
                char *a = cs, *b = p;
                while (a < b && (*a == ' ' || *a == '\t')) a++;
                while (b > a && (b[-1] == ' ' || b[-1] == '\t')) b--;
                if (!(*p == '\0' && a == b && cs == p)) {  // skip empty cell after trailing '|'
                    if (nc < MD_MAXC) { *b = '\0'; cells[nrows][nc++] = a; }
                }
                if (*p == '\0') break;
                cs = p + 1;
            }
        }
        if (nc > 0) {
            int issep = 1;
            for (int c = 0; c < nc; c++) if (!md_is_separator_cell(cells[nrows][c])) { issep = 0; break; }
            if (issep && sep_row < 0) sep_row = nrows;
            ncol[nrows++] = nc;
        }
        line = nl ? nl + 1 : NULL;
    }
    if (nrows == 0) return;

    int ncols = 0;
    for (int r = 0; r < nrows; r++) if (ncol[r] > ncols) ncols = ncol[r];
    int width[MD_MAXC];
    for (int c = 0; c < ncols; c++) width[c] = 1;
    for (int r = 0; r < nrows; r++) {
        if (r == sep_row) continue;
        for (int c = 0; c < ncol[r] && c < ncols; c++) {
            int w = md_cell_width(cells[r][c]);
            if (w > width[c]) width[c] = w;
        }
    }

    // Each column between pipes spans width+2 cells (a space on each side).
    #define MD_BORDER(L, M, R) do { \
        printf(ANSI_DIM L); \
        for (int c = 0; c < ncols; c++) { \
            for (int k = 0; k < width[c] + 2; k++) printf("─"); \
            printf(c < ncols - 1 ? M : R); \
        } \
        printf(ANSI_RESET "\n"); \
    } while (0)

    printf("\n");
    MD_BORDER("┌", "┬", "┐");
    int header_done = 0;
    for (int r = 0; r < nrows; r++) {
        if (r == sep_row) continue;
        printf(ANSI_DIM "│" ANSI_RESET);
        for (int c = 0; c < ncols; c++) {
            const char *txt = (c < ncol[r]) ? cells[r][c] : "";
            printf(" ");
            if (!header_done) printf(ANSI_BOLD);
            md_emit_cell(txt);
            if (!header_done) printf(ANSI_RESET);
            for (int k = md_cell_width(txt); k < width[c]; k++) printf(" ");
            printf(" " ANSI_DIM "│" ANSI_RESET);
        }
        printf("\n");
        if (!header_done) { header_done = 1; MD_BORDER("├", "┼", "┤"); }
    }
    MD_BORDER("└", "┴", "┘");
    #undef MD_BORDER
}

static void md_flush(void) {
    // Emit any trailing marker bytes held for lookahead — at end of stream they
    // are just literal text with no completion coming.
    for (int k = 0; k < g_md.pending_len; k++) putchar(g_md.pending[k]);
    g_md.pending_len = 0;
    if (!g_md.in_table) return;
    if (g_md.table_line_start) {
        while (g_md.table_len > 0 &&
               (g_md.table_buf[g_md.table_len-1] == ' ' || g_md.table_buf[g_md.table_len-1] == '\t'))
            g_md.table_len--;
    }
    md_flush_table();
}

static void md_print(const char *text_in) {
    // Combine bytes held from the previous chunk for cross-chunk lookahead, so a
    // multi-char marker (``` ** ~~) split across SSE tokens is still recognized.
    int plen = g_md.pending_len;
    int tlen = (int)strlen(text_in);
    int total = plen + tlen;
    if (total == 0) return;
    char *combined = malloc((size_t)total + 1);
    if (!combined) { fputs(text_in, stdout); return; }  // best effort on OOM
    memcpy(combined, g_md.pending, (size_t)plen);
    memcpy(combined + plen, text_in, (size_t)tlen);
    combined[total] = '\0';
    g_md.pending_len = 0;

    // Hold back trailing bytes that could be the start of a longer marker:
    // 1-2 backticks (a 3rd would make a ``` fence), or a lone * / ~ (vs ** / ~~).
    int hold = 0;
    char last = combined[total - 1];
    if (last == '`') {                       // 1-2 backticks: a 3rd makes a ``` fence
        int k = 0;
        while (k < total && combined[total - 1 - k] == '`') k++;
        if (k < 3) hold = k;
    } else if (last == '*' && !(total >= 2 && combined[total - 2] == '*')) {
        hold = 1;                            // lone * (vs **)
    } else if (last == '~' && !(total >= 2 && combined[total - 2] == '~')) {
        hold = 1;                            // lone ~ (vs ~~)
    }
    // The header and rule handlers consume a whole line at once. If the trailing
    // line is unterminated (no \n yet) and looks like a header (#…) or a rule
    // (only -, *, _ and spaces), hold the whole line so its handler sees it
    // complete next chunk instead of emitting a broken partial.
    if (!g_md.code_block) {
        int ls = total;
        while (ls > 0 && combined[ls - 1] != '\n') ls--;
        int at_line_start = (ls == 0) ? g_md.line_start : 1;
        if (at_line_start) {
            int q = ls;
            while (q < total && (combined[q] == ' ' || combined[q] == '\t')) q++;
            char m = (q < total) ? combined[q] : 0;
            int hold_line = 0;
            if (m == '#') {
                hold_line = 1;                       // header
            } else if (m == '-' || m == '*' || m == '_') {
                hold_line = 1;                       // possible horizontal rule
                for (int t = q; t < total; t++)
                    if (combined[t] != m && combined[t] != ' ' && combined[t] != '\t') { hold_line = 0; break; }
            }
            int n = total - ls;
            if (hold_line && n <= (int)sizeof(g_md.pending) - 1 && n > hold) hold = n;
        }
    }
    int proc_len = total - hold;

    const char *text = combined;
    for (int i = 0; i < proc_len; i++) {
        char c = text[i];

        // Markdown table: buffer rows until the table ends, then render aligned.
        if (g_md.in_table) {
            if (g_md.table_line_start) {
                if (c == ' ' || c == '\t') { md_tbl_append(c); continue; }
                if (c == '|') { g_md.table_line_start = 0; md_tbl_append(c); continue; }
                // Line does not continue the table: drop the tentative leading
                // whitespace, render the table, then fall through to handle c.
                while (g_md.table_len > 0 &&
                       (g_md.table_buf[g_md.table_len-1] == ' ' || g_md.table_buf[g_md.table_len-1] == '\t'))
                    g_md.table_len--;
                md_flush_table();
                g_md.line_start = 1;
            } else {
                md_tbl_append(c);
                if (c == '\n') g_md.table_line_start = 1;
                continue;
            }
        }

        // Skip language tag after opening ``` (may span tokens)
        if (g_md.skip_lang) {
            if (c == '\n') {
                g_md.skip_lang = 0;
                printf(ANSI_CODEBLK ANSI_CODEBLK_LINE "\n");
                g_md.line_start = 1;
                continue;
            }
            if (md_lang_char(c)) {
                continue;
            }
            g_md.skip_lang = 0;
            if (c == ' ' || c == '\t') {
                continue;
            }
        }

        // Code block toggle: ```
        if (c == '`' && text[i+1] == '`' && text[i+2] == '`') {
            if (g_md.code_block) {
                printf(ANSI_RESET "\n");
                g_md.code_block = 0;
            } else {
                g_md.code_block = 1;
                g_md.skip_lang = 1;  // eat language tag until newline
            }
            i += 2;
            continue;
        }

        // Inside code block: print with full-width background
        if (g_md.code_block) {
            printf(ANSI_CODEBLK);
            if (c == '\n') {
                printf(ANSI_CODEBLK_LINE "\n");
            } else {
                putchar(c);
            }
            continue;
        }

        // Inline code toggle: `
        if (c == '`') {
            if (g_md.code_inline) {
                printf(ANSI_RESET);
                g_md.code_inline = 0;
            } else {
                printf(ANSI_CODE);
                g_md.code_inline = 1;
            }
            continue;
        }

        // Inside inline code: print verbatim
        if (g_md.code_inline) {
            putchar(c);
            continue;
        }

        // Table start: line begins with optional spaces then '|' → buffer it
        if (g_md.line_start) {
            int p = i;
            while (text[p] == ' ' || text[p] == '\t') p++;
            if (text[p] == '|') {
                g_md.in_table = 1;
                g_md.table_line_start = 0;
                g_md.table_len = 0;
                md_tbl_append(c);
                continue;
            }
        }

        // Horizontal rule: a whole line of only -, *, or _ (3+), optional spaces
        if (g_md.line_start && (c == '-' || c == '*' || c == '_')) {
            int p = i, cnt = 0, only = 1;
            char hc = c;
            while (text[p] && text[p] != '\n') {
                if (text[p] == hc) cnt++;
                else if (text[p] != ' ' && text[p] != '\t') { only = 0; break; }
                p++;
            }
            if (only && cnt >= 3 && text[p] == '\n') {
                printf(ANSI_DIM);
                for (int k = 0; k < 48; k++) printf("─");
                printf(ANSI_RESET "\n");
                i = p;  // consume through the newline (loop's i++ steps past it)
                g_md.line_start = 1;
                continue;
            }
        }

        // Blockquote: > (or >>) at line start → colored gutter + dim text
        if (g_md.line_start && c == '>') {
            while (text[i] == '>') i++;
            while (text[i] == ' ') i++;
            printf("\033[2m\033[36m│\033[0m " ANSI_DIM);
            g_md.quote = 1;
            g_md.line_start = 0;
            i--;  // loop's i++ lands on the first content char
            continue;
        }

        // Headers at line start: # ## ### — hide markers, show text bold blue
        if (g_md.line_start && c == '#') {
            while (text[i] == '#') i++;  // skip all #
            while (text[i] == ' ') i++;  // skip space after #
            printf(ANSI_HEADER);
            while (text[i] && text[i] != '\n') { putchar(text[i]); i++; }
            printf(ANSI_RESET);
            if (text[i] == '\n') { putchar('\n'); g_md.line_start = 1; }
            continue;
        }

        // Bullet lists: - or * at line start (possibly indented with spaces)
        // Count leading spaces for indent level, then check for bullet marker
        if (g_md.line_start && (c == '-' || c == '*' || c == ' ')) {
            // Peek ahead: count indent, find marker
            int indent = 0;
            int peek = i;
            while (text[peek] == ' ' || text[peek] == '\t') { indent++; peek++; }
            char marker = text[peek];
            if ((marker == '-' || marker == '*') && marker != '\0') {
                char after = text[peek + 1];
                // Bullet: marker followed by space, end of token, or tab
                // For *, must not be ** (bold)
                if (marker == '-' && (after == ' ' || after == '\0')) {
                    int depth = indent / 2;
                    for (int d = 0; d < depth + 1; d++) printf("  ");
                    printf("\033[33m•\033[0m ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--; // loop will i++
                    g_md.line_start = 0;
                    continue;
                }
                if (marker == '*' && after != '*' && (after == ' ' || after == '\0' || after == '\t')) {
                    int depth = indent / 2;
                    for (int d = 0; d < depth + 1; d++) printf("  ");
                    printf("\033[33m•\033[0m ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--;
                    g_md.line_start = 0;
                    continue;
                }
            }
            // Not a bullet — fall through to normal handling
        }

        // Numbered lists at line start: 1. item → colored number
        if (g_md.line_start && c >= '0' && c <= '9') {
            int num_start = i;
            while (text[i] >= '0' && text[i] <= '9') i++;
            if (text[i] == '.' && text[i+1] == ' ') {
                printf("  \033[33m");  // yellow
                for (int j = num_start; j <= i; j++) putchar(text[j]);
                printf("\033[0m");
                i++; // skip space
                g_md.line_start = 0;
                continue;
            }
            // Not a list, rewind and print normally
            i = num_start;
            c = text[i];
        }

        // Bold: **
        if (c == '*' && text[i+1] == '*') {
            if (g_md.bold) {
                printf(ANSI_RESET);
                g_md.bold = 0;
            } else {
                printf(ANSI_BOLD);
                g_md.bold = 1;
            }
            i++;
            continue;
        }

        // Italic: single * (but not **)
        if (c == '*' && text[i+1] != '*') {
            if (g_md.italic) {
                printf(ANSI_RESET);
                g_md.italic = 0;
            } else {
                printf(ANSI_ITALIC);
                g_md.italic = 1;
            }
            continue;
        }

        // Strikethrough: ~~text~~
        if (c == '~' && text[i+1] == '~') {
            if (g_md.strike) { printf(ANSI_RESET); g_md.strike = 0; }
            else { printf("\033[9m"); g_md.strike = 1; }
            i++;
            continue;
        }

        // Track line starts
        if (c == '\n') {
            if (g_md.quote) { printf(ANSI_RESET); g_md.quote = 0; }
            g_md.line_start = 1;
        } else {
            g_md.line_start = 0;
        }

        putchar(c);
    }

    // Stash the held trailing bytes for the next chunk. Inside a table they're
    // raw buffer content (rendered on flush), so append them there instead.
    if (hold > 0) {
        if (g_md.in_table) {
            for (int k = proc_len; k < total; k++) md_tbl_append(combined[k]);
        } else {
            memcpy(g_md.pending, combined + proc_len, (size_t)hold);
            g_md.pending_len = hold;
        }
    }
    free(combined);
}

static int json_int_field(const char *json, const char *field, int fallback) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", field);
    char *p = strstr(json, pattern);
    if (!p) return fallback;
    return atoi(p + strlen(pattern));
}

static double json_double_field(const char *json, const char *field, double fallback) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", field);
    char *p = strstr(json, pattern);
    if (!p) return fallback;
    return atof(p + strlen(pattern));
}

// Stream SSE response, accumulate text, return malloc'd response string
static char *stream_response(int sock, int show_thinking) {
    FILE *stream = fdopen(sock, "r");
    if (!stream) { close(sock); return NULL; }

    int header_done = 0, in_think = 0, tokens = 0;
    int mtp_drafts = -1, mtp_accepted = -1;  // MTP shadow-draft stats from final chunk usage
    int usage_available = 0;
    int usage_total_tokens = 0, usage_think_tokens = 0, usage_response_tokens = 0;
    double usage_ttft_ms = 0, usage_generation_ms = 0, usage_think_ms = 0, usage_response_ms = 0;
    double usage_experts_mib_per_sec = 0.0;
    double usage_experts_mib_per_sec_per_expert = 0.0;
    double t_start = now_ms(), t_first = 0;
    md_reset();  // fresh markdown state for each response

    char *response = calloc(1, MAX_RESPONSE);
    int resp_len = 0;

    char line[65536];
    while (fgets(line, sizeof(line), stream)) {
        if (!header_done) {
            if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) header_done = 1;
            continue;
        }
        if (strncmp(line, "data: ", 6) != 0) continue;
        if (strncmp(line + 6, "[DONE]", 6) == 0) break;

        // Final chunk carries MTP shadow-draft usage stats (no content delta).
        char *mk = strstr(line + 6, "\"mtp_drafts\":");
        if (mk) {
            mtp_drafts = atoi(mk + 13);
            char *ak = strstr(line + 6, "\"mtp_accepted\":");
            if (ak) mtp_accepted = atoi(ak + 15);
        }
        if (strstr(line + 6, "\"completion_tokens\":")) {
            usage_available = 1;
            usage_total_tokens = json_int_field(line + 6, "completion_tokens", 0);
            usage_think_tokens = json_int_field(line + 6, "thinking_tokens", 0);
            usage_response_tokens = json_int_field(line + 6, "response_tokens", 0);
            usage_ttft_ms = json_double_field(line + 6, "ttft_ms", 0.0);
            usage_generation_ms = json_double_field(line + 6, "generation_ms", 0.0);
            usage_think_ms = json_double_field(line + 6, "thinking_ms", 0.0);
            usage_response_ms = json_double_field(line + 6, "response_ms", 0.0);
            usage_experts_mib_per_sec = json_double_field(line + 6, "experts_mib_per_sec", 0.0);
            usage_experts_mib_per_sec_per_expert = json_double_field(line + 6, "experts_mib_per_sec_per_expert", 0.0);
        }

        int is_reasoning_delta = 0;
        char *ck = strstr(line + 6, "\"reasoning_content\":\"");
        if (ck) {
            ck += 21;
            is_reasoning_delta = 1;
        } else {
            ck = strstr(line + 6, "\"content\":\"");
            if (ck) ck += 11;
        }
        if (!ck) continue;

        char decoded[4096]; int di = 0;
        for (int i = 0; ck[i] && ck[i] != '"' && di < 4095; i++) {
            if (ck[i] == '\\' && ck[i+1]) {
                i++;
                switch (ck[i]) {
                    case 'n': decoded[di++]='\n'; break;
                    case 't': decoded[di++]='\t'; break;
                    case '"': decoded[di++]='"'; break;
                    case '\\': decoded[di++]='\\'; break;
                    default: decoded[di++]=ck[i]; break;
                }
            } else decoded[di++] = ck[i];
        }
        decoded[di] = 0;
        if (!di) continue;

        if (is_reasoning_delta) in_think = 1;
        else in_think = 0;
        if (strstr(decoded, "<think>")) in_think = 1;
        if (strstr(decoded, "</think>")) { in_think = 0; tokens++; continue; }
        tokens++;
        if (!t_first) t_first = now_ms();

        // Accumulate non-thinking response
        if (!is_reasoning_delta && !in_think && resp_len + di < MAX_RESPONSE - 1) {
            memcpy(response + resp_len, decoded, di);
            resp_len += di;
            response[resp_len] = 0;
        }

        if (is_reasoning_delta && !show_thinking) continue;
        if (is_reasoning_delta || in_think) printf(ANSI_DIM "%s" ANSI_RESET, decoded);
        else md_print(decoded);
        fflush(stdout);
    }
    fclose(stream);

    md_flush();           // render a table left buffered at end of response
    printf(ANSI_RESET);  // ensure no style leaks
    double t_end = now_ms();
    double ttft_ms = t_first > 0 ? t_first - t_start : 0;
    double gen_time = t_first > 0 ? t_end - t_first : 0;
    int gen_tokens = tokens > 1 ? tokens - 1 : 0;
    printf("\n\n");
    if (usage_available && usage_total_tokens > 0 && usage_generation_ms > 0) {
        char detail[192] = "";
        char *w = detail;
        size_t rem = sizeof(detail);
        if (usage_think_tokens > 0) {
            int n = snprintf(w, rem, "%d@%.1ftok/s think",
                             usage_think_tokens,
                             usage_think_ms > 0 ? usage_think_tokens * 1000.0 / usage_think_ms : 0.0);
            w += n; rem = (n > 0 && (size_t)n < rem) ? rem - (size_t)n : 0;
        }
        if (usage_response_tokens > 0 && rem > 0) {
            snprintf(w, rem, "%s%d@%.1ftok/s response",
                     detail[0] ? ", " : "",
                     usage_response_tokens,
                     usage_response_ms > 0 ? usage_response_tokens * 1000.0 / usage_response_ms : 0.0);
        }
        char mtp[64] = "";
        if (mtp_drafts > 0)
            snprintf(mtp, sizeof(mtp), ", MTP %.0f%% (%d/%d)",
                     100.0 * mtp_accepted / mtp_drafts, mtp_accepted, mtp_drafts);
        char experts[64] = "";
        if (usage_experts_mib_per_sec > 0.0)
            snprintf(experts, sizeof(experts), ", experts %.1f MiB/s, %.1f MiB/s/expert",
                     usage_experts_mib_per_sec, usage_experts_mib_per_sec_per_expert);
        printf("[%d tokens, %.1f tok/s, TTFT %.1fs%s%s%s%s%s]\n\n",
               usage_total_tokens, usage_total_tokens * 1000.0 / usage_generation_ms,
               usage_ttft_ms / 1000.0,
               detail[0] ? " (" : "",
               detail,
               detail[0] ? ")" : "",
               experts,
               mtp);
    } else if (gen_tokens > 0 && gen_time > 0) {
        char mtp[64] = "";
        if (mtp_drafts > 0)
            snprintf(mtp, sizeof(mtp), ", MTP %.0f%% (%d/%d)",
                     100.0 * mtp_accepted / mtp_drafts, mtp_accepted, mtp_drafts);
        printf("[%d tokens, %.1f tok/s, TTFT %.1fs%s]\n\n",
               tokens, gen_tokens * 1000.0 / gen_time,
               ttft_ms / 1000.0, mtp);
    }

    return response;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    int port = 8000;
    int max_tokens = 8192;
    int reasoning_enabled = flag_enabled(getenv("FLASHCHAT_REASONING"));
    int show_thinking = flag_enabled(getenv("FLASHCHAT_SHOW_THINKING"));
    const char *resume_id = NULL;
    const char *host = getenv("FLASHCHAT_SERVER_HOST");
    if (!host || !host[0] || strcmp(host, "0.0.0.0") == 0 || strcmp(host, "::") == 0) {
        host = "127.0.0.1";
    }

    static struct option long_options[] = {
        {"host",        required_argument, 0, 'H'},
        {"port",        required_argument, 0, 'p'},
        {"max-tokens",  required_argument, 0, 't'},
        {"reasoning",   no_argument,       0, 1000},
        {"no-reasoning", no_argument,      0, 1001},
        {"show-think",  no_argument,       0, 's'},
        {"resume",      required_argument, 0, 'r'},
        {"sessions",    no_argument,       0, 'l'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    init_sessions_dir();

    int c;
    while ((c = getopt_long(argc, argv, "H:p:t:sr:lh", long_options, NULL)) != -1) {
        switch (c) {
            case 'H': host = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 't': max_tokens = atoi(optarg); break;
            case 1000: reasoning_enabled = 1; break;
            case 1001: reasoning_enabled = 0; break;
            case 's': show_thinking = 1; break;
            case 'r': resume_id = optarg; break;
            case 'l': session_list(); return 0;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  --host HOST      Server host (default: 127.0.0.1)\n");
                printf("  --port N         Server port (default: 8000)\n");
                printf("  --max-tokens N   Max response tokens (default: 8192)\n");
                printf("  --reasoning      Enable model reasoning mode\n");
                printf("  --no-reasoning   Disable model reasoning mode\n");
                printf("  --show-think     Show <think> blocks (dimmed)\n");
                printf("  --resume ID      Resume a previous session\n");
                printf("  --sessions       List saved sessions\n");
                printf("  --help           This message\n");
                return 0;
            default: return 1;
        }
    }

    char session_id[64];
    if (resume_id) {
        strncpy(session_id, resume_id, sizeof(session_id) - 1);
        session_id[sizeof(session_id) - 1] = 0;
    } else {
        generate_session_id(session_id, sizeof(session_id));
    }

    const char *model_name = getenv("FLASHCHAT_MODEL");
    printf("==================================================\n");
    printf("  %s Chat (Flashchat)\n", model_name ? model_name : "Flashchat");
    printf("==================================================\n");
    printf("  Server:  http://%s:%d\n", host, port);
    printf("  Session: %s%s\n", session_id, resume_id ? " (resumed)" : "");
    printf("\n  Commands: /quit /exit /clear /sessions\n");
    printf("==================================================\n\n");

    // Health check
    int sock = connect_to_server(host, port);
    if (sock < 0) {
        fprintf(stderr, "Server not running at %s:%d.\n", host, port);
        fprintf(stderr, "Start it: ./infer --serve %d\n\n", port);
        return 1;
    }
    close(sock);

    // Resume: load and display previous conversation
    if (resume_id) {
        int turns = session_load(session_id);
        if (turns == 0) {
            printf("No session found with ID: %s\n\n", session_id);
        }
        // Note: server-side KV cache may not match if server restarted.
        // The conversation will continue but model won't "remember" old context
        // unless we re-prefill (TODO: detect server restart and replay).
    }

    printf("Ready to chat.\n\n");

    // Set up linenoise: history, hints
    linenoiseSetMultiLine(1);  // allow multi-line input with arrow keys
    linenoiseHistoryLoad(g_history_path);
    linenoiseHistorySetMaxLen(500);

    for (;;) {
        char *line = linenoise("> ");
        if (!line) {
            printf("\n");
            break;
        }

        size_t len = strlen(line);
        if (len == 0) { free(line); continue; }

        // Add to history
        linenoiseHistoryAdd(line);
        linenoiseHistorySave(g_history_path);

        char input_line[MAX_INPUT_LINE];
        strncpy(input_line, line, MAX_INPUT_LINE - 1);
        input_line[MAX_INPUT_LINE - 1] = 0;
        free(line);

        if (strcmp(input_line, "/quit") == 0 || strcmp(input_line, "/exit") == 0) {
            printf("Goodbye.\n");
            break;
        }
        if (strcmp(input_line, "/clear") == 0) {
            generate_session_id(session_id, sizeof(session_id));
            printf("[new session: %s]\n\n", session_id);
            continue;
        }
        if (strcmp(input_line, "/sessions") == 0) {
            session_list();
            continue;
        }

        // Save user turn
        session_save_turn(session_id, "user", input_line);

        sock = send_chat_request(host, port, input_line, max_tokens, session_id, reasoning_enabled);
        if (sock < 0) continue;

        printf("\n");
        char *response = stream_response(sock, show_thinking);

        // Save assistant turn
        if (response && strlen(response) > 0) {
            session_save_turn(session_id, "assistant", response);
        }

        // ---- Tool call handling ----
        // Detect <tool_call>{"name":"bash","arguments":{"command":"..."}}
        // Execute the command, feed output back as a continuation
        while (response && strstr(response, "<tool_call>")) {
            char *tc_start = strstr(response, "<tool_call>");
            char *tc_end = strstr(tc_start, "</tool_call>");
            if (!tc_start || !tc_end) break;

            // Extract content between tags
            tc_start += 11;  // skip <tool_call>
            char tc_body[4096] = {0};
            int tc_len = (int)(tc_end - tc_start);
            if (tc_len > 4095) tc_len = 4095;
            memcpy(tc_body, tc_start, tc_len);

            // Parse command — handle multiple formats the model might produce:
            // 1. JSON: {"name":"bash","arguments":{"command":"ls -la"}}
            // 2. XML-ish: <function=bash><arg_key>command</arg_key><arg_value>ls -la</arg_value>
            // 3. Simple: just a command string
            char command[4096] = {0};
            int ci = 0;

            char *cmd_key = strstr(tc_body, "\"command\"");
            if (cmd_key) {
                // JSON format: find value after "command":"
                cmd_key = strchr(cmd_key + 9, '"');
                if (cmd_key) {
                    cmd_key++;
                    for (int i = 0; cmd_key[i] && cmd_key[i] != '"' && ci < 4095; i++) {
                        if (cmd_key[i] == '\\' && cmd_key[i+1]) {
                            i++;
                            switch (cmd_key[i]) {
                                case 'n': command[ci++] = '\n'; break;
                                case '"': command[ci++] = '"'; break;
                                case '\\': command[ci++] = '\\'; break;
                                default: command[ci++] = cmd_key[i]; break;
                            }
                        } else {
                            command[ci++] = cmd_key[i];
                        }
                    }
                }
            }
            // Fallback: look for <arg_value>...</arg_value> (model's XML format)
            if (ci == 0) {
                char *av = strstr(tc_body, "<arg_value>");
                if (av) {
                    av += 11;
                    char *av_end = strstr(av, "</arg_value>");
                    if (!av_end) av_end = strstr(av, "<");
                    if (av_end) {
                        int avlen = (int)(av_end - av);
                        if (avlen > 4095) avlen = 4095;
                        memcpy(command, av, avlen);
                        ci = avlen;
                        // Trim whitespace
                        while (ci > 0 && (command[ci-1] == '\n' || command[ci-1] == ' ')) ci--;
                        command[ci] = 0;
                    }
                }
            }
            // Fallback: look for function=bash followed by any command-like text
            if (ci == 0) {
                char *fn = strstr(tc_body, "bash");
                if (fn) {
                    // Take everything after "bash" that looks like a command
                    fn += 4;
                    while (*fn && (*fn == '>' || *fn == '\n' || *fn == ' ' || *fn == '"')) fn++;
                    while (*fn && *fn != '<' && *fn != '"' && ci < 4095) {
                        command[ci++] = *fn++;
                    }
                    while (ci > 0 && (command[ci-1] == '\n' || command[ci-1] == ' ')) ci--;
                    command[ci] = 0;
                }
            }

            if (ci == 0) break;

            // Show the command and ask for confirmation
            printf("\033[33m$ %s\033[0m\n", command);
            printf("\033[2m[execute? y/n] \033[0m");
            fflush(stdout);
            int ch = getchar();
            while (getchar() != '\n');  // consume rest of line
            if (ch != 'y' && ch != 'Y') {
                printf("\033[2m[skipped]\033[0m\n");
                free(response);
                response = NULL;
                break;
            }

            // Execute
            FILE *proc = popen(command, "r");
            char output[65536] = {0};
            int out_len = 0;
            if (proc) {
                while (out_len < 65535) {
                    int ch = fgetc(proc);
                    if (ch == EOF) break;
                    output[out_len++] = (char)ch;
                }
                output[out_len] = 0;
                pclose(proc);
            }

            // Print output
            if (out_len > 0) {
                printf("\033[2m%s\033[0m", output);
                if (output[out_len-1] != '\n') printf("\n");
            }

            // Send tool response back to model as a continuation
            // Format: <tool_response>\n{output}\n</tool_response>
            char *tool_msg = malloc(out_len + 256);
            snprintf(tool_msg, out_len + 256, "<tool_response>\n%s</tool_response>", output);

            free(response);
            sock = send_chat_request(host, port, tool_msg, max_tokens, session_id, reasoning_enabled);
            free(tool_msg);
            if (sock < 0) { response = NULL; break; }

            printf("\n");
            response = stream_response(sock, show_thinking);

            if (response && strlen(response) > 0) {
                session_save_turn(session_id, "assistant", response);
            }
        }

        free(response);
    }

    return 0;
}
