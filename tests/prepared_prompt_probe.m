#include <stdio.h>
#include <stdarg.h>

static int probe_quiet = 0;

static int probe_fprintf(FILE *stream, const char *format, ...) {
    if (probe_quiet && stream == stderr) return 0;
    va_list args;
    va_start(args, format);
    int result = vfprintf(stream, format, args);
    va_end(args);
    return result;
}

#define fprintf probe_fprintf
#define main flashchat_infer_main
#include "../metal_infer/infer.m"
#undef main
#undef fprintf
#include <assert.h>
#include <time.h>

#define PROBE_PAIRS 20

typedef struct {
    PromptTokens *tokens;
    const char *system;
    int system_tokens;
    uint64_t system_hash;
    ToolValidationResult validation;
    int validation_status;
    int hit;
    double prepare_ms, validation_ms;
} ProbeResult;

static int probe_measure = 0;

static double probe_clock_ms(void) {
    if (!probe_measure) return 0;
    struct timespec ts;
    assert(clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Reproduce 9c62991's serve_loop preparation: render/hash/count the system,
// render it again, then tokenize the entire assembled prompt. The shared renderer,
// tokenizer and validator are unchanged by exp28; diagnostics are excluded in both paths.
static ProbeResult probe_prepare(ApiRequest *req, PreparedPrompt *cache, int use_cache) {
    ProbeResult r = {.hit = -1};
    PreparedPrompt scratch = {0};
    double start = probe_clock_ms();
    if (use_cache) {
        const PreparedPrompt *p = prepare_request_prompt(cache, &scratch, req, &r.hit);
        assert(p == cache && "probe input must fit the bounded cache");
        r.system = p->system;
        r.system_tokens = p->tokens->count;
        r.system_hash = hash_string_djb2(p->system);
        r.tokens = prepared_prompt_tokens(p, req->conversation_text);
    } else {
        r.system = build_system_prompt_for_request(req, NULL);
        assert(r.system);
        r.system_hash = hash_string_djb2(r.system);
        r.system_tokens = count_sys_prompt_tokens(r.system);
        char *second_system = build_system_prompt_for_request(req, NULL);
        size_t size = strlen(second_system) + strlen(req->conversation_text) + 128;
        char *assembled = malloc(size);
        assert(assembled);
        snprintf(assembled, size, "<|im_start|>system\n%s<|im_end|>\n%s",
                 second_system, req->conversation_text);
        r.tokens = encode_prompt_text_to_tokens(assembled);
        free(assembled);
        free(second_system);
    }
    assert(r.tokens);
    double prepared = probe_clock_ms();
    if (req->tool_count) {
        ParsedToolCall call = {.is_tool_call = 1, .arguments = "{\"path\":\"src/main.c\",\"limit\":10}"};
        strcpy(call.name, req->tools[0].name);
        // The same request supplies exact serialized inputs to both paths. The
        // legacy path must parse its schema even when the new path ran first.
        int saved = req->tools[0].schema_prepared;
        if (!use_cache) req->tools[0].schema_prepared = 0;
        r.validation_status = validate_parsed_tool_call(req, &call, &r.validation);
        req->tools[0].schema_prepared = saved;
    }
    double finished = probe_clock_ms();
    r.prepare_ms = prepared - start;
    r.validation_ms = finished - prepared;
    return r;
}

static ApiRequest *probe_request(int with_tools) {
    NSMutableArray *tools = [NSMutableArray array];
    NSMutableString *system = [NSMutableString string];
    if (with_tools) {
        while (system.length < 4096) {
            [system appendString:@"Inspect the relevant files before editing. Preserve existing behavior, "
                @"explain concrete findings, and use the supplied tools to verify assumptions.\n"];
        }
        for (int i = 0; i < 24; i++) {
            NSString *name = [NSString stringWithFormat:@"inspect_%02d", i];
            [tools addObject:@{@"type": @"function", @"function": @{
                @"name": name,
                @"description": @"Inspect project files and return matching source lines with their paths. "
                    @"Use a relative project path and a bounded result limit. Missing files are reported "
                    @"as errors. This tool reads source files without changing their contents. Include "
                    @"a query to narrow matches, and context lines when adjacent code helps interpretation.",
                @"parameters": @{@"type": @"object", @"additionalProperties": @NO,
                    @"required": @[@"path", @"limit"], @"properties": @{
                        @"path": @{@"type": @"string", @"description": @"Relative path of the source file to inspect."},
                        @"limit": @{@"type": @"integer", @"minimum": @1, @"maximum": @100,
                                   @"description": @"Maximum number of matches returned."},
                        @"query": @{@"type": @"string", @"description": @"Optional exact text to find in the file."},
                        @"context": @{@"type": @"integer", @"minimum": @0, @"maximum": @20},
                        @"include": @{@"type": @"array", @"items": @{@"type": @"string"}}}}}}];
        }
    }
    NSMutableArray *messages = [NSMutableArray array];
    if (with_tools) [messages addObject:@{@"role": @"system", @"content": system}];
    [messages addObject:@{@"role": @"user", @"content": @"Summarize src/file_0000.c in three lines."}];
    ApiRequest *req = calloc(1, sizeof(*req));
    assert(req);
    api_request_init(req, API_KIND_CHAT);
    char *error = NULL;
    int rc = fill_request_from_chat_json(@{@"messages": messages, @"tools": tools,
        @"reasoning": @NO, @"temperature": @0, @"stream": @NO}, req, &error);
    assert(rc == 0 && !error);
    return req;
}

static int probe_compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double probe_median(double *values) {
    qsort(values, PROBE_PAIRS, sizeof(*values), probe_compare_double);
    return (values[PROBE_PAIRS / 2 - 1] + values[PROBE_PAIRS / 2]) / 2;
}

static void probe_case(const char *name, int with_tools, int invalidate) {
    ApiRequest *req = probe_request(with_tools);
    NSString *original_system = [NSString stringWithUTF8String:req->system_prompt ?: ""];
    NSString *original_conversation = [NSString stringWithUTF8String:req->conversation_text];
    PreparedPrompt cache = {0};
    double before[PROBE_PAIRS], after[PROBE_PAIRS], saved[PROBE_PAIRS];
    int count = probe_measure ? PROBE_PAIRS : 2;
    int hits = 0;
    // One untimed pair warms the tokenizer, allocator and entry. Miss cases
    // replace their system bytes every pair. The changing suffix is never cached.
    for (int i = -1; i < count; i++) {
        @autoreleasepool {
            NSString *file = [NSString stringWithFormat:@"file_%04d", i + 1];
            free(req->conversation_text);
            req->conversation_text = dup_nsstring([original_conversation
                                      stringByReplacingOccurrencesOfString:@"file_0000" withString:file]);
            if (invalidate) {
                free(req->system_prompt);
                req->system_prompt = dup_nsstring([original_system stringByAppendingFormat:@"\nRevision: %04d", i + 1]);
            }
            int measure = probe_measure;
            if (i < 0) probe_measure = 0;
            ProbeResult old, new;
            if (i % 2 == 0) {
                old = probe_prepare(req, &cache, 0);
                new = probe_prepare(req, &cache, 1);
            } else {
                new = probe_prepare(req, &cache, 1);
                old = probe_prepare(req, &cache, 0);
            }
            probe_measure = measure;
            assert(!strcmp(old.system, new.system));
            assert(old.system_hash == new.system_hash && old.system_tokens == new.system_tokens);
            assert(old.tokens->count == new.tokens->count);
            assert(!memcmp(old.tokens->ids, new.tokens->ids, old.tokens->count * sizeof(uint32_t)));
            assert(old.validation_status == new.validation_status);
            assert(!memcmp(&old.validation, &new.validation, sizeof(old.validation)));
            assert(new.hit == (i >= 0 && !invalidate));
            if (i >= 0) {
                hits += new.hit;
                if (probe_measure) {
                    before[i] = old.prepare_ms + old.validation_ms;
                    after[i] = new.prepare_ms + new.validation_ms;
                    saved[i] = before[i] - after[i];
                    ProbeResult *paths[] = {&old, &new};
                    for (int j = 0; j < 2; j++) {
                        ProbeResult *r = paths[j];
                        printf("%s\t%d\t%s\t%d\t%zu\t%d\t%d\t%.6f\t%.6f\t%.6f\n",
                               name, i + 1, j ? "exp28" : "before", r->hit,
                               strlen(r->system), r->system_tokens, r->tokens->count,
                               r->prepare_ms, r->validation_ms, r->prepare_ms + r->validation_ms);
                    }
                }
            }
            free((void *)old.system);
            free(old.tokens->ids); free(old.tokens);
            free(new.tokens->ids); free(new.tokens);
        }
    }
    if (probe_measure) {
        double old_ms = probe_median(before), new_ms = probe_median(after);
        double saved_ms = probe_median(saved);
        printf("# %s: before=%.3f ms exp28=%.3f ms paired_saving=%.3f ms "
               "saving_p10_p90=[%.3f,%.3f] ms change=%+.1f%% hits=%d/%d\n",
               name, old_ms, new_ms, saved_ms, saved[1], saved[17],
               (new_ms / old_ms - 1) * 100, hits, count);
    } else {
        printf("PASS %s: exact bytes/tokens/counts/schema results; system=%zu bytes/%d tokens; hits=%d/%d\n",
               name, strlen(cache.system), cache.tokens->count, hits, count);
    }
    prepared_prompt_free(&cache);
    api_request_free(req);
    free(req);
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "--check") && strcmp(argv[1], "--measure"))) {
        fprintf(stderr, "Usage: %s --check|--measure /absolute/path/to/vocab.bin\n"
                        "--check reads no clock. --measure runs 20 alternating pairs per case; requires benchmark approval.\n", argv[0]);
        return 2;
    }
    @autoreleasepool {
        if (bpe_load(&g_tokenizer, argv[2]) != 0) return 1;
        g_tokenizer_loaded = 1;
        g_tokenizer_generation++;
        g_cfg.thinking_capable = 1;
        g_cfg.bits = 4;
        kApiModelId = "exp28-preparation-probe";
        char temp[] = "/tmp/flashchat-exp28-probe-XXXXXX";
        assert(mkdtemp(temp));
        snprintf(g_custom_system_prompt_path, sizeof(g_custom_system_prompt_path), "%s/system.md", temp);
        // Suppress tokenizer diagnostics in both paths without hiding assertions.
        probe_quiet = 1;
        probe_measure = !strcmp(argv[1], "--measure");
        if (probe_measure) puts("case\tpair\tpath\tcache_hit\tsystem_bytes\tsystem_tokens\tprompt_tokens\tprepare_ms\tvalidate_ms\ttotal_ms");
        probe_case("small_prefix_hit", 0, 0);
        probe_case("24_tools_hit", 1, 0);
        probe_case("24_tools_miss", 1, 1);
        bpe_free(&g_tokenizer);
        assert(rmdir(temp) == 0);
    }
    return 0;
}
