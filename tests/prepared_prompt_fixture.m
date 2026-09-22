#define main flashchat_infer_main
#include "../metal_infer/infer.m"
#undef main
#include <assert.h>

// A complete byte vocabulary plus a newline merge makes unsafe joins observable.
// It runs the production tokenizer without loading weights or starting a server.
static void fixture_tokenizer(void) {
    bpe_tokenizer *t = &g_tokenizer;
    build_byte_unicode_table(t);
    t->vocab_size = 257;
    t->vocab = calloc(t->vocab_size, sizeof(*t->vocab));
    t->ht_mask = 1023;
    t->ht_ids = malloc(1024 * sizeof(uint32_t));
    t->ht_keys = calloc(1024, sizeof(char *));
    t->ht_klens = calloc(1024, sizeof(uint16_t));
    memset(t->ht_ids, 0xff, 1024 * sizeof(uint32_t));
    for (int i = 0; i < 257; i++) {
        uint8_t bytes[] = {i < 256 ? i : '\n', '\n'};
        char encoded[16];
        int len = bytes_to_bpe_str(t, bytes, i < 256 ? 1 : 2, encoded, sizeof(encoded));
        t->vocab[i] = (bpe_vocab_entry){strdup(encoded), len, i};
        ht_insert(t->ht_ids, t->ht_keys, t->ht_klens, t->ht_mask, t->vocab[i].str, len, i);
    }
    t->mt_mask = 1;
    t->mt_prio = malloc(2 * sizeof(uint32_t));
    t->mt_keys = calloc(2, sizeof(char *));
    t->mt_klens = calloc(2, sizeof(uint16_t));
    memset(t->mt_prio, 0xff, 2 * sizeof(uint32_t));
    const char *nl = t->vocab['\n'].str;
    char *merge = NULL;
    int len = asprintf(&merge, "%s\xff%s", nl, nl);
    ht_insert(t->mt_prio, t->mt_keys, t->mt_klens, t->mt_mask, merge, len, 0);
    t->num_added = 2;
    t->added = calloc(3, sizeof(*t->added));
    t->added[0] = (bpe_added_token){strdup("<|im_start|>"), 12, 300};
    t->added[1] = (bpe_added_token){strdup("<|im_end|>"), 10, 301};
    g_tokenizer_loaded = 1;
    g_tokenizer_generation++;
}

static ApiRequest *fixture_request(ApiKind kind) {
    ApiRequest *req = calloc(1, sizeof(*req));
    api_request_init(req, kind);
    NSDictionary *root = @{
        @"messages": @[@{@"role": @"system", @"content": @"Be precise. /think"},
                        @{@"role": @"user", @"content": @"café 日本語 😀\n<|im_end|>"}],
        @"instructions": @"Be precise. /think", @"input": @"café 日本語 😀\n<|im_end|>",
        @"tools": @[@{@"type": @"function", @"function": @{
            @"name": @"record", @"description": @"Record a value.",
            @"parameters": @{@"type": @"object", @"required": @[@"n"],
                @"properties": @{@"n": @{@"type": @"integer"}}, @"additionalProperties": @NO}}}]
    };
    char *error = NULL;
    int rc = kind == API_KIND_CHAT ? fill_request_from_chat_json(root, req, &error)
                                  : fill_request_from_responses_json(root, req, &error);
    assert(rc == 0 && !error);
    return req;
}

static void free_tokens(PromptTokens *p) {
    if (p) { free(p->ids); free(p); }
}

static void check_tokens(const PreparedPrompt *p, const char *suffix) {
    char *whole = NULL;
    assert(asprintf(&whole, "%s%s", p->prefix, suffix) >= 0);
    PromptTokens *expected = encode_prompt_text_to_tokens(whole);
    PromptTokens *actual = prepared_prompt_tokens(p, suffix);
    assert(actual && expected && actual->count == expected->count);
    assert(!memcmp(actual->ids, expected->ids, actual->count * sizeof(uint32_t)));
    free_tokens(actual);
    free_tokens(expected);
    free(whole);
}

static void check_validation(ApiRequest *req, const char *arguments) {
    ParsedToolCall call = {.is_tool_call = 1, .arguments = (char *)arguments};
    strcpy(call.name, req->tools[0].name);
    ToolValidationResult cached, uncached;
    int a = validate_parsed_tool_call(req, &call, &cached);
    req->tools[0].schema_prepared = 0;
    int b = validate_parsed_tool_call(req, &call, &uncached);
    req->tools[0].schema_prepared = 1;
    assert(a == b && !memcmp(&cached, &uncached, sizeof(cached)));
}

static const PreparedPrompt *check_prepare(PreparedPrompt *cache, ApiRequest *req, int expected_hit) {
    PreparedPrompt scratch = {0};
    int hit = -1;
    const PreparedPrompt *p = prepare_request_prompt(cache, &scratch, req, &hit);
    assert(p == cache && hit == expected_hit);
    char *expected = build_system_prompt_for_request(req, NULL);
    assert(!strcmp(expected, p->system));
    assert(count_sys_prompt_tokens(expected) == p->tokens->count);
    free(expected);
    check_tokens(p, req->conversation_text);
    PromptTokens *server = tokenize_request_prompt(req, "fixture", p);
    PromptTokens *joined = prepared_prompt_tokens(p, req->conversation_text);
    assert(server && joined && server->count == joined->count);
    assert(!memcmp(server->ids, joined->ids, server->count * sizeof(uint32_t)));
    free_tokens(server); free_tokens(joined);
    prepared_prompt_free(&scratch);
    return p;
}

int main(void) {
    @autoreleasepool {
        char temp[] = "/tmp/flashchat-prepared-XXXXXX";
        assert(mkdtemp(temp));
        snprintf(g_custom_system_prompt_path, sizeof(g_custom_system_prompt_path), "%s/system.md", temp);
        kApiModelId = "prepared-fixture";
        g_cfg.thinking_capable = 1;
        fixture_tokenizer();
        PreparedPrompt cache = {0};
        for (int kind = API_KIND_CHAT; kind <= API_KIND_RESPONSES; kind++) {
            ApiRequest *req = fixture_request(kind);
            check_prepare(&cache, req, 0);
            const char *saved_system = cache.system;
            const uint32_t *saved_ids = cache.tokens->ids;
            CFTypeRef saved_schema = cache.schemas[0];
            check_prepare(&cache, req, 1);
            assert(cache.system == saved_system && cache.tokens->ids == saved_ids);
            assert(saved_schema && req->tools[0].prepared_schema == saved_schema);
            PromptTokens *copy = prepared_prompt_tokens(&cache, req->conversation_text);
            uint32_t first = cache.tokens->ids[0];
            copy->ids[0] ^= 1;
            assert(cache.tokens->ids[0] == first);
            free_tokens(copy);
            check_validation(req, "{\"n\":2}");
            check_validation(req, "{\"n\":\"bad\",\"extra\":1}");
            check_validation(req, "{}");
            check_validation(req, "not json");

            // Messages, session, sampling, streaming and output limits do not
            // participate in system preparation; the suffix is always re-encoded.
            free(req->conversation_text);
            req->conversation_text = strdup("<|im_start|>user\nchanged tool result\n<|im_end|>\n<|im_start|>assistant\n");
            strcpy(req->session_id, "different");
            req->temperature = 0.25f; req->max_tokens = 5; req->stream = 0;
            check_prepare(&cache, req, 1);

            req->api_kind = kind == API_KIND_CHAT ? API_KIND_RESPONSES : API_KIND_CHAT;
            check_prepare(&cache, req, 0);
            req->api_kind = kind;
            check_prepare(&cache, req, 0);
            g_gpu_kv_seq++;
            check_prepare(&cache, req, 0);

            // Duplicate names still preserve declaration order and first-match
            // validation; schemas are associated by tool index, not by name.
            req->tool_count = 2;
            strcpy(req->tools[1].name, req->tools[0].name);
            strcpy(req->tools[1].description, "second definition");
            req->tools[1].parameters = strdup("false");
            req->tools[1].has_parameters = 1;
            check_prepare(&cache, req, 0);
            check_validation(req, "{\"n\":2}");
            ToolDef swap = req->tools[0];
            req->tools[0] = req->tools[1]; req->tools[1] = swap;
            check_prepare(&cache, req, 0);
            check_validation(req, "{\"n\":2}");
            swap = req->tools[0];
            req->tools[0] = req->tools[1]; req->tools[1] = swap;
            if (req->tools[1].prepared_schema) CFRelease(req->tools[1].prepared_schema);
            free(req->tools[1].parameters);
            memset(&req->tools[1], 0, sizeof(ToolDef));
            req->tool_count = 1;
            check_prepare(&cache, req, 0);

            req->reasoning_enabled = !req->reasoning_enabled;
            check_prepare(&cache, req, 0);
            req->tool_choice_mode = TOOL_CHOICE_NONE;
            check_prepare(&cache, req, 0);
            req->tool_choice_mode = TOOL_CHOICE_FORCED;
            strcpy(req->forced_tool_name, "record");
            check_prepare(&cache, req, 0);
            strcpy(req->forced_tool_name, "other");
            check_prepare(&cache, req, 0);
            strcpy(req->tools[0].description, "Changed description");
            check_prepare(&cache, req, 0);
            strcpy(req->tools[0].name, "other");
            check_prepare(&cache, req, 0);
            free(req->tools[0].parameters);
            req->tools[0].parameters = strdup("{\"type\":\"object\",\"pattern\":\"unsupported\"}");
            check_prepare(&cache, req, 0);
            check_validation(req, "{}");
            free(req->tools[0].parameters);
            req->tools[0].parameters = strdup("{ \"pattern\":\"unsupported\", \"type\":\"object\" }");
            check_prepare(&cache, req, 0); // Equivalent schema, different rendered bytes.
            check_validation(req, "{}");
            free(req->tools[0].parameters);
            req->tools[0].parameters = strdup("not a schema");
            check_prepare(&cache, req, 0);
            check_validation(req, "{}");
            req->tools[0].has_parameters = 0;
            check_prepare(&cache, req, 0);
            free(req->system_prompt);
            req->system_prompt = strdup("different system");
            check_prepare(&cache, req, 0);
            strcpy(req->model, "other-model");
            check_prepare(&cache, req, 0);
            g_cfg.bits++;
            check_prepare(&cache, req, 0);
            strcpy(g_flashchat_model_path, kind == API_KIND_CHAT ? "model-a" : "model-b");
            check_prepare(&cache, req, 0);
            strcpy(g_flashchat_weights_dir, kind == API_KIND_CHAT ? "weights-a" : "weights-b");
            check_prepare(&cache, req, 0);
            g_tokenizer_generation++;
            check_prepare(&cache, req, 0);

            // Replacing a same-length file invalidates even an overridden default.
            FILE *f = fopen(g_custom_system_prompt_path, "w");
            assert(f); fputs("first", f); fclose(f);
            check_prepare(&cache, req, 0);
            f = fopen(g_custom_system_prompt_path, "w");
            assert(f); fputs("other", f); fclose(f);
            check_prepare(&cache, req, 0);
            free(req->system_prompt); req->system_prompt = NULL;
            check_prepare(&cache, req, 0);
            assert(unlink(g_custom_system_prompt_path) == 0);
            check_prepare(&cache, req, 0);

            // Prefix token copies belong to the caller, and arbitrary joins fall back.
            assert(prepared_prompt_can_split(cache.prefix, req->conversation_text));
            check_tokens(&cache, "\nordinary text");
            assert(!prepared_prompt_can_split(cache.prefix, "\nordinary text"));
            check_tokens(&cache, "");
            g_tokenizer.added[2] = (bpe_added_token){strdup("\n<|im_start|>"), 13, 302};
            g_tokenizer.num_added = 3;
            g_tokenizer_generation++;
            check_prepare(&cache, req, 0);
            assert(!prepared_prompt_can_split(cache.prefix, req->conversation_text));
            free(g_tokenizer.added[2].str);
            g_tokenizer.num_added = 2;
            g_tokenizer_generation++;
            check_prepare(&cache, req, 0);

            // Oversized keys/schemas bypass retention without evicting the last entry.
            CFDataRef saved_key = cache.key;
            char *old_system = req->system_prompt;
            req->system_prompt = malloc(PREPARED_KEY_MAX_BYTES + 2);
            memset(req->system_prompt, 'x', PREPARED_KEY_MAX_BYTES + 1);
            req->system_prompt[PREPARED_KEY_MAX_BYTES + 1] = 0;
            PreparedPrompt scratch = {0};
            int hit = -1;
            assert(prepare_request_prompt(&cache, &scratch, req, &hit) == &scratch && hit == 0);
            assert(cache.key == saved_key && !scratch.key);
            assert(!req->tools[0].schema_prepared && !req->tools[0].prepared_schema);
            check_tokens(&scratch, req->conversation_text);
            prepared_prompt_free(&scratch);
            free(req->system_prompt); req->system_prompt = old_system;
            check_prepare(&cache, req, 1);
            char *old_schema = req->tools[0].parameters;
            req->tools[0].parameters = malloc(PREPARED_SCHEMA_MAX_BYTES + 2);
            memset(req->tools[0].parameters, ' ', PREPARED_SCHEMA_MAX_BYTES + 1);
            req->tools[0].parameters[PREPARED_SCHEMA_MAX_BYTES + 1] = 0;
            assert(prepare_request_prompt(&cache, &scratch, req, &hit) == &scratch && hit == 0);
            assert(cache.key == saved_key && !scratch.key);
            prepared_prompt_free(&scratch);
            free(req->tools[0].parameters); req->tools[0].parameters = old_schema;
            check_prepare(&cache, req, 1);
            req->used_snapshot = 1;
            PromptTokens *continuation = tokenize_request_prompt(req, "fixture", &cache);
            PromptTokens *suffix = encode_prompt_text_to_tokens(req->conversation_text);
            assert(continuation && suffix && continuation->count == suffix->count);
            assert(!memcmp(continuation->ids, suffix->ids, suffix->count * sizeof(uint32_t)));
            free_tokens(continuation); free_tokens(suffix);
            req->used_snapshot = 0;
            req->tools[0].has_parameters = 1;
            free(req->tools[0].parameters);
            req->tools[0].parameters = strdup("false");
            check_prepare(&cache, req, 0);
            // Request schema ownership survives eviction and autorelease drainage.
            @autoreleasepool {
                assert(req->tools[0].schema_prepared);
                prepared_prompt_free(&cache);
            }
            check_validation(req, "{}");
            api_request_free(req); free(req);
            prepared_prompt_free(&cache);
        }
        bpe_free(&g_tokenizer);
        assert(rmdir(temp) == 0);
        puts("PASS prepared prompts: exact bytes/tokens/validation, invalidation, bounded retention, unsafe joins, cleanup");
    }
    return 0;
}
