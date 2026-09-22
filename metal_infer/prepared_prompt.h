// Private to infer.m. One inference owner; entries never outlive serve_loop.
#define PREPARED_KEY_MAX_BYTES (256 * 1024)
#define PREPARED_TEXT_MAX_BYTES (512 * 1024)
#define PREPARED_TOKEN_MAX_BYTES (2 * 1024 * 1024)
#define PREPARED_SCHEMA_MAX_BYTES (64 * 1024)

typedef struct {
    CFDataRef key;
    char *system;
    char *prefix;
    PromptTokens *tokens;
    PromptBuildInfo build_info;
    CFTypeRef schemas[MAX_TOOLS];
    int schemas_prepared;
} PreparedPrompt;

static void prepared_prompt_free(PreparedPrompt *p) {
    if (p->key) CFRelease(p->key);
    for (int i = 0; i < MAX_TOOLS; i++) {
        if (p->schemas[i]) CFRelease(p->schemas[i]);
    }
    free(p->system);
    free(p->prefix);
    if (p->tokens) { free(p->tokens->ids); free(p->tokens); }
    memset(p, 0, sizeof(*p));
}

// Length framing distinguishes null, empty, and adjacent strings without hashing.
static BOOL prepared_key_string(NSMutableData *key, const char *s) {
    uint64_t len = s ? strlen(s) : UINT64_MAX;
    size_t bytes = s ? (size_t)len : 0;
    if (bytes > PREPARED_KEY_MAX_BYTES ||
        key.length + sizeof(len) + bytes > PREPARED_KEY_MAX_BYTES) return NO;
    [key appendBytes:&len length:sizeof(len)];
    if (bytes) [key appendBytes:s length:bytes];
    return YES;
}

static NSData *prepared_prompt_key(const ApiRequest *req, const char *base) {
    NSMutableData *key = [NSMutableData data];
    // Model/tokenizer inputs are immutable during a server lifetime. A successful
    // tokenizer load advances its generation even if the allocator reuses addresses.
    uint64_t flags[] = {1 /* template/preparation revision */, g_tokenizer_generation,
        g_cfg.bits, GPU_KV_SEQ, req->api_kind, req->reasoning_enabled,
        req->tool_choice_mode, req->tool_count};
    [key appendBytes:flags length:sizeof(flags)];
    const char *strings[] = {g_flashchat_model_path, g_flashchat_weights_dir,
        kApiModelId, req->model, base, req->system_prompt, req->forced_tool_name};
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); i++) {
        if (!prepared_key_string(key, strings[i])) return nil;
    }
    size_t schema_bytes = 0;
    for (int i = 0; i < req->tool_count; i++) {
        const ToolDef *tool = &req->tools[i];
        schema_bytes += tool->parameters ? strlen(tool->parameters) : 0;
        if (schema_bytes > PREPARED_SCHEMA_MAX_BYTES) return nil;
        [key appendBytes:&tool->has_parameters length:sizeof(tool->has_parameters)];
        if (!prepared_key_string(key, tool->name) ||
            !prepared_key_string(key, tool->description) ||
            !prepared_key_string(key, tool->parameters)) return nil;
    }
    return key;
}

static void prepared_prompt_attach_schemas(const PreparedPrompt *p, ApiRequest *req) {
    if (!p->schemas_prepared) return;
    for (int i = 0; i < req->tool_count; i++) {
        ToolDef *tool = &req->tools[i];
        if (tool->prepared_schema) CFRelease(tool->prepared_schema);
        tool->prepared_schema = p->schemas[i] ? CFRetain(p->schemas[i]) : NULL;
        tool->schema_prepared = 1;
    }
}

// Oversized/failed preparations never replace a usable entry. The caller owns
// scratch until the request is finished; request schema references own a retain.
static const PreparedPrompt *prepare_request_prompt(PreparedPrompt *cache,
                                                     PreparedPrompt *scratch,
                                                     ApiRequest *req, int *hit) {
    if (hit) *hit = 0;
    for (int i = 0; i < req->tool_count; i++) {
        if (req->tools[i].prepared_schema) CFRelease(req->tools[i].prepared_schema);
        req->tools[i].prepared_schema = NULL;
        req->tools[i].schema_prepared = 0;
    }
    @autoreleasepool {
        init_tokenizer();
        char *base = load_system_prompt();
        NSData *key = prepared_prompt_key(req, base);
        if (g_tokenizer_loaded && key && cache->key &&
            [key isEqualToData:(__bridge NSData *)cache->key]) {
            free(base);
            prepared_prompt_attach_schemas(cache, req);
            if (hit) *hit = 1;
            return cache;
        }
        scratch->system = build_system_prompt_with_base(req, base, &scratch->build_info);
        free(base);
        if (!scratch->system) return NULL;
        size_t size = strlen(scratch->system) + 64;
        scratch->prefix = malloc(size);
        if (!scratch->prefix) return NULL;
        snprintf(scratch->prefix, size, "<|im_start|>system\n%s<|im_end|>\n", scratch->system);
        scratch->tokens = encode_prompt_text_to_tokens(scratch->prefix);
        if (!scratch->tokens) return NULL;
        size_t token_bytes = (size_t)scratch->tokens->count * sizeof(uint32_t);
        if (!key || size > PREPARED_TEXT_MAX_BYTES || token_bytes > PREPARED_TOKEN_MAX_BYTES)
            return scratch;

        // The tokenizer allocates for worst-case expansion. Retain only used IDs.
        uint32_t *ids = malloc(token_bytes ?: 1);
        if (!ids) return scratch;
        memcpy(ids, scratch->tokens->ids, token_bytes);
        free(scratch->tokens->ids);
        scratch->tokens->ids = ids;
        for (int i = 0; i < req->tool_count; i++) {
            const ToolDef *tool = &req->tools[i];
            if (!tool->has_parameters || !tool->parameters || !tool->parameters[0]) continue;
            NSData *data = [NSData dataWithBytes:tool->parameters length:strlen(tool->parameters)];
            id schema = [NSJSONSerialization JSONObjectWithData:data
                                         options:NSJSONReadingFragmentsAllowed error:NULL];
            scratch->schemas[i] = CFBridgingRetain(schema);
        }
        scratch->schemas_prepared = 1;
        scratch->key = CFBridgingRetain([key copy]);
        prepared_prompt_free(cache);
        *cache = *scratch;
        memset(scratch, 0, sizeof(*scratch));
        prepared_prompt_attach_schemas(cache, req);
        return cache;
    }
}

// bpe_encode splits at added tokens before ordinary text pretokenization. A split
// is safe only if the suffix starts at one and no added token crosses the join.
static BOOL prepared_prompt_can_split(const char *prefix, const char *suffix) {
    size_t left = strlen(prefix), right = strlen(suffix);
    BOOL boundary = NO;
    for (uint32_t i = 0; i < g_tokenizer.num_added; i++) {
        const bpe_added_token *token = &g_tokenizer.added[i];
        size_t len = token->len;
        if (len && len <= right && !memcmp(suffix, token->str, len)) boundary = YES;
        for (size_t n = 1; n < len && n <= left; n++) {
            if (len - n <= right && !memcmp(prefix + left - n, token->str, n) &&
                !memcmp(suffix, token->str + n, len - n)) return NO;
        }
    }
    return boundary;
}

static PromptTokens *prepared_prompt_tokens(const PreparedPrompt *p, const char *suffix) {
    if (prepared_prompt_can_split(p->prefix, suffix)) {
        PromptTokens *tail = encode_prompt_text_to_tokens(suffix);
        if (!tail) return NULL;
        size_t count = (size_t)p->tokens->count + tail->count;
        uint32_t *ids = malloc(count * sizeof(uint32_t));
        if (!ids) { free(tail->ids); free(tail); return NULL; }
        memcpy(ids, p->tokens->ids, p->tokens->count * sizeof(uint32_t));
        memcpy(ids + p->tokens->count, tail->ids, tail->count * sizeof(uint32_t));
        free(tail->ids);
        tail->ids = ids;
        tail->count = (int)count;
        return tail;
    }
    size_t size = strlen(p->prefix) + strlen(suffix) + 1;
    char *text = malloc(size);
    if (!text) return NULL;
    snprintf(text, size, "%s%s", p->prefix, suffix);
    PromptTokens *result = encode_prompt_text_to_tokens(text);
    free(text);
    return result;
}
