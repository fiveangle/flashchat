#define main flashchat_infer_main
#include "../metal_infer/infer.m"
#undef main
#include <assert.h>

int main(void) {
    @autoreleasepool {
        g_progress = (server_progress_t){.cached_tokens = 50, .prompt_tokens = 20};
        server_progress_prefill(62, 1, 1, 2, 2);
        assert(g_progress.prefill_done == 12 && g_progress.context_used == 62);
        server_progress_prefill(50, 0, 0, 0, 0);
        assert(g_progress.prefill_done == 0 && g_progress.context_used == 50);
        ConversationCache cache = {0};
        uint32_t input[] = {10, 20, 30, 40, 50};
        PromptTokens prompt = {.ids = input, .count = 5};
        assert(conversation_cache_begin(&cache, &prompt, "one", 8) == 0);
        cache.live_valid = 1;
        cache.checkpoint_count = 3;
        uint32_t next[] = {10, 20, 30, 40, 50, 60};
        PromptTokens extension = {.ids = next, .count = 6};
        int restore = -1;
        assert(conversation_cache_match(&cache, &extension, "one", &restore) == 5 && !restore);
        next[4] = 99; // Client rewrites the latest assistant turn.
        assert(conversation_cache_match(&cache, &extension, "one", &restore) == 3 && restore);
        next[1] = 99; // Earlier history, system, or tools changed.
        assert(conversation_cache_match(&cache, &extension, "one", &restore) == 0);
        next[1] = 20;
        assert(conversation_cache_match(&cache, &extension, "two", &restore) == 0);
        extension.count = 2; // A shortened prompt cannot restore a longer checkpoint.
        assert(conversation_cache_match(&cache, &extension, "one", &restore) == 0);
        extension.count = 6;
        next[4] = 50;
        cache.live_valid = 0; // Speculation/repair: only the faithful checkpoint is reusable.
        assert(conversation_cache_match(&cache, &extension, "one", &restore) == 3 && restore);

        g_cfg.num_layers = 2;
        g_cfg.num_linear_layers = 1;
        g_cfg.full_attn_interval = 2;
        g_cfg.linear_conv_kernel_dim = 2;
        g_cfg.linear_conv_dim = 2;
        g_cfg.linear_num_v_heads = 1;
        g_cfg.linear_value_dim = 2;
        g_cfg.linear_key_dim = 2;
        float conv[] = {1, 2}, ssm[] = {3, 4, 5, 6};
        LinearAttnState recurrent = {.conv_state = conv, .ssm_state = ssm};
        void *states[] = {&recurrent, NULL};
        float attention[] = {7, 8, 9, 10, 11};
        KVCache kv = {.len = 3, .k_cache = attention};
        KVCache *caches[] = {NULL, &kv};
        assert(conversation_checkpoint_capture(&cache, 3, states, caches) == 0);
        conv[0] = 100; ssm[3] = 200; kv.len = 5;
        assert(conversation_checkpoint_restore(&cache, states, caches) == 0);
        assert(conv[0] == 1 && ssm[3] == 6 && kv.len == 3);
        assert(attention[0] == 7 && attention[4] == 11); // KV bytes neither copied nor erased.
        kv.len = 2;
        assert(conversation_checkpoint_restore(&cache, states, caches) == -1);
        kv.len = 3;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        assert(device && "Metal device required for checkpoint coherency test");
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal; kernel void noop() {}"
                                                     options:nil error:&error];
        assert(library);
        MetalCtx metal = {0};
        metal.delta_net_step = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"noop"] error:&error];
        assert(metal.delta_net_step);
        metal.buf_delta_state[0] = [device newBufferWithLength:serve_gpu_delta_snapshot_size() options:MTLResourceStorageModeShared];
        metal.buf_conv_state[0] = [device newBufferWithLength:serve_gpu_conv_snapshot_size() options:MTLResourceStorageModeShared];
        assert(metal.buf_delta_state[0] && metal.buf_conv_state[0]);
        g_metal = &metal;
        memset([metal.buf_delta_state[0] contents], 0x35, serve_gpu_delta_snapshot_size());
        memset([metal.buf_conv_state[0] contents], 0x52, serve_gpu_conv_snapshot_size());
        assert(conversation_checkpoint_capture(&cache, 3, states, caches) == 0);
        memset([metal.buf_delta_state[0] contents], 0, serve_gpu_delta_snapshot_size());
        memset([metal.buf_conv_state[0] contents], 0, serve_gpu_conv_snapshot_size());
        conv[1] = 300; kv.len = 5;
        assert(conversation_checkpoint_restore(&cache, states, caches) == 0);
        assert(conv[1] == 2 && kv.len == 3);
        assert(!memcmp([metal.buf_delta_state[0] contents], cache.gpu_delta[0], serve_gpu_delta_snapshot_size()));
        assert(!memcmp([metal.buf_conv_state[0] contents], cache.gpu_conv[0], serve_gpu_conv_snapshot_size()));
        conversation_cache_invalidate(&cache); // Cancellation/reset must invalidate both choices.
        assert(conversation_cache_match(&cache, &extension, "one", &restore) == 0);
        conversation_cache_free(&cache);
        g_metal = NULL;
        puts("PASS: exact prefix selection, changed/shortened input, session changes, CPU/GPU recurrent rollback, attention lengths, invalidation");
    }
    return 0;
}
