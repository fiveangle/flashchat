#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

static uint32_t request_sampling_random(void) { return UINT32_MAX; }
#define arc4random request_sampling_random
#define main flashchat_infer_main
#include "../metal_infer/infer.m"
#undef main
#undef arc4random

int main(void) {
    @autoreleasepool {
        kApiModelId = "sampling-fixture";
        g_cfg.vocab_size = 4;
        g_cfg.think_start_token = 2;
        g_cfg.think_end_token = 3;
        const float logits[] = {2.0f, 1.0f, -100.0f, -100.0f};
        const int counts[] = {1, 0, 0, 0};
        for (int kind = API_KIND_CHAT; kind <= API_KIND_RESPONSES; kind++) {
            for (int tools = 0; tools <= 1; tools++) {
                for (int forced = 0; forced <= 1; forced++) {
                    ApiRequest req;
                    api_request_init(&req, kind);
                    req.tool_count = tools;
                    req.tool_choice_mode = forced ? TOOL_CHOICE_FORCED : TOOL_CHOICE_AUTO;
                    req.reasoning_enabled = 0;
                    req.temperature = 1.0f;
                    req.top_p = 1.0f;
                    req.top_k = 2;
                    req.min_p = 0;
                    req.presence_penalty = 0;
                    req.repetition_penalty = 1;
                    // Fixed random draw selects the lower-ranked candidate;
                    // any hidden greedy override would return token 0 instead.
                    assert(sample_request_token(&req, logits, NULL) == 1);
                    req.temperature = 0;
                    assert(sample_request_token(&req, logits, NULL) == 0);
                    req.presence_penalty = 2;
                    assert(sample_request_token(&req, logits, counts) == 1);
                    req.presence_penalty = 0;
                    req.repetition_penalty = 3;
                    assert(sample_request_token(&req, logits, counts) == 1);
                }
            }
        }
        puts("PASS request sampling: temperature and penalties preserved with tools and forced calls");
    }
    return 0;
}
