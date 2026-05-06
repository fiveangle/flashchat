#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

// ============================================================================
// Compile-time maximums
// ============================================================================

#define MAX_HIDDEN_DIM              4096
#define MAX_NUM_LAYERS              64
#define MAX_NUM_EXPERTS             512
#define MAX_NUM_ATTN_HEADS          64
#define MAX_NUM_KV_HEADS            8
#define MAX_HEAD_DIM                256
#define MAX_VOCAB_SIZE              300000
#define MAX_MOE_INTERMEDIATE        1024
#define MAX_LINEAR_NUM_V_HEADS      64
#define MAX_LINEAR_NUM_K_HEADS      16
#define MAX_LINEAR_KEY_DIM          128
#define MAX_LINEAR_VALUE_DIM        128
#define MAX_FULL_ATTN_LAYERS        16
#define MAX_LINEAR_LAYERS           48
#define MAX_SEQ_LEN                 1048576
#define MAX_K                       8

// ============================================================================
// Model configuration struct
// ============================================================================

typedef struct {
    char model_id[64];
    char model_name[64];
    char hf_repo[128];

    int hidden_dim;
    int num_layers;
    int num_attn_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rms_norm_eps;

    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate;
    int shared_intermediate;
    int full_attn_interval;
    int num_full_attn_layers;
    int num_linear_layers;

    int group_size;
    int bits;

    int linear_num_v_heads;
    int linear_num_k_heads;
    int linear_key_dim;
    int linear_value_dim;
    int linear_conv_kernel_dim;

    int linear_total_key;
    int linear_total_value;
    int linear_conv_dim;

    float rope_theta;
    float partial_rotary;
    int rotary_dim;

    int expert_size;
    int gate_w_off;
    int gate_s_off;
    int gate_b_off;
    int up_w_off;
    int up_s_off;
    int up_b_off;
    int down_w_off;
    int down_s_off;
    int down_b_off;

    int gate_w_size;
    int gate_s_size;
    int gate_b_size;
    int up_w_size;
    int up_s_size;
    int up_b_size;
    int down_w_size;
    int down_s_size;
    int down_b_size;

    int eos_token_1;
    int eos_token_2;
    int think_start_token;
    int think_end_token;

    char extract_weights_script[128];
    char repack_experts_script[128];
} ModelConfig;

// ============================================================================
// JSON helpers
// ============================================================================

static const char *json_find_key(const char *json, const char *key) {
    size_t key_len = strlen(key);
    const char *p = json;
    while ((p = strstr(p, key)) != NULL) {
        if (p > json && p[-1] == '"' && p[key_len] == '"') {
            const char *colon = p + key_len + 1;
            while (*colon && (*colon == ' ' || *colon == '\t' || *colon == '\n' || *colon == '\r')) colon++;
            if (*colon == ':') return colon + 1;
        }
        p += key_len;
    }
    return NULL;
}

static int json_parse_int(const char *p, int *out) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':')) p++;
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) return -1;
    *out = (int)v;
    return 0;
}

static int json_parse_float(const char *p, float *out) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':')) p++;
    char *end;
    double v = strtod(p, &end);
    if (end == p) return -1;
    *out = (float)v;
    return 0;
}

static int json_parse_string(const char *p, char *out, size_t out_len) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':')) p++;
    if (*p != '"') return -1;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < out_len - 1) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return 0;
}

// ============================================================================
// Config loader
// ============================================================================

static const char *resolve_model_config_path(void) {
    const char *env_path = getenv("FLASHCHAT_MODEL_CONFIG");
    if (env_path && env_path[0]) return env_path;
    if (access("assets/model_configs.json", R_OK) == 0) return "assets/model_configs.json";
    return "../assets/model_configs.json";
}

static int load_default_model_id(const char *json_path, char *out, size_t out_len) {
    FILE *f = fopen(json_path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(size + 1);
    if (!json) { fclose(f); return -1; }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    const char *p = json_find_key(json, "default_model");
    int rc = -1;
    if (p && json_parse_string(p, out, out_len) == 0 && out[0]) {
        rc = 0;
    }
    free(json);
    return rc;
}

static int load_model_config(const char *json_path, const char *model_id, ModelConfig *cfg) {
    FILE *f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", json_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(size + 1);
    if (!json) { fclose(f); return -1; }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    char model_key[80];
    snprintf(model_key, sizeof(model_key), "\"%s\"", model_id);
    const char *model_start = strstr(json, model_key);
    if (!model_start) {
        fprintf(stderr, "ERROR: Model '%s' not found in %s\n", model_id, json_path);
        free(json);
        return -1;
    }

    const char *p;
#define CFG_INT(field, key_name) \
    p = json_find_key(model_start, key_name); \
    if (p && json_parse_int(p, &cfg->field) != 0) { \
        fprintf(stderr, "WARNING: Failed to parse %s for model %s\n", key_name, model_id); \
    }

#define CFG_FLOAT(field, key_name) \
    p = json_find_key(model_start, key_name); \
    if (p && json_parse_float(p, &cfg->field) != 0) { \
        fprintf(stderr, "WARNING: Failed to parse %s for model %s\n", key_name, model_id); \
    }

#define CFG_STR(field, key_name) \
    p = json_find_key(model_start, key_name); \
    if (p && json_parse_string(p, cfg->field, sizeof(cfg->field)) != 0) { \
        fprintf(stderr, "WARNING: Failed to parse %s for model %s\n", key_name, model_id); \
    }

    CFG_STR(model_id, "id");
    if (strlen(cfg->model_id) == 0) strncpy(cfg->model_id, model_id, sizeof(cfg->model_id) - 1);
    CFG_STR(model_name, "name");
    CFG_STR(hf_repo, "hf_repo");

    CFG_INT(hidden_dim, "hidden_size");
    CFG_INT(num_layers, "num_hidden_layers");
    CFG_INT(num_attn_heads, "num_attention_heads");
    CFG_INT(num_kv_heads, "num_key_value_heads");
    CFG_INT(head_dim, "head_dim");
    CFG_INT(vocab_size, "vocab_size");
    CFG_FLOAT(rms_norm_eps, "rms_norm_eps");

    CFG_INT(num_experts, "num_experts");
    CFG_INT(num_experts_per_tok, "num_experts_per_tok");
    CFG_INT(moe_intermediate, "moe_intermediate_size");
    CFG_INT(shared_intermediate, "shared_expert_intermediate_size");
    CFG_INT(full_attn_interval, "full_attention_interval");

    p = json_find_key(model_start, "quantization");
    if (p) {
        const char *bits_p = json_find_key(p, "bits");
        if (bits_p) json_parse_int(bits_p, &cfg->bits);
        const char *gs_p = json_find_key(p, "group_size");
        if (gs_p) json_parse_int(gs_p, &cfg->group_size);
    }

    CFG_INT(linear_num_v_heads, "linear_num_value_heads");
    CFG_INT(linear_num_k_heads, "linear_num_key_heads");
    CFG_INT(linear_key_dim, "linear_key_head_dim");
    CFG_INT(linear_value_dim, "linear_value_head_dim");
    CFG_INT(linear_conv_kernel_dim, "linear_conv_kernel_dim");

    CFG_FLOAT(rope_theta, "rope_theta");
    CFG_FLOAT(partial_rotary, "partial_rotary_factor");

    p = json_find_key(model_start, "special_tokens");
    if (p) {
        const char *tp;
        tp = json_find_key(p, "eos_1"); if (tp) json_parse_int(tp, &cfg->eos_token_1);
        tp = json_find_key(p, "eos_2"); if (tp) json_parse_int(tp, &cfg->eos_token_2);
        tp = json_find_key(p, "think_start"); if (tp) json_parse_int(tp, &cfg->think_start_token);
        tp = json_find_key(p, "think_end"); if (tp) json_parse_int(tp, &cfg->think_end_token);
    }

    p = json_find_key(model_start, "scripts");
    if (p) {
        const char *sp;
        sp = json_find_key(p, "extract_weights"); if (sp) json_parse_string(sp, cfg->extract_weights_script, sizeof(cfg->extract_weights_script));
        sp = json_find_key(p, "repack_experts"); if (sp) json_parse_string(sp, cfg->repack_experts_script, sizeof(cfg->repack_experts_script));
    }

#undef CFG_INT
#undef CFG_FLOAT
#undef CFG_STR

    free(json);

    cfg->linear_total_key   = cfg->linear_num_k_heads * cfg->linear_key_dim;
    cfg->linear_total_value = cfg->linear_num_v_heads * cfg->linear_value_dim;
    cfg->linear_conv_dim    = cfg->linear_total_key * 2 + cfg->linear_total_value;
    cfg->rotary_dim         = (int)(cfg->head_dim * cfg->partial_rotary);
    cfg->num_full_attn_layers = cfg->num_layers / cfg->full_attn_interval;
    cfg->num_linear_layers    = cfg->num_layers - cfg->num_full_attn_layers;

    int gs = cfg->group_size;
    int gate_w = (cfg->moe_intermediate * cfg->hidden_dim) / 2;
    int gate_s = cfg->moe_intermediate * (cfg->hidden_dim / gs) * 2;
    int gate_b = gate_s;
    int up_w   = gate_w;
    int up_s   = gate_s;
    int up_b   = gate_s;
    int down_w = (cfg->hidden_dim * cfg->moe_intermediate) / 2;
    int down_s = cfg->hidden_dim * (cfg->moe_intermediate / gs) * 2;
    int down_b = down_s;
    cfg->expert_size = gate_w + gate_s + gate_b + up_w + up_s + up_b + down_w + down_s + down_b;
    cfg->gate_w_off = 0;
    cfg->gate_s_off = gate_w;
    cfg->gate_b_off = gate_w + gate_s;
    cfg->up_w_off   = gate_w + gate_s + gate_b;
    cfg->up_s_off   = cfg->up_w_off + up_w;
    cfg->up_b_off   = cfg->up_s_off + up_s;
    cfg->down_w_off = cfg->up_b_off + up_b;
    cfg->down_s_off = cfg->down_w_off + down_w;
    cfg->down_b_off = cfg->down_s_off + down_s;

    cfg->gate_w_size = gate_w;
    cfg->gate_s_size = gate_s;
    cfg->gate_b_size = gate_b;
    cfg->up_w_size   = up_w;
    cfg->up_s_size   = up_s;
    cfg->up_b_size   = up_b;
    cfg->down_w_size = down_w;
    cfg->down_s_size = down_s;
    cfg->down_b_size = down_b;

    if (cfg->hidden_dim > MAX_HIDDEN_DIM) {
        fprintf(stderr, "ERROR: hidden_dim %d exceeds MAX_HIDDEN_DIM %d\n", cfg->hidden_dim, MAX_HIDDEN_DIM);
        return -1;
    }
    if (cfg->num_layers > MAX_NUM_LAYERS) {
        fprintf(stderr, "ERROR: num_layers %d exceeds MAX_NUM_LAYERS %d\n", cfg->num_layers, MAX_NUM_LAYERS);
        return -1;
    }
    if (cfg->num_experts > MAX_NUM_EXPERTS) {
        fprintf(stderr, "ERROR: num_experts %d exceeds MAX_NUM_EXPERTS %d\n", cfg->num_experts, MAX_NUM_EXPERTS);
        return -1;
    }
    if (cfg->num_attn_heads > MAX_NUM_ATTN_HEADS) {
        fprintf(stderr, "ERROR: num_attn_heads %d exceeds MAX_NUM_ATTN_HEADS %d\n", cfg->num_attn_heads, MAX_NUM_ATTN_HEADS);
        return -1;
    }
    if (cfg->head_dim > MAX_HEAD_DIM) {
        fprintf(stderr, "ERROR: head_dim %d exceeds MAX_HEAD_DIM %d\n", cfg->head_dim, MAX_HEAD_DIM);
        return -1;
    }
    if (cfg->moe_intermediate > MAX_MOE_INTERMEDIATE) {
        fprintf(stderr, "ERROR: moe_intermediate %d exceeds MAX_MOE_INTERMEDIATE %d\n", cfg->moe_intermediate, MAX_MOE_INTERMEDIATE);
        return -1;
    }

    printf("[config] Loaded model: %s (%s)\n", cfg->model_name, cfg->model_id);
    printf("[config]  hidden_dim=%d, layers=%d, experts=%d, vocab=%d\n",
           cfg->hidden_dim, cfg->num_layers, cfg->num_experts, cfg->vocab_size);
    printf("[config]  expert_size=%d bytes\n", cfg->expert_size);
    return 0;
}

extern ModelConfig g_cfg;

#endif
