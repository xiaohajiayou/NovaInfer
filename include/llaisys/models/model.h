#ifndef LLAISYS_MODELS_MODEL_H
#define LLAISYS_MODELS_MODEL_H

#include "../runtime/infer_types.h"
#include "qwen2.h"

__C {
    typedef enum LlaisysModelType {
        LLAISYS_MODEL_TYPE_UNKNOWN = 0,
        LLAISYS_MODEL_TYPE_QWEN2 = 1,
        LLAISYS_MODEL_TYPE_MOCK = 2,
    } LlaisysModelType;

    struct LlaisysModelCreateParams {
        LlaisysModelType model_type;
        const void *meta;
        llaisysDeviceType_t device;
        int *device_ids;
        int ndevice;
    };

    struct LlaisysKvStateCreateParams {
        int32_t kv_cache_block_size;
        int32_t max_model_len;
        int32_t kv_cache_capacity_tokens;
    };

    struct LlaisysParallelContextCreateParams {
        int32_t tensor_parallel_size;
        int32_t rank;
        int32_t local_rank;
        const char *distributed_backend;
        const char *init_method;
        int *device_ids;
        int32_t ndevice;
    };

    struct LlaisysKvStats {
        int64_t capacity_tokens;
        int64_t used_tokens;
        int64_t free_tokens;
        int64_t peak_used_tokens;
    };

    typedef enum AttentionPhase {
        ATTENTION_PHASE_PREFILL = 0,
        ATTENTION_PHASE_DECODE = 1,
    } AttentionPhase;

    struct AttentionMetadata {
        int32_t phase;
        llaisysTensor_t cu_seqlens_q;
        llaisysTensor_t cu_seqlens_k;
        int32_t max_seqlen_q;
        int32_t max_seqlen_k;
        llaisysTensor_t slot_mapping;
        llaisysTensor_t block_tables;
        int32_t block_table_width;
        llaisysTensor_t cudnn_seq_lens_q;
        llaisysTensor_t cudnn_seq_lens_kv;
        llaisysTensor_t cudnn_page_table;
        llaisysTensor_t cudnn_qo_ragged_offset;
        int32_t cudnn_b_exec;
        int32_t cudnn_warmup_b;
    };

    struct ModelForwardInput {
        llaisysTensor_t input_ids;
        llaisysTensor_t pos_ids;
        llaisysTensor_t logits_indices;
        struct AttentionMetadata attention;
    };

    struct ModelForwardOutput {
        llaisysTensor_t logits;
    };

    struct SamplerInput {
        llaisysTensor_t logits;
        llaisysTensor_t temperatures;
        llaisysTensor_t top_ps;
        llaisysTensor_t top_ks;
        llaisysTensor_t seeds;
        llaisysTensor_t has_seeds;
    };

    struct SamplerOutput {
        llaisysTensor_t sampled_ids;
    };

    struct LlaisysModel;
    struct LlaisysKvState;
    struct LlaisysParallelContext;

    __export struct LlaisysModel *llaisysModelCreate(const struct LlaisysModelCreateParams *params);
    __export void llaisysModelDestroy(struct LlaisysModel *model);
    __export LlaisysModelType llaisysModelType(const struct LlaisysModel *model);

    __export struct LlaisysKvState *llaisysKvStateCreate(const struct LlaisysKvStateCreateParams *params);
    __export void llaisysKvStateDestroy(struct LlaisysKvState *kv_state);

    __export struct LlaisysParallelContext *llaisysParallelContextCreate(
        const struct LlaisysParallelContextCreateParams *params);
    __export void llaisysParallelContextDestroy(struct LlaisysParallelContext *parallel_context);
    __export int32_t llaisysModelBindParallelContext(struct LlaisysModel *model,
                                                     struct LlaisysParallelContext *parallel_context);

    __export void *llaisysModelWeights(struct LlaisysModel *model);
    __export int llaisysModelReplaceWeight(struct LlaisysModel *model,
                                           const char *field_name,
                                           int32_t layer_idx,
                                           llaisysTensor_t new_weight);

    __export int32_t llaisysModelForward(struct LlaisysModel *model,
                                         struct LlaisysKvState *kv_state,
                                         const struct ModelForwardInput *input,
                                         struct ModelForwardOutput *output);
    __export int32_t llaisysSamplerSample(const struct SamplerInput *input,
                                          struct SamplerOutput *output);

    __export int llaisysKvStateRequestFree(struct LlaisysKvState *kv_state, int64_t seq_id);
    __export int llaisysKvStateStats(struct LlaisysKvState *kv_state, struct LlaisysKvStats *out_stats);
    __export int llaisysKvStateResetPrefixCache(struct LlaisysKvState *kv_state);
}

#endif // LLAISYS_MODELS_MODEL_H
