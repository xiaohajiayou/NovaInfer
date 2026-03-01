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

    typedef enum LlaisysKvCacheLayout {
        LLAISYS_KV_CACHE_LAYOUT_SLOT = 0,
        LLAISYS_KV_CACHE_LAYOUT_BLOCK = 1,
    } LlaisysKvCacheLayout;

    struct LlaisysModelCreateParams {
        LlaisysModelType model_type;
        const void *meta; // points to model-specific meta struct
        llaisysDeviceType_t device;
        int *device_ids;
        int ndevice;
    };

    struct LlaisysRuntimeCreateParams {
        int32_t kv_cache_layout;          // LlaisysKvCacheLayout, <0 means default(BLOCK)
        int32_t kv_cache_block_size;      // <=0 means default(16)
        int32_t max_model_len;            // <=0 means use model meta.maxseq
        int32_t kv_cache_capacity_tokens; // <=0 means use max_model_len
    };

    struct LlaisysKvStats {
        int64_t capacity_tokens;
        int64_t used_tokens;
        int64_t free_tokens;
        int64_t peak_used_tokens;
    };

    typedef enum AttentionMode {
        ATTENTION_MODE_SLOT = 0,
        ATTENTION_MODE_BLOCK = 1,
    } AttentionMode;

    struct AttentionMetadata {
        int32_t mode; // AttentionMode
        llaisysTensor_t seq_ids;       // [n_tokens], i64, host metadata
        llaisysTensor_t q_seq_rows;    // [n_tokens], i32, BLOCK metadata (device checked by model)
        llaisysTensor_t q_pos;         // [n_tokens], i32, BLOCK metadata (device checked by model)
        llaisysTensor_t slot_mapping;  // [n_tokens], i32, BLOCK metadata (device checked by model)
        llaisysTensor_t context_lens;  // [n_batch_seq], i32, BLOCK metadata (device checked by model)
        llaisysTensor_t batch_seq_ids; // [n_batch_seq], i64, host metadata (BLOCK)
        llaisysTensor_t block_tables;  // [n_batch_seq * block_table_width], i32, BLOCK metadata (device checked by model)
        llaisysTensor_t pos_ids_host;  // [n_tokens], i64, optional host mirror for pos_ids
        int32_t block_table_width;
    };

    struct ModelForwardInput {
        llaisysTensor_t input_ids;   // [n_tokens], i64
        llaisysTensor_t pos_ids;     // [n_tokens], i64 (BLOCK required)
        llaisysTensor_t logits_mask; // [n_tokens], i8
        struct AttentionMetadata attention;
    };

    struct ModelForwardOutput {
        llaisysTensor_t output_ids; // [n_outputs], i64
        llaisysTensor_t logits;     // optional [n_outputs, vocab], f32
        int32_t n_outputs;          // number of valid rows in output_ids/logits
    };

    struct SamplerInput {
        llaisysTensor_t logits;       // optional [n_outputs, vocab], f32
        llaisysTensor_t output_ids;   // [n_outputs], i64; selects rows from logits
        llaisysTensor_t temperatures; // [n_outputs], f32
        llaisysTensor_t top_ps;       // [n_outputs], f32
        llaisysTensor_t top_ks;       // [n_outputs], i32
        llaisysTensor_t seeds;        // [n_outputs], i64
        llaisysTensor_t has_seeds;    // [n_outputs], i8
    };

    struct SamplerOutput {
        llaisysTensor_t sampled_ids; // [n_outputs], i64
    };

    struct LlaisysModel;
    struct LlaisysRuntime;

    __export struct LlaisysModel *llaisysModelCreate(const struct LlaisysModelCreateParams *params,
                                                     struct LlaisysRuntime *runtime);
    __export void llaisysModelDestroy(struct LlaisysModel *model);
    __export LlaisysModelType llaisysModelType(const struct LlaisysModel *model);
    __export struct LlaisysRuntime *llaisysRuntimeCreate(const struct LlaisysRuntimeCreateParams *params);
    __export void llaisysRuntimeDestroy(struct LlaisysRuntime *runtime);

    __export void *llaisysModelWeights(struct LlaisysModel *model);
    // Safely replace one weight slot:
    //  0 success
    // -1 invalid input
    // -2 unsupported model type
    // -3 unknown field name
    // -4 invalid layer index for per-layer field
    __export int llaisysModelReplaceWeight(struct LlaisysModel *model,
                                           const char *field_name,
                                           int32_t layer_idx,
                                           llaisysTensor_t new_weight);

    // Return codes:
    //  0  success
    //  1  no KV slot / out of context window
    //  2  aborted (reserved)
    // -1  invalid input
    // < -1 internal error
    __export int32_t llaisysModelForward(struct LlaisysModel *model,
                                         const struct ModelForwardInput *input,
                                         struct ModelForwardOutput *output);
    __export int32_t llaisysSamplerSample(const struct SamplerInput *input,
                                          struct SamplerOutput *output);

    // KV status (mirrors runtime::kv_cache::KvStatus; exported as int in C API):
    // 0: OK
    // 1: OOM_SLOT
    // 2: INVALID_SEQ
    // 3: INVALID_POS
    // 4: EMPTY_RANGE
    // 5: INTERNAL_ERROR
    __export int llaisysRuntimeKvSeqCp(struct LlaisysRuntime *runtime, int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1);
    __export int llaisysRuntimeKvSeqRm(struct LlaisysRuntime *runtime, int64_t seq_id, int64_t p0, int64_t p1);
    __export int llaisysRuntimeKvSeqAdd(struct LlaisysRuntime *runtime, int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);
    __export int llaisysRuntimeKvSeqKeep(struct LlaisysRuntime *runtime, int64_t seq_id);
    __export int64_t llaisysRuntimeKvSeqPosMax(struct LlaisysRuntime *runtime, int64_t seq_id);
    // Free all KV entries that belong to one request/sequence id.
    // Return code follows KV status mapping above.
    __export int llaisysRuntimeRequestFree(struct LlaisysRuntime *runtime, int64_t seq_id);
    // 0 success, <0 invalid input/internal error.
    __export int llaisysRuntimeKvStats(struct LlaisysRuntime *runtime, struct LlaisysKvStats *out_stats);
    // Reset prefix-cache related metadata (no-op when prefix cache disabled/not implemented).
    // Return code follows KV status mapping above.
    __export int llaisysRuntimeKvResetPrefixCache(struct LlaisysRuntime *runtime);
}

#endif // LLAISYS_MODELS_MODEL_H
