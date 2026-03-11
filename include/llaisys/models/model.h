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

    struct LlaisysParallelInitParams {
        int32_t tensor_parallel_size;     // >=1
        int32_t pipeline_parallel_size;   // >=1
        int32_t world_size;               // >=1
        int32_t rank;                     // [0, world_size)
        int32_t local_rank;               // >=0
        const char *distributed_executor_backend; // optional, e.g. "uni"
        const char *distributed_backend;          // optional, e.g. "nccl"
        const char *master_addr;                  // optional
        int32_t master_port;                      // optional
        int32_t node_rank;                        // optional
        int32_t nnodes;                           // optional
        const char *init_method;                  // optional
        const char *tp_group_name;                // optional
        int32_t use_single_process_tp;            // bool(0/1)
        int *device_ids;                          // logical CUDA device ids
        int32_t ndevice;                          // number of device_ids
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

    typedef enum AttentionPhase {
        ATTENTION_PHASE_PREFILL = 0,
        ATTENTION_PHASE_DECODE = 1,
    } AttentionPhase;

    struct AttentionMetadata {
        int32_t mode; // AttentionMode
        int32_t phase; // AttentionPhase
        llaisysTensor_t seq_ids; // [n_tokens], i64, SLOT metadata
        llaisysTensor_t pos_ids_host;    // [n_tokens], i64, SLOT metadata
        llaisysTensor_t cu_seqlens_q;    // [n_batch_seq + 1], i32, BLOCK required
        llaisysTensor_t cu_seqlens_k;    // [n_batch_seq + 1], i32, BLOCK required
        int32_t max_seqlen_q;            // BLOCK required
        int32_t max_seqlen_k;            // BLOCK required
        llaisysTensor_t slot_mapping;    // [n_tokens], i32, BLOCK required
        llaisysTensor_t block_tables;    // [n_batch_seq * block_table_width], i32, BLOCK metadata
        int32_t block_table_width;
        // CUDNN-only BLOCK metadata. These fields are ignored by native BLOCK/SLOT paths.
        llaisysTensor_t cudnn_seq_lens_q;   // [cudnn_b_exec], i32
        llaisysTensor_t cudnn_seq_lens_kv;  // [cudnn_b_exec], i32
        llaisysTensor_t cudnn_page_table;   // [cudnn_b_exec * block_table_width], i32
        llaisysTensor_t cudnn_qo_ragged_offset; // [cudnn_b_exec + 1], i32, prefill-only
        int32_t cudnn_b_exec;               // rows in cudnn_* metadata
    };

    struct ModelForwardInput {
        llaisysTensor_t input_ids;      // [n_tokens], i64
        llaisysTensor_t pos_ids;        // [n_tokens], i64, required
        llaisysTensor_t logits_indices; // [n_logits], i64, required
        struct AttentionMetadata attention;
    };

    struct ModelForwardOutput {
        llaisysTensor_t logits; // [n_logits, vocab], f32
    };

    struct SamplerInput {
        llaisysTensor_t logits;       // [n_outputs, vocab], f32
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

    __export struct LlaisysModel *llaisysModelCreate(const struct LlaisysModelCreateParams *params);
    __export void llaisysModelDestroy(struct LlaisysModel *model);
    __export LlaisysModelType llaisysModelType(const struct LlaisysModel *model);
    __export struct LlaisysRuntime *llaisysRuntimeCreate(const struct LlaisysRuntimeCreateParams *params);
    __export void llaisysRuntimeDestroy(struct LlaisysRuntime *runtime);
    // Return: 0 success, -1 invalid input, -2 unsupported config.
    __export int32_t llaisysRuntimeParallelInit(struct LlaisysRuntime *runtime,
                                                const struct LlaisysParallelInitParams *params);
    // Return the runtime compute stream bound to (device_type, device_id) in current thread.
    // nullptr on invalid input or failure.
    __export llaisysStream_t llaisysRuntimeGetComputeStream(struct LlaisysRuntime *runtime,
                                                            llaisysDeviceType_t device_type,
                                                            int device_id);

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
                                         struct LlaisysRuntime *runtime,
                                         const struct ModelForwardInput *input,
                                         struct ModelForwardOutput *output);
    // Build BLOCK attention runtime metadata on target device.
    // Return: 0 success, -1 invalid input, -2 internal error.
    __export int32_t llaisysRuntimeBuildBlockAttentionMetadata(
        struct LlaisysRuntime *runtime,
        llaisysTensor_t req_num_scheduled_tokens, // [n_batch_seq], i32
        llaisysTensor_t req_num_computed_tokens,  // [n_batch_seq], i32
        llaisysTensor_t block_tables,             // [n_batch_seq * block_table_width], i32
        int32_t block_table_width,
        int32_t ntoken,
        llaisysTensor_t query_start_loc, // [n_batch_seq + 1], i32 (out)
        llaisysTensor_t seq_lens,        // [n_batch_seq], i32 (out)
        llaisysTensor_t slot_mapping);   // [ntoken], i32 (out)
    __export int32_t llaisysSamplerSample(struct LlaisysRuntime *runtime,
                                          const struct SamplerInput *input,
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
