#ifndef LLAISYS_RUNTIME_INFER_TYPES_H
#define LLAISYS_RUNTIME_INFER_TYPES_H

#include "../../llaisys.h"

__C {
    // SoA batch layout aligned with llama.cpp-style token list input.
    // Field groups:
    // 1) Common decode inputs (used by both Unified/SLOT and Paged/BLOCK).
    // 2) Paged/BLOCK explicit mapping inputs (required in BLOCK mode).
    struct LlaisysBatch {
        int32_t n_tokens;
        // ---- Common fields (Unified/SLOT required; Paged/BLOCK also consumes by default) ----
        int64_t *token;    // [n_tokens], token ids; used when embd == NULL
        float *embd;       // optional embedding path (reserved, currently unused)
        int64_t *pos;      // [n_tokens], optional absolute positions
        int32_t *n_seq_id; // [n_tokens], optional seq-id count per token
        int64_t **seq_id;  // [n_tokens][n_seq_id[i]], optional seq-id set per token
        int8_t *logits;    // [n_tokens], non-zero means keep logits for this token

        // ---- Paged/BLOCK explicit mapping fields (required in BLOCK mode) ----
        int32_t *slot_mapping;    // [n_tokens], physical KV slot index per token
        int32_t *context_lens;    // [n_batch_seq], valid context length per sequence row
        int64_t *batch_seq_ids;   // [n_batch_seq], seq id corresponding to each row
        int32_t *block_tables;    // [n_batch_seq * block_table_width], row-major, -1 padded
        int32_t n_batch_seq;      // row count for context_lens/batch_seq_ids/block_tables
        int32_t block_table_width; // block columns per row
    };

    __export struct LlaisysBatch llaisysBatchInit(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
    __export struct LlaisysBatch llaisysBatchGetOne(int64_t *token, int32_t n_tokens);
    __export void llaisysBatchFree(struct LlaisysBatch batch);
}

#endif // LLAISYS_RUNTIME_INFER_TYPES_H
