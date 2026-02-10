#ifndef LLAISYS_RUNTIME_INFER_TYPES_H
#define LLAISYS_RUNTIME_INFER_TYPES_H

#include "../../llaisys.h"

__C {
    // SoA batch layout aligned with llama.cpp-style token list input.
    struct LlaisysBatch {
        int32_t n_tokens;
        int64_t *token;    // used when embd == NULL
        float *embd;       // optional embedding path (reserved)
        int64_t *pos;      // optional position ids
        int32_t *n_seq_id; // optional sequence-id count per token
        int64_t **seq_id;  // optional seq-id sets per token
        int8_t *logits;    // non-zero means keep logits for this token
    };

    __export struct LlaisysBatch llaisysBatchInit(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
    __export struct LlaisysBatch llaisysBatchGetOne(int64_t *token, int32_t n_tokens);
    __export void llaisysBatchFree(struct LlaisysBatch batch);
}

#endif // LLAISYS_RUNTIME_INFER_TYPES_H
