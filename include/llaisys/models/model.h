#ifndef LLAISYS_MODELS_MODEL_H
#define LLAISYS_MODELS_MODEL_H

#include "../runtime/infer_types.h"
#include "qwen2.h"

__C {
    typedef enum LlaisysModelType {
        LLAISYS_MODEL_TYPE_UNKNOWN = 0,
        LLAISYS_MODEL_TYPE_QWEN2 = 1,
    } LlaisysModelType;

    struct LlaisysModelCreateParams {
        LlaisysModelType model_type;
        const void *meta; // points to model-specific meta struct
        llaisysDeviceType_t device;
        int *device_ids;
        int ndevice;
    };

    struct LlaisysModel;

    __export struct LlaisysModel *llaisysModelCreate(const struct LlaisysModelCreateParams *params);
    __export void llaisysModelDestroy(struct LlaisysModel *model);
    __export LlaisysModelType llaisysModelType(const struct LlaisysModel *model);

    __export void *llaisysModelWeights(struct LlaisysModel *model);

    // Return codes:
    //  0  success
    //  1  no KV slot / out of context window
    //  2  aborted (reserved)
    // -1  invalid input
    // < -1 internal error
    __export int32_t llaisysModelDecode(struct LlaisysModel *model, struct LlaisysBatch batch);

    __export float *llaisysModelGetLogits(struct LlaisysModel *model);
    __export float *llaisysModelGetLogitsIth(struct LlaisysModel *model, int32_t i);
    __export int32_t llaisysModelNOutputs(struct LlaisysModel *model);
    __export const int32_t *llaisysModelOutputIds(struct LlaisysModel *model);

    // KV status:
    // 0 OK, 1 OOM_SLOT, 2 INVALID_SEQ, 3 INVALID_POS, 4 EMPTY_RANGE, 5 INTERNAL_ERROR
    __export int llaisysModelKvSeqCp(struct LlaisysModel *model, int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1);
    __export int llaisysModelKvSeqRm(struct LlaisysModel *model, int64_t seq_id, int64_t p0, int64_t p1);
    __export int llaisysModelKvSeqAdd(struct LlaisysModel *model, int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);
    __export int llaisysModelKvSeqKeep(struct LlaisysModel *model, int64_t seq_id);
    __export int64_t llaisysModelKvSeqPosMax(struct LlaisysModel *model, int64_t seq_id);
}

#endif // LLAISYS_MODELS_MODEL_H
