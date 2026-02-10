#include "llaisys/models/model.h"

#include "qwen2/qwen2_model.hpp"

#include <cstring>
#include <memory>
#include <new>

namespace {

using llaisys::models::qwen2::Qwen2Model;

struct LlaisysModelImpl {
    LlaisysModelType type{LLAISYS_MODEL_TYPE_UNKNOWN};
    std::unique_ptr<Qwen2Model> qwen2{};
};

void free_batch_storage(struct LlaisysBatch &batch) {
    if (batch.seq_id != nullptr) {
        const int32_t n_tokens = batch.n_tokens > 0 ? batch.n_tokens : 0;
        for (int32_t i = 0; i < n_tokens; ++i) {
            delete[] batch.seq_id[i];
        }
    }

    delete[] batch.token;
    delete[] batch.embd;
    delete[] batch.pos;
    delete[] batch.n_seq_id;
    delete[] batch.seq_id;
    delete[] batch.logits;

    batch.n_tokens = 0;
    batch.token = nullptr;
    batch.embd = nullptr;
    batch.pos = nullptr;
    batch.n_seq_id = nullptr;
    batch.seq_id = nullptr;
    batch.logits = nullptr;
}

} // namespace

__C {

struct LlaisysModel {
    std::unique_ptr<LlaisysModelImpl> impl;
};

__export struct LlaisysBatch llaisysBatchInit(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
    struct LlaisysBatch batch{};
    if (n_tokens <= 0) {
        return batch;
    }

    const int32_t seq_cap = n_seq_max > 0 ? n_seq_max : 1;

    batch.n_tokens = n_tokens;
    try {
        if (embd > 0) {
            batch.embd = new float[static_cast<size_t>(n_tokens) * static_cast<size_t>(embd)]();
        } else {
            batch.token = new int64_t[n_tokens]();
        }

        batch.pos = new int64_t[n_tokens]();
        batch.n_seq_id = new int32_t[n_tokens]();
        batch.seq_id = new int64_t *[n_tokens]();
        batch.logits = new int8_t[n_tokens]();

        for (int32_t i = 0; i < n_tokens; ++i) {
            batch.seq_id[i] = new int64_t[seq_cap]();
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }
    } catch (const std::bad_alloc &) {
        free_batch_storage(batch);
        return {};
    }

    return batch;
}

__export struct LlaisysBatch llaisysBatchGetOne(int64_t *token, int32_t n_tokens) {
    struct LlaisysBatch batch = llaisysBatchInit(n_tokens, 0, 1);
    if (batch.n_tokens <= 0 || batch.token == nullptr) {
        return batch;
    }

    if (token != nullptr) {
        std::memcpy(batch.token, token, static_cast<size_t>(n_tokens) * sizeof(int64_t));
    }
    if (n_tokens > 0 && batch.logits != nullptr) {
        batch.logits[n_tokens - 1] = 1;
    }

    return batch;
}

__export void llaisysBatchFree(struct LlaisysBatch batch) {
    free_batch_storage(batch);
}

__export struct LlaisysModel *llaisysModelCreate(const struct LlaisysModelCreateParams *params) {
    if (params == nullptr) {
        return nullptr;
    }

    try {
        auto *handle = new LlaisysModel{};
        handle->impl = std::make_unique<LlaisysModelImpl>();
        handle->impl->type = params->model_type;

        switch (params->model_type) {
        case LLAISYS_MODEL_TYPE_QWEN2: {
            if (params->meta == nullptr) {
                delete handle;
                return nullptr;
            }
            const auto *meta = reinterpret_cast<const LlaisysQwen2Meta *>(params->meta);
            handle->impl->qwen2 = std::make_unique<Qwen2Model>(*meta, params->device, params->device_ids, params->ndevice);
            return handle;
        }
        default:
            delete handle;
            return nullptr;
        }
    } catch (...) {
        return nullptr;
    }
}

__export void llaisysModelDestroy(struct LlaisysModel *model) {
    delete model;
}

__export LlaisysModelType llaisysModelType(const struct LlaisysModel *model) {
    if (model == nullptr || model->impl == nullptr) {
        return LLAISYS_MODEL_TYPE_UNKNOWN;
    }
    return model->impl->type;
}

__export void *llaisysModelWeights(struct LlaisysModel *model) {
    if (model == nullptr || model->impl == nullptr) {
        return nullptr;
    }

    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->weights() : nullptr;
    default:
        return nullptr;
    }
}

__export int32_t llaisysModelDecode(struct LlaisysModel *model, struct LlaisysBatch batch) {
    if (model == nullptr || model->impl == nullptr) {
        return -1;
    }

    try {
        switch (model->impl->type) {
        case LLAISYS_MODEL_TYPE_QWEN2:
            return model->impl->qwen2 ? model->impl->qwen2->decode(batch) : -1;
        default:
            return -1;
        }
    } catch (const std::invalid_argument &) {
        return -1;
    } catch (...) {
        return -2;
    }
}

__export float *llaisysModelGetLogits(struct LlaisysModel *model) {
    if (model == nullptr || model->impl == nullptr) {
        return nullptr;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->logits() : nullptr;
    default:
        return nullptr;
    }
}

__export float *llaisysModelGetLogitsIth(struct LlaisysModel *model, int32_t i) {
    if (model == nullptr || model->impl == nullptr) {
        return nullptr;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->logits_ith(i) : nullptr;
    default:
        return nullptr;
    }
}

__export int32_t llaisysModelNOutputs(struct LlaisysModel *model) {
    if (model == nullptr || model->impl == nullptr) {
        return 0;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->n_outputs() : 0;
    default:
        return 0;
    }
}

__export const int32_t *llaisysModelOutputIds(struct LlaisysModel *model) {
    if (model == nullptr || model->impl == nullptr) {
        return nullptr;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->output_ids() : nullptr;
    default:
        return nullptr;
    }
}

__export int llaisysModelKvSeqCp(struct LlaisysModel *model, int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) {
    if (model == nullptr || model->impl == nullptr) {
        return 5;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->kv_seq_cp(dst_seq, src_seq, p0, p1) : 5;
    default:
        return 5;
    }
}

__export int llaisysModelKvSeqRm(struct LlaisysModel *model, int64_t seq_id, int64_t p0, int64_t p1) {
    if (model == nullptr || model->impl == nullptr) {
        return 5;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->kv_seq_rm(seq_id, p0, p1) : 5;
    default:
        return 5;
    }
}

__export int llaisysModelKvSeqAdd(struct LlaisysModel *model, int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    if (model == nullptr || model->impl == nullptr) {
        return 5;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->kv_seq_add(seq_id, p0, p1, delta) : 5;
    default:
        return 5;
    }
}

__export int llaisysModelKvSeqKeep(struct LlaisysModel *model, int64_t seq_id) {
    if (model == nullptr || model->impl == nullptr) {
        return 5;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->kv_seq_keep(seq_id) : 5;
    default:
        return 5;
    }
}

__export int64_t llaisysModelKvSeqPosMax(struct LlaisysModel *model, int64_t seq_id) {
    if (model == nullptr || model->impl == nullptr) {
        return -1;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->kv_seq_pos_max(seq_id) : -1;
    default:
        return -1;
    }
}

} // extern "C"
