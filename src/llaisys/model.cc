#include "llaisys/models/model.h"

#include "qwen2/qwen2_model.hpp"
#include "runtime/weights/weights.hpp"

#include <cstring>
#include <unordered_map>
#include <memory>
#include <new>
#include <vector>

namespace {

using llaisys::models::qwen2::Qwen2Model;
using KvStatus = llaisys::runtime::kv_cache::KvStatus;

int to_kv_code(KvStatus status) {
    return static_cast<int>(status);
}

class MockModel {
public:
    int32_t decode(const LlaisysBatch &batch) {
        if (batch.n_tokens <= 0 || batch.embd != nullptr || batch.token == nullptr) {
            return -1;
        }

        const size_t ntoken = static_cast<size_t>(batch.n_tokens);
        for (size_t i = 0; i < ntoken; ++i) {
            const int32_t nseq = batch.n_seq_id ? batch.n_seq_id[i] : 1;
            if (nseq != 1) {
                return -1;
            }
            if (batch.seq_id && batch.seq_id[i] == nullptr) {
                return -1;
            }
        }

        output_ids_.clear();
        output_logits_.clear();
        for (size_t i = 0; i < ntoken; ++i) {
            const int64_t seq_id = (batch.seq_id && batch.seq_id[i]) ? batch.seq_id[i][0] : 0;
            auto it = seq_pos_max_.find(seq_id);
            if (it == seq_pos_max_.end()) {
                it = seq_pos_max_.emplace(seq_id, -1).first;
            }
            int64_t &pos_max = it->second;
            const int64_t expected_pos = pos_max + 1;
            if (batch.pos && batch.pos[i] != expected_pos) {
                return -1;
            }
            pos_max = expected_pos;

            const bool collect = batch.logits ? (batch.logits[i] != 0) : (i + 1 == ntoken);
            if (collect) {
                output_ids_.push_back(static_cast<int32_t>(i));
                const float v0 = static_cast<float>(batch.token[i] % 97);
                output_logits_.push_back(v0);
                output_logits_.push_back(v0 + 1.0f);
                output_logits_.push_back(v0 + 2.0f);
                output_logits_.push_back(v0 + 3.0f);
            }
        }
        return 0;
    }

    float *logits() noexcept {
        return output_logits_.empty() ? nullptr : output_logits_.data();
    }
    float *logits_ith(int32_t i) noexcept {
        if (i < 0 || static_cast<size_t>(i) >= output_ids_.size()) {
            return nullptr;
        }
        return output_logits_.data() + static_cast<size_t>(i) * 4;
    }
    int32_t n_outputs() const noexcept {
        return static_cast<int32_t>(output_ids_.size());
    }
    const int32_t *output_ids() const noexcept {
        return output_ids_.empty() ? nullptr : output_ids_.data();
    }

    KvStatus kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) {
        auto it = seq_pos_max_.find(src_seq);
        if (it == seq_pos_max_.end()) {
            return KvStatus::INVALID_SEQ;
        }
        if (p0 < 0 || p1 < 0 || p0 > p1 || p1 > it->second + 1) {
            return KvStatus::INVALID_POS;
        }
        if (p0 == p1) {
            return KvStatus::EMPTY_RANGE;
        }
        seq_pos_max_[dst_seq] = static_cast<int64_t>(p1 - p0 - 1);
        return KvStatus::OK;
    }
    KvStatus kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
        auto it = seq_pos_max_.find(seq_id);
        if (it == seq_pos_max_.end()) {
            return KvStatus::INVALID_SEQ;
        }
        if (p0 < 0 || p1 < 0 || p0 > p1 || p1 > it->second + 1) {
            return KvStatus::INVALID_POS;
        }
        if (p0 == p1) {
            return KvStatus::EMPTY_RANGE;
        }
        if (p0 == 0 && p1 == it->second + 1) {
            seq_pos_max_.erase(it);
        } else {
            it->second = p0 - 1;
        }
        return KvStatus::OK;
    }
    KvStatus kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
        auto it = seq_pos_max_.find(seq_id);
        if (it == seq_pos_max_.end()) {
            return KvStatus::INVALID_SEQ;
        }
        if (p0 < 0 || p1 < 0 || p0 > p1 || p1 > it->second + 1) {
            return KvStatus::INVALID_POS;
        }
        if (p0 == p1) {
            return KvStatus::EMPTY_RANGE;
        }
        if (delta != 0) {
            return KvStatus::INVALID_POS;
        }
        return KvStatus::OK;
    }
    KvStatus kv_seq_keep(int64_t seq_id) {
        if (seq_pos_max_.find(seq_id) == seq_pos_max_.end()) {
            return KvStatus::INVALID_SEQ;
        }
        std::vector<int64_t> remove_keys;
        remove_keys.reserve(seq_pos_max_.size());
        for (const auto &kv : seq_pos_max_) {
            if (kv.first != seq_id) {
                remove_keys.push_back(kv.first);
            }
        }
        for (const int64_t k : remove_keys) {
            seq_pos_max_.erase(k);
        }
        return KvStatus::OK;
    }
    int64_t kv_seq_pos_max(int64_t seq_id) const noexcept {
        auto it = seq_pos_max_.find(seq_id);
        if (it == seq_pos_max_.end()) {
            return -1;
        }
        return it->second;
    }

private:
    std::unordered_map<int64_t, int64_t> seq_pos_max_{};
    std::vector<float> output_logits_{};
    std::vector<int32_t> output_ids_{};
};

struct LlaisysModelImpl {
    LlaisysModelType type{LLAISYS_MODEL_TYPE_UNKNOWN};
    std::unique_ptr<Qwen2Model> qwen2{};
    std::unique_ptr<MockModel> mock{};
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

            llaisys::runtime::kv_cache::KvCacheLayout kv_layout =
                llaisys::runtime::kv_cache::KvCacheLayout::BLOCK;
            if (params->kv_cache_layout == LLAISYS_KV_CACHE_LAYOUT_SLOT) {
                kv_layout = llaisys::runtime::kv_cache::KvCacheLayout::SLOT;
            } else if (params->kv_cache_layout >= 0 && params->kv_cache_layout != LLAISYS_KV_CACHE_LAYOUT_BLOCK) {
                delete handle;
                return nullptr;
            }

            size_t kv_block_size = 16;
            if (params->kv_cache_block_size > 0) {
                kv_block_size = static_cast<size_t>(params->kv_cache_block_size);
            }

            auto meta = *reinterpret_cast<const LlaisysQwen2Meta *>(params->meta);
            if (params->max_model_len > 0) {
                meta.maxseq = static_cast<int64_t>(params->max_model_len);
            }
            size_t kv_cache_capacity_tokens = static_cast<size_t>(meta.maxseq);
            if (params->kv_cache_capacity_tokens > 0) {
                kv_cache_capacity_tokens = static_cast<size_t>(params->kv_cache_capacity_tokens);
            }
            handle->impl->qwen2 =
                std::make_unique<Qwen2Model>(meta, params->device, params->device_ids, params->ndevice, kv_layout,
                                             kv_block_size, kv_cache_capacity_tokens);
            return handle;
        }
        case LLAISYS_MODEL_TYPE_MOCK: {
            handle->impl->mock = std::make_unique<MockModel>();
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
    case LLAISYS_MODEL_TYPE_MOCK:
        return nullptr;
    default:
        return nullptr;
    }
}

__export int llaisysModelReplaceWeight(struct LlaisysModel *model,
                                       const char *field_name,
                                       int32_t layer_idx,
                                       llaisysTensor_t new_weight) {
    if (model == nullptr || model->impl == nullptr || field_name == nullptr) {
        return -1;
    }
    if (model->impl->type != LLAISYS_MODEL_TYPE_QWEN2 || !model->impl->qwen2) {
        return -2;
    }

    LlaisysQwen2Weights *w = model->impl->qwen2->weights();
    if (w == nullptr) {
        return -1;
    }

    auto replace = [&](llaisysTensor_t *slot) -> int {
        if (slot == nullptr) {
            return -3;
        }
        llaisys::runtime::weights::replace_slot(slot, new_weight);
        return 0;
    };

    if (std::strcmp(field_name, "in_embed") == 0) {
        return replace(&w->in_embed);
    }
    if (std::strcmp(field_name, "out_embed") == 0) {
        return replace(&w->out_embed);
    }
    if (std::strcmp(field_name, "out_norm_w") == 0) {
        return replace(&w->out_norm_w);
    }

    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= model->impl->qwen2->nlayer()) {
        return -4;
    }

    if (std::strcmp(field_name, "attn_norm_w") == 0) {
        return replace(&w->attn_norm_w[layer_idx]);
    }
    if (std::strcmp(field_name, "attn_q_w") == 0) {
        return replace(&w->attn_q_w[layer_idx]);
    }
    if (std::strcmp(field_name, "attn_q_b") == 0) {
        return replace(&w->attn_q_b[layer_idx]);
    }
    if (std::strcmp(field_name, "attn_k_w") == 0) {
        return replace(&w->attn_k_w[layer_idx]);
    }
    if (std::strcmp(field_name, "attn_k_b") == 0) {
        return replace(&w->attn_k_b[layer_idx]);
    }
    if (std::strcmp(field_name, "attn_v_w") == 0) {
        return replace(&w->attn_v_w[layer_idx]);
    }
    if (std::strcmp(field_name, "attn_v_b") == 0) {
        return replace(&w->attn_v_b[layer_idx]);
    }
    if (std::strcmp(field_name, "attn_o_w") == 0) {
        return replace(&w->attn_o_w[layer_idx]);
    }
    if (std::strcmp(field_name, "mlp_norm_w") == 0) {
        return replace(&w->mlp_norm_w[layer_idx]);
    }
    if (std::strcmp(field_name, "mlp_gate_w") == 0) {
        return replace(&w->mlp_gate_w[layer_idx]);
    }
    if (std::strcmp(field_name, "mlp_up_w") == 0) {
        return replace(&w->mlp_up_w[layer_idx]);
    }
    if (std::strcmp(field_name, "mlp_down_w") == 0) {
        return replace(&w->mlp_down_w[layer_idx]);
    }

    return -3;
}

__export int32_t llaisysModelDecode(struct LlaisysModel *model, struct LlaisysBatch batch) {
    if (model == nullptr || model->impl == nullptr) {
        return -1;
    }

    try {
        switch (model->impl->type) {
        case LLAISYS_MODEL_TYPE_QWEN2:
            return model->impl->qwen2 ? model->impl->qwen2->decode(batch) : -1;
        case LLAISYS_MODEL_TYPE_MOCK:
            return model->impl->mock ? model->impl->mock->decode(batch) : -1;
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
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock ? model->impl->mock->logits() : nullptr;
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
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock ? model->impl->mock->logits_ith(i) : nullptr;
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
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock ? model->impl->mock->n_outputs() : 0;
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
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock ? model->impl->mock->output_ids() : nullptr;
    default:
        return nullptr;
    }
}

__export int llaisysModelKvSeqCp(struct LlaisysModel *model, int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) {
    if (model == nullptr || model->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2
                   ? to_kv_code(model->impl->qwen2->kv_seq_cp(dst_seq, src_seq, p0, p1))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock
                   ? to_kv_code(model->impl->mock->kv_seq_cp(dst_seq, src_seq, p0, p1))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    default:
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
}

__export int llaisysModelKvSeqRm(struct LlaisysModel *model, int64_t seq_id, int64_t p0, int64_t p1) {
    if (model == nullptr || model->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2
                   ? to_kv_code(model->impl->qwen2->kv_seq_rm(seq_id, p0, p1))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock
                   ? to_kv_code(model->impl->mock->kv_seq_rm(seq_id, p0, p1))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    default:
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
}

__export int llaisysModelKvSeqAdd(struct LlaisysModel *model, int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    if (model == nullptr || model->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2
                   ? to_kv_code(model->impl->qwen2->kv_seq_add(seq_id, p0, p1, delta))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock
                   ? to_kv_code(model->impl->mock->kv_seq_add(seq_id, p0, p1, delta))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    default:
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
}

__export int llaisysModelKvSeqKeep(struct LlaisysModel *model, int64_t seq_id) {
    if (model == nullptr || model->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2
                   ? to_kv_code(model->impl->qwen2->kv_seq_keep(seq_id))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock
                   ? to_kv_code(model->impl->mock->kv_seq_keep(seq_id))
                   : to_kv_code(KvStatus::INTERNAL_ERROR);
    default:
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
}

__export int64_t llaisysModelKvSeqPosMax(struct LlaisysModel *model, int64_t seq_id) {
    if (model == nullptr || model->impl == nullptr) {
        return -1;
    }
    switch (model->impl->type) {
    case LLAISYS_MODEL_TYPE_QWEN2:
        return model->impl->qwen2 ? model->impl->qwen2->kv_seq_pos_max(seq_id) : -1;
    case LLAISYS_MODEL_TYPE_MOCK:
        return model->impl->mock ? model->impl->mock->kv_seq_pos_max(seq_id) : -1;
    default:
        return -1;
    }
}

} // extern "C"
