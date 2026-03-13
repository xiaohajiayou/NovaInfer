#include "llaisys/models/model.h"

#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "llaisys_tensor.hpp"
#include "qwen2/qwen2_model.hpp"
#include "kv_cache/paged_kv.hpp"
#include "weights/weights.hpp"
#include "../core/context/context.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

using llaisys::models::qwen2::Qwen2Model;
using KvStatus = llaisys::kv_cache::KvStatus;
int to_kv_code(KvStatus status) {
    return static_cast<int>(status);
}

class MockModel {};

struct LlaisysModelImpl {
    LlaisysModelType type{LLAISYS_MODEL_TYPE_UNKNOWN};
    std::unique_ptr<Qwen2Model> qwen2{};
    std::unique_ptr<MockModel> mock{};

    int32_t forward(const ModelForwardInput &input, ModelForwardOutput *output) {
        switch (type) {
        case LLAISYS_MODEL_TYPE_QWEN2:
            return qwen2 ? qwen2->forward(input, output) : -1;
        case LLAISYS_MODEL_TYPE_MOCK:
            return -2;
        default:
            return -1;
        }
    }

};

struct LlaisysKvStateImpl {
    LlaisysKvStateCreateParams params{};
    std::weak_ptr<LlaisysModelImpl> bound_model{};
    int64_t kv_peak_used_tokens{0};

    bool ensure_model_bound(const std::shared_ptr<LlaisysModelImpl> &impl) {
        if (!impl) {
            return false;
        }

        if (auto current = bound_model.lock(); current && current.get() == impl.get()) {
            return true;
        }

        bound_model.reset();
        kv_peak_used_tokens = 0;

        switch (impl->type) {
        case LLAISYS_MODEL_TYPE_QWEN2: {
            if (!impl->qwen2) {
                return false;
            }

            const size_t kv_block_size =
                params.kv_cache_block_size > 0 ? static_cast<size_t>(params.kv_cache_block_size) : static_cast<size_t>(16);
            const size_t kv_cache_capacity_tokens = params.kv_cache_capacity_tokens > 0
                                                        ? static_cast<size_t>(params.kv_cache_capacity_tokens)
                                                        : static_cast<size_t>(0);
            const int64_t max_model_len = params.max_model_len > 0 ? static_cast<int64_t>(params.max_model_len) : int64_t{0};
            if (impl->qwen2->configure_runtime(kv_block_size, kv_cache_capacity_tokens, max_model_len) != 0 ||
                impl->qwen2->kv_cache() == nullptr) {
                return false;
            }
            break;
        }
        case LLAISYS_MODEL_TYPE_MOCK:
            if (!impl->mock) {
                return false;
            }
            break;
        default:
            return false;
        }
        bound_model = impl;
        return true;
    }

    bool has_live_model_binding() {
        return !bound_model.expired();
    }

    KvStatus request_free(int64_t seq_id) {
        if (!has_live_model_binding()) {
            return KvStatus::INTERNAL_ERROR;
        }
        auto current = bound_model.lock();
        if (!current) {
            return KvStatus::INTERNAL_ERROR;
        }
        switch (current->type) {
        case LLAISYS_MODEL_TYPE_QWEN2:
            return current->qwen2 && current->qwen2->kv_cache()
                       ? current->qwen2->kv_cache()->request_free(seq_id)
                       : KvStatus::INTERNAL_ERROR;
        case LLAISYS_MODEL_TYPE_MOCK:
            return KvStatus::INVALID_SEQ;
        default:
            return KvStatus::INTERNAL_ERROR;
        }
    }

    int kv_stats(LlaisysKvStats *out_stats) noexcept {
        if (!has_live_model_binding() || out_stats == nullptr) {
            return -1;
        }
        auto current = bound_model.lock();
        if (!current) {
            return -1;
        }
        switch (current->type) {
        case LLAISYS_MODEL_TYPE_QWEN2: {
            auto *cache = current->qwen2 ? current->qwen2->kv_cache() : nullptr;
            if (cache == nullptr) {
                return -1;
            }
            std::vector<int32_t> used_slots;
            cache->used_slots(&used_slots);
            out_stats->capacity_tokens = static_cast<int64_t>(current->qwen2->kv_cache_capacity_tokens());
            out_stats->used_tokens = static_cast<int64_t>(used_slots.size());
            out_stats->free_tokens = std::max<int64_t>(0, out_stats->capacity_tokens - out_stats->used_tokens);
            if (out_stats->used_tokens > kv_peak_used_tokens) {
                kv_peak_used_tokens = out_stats->used_tokens;
            }
            out_stats->peak_used_tokens = kv_peak_used_tokens;
            return 0;
        }
        case LLAISYS_MODEL_TYPE_MOCK:
            out_stats->capacity_tokens = 0;
            out_stats->used_tokens = 0;
            out_stats->free_tokens = 0;
            out_stats->peak_used_tokens = 0;
            return 0;
        default:
            return -1;
        }
    }

    KvStatus kv_reset_prefix_cache() {
        if (!has_live_model_binding()) {
            return KvStatus::INTERNAL_ERROR;
        }
        auto current = bound_model.lock();
        if (!current) {
            return KvStatus::INTERNAL_ERROR;
        }
        switch (current->type) {
        case LLAISYS_MODEL_TYPE_QWEN2:
            return current->qwen2 && current->qwen2->kv_cache()
                       ? current->qwen2->kv_cache()->reset_prefix_cache()
                       : KvStatus::INTERNAL_ERROR;
        case LLAISYS_MODEL_TYPE_MOCK:
            return KvStatus::OK;
        default:
            return KvStatus::INTERNAL_ERROR;
        }
    }
};

} // namespace

__C {

struct LlaisysModel {
    std::shared_ptr<LlaisysModelImpl> impl;
};

struct LlaisysKvState {
    std::unique_ptr<LlaisysKvStateImpl> impl;
};

__export struct LlaisysKvState *llaisysKvStateCreate(const struct LlaisysKvStateCreateParams *params) {
    if (params == nullptr) {
        return nullptr;
    }
    try {
        auto *kv_state = new LlaisysKvState{};
        kv_state->impl = std::make_unique<LlaisysKvStateImpl>();
        kv_state->impl->params = *params;
        return kv_state;
    } catch (...) {
        return nullptr;
    }
}

__export struct LlaisysModel *llaisysModelCreate(const struct LlaisysModelCreateParams *params) {
    if (params == nullptr) {
        return nullptr;
    }
    try {
        auto *handle = new LlaisysModel{};
        handle->impl = std::make_shared<LlaisysModelImpl>();
        handle->impl->type = params->model_type;

        switch (params->model_type) {
        case LLAISYS_MODEL_TYPE_QWEN2: {
            if (params->meta == nullptr) {
                delete handle;
                return nullptr;
            }
            auto meta = *reinterpret_cast<const LlaisysQwen2Meta *>(params->meta);
            handle->impl->qwen2 = std::make_unique<Qwen2Model>(meta, params->device, params->device_ids, params->ndevice);
            break;
        }
        case LLAISYS_MODEL_TYPE_MOCK: {
            handle->impl->mock = std::make_unique<MockModel>();
            break;
        }
        default:
            delete handle;
            return nullptr;
        }

        return handle;
    } catch (...) {
        return nullptr;
    }
}

__export void llaisysModelDestroy(struct LlaisysModel *model) {
    delete model;
}

__export void llaisysKvStateDestroy(struct LlaisysKvState *kv_state) {
    delete kv_state;
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
        llaisys::weights::replace_slot(slot, new_weight);
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

__export int32_t llaisysModelForward(struct LlaisysModel *model,
                                     struct LlaisysKvState *kv_state,
                                     const struct ModelForwardInput *input,
                                     struct ModelForwardOutput *output) {
    LLAISYS_NVTX_SCOPE("api/model_forward");
    if (model == nullptr || model->impl == nullptr || kv_state == nullptr || kv_state->impl == nullptr || input == nullptr ||
        input->input_ids == nullptr) {
        return -1;
    }
    try {
        if (model->impl->type == LLAISYS_MODEL_TYPE_QWEN2 && model->impl->qwen2 != nullptr) {
            if (!model->impl->qwen2->bind_kv_state_handle(static_cast<const void *>(kv_state))) {
                std::fprintf(stderr,
                             "[ERROR] Qwen2: kv_state handle changed after first forward (current=%p)\n",
                             static_cast<const void *>(kv_state));
                return -1;
            }
        }
        if (!kv_state->impl->ensure_model_bound(model->impl)) {
            return -1;
        }
        return model->impl->forward(*input, output);
    } catch (const std::invalid_argument &) {
        return -1;
    } catch (...) {
        return -2;
    }
}

__export int llaisysKvStateRequestFree(struct LlaisysKvState *kv_state, int64_t seq_id) {
    if (kv_state == nullptr || kv_state->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(kv_state->impl->request_free(seq_id));
}

__export int llaisysKvStateStats(struct LlaisysKvState *kv_state, struct LlaisysKvStats *out_stats) {
    if (kv_state == nullptr || kv_state->impl == nullptr || out_stats == nullptr) {
        return -1;
    }
    return kv_state->impl->kv_stats(out_stats);
}

__export int llaisysKvStateResetPrefixCache(struct LlaisysKvState *kv_state) {
    if (kv_state == nullptr || kv_state->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(kv_state->impl->kv_reset_prefix_cache());
}

} // extern "C"
