#include "llaisys/models/model.h"

#include "../core/context/context.hpp"
#include "../utils.hpp"
#include "kv_cache/paged_kv.hpp"
#include "llaisys_tensor.hpp"
#include "qwen2/qwen2_model.hpp"
#include "weights/weights.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#ifdef ENABLE_NCCL_API
#include <nccl.h>
#endif

namespace {

using llaisys::models::qwen2::Qwen2Model;
using KvStatus = llaisys::kv_cache::KvStatus;

int to_kv_code(KvStatus status) {
    return static_cast<int>(status);
}

std::string str_or_empty(const char *s) {
    return s == nullptr ? std::string{} : std::string{s};
}

class MockModel {};

struct LlaisysParallelContextImpl {
    int32_t tensor_parallel_size{1};
    int32_t rank{0};
    int32_t local_rank{0};
    std::string distributed_backend{};
    std::string init_method{};
    std::vector<int> device_ids{};
#ifdef ENABLE_NCCL_API
    ncclComm_t nccl_comm{nullptr};
#endif

    ~LlaisysParallelContextImpl() {
#ifdef ENABLE_NCCL_API
        if (nccl_comm != nullptr) {
            ncclCommDestroy(nccl_comm);
            nccl_comm = nullptr;
        }
#endif
    }

    int initialize() {
        if (tensor_parallel_size <= 1) {
            return 0;
        }
#ifndef ENABLE_NCCL_API
        std::fprintf(stderr, "[ERROR] TP requires ENABLE_NCCL_API build\n");
        return -1;
#else
        if (!distributed_backend.empty() && distributed_backend != "nccl") {
            std::fprintf(stderr, "[ERROR] unsupported distributed_backend=%s\n", distributed_backend.c_str());
            return -1;
        }
        if (static_cast<int32_t>(device_ids.size()) != tensor_parallel_size) {
            std::fprintf(stderr,
                         "[ERROR] parallel context invalid device_ids: tp_size=%d rank=%d local_rank=%d ndevice=%d\n",
                         tensor_parallel_size,
                         rank,
                         local_rank,
                         static_cast<int>(device_ids.size()));
            return -1;
        }

        std::string id_file;
        if (!init_method.empty() && init_method.rfind("file://", 0) == 0) {
            id_file = init_method.substr(7);
        }
        if (id_file.empty()) {
            id_file = "/tmp/llaisys_tp_nccl.id";
        }

        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, device_ids[static_cast<size_t>(rank)]);

        ncclUniqueId uid{};
        if (rank == 0) {
            const ncclResult_t uid_rc = ncclGetUniqueId(&uid);
            if (uid_rc != ncclSuccess) {
                std::fprintf(stderr, "[ERROR] ncclGetUniqueId failed: %s\n", ncclGetErrorString(uid_rc));
                return -1;
            }
            std::ofstream ofs(id_file, std::ios::binary | std::ios::trunc);
            if (!ofs.good()) {
                std::fprintf(stderr, "[ERROR] open NCCL id file failed: %s\n", id_file.c_str());
                return -1;
            }
            ofs.write(reinterpret_cast<const char *>(&uid), sizeof(uid));
            ofs.flush();
            if (!ofs.good()) {
                std::fprintf(stderr, "[ERROR] write NCCL id file failed: %s\n", id_file.c_str());
                return -1;
            }
        } else {
            bool ok = false;
            constexpr int kMaxRetry = 2000;
            for (int i = 0; i < kMaxRetry; ++i) {
                std::ifstream ifs(id_file, std::ios::binary);
                if (ifs.good()) {
                    ifs.read(reinterpret_cast<char *>(&uid), sizeof(uid));
                    if (ifs.gcount() == static_cast<std::streamsize>(sizeof(uid))) {
                        ok = true;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!ok) {
                std::fprintf(stderr,
                             "[ERROR] read NCCL id file timeout: path=%s tp_size=%d rank=%d local_rank=%d\n",
                             id_file.c_str(),
                             tensor_parallel_size,
                             rank,
                             local_rank);
                return -1;
            }
        }

        const ncclResult_t init_rc = ncclCommInitRank(&nccl_comm, tensor_parallel_size, uid, rank);
        if (init_rc != ncclSuccess || nccl_comm == nullptr) {
            std::fprintf(stderr,
                         "[ERROR] ncclCommInitRank failed: %s tp_size=%d rank=%d dev=%d\n",
                         ncclGetErrorString(init_rc),
                         tensor_parallel_size,
                         rank,
                         device_ids[static_cast<size_t>(rank)]);
            return -1;
        }
        return 0;
#endif
    }
};

struct LlaisysModelImpl {
    LlaisysModelType type{LLAISYS_MODEL_TYPE_UNKNOWN};
    std::unique_ptr<Qwen2Model> qwen2{};
    std::unique_ptr<MockModel> mock{};
    std::shared_ptr<LlaisysParallelContextImpl> parallel_context{};

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

    int32_t bind_parallel_context(const std::shared_ptr<LlaisysParallelContextImpl> &ctx) {
        if (!ctx) {
            return -1;
        }
        if (parallel_context && parallel_context.get() != ctx.get()) {
            std::fprintf(stderr, "[ERROR] model already bound to a different parallel context\n");
            return -1;
        }
        switch (type) {
        case LLAISYS_MODEL_TYPE_QWEN2: {
            if (!qwen2) {
                return -1;
            }
#ifdef ENABLE_NCCL_API
            void *nccl_comm = reinterpret_cast<void *>(ctx->nccl_comm);
#else
            void *nccl_comm = nullptr;
#endif
            const int rc = qwen2->bind_parallel_context(
                ctx->tensor_parallel_size,
                ctx->rank,
                ctx->local_rank,
                ctx->device_ids.empty() ? nullptr : ctx->device_ids.data(),
                static_cast<int32_t>(ctx->device_ids.size()),
                nccl_comm);
            if (rc != 0) {
                return rc;
            }
            parallel_context = ctx;
            return 0;
        }
        case LLAISYS_MODEL_TYPE_MOCK:
            parallel_context = ctx;
            return 0;
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
            if (!impl->qwen2 || !impl->parallel_context) {
                return false;
            }
            const size_t kv_block_size =
                params.kv_cache_block_size > 0 ? static_cast<size_t>(params.kv_cache_block_size) : static_cast<size_t>(16);
            const size_t kv_cache_capacity_tokens =
                params.kv_cache_capacity_tokens > 0 ? static_cast<size_t>(params.kv_cache_capacity_tokens) : static_cast<size_t>(0);
            const int64_t max_model_len = params.max_model_len > 0 ? static_cast<int64_t>(params.max_model_len) : int64_t{0};
            if (impl->qwen2->configure_runtime(kv_block_size, kv_cache_capacity_tokens, max_model_len) != 0 ||
                impl->qwen2->kv_cache() == nullptr) {
                return false;
            }
            break;
        }
        case LLAISYS_MODEL_TYPE_MOCK:
            if (!impl->mock || !impl->parallel_context) {
                return false;
            }
            break;
        default:
            return false;
        }
        bound_model = impl;
        return true;
    }

    KvStatus request_free(int64_t seq_id) {
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
        if (out_stats == nullptr) {
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
            kv_peak_used_tokens = std::max<int64_t>(kv_peak_used_tokens, out_stats->used_tokens);
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

struct LlaisysParallelContext {
    std::shared_ptr<LlaisysParallelContextImpl> impl;
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

__export void llaisysKvStateDestroy(struct LlaisysKvState *kv_state) {
    delete kv_state;
}

__export struct LlaisysParallelContext *llaisysParallelContextCreate(
    const struct LlaisysParallelContextCreateParams *params) {
    if (params == nullptr) {
        return nullptr;
    }
    try {
        const int32_t tp_size = params->tensor_parallel_size > 0 ? params->tensor_parallel_size : int32_t{1};
        const int32_t rank = params->rank >= 0 ? params->rank : int32_t{0};
        const int32_t local_rank = params->local_rank >= 0 ? params->local_rank : int32_t{0};
        const int32_t ndevice = params->ndevice;
        if (tp_size < 1 || rank < 0 || rank >= tp_size || local_rank < 0 || ndevice < 0) {
            return nullptr;
        }
        if (ndevice > 0 && params->device_ids == nullptr) {
            return nullptr;
        }
        if (tp_size > 1 && ndevice != tp_size) {
            std::fprintf(stderr,
                         "[ERROR] parallel context invalid device_ids: tp_size=%d rank=%d local_rank=%d ndevice=%d\n",
                         tp_size,
                         rank,
                         local_rank,
                         ndevice);
            return nullptr;
        }

        auto *handle = new LlaisysParallelContext{};
        handle->impl = std::make_shared<LlaisysParallelContextImpl>();
        handle->impl->tensor_parallel_size = tp_size;
        handle->impl->rank = rank;
        handle->impl->local_rank = local_rank;
        handle->impl->distributed_backend = str_or_empty(params->distributed_backend);
        handle->impl->init_method = str_or_empty(params->init_method);
        handle->impl->device_ids.reserve(static_cast<size_t>(std::max(0, ndevice)));
        for (int32_t i = 0; i < ndevice; ++i) {
            handle->impl->device_ids.push_back(params->device_ids[i]);
        }
        if (handle->impl->initialize() != 0) {
            delete handle;
            return nullptr;
        }
        return handle;
    } catch (...) {
        return nullptr;
    }
}

__export void llaisysParallelContextDestroy(struct LlaisysParallelContext *parallel_context) {
    delete parallel_context;
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
        case LLAISYS_MODEL_TYPE_MOCK:
            handle->impl->mock = std::make_unique<MockModel>();
            break;
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

__export int32_t llaisysModelBindParallelContext(struct LlaisysModel *model,
                                                 struct LlaisysParallelContext *parallel_context) {
    if (model == nullptr || model->impl == nullptr || parallel_context == nullptr || parallel_context->impl == nullptr) {
        return -1;
    }
    try {
        return model->impl->bind_parallel_context(parallel_context->impl);
    } catch (...) {
        return -1;
    }
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
