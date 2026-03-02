#include "llaisys/models/model.h"

#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "llaisys_tensor.hpp"
#include "qwen2/qwen2_model.hpp"
#include "runtime/kv_cache/paged_kv.hpp"
#include "runtime/kv_cache/unified_kv.hpp"
#include "runtime/weights/weights.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

using llaisys::models::qwen2::Qwen2Model;
using KvStatus = llaisys::runtime::kv_cache::KvStatus;
using KvCacheBase = llaisys::runtime::kv_cache::KvCacheBase;
using KvCacheLayout = llaisys::runtime::kv_cache::KvCacheLayout;

int to_kv_code(KvStatus status) {
    return static_cast<int>(status);
}

llaisys::tensor_t kv_layer_k_from_cache(KvCacheBase *cache, KvCacheLayout layout, size_t layer) {
    if (cache == nullptr) {
        return nullptr;
    }
    if (layout == KvCacheLayout::SLOT) {
        auto *impl = dynamic_cast<llaisys::runtime::kv_cache::UnifiedKvImpl *>(cache);
        return impl ? impl->layer_k(layer) : nullptr;
    }
    auto *impl = dynamic_cast<llaisys::runtime::kv_cache::PagedKvImpl *>(cache);
    return impl ? impl->layer_k(layer) : nullptr;
}

llaisys::tensor_t kv_layer_v_from_cache(KvCacheBase *cache, KvCacheLayout layout, size_t layer) {
    if (cache == nullptr) {
        return nullptr;
    }
    if (layout == KvCacheLayout::SLOT) {
        auto *impl = dynamic_cast<llaisys::runtime::kv_cache::UnifiedKvImpl *>(cache);
        return impl ? impl->layer_v(layer) : nullptr;
    }
    auto *impl = dynamic_cast<llaisys::runtime::kv_cache::PagedKvImpl *>(cache);
    return impl ? impl->layer_v(layer) : nullptr;
}

class MockModel {
public:
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

    KvStatus request_free(int64_t seq_id) {
        auto it = seq_pos_max_.find(seq_id);
        if (it == seq_pos_max_.end()) {
            return KvStatus::INVALID_SEQ;
        }
        seq_pos_max_.erase(it);
        return KvStatus::OK;
    }

private:
    std::unordered_map<int64_t, int64_t> seq_pos_max_{};
};

class RuntimeKVCacheManager {
public:
    virtual ~RuntimeKVCacheManager() = default;
    virtual KvStatus kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) = 0;
    virtual KvStatus kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1) = 0;
    virtual KvStatus kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) = 0;
    virtual KvStatus kv_seq_keep(int64_t seq_id) = 0;
    virtual int64_t kv_seq_pos_max(int64_t seq_id) const noexcept = 0;
    virtual KvStatus request_free(int64_t seq_id) = 0;
    virtual int kv_stats(LlaisysKvStats *out_stats) noexcept = 0;
    virtual KvStatus kv_reset_prefix_cache() = 0;
};

class ModelKVCacheManager final : public RuntimeKVCacheManager {
public:
    ModelKVCacheManager(KvCacheBase *kv_cache,
                        KvCacheLayout kv_layout,
                        size_t kv_nlayer,
                        size_t kv_nkvh,
                        size_t kv_dh,
                        llaisysDataType_t kv_dtype,
                        size_t kv_capacity_tokens)
        : kv_cache_(kv_cache),
          kv_layout_(kv_layout),
          kv_nlayer_(kv_nlayer),
          kv_nkvh_(kv_nkvh),
          kv_dh_(kv_dh),
          kv_dtype_(kv_dtype),
          kv_capacity_tokens_(kv_capacity_tokens) {}

    KvStatus kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) override {
        if (kv_cache_ == nullptr) {
            return KvStatus::INTERNAL_ERROR;
        }
        std::vector<int32_t> src_slots;
        std::vector<int32_t> dst_slots;
        const KvStatus rc = kv_cache_->seq_cp(dst_seq, src_seq, p0, p1, &src_slots, &dst_slots);
        if (rc != KvStatus::OK) {
            return rc;
        }
        const size_t copy_len = src_slots.size();
        const size_t stride_elems = kv_nkvh_ * kv_dh_;
        const size_t stride_bytes = stride_elems * llaisys::utils::dsize(kv_dtype_);
        for (size_t i = 0; i < copy_len; ++i) {
            const int32_t src_slot = src_slots[i];
            const int32_t dst_slot = dst_slots[i];
            if (src_slot == dst_slot) {
                continue;
            }
            for (size_t layer = 0; layer < kv_nlayer_; ++layer) {
                llaisys::tensor_t layer_k_cache = kv_layer_k_from_cache(kv_cache_, kv_layout_, layer);
                llaisys::tensor_t layer_v_cache = kv_layer_v_from_cache(kv_cache_, kv_layout_, layer);
                if (layer_k_cache == nullptr || layer_v_cache == nullptr) {
                    return KvStatus::INTERNAL_ERROR;
                }
                std::byte *k_dst =
                    layer_k_cache->data() + static_cast<ptrdiff_t>(dst_slot) * static_cast<ptrdiff_t>(stride_bytes);
                std::byte *v_dst =
                    layer_v_cache->data() + static_cast<ptrdiff_t>(dst_slot) * static_cast<ptrdiff_t>(stride_bytes);
                const std::byte *k_src =
                    layer_k_cache->data() + static_cast<ptrdiff_t>(src_slot) * static_cast<ptrdiff_t>(stride_bytes);
                const std::byte *v_src =
                    layer_v_cache->data() + static_cast<ptrdiff_t>(src_slot) * static_cast<ptrdiff_t>(stride_bytes);
                std::memcpy(k_dst, k_src, stride_bytes);
                std::memcpy(v_dst, v_src, stride_bytes);
            }
        }
        return KvStatus::OK;
    }

    KvStatus kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1) override {
        return kv_cache_ ? kv_cache_->seq_rm(seq_id, p0, p1) : KvStatus::INTERNAL_ERROR;
    }

    KvStatus kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) override {
        return kv_cache_ ? kv_cache_->seq_add(seq_id, p0, p1, delta) : KvStatus::INTERNAL_ERROR;
    }

    KvStatus kv_seq_keep(int64_t seq_id) override {
        return kv_cache_ ? kv_cache_->seq_keep(seq_id) : KvStatus::INTERNAL_ERROR;
    }

    int64_t kv_seq_pos_max(int64_t seq_id) const noexcept override {
        return kv_cache_ ? kv_cache_->seq_pos_max(seq_id) : -1;
    }

    KvStatus request_free(int64_t seq_id) override {
        return kv_cache_ ? kv_cache_->request_free(seq_id) : KvStatus::INTERNAL_ERROR;
    }

    int kv_stats(LlaisysKvStats *out_stats) noexcept override {
        if (out_stats == nullptr || kv_cache_ == nullptr) {
            return -1;
        }
        std::vector<int32_t> used_slots;
        kv_cache_->used_slots(&used_slots);
        out_stats->capacity_tokens = static_cast<int64_t>(kv_capacity_tokens_);
        out_stats->used_tokens = static_cast<int64_t>(used_slots.size());
        out_stats->free_tokens = std::max<int64_t>(0, out_stats->capacity_tokens - out_stats->used_tokens);
        if (out_stats->used_tokens > kv_peak_used_tokens_) {
            kv_peak_used_tokens_ = out_stats->used_tokens;
        }
        out_stats->peak_used_tokens = kv_peak_used_tokens_;
        return 0;
    }

    KvStatus kv_reset_prefix_cache() override {
        return kv_cache_ ? kv_cache_->reset_prefix_cache() : KvStatus::INTERNAL_ERROR;
    }

private:
    KvCacheBase *kv_cache_{nullptr};
    KvCacheLayout kv_layout_{KvCacheLayout::BLOCK};
    size_t kv_nlayer_{0};
    size_t kv_nkvh_{0};
    size_t kv_dh_{0};
    llaisysDataType_t kv_dtype_{LLAISYS_DTYPE_F32};
    size_t kv_capacity_tokens_{0};
    int64_t kv_peak_used_tokens_{0};
};

class MockKVCacheManager final : public RuntimeKVCacheManager {
public:
    explicit MockKVCacheManager(MockModel *mock) : mock_(mock) {}

    KvStatus kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) override {
        return mock_ ? mock_->kv_seq_cp(dst_seq, src_seq, p0, p1) : KvStatus::INTERNAL_ERROR;
    }
    KvStatus kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1) override {
        return mock_ ? mock_->kv_seq_rm(seq_id, p0, p1) : KvStatus::INTERNAL_ERROR;
    }
    KvStatus kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) override {
        return mock_ ? mock_->kv_seq_add(seq_id, p0, p1, delta) : KvStatus::INTERNAL_ERROR;
    }
    KvStatus kv_seq_keep(int64_t seq_id) override {
        return mock_ ? mock_->kv_seq_keep(seq_id) : KvStatus::INTERNAL_ERROR;
    }
    int64_t kv_seq_pos_max(int64_t seq_id) const noexcept override {
        return mock_ ? mock_->kv_seq_pos_max(seq_id) : -1;
    }
    KvStatus request_free(int64_t seq_id) override {
        return mock_ ? mock_->request_free(seq_id) : KvStatus::INTERNAL_ERROR;
    }
    int kv_stats(LlaisysKvStats *out_stats) noexcept override {
        if (out_stats == nullptr) {
            return -1;
        }
        out_stats->capacity_tokens = 0;
        out_stats->used_tokens = 0;
        out_stats->free_tokens = 0;
        out_stats->peak_used_tokens = 0;
        return 0;
    }
    KvStatus kv_reset_prefix_cache() override { return KvStatus::OK; }

private:
    MockModel *mock_{nullptr};
};

struct LlaisysModelImpl {
    LlaisysModelType type{LLAISYS_MODEL_TYPE_UNKNOWN};
    std::unique_ptr<Qwen2Model> qwen2{};
    std::unique_ptr<MockModel> mock{};

    int configure_runtime(const LlaisysRuntimeCreateParams &params) {
        switch (type) {
        case LLAISYS_MODEL_TYPE_QWEN2: {
            if (!qwen2) {
                return -1;
            }
            llaisys::runtime::kv_cache::KvCacheLayout kv_layout = llaisys::runtime::kv_cache::KvCacheLayout::BLOCK;
            if (params.kv_cache_layout == LLAISYS_KV_CACHE_LAYOUT_SLOT) {
                kv_layout = llaisys::runtime::kv_cache::KvCacheLayout::SLOT;
            } else if (params.kv_cache_layout >= 0 && params.kv_cache_layout != LLAISYS_KV_CACHE_LAYOUT_BLOCK) {
                return -1;
            }
            const size_t kv_block_size =
                params.kv_cache_block_size > 0 ? static_cast<size_t>(params.kv_cache_block_size) : static_cast<size_t>(16);
            const size_t kv_cache_capacity_tokens = params.kv_cache_capacity_tokens > 0
                                                        ? static_cast<size_t>(params.kv_cache_capacity_tokens)
                                                        : static_cast<size_t>(0);
            const int64_t max_model_len = params.max_model_len > 0 ? static_cast<int64_t>(params.max_model_len) : int64_t{0};
            return qwen2->configure_runtime(kv_layout, kv_block_size, kv_cache_capacity_tokens, max_model_len);
        }
        case LLAISYS_MODEL_TYPE_MOCK:
            return 0;
        default:
            return -1;
        }
    }

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

struct LlaisysRuntimeImpl {
    LlaisysRuntimeCreateParams params{};
    LlaisysModelType model_type{LLAISYS_MODEL_TYPE_UNKNOWN};
    std::shared_ptr<LlaisysModelImpl> model_owner{};
    std::unique_ptr<RuntimeKVCacheManager> kv_cache_manager{};

    bool bind_model(const std::shared_ptr<LlaisysModelImpl> &impl) {
        if (!impl || model_type != LLAISYS_MODEL_TYPE_UNKNOWN) {
            return false;
        }
        model_owner = impl;
        model_type = impl->type;
        switch (model_type) {
        case LLAISYS_MODEL_TYPE_QWEN2: {
            if (!impl->qwen2 || impl->qwen2->kv_cache() == nullptr) {
                unbind_model();
                return false;
            }
            kv_cache_manager = std::make_unique<ModelKVCacheManager>(impl->qwen2->kv_cache(),
                                                                     impl->qwen2->kv_layout(),
                                                                     impl->qwen2->nlayer(),
                                                                     impl->qwen2->nkvh(),
                                                                     impl->qwen2->dh(),
                                                                     impl->qwen2->dtype(),
                                                                     impl->qwen2->kv_cache_capacity_tokens());
            return true;
        }
        case LLAISYS_MODEL_TYPE_MOCK:
            if (!impl->mock) {
                unbind_model();
                return false;
            }
            kv_cache_manager = std::make_unique<MockKVCacheManager>(impl->mock.get());
            return true;
        default:
            unbind_model();
            return false;
        }
    }

    void unbind_model() {
        model_owner.reset();
        model_type = LLAISYS_MODEL_TYPE_UNKNOWN;
        kv_cache_manager.reset();
    }

    KvStatus kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) {
        return kv_cache_manager ? kv_cache_manager->kv_seq_cp(dst_seq, src_seq, p0, p1) : KvStatus::INTERNAL_ERROR;
    }

    KvStatus kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
        return kv_cache_manager ? kv_cache_manager->kv_seq_rm(seq_id, p0, p1) : KvStatus::INTERNAL_ERROR;
    }

    KvStatus kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
        return kv_cache_manager ? kv_cache_manager->kv_seq_add(seq_id, p0, p1, delta) : KvStatus::INTERNAL_ERROR;
    }

    KvStatus kv_seq_keep(int64_t seq_id) {
        return kv_cache_manager ? kv_cache_manager->kv_seq_keep(seq_id) : KvStatus::INTERNAL_ERROR;
    }

    int64_t kv_seq_pos_max(int64_t seq_id) const noexcept {
        return kv_cache_manager ? kv_cache_manager->kv_seq_pos_max(seq_id) : -1;
    }

    KvStatus request_free(int64_t seq_id) {
        return kv_cache_manager ? kv_cache_manager->request_free(seq_id) : KvStatus::INTERNAL_ERROR;
    }

    int kv_stats(LlaisysKvStats *out_stats) noexcept {
        return kv_cache_manager ? kv_cache_manager->kv_stats(out_stats) : -1;
    }

    KvStatus kv_reset_prefix_cache() {
        return kv_cache_manager ? kv_cache_manager->kv_reset_prefix_cache() : KvStatus::INTERNAL_ERROR;
    }
};

} // namespace

__C {

struct LlaisysModel {
    std::shared_ptr<LlaisysModelImpl> impl;
};

struct LlaisysRuntime {
    std::unique_ptr<LlaisysRuntimeImpl> impl;
};

__export struct LlaisysRuntime *llaisysRuntimeCreate(const struct LlaisysRuntimeCreateParams *params) {
    if (params == nullptr) {
        return nullptr;
    }
    try {
        auto *runtime = new LlaisysRuntime{};
        runtime->impl = std::make_unique<LlaisysRuntimeImpl>();
        runtime->impl->params = *params;
        return runtime;
    } catch (...) {
        return nullptr;
    }
}

__export struct LlaisysModel *llaisysModelCreate(const struct LlaisysModelCreateParams *params,
                                                 struct LlaisysRuntime *runtime) {
    if (params == nullptr || runtime == nullptr || runtime->impl == nullptr) {
        return nullptr;
    }
    if (runtime->impl->model_type != LLAISYS_MODEL_TYPE_UNKNOWN) {
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

        if (handle->impl->configure_runtime(runtime->impl->params) != 0) {
            delete handle;
            return nullptr;
        }
        if (!runtime->impl->bind_model(handle->impl)) {
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

__export void llaisysRuntimeDestroy(struct LlaisysRuntime *runtime) {
    if (runtime != nullptr && runtime->impl != nullptr) {
        runtime->impl->unbind_model();
    }
    delete runtime;
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

__export int32_t llaisysModelForward(struct LlaisysModel *model,
                                     const struct ModelForwardInput *input,
                                     struct ModelForwardOutput *output) {
    if (model == nullptr || model->impl == nullptr || input == nullptr || input->input_ids == nullptr) {
        return -1;
    }
    try {
        return model->impl->forward(*input, output);
    } catch (const std::invalid_argument &) {
        return -1;
    } catch (...) {
        return -2;
    }
}

__export int32_t llaisysSamplerSample(const struct SamplerInput *input,
                                      struct SamplerOutput *output) {
    if (input == nullptr || output == nullptr || input->logits == nullptr || output->sampled_ids == nullptr) {
        return -1;
    }
    try {
        const llaisys::tensor_t logits = input->logits->tensor;
        llaisys::tensor_t sampled_ids = output->sampled_ids->tensor;
        if (logits == nullptr || sampled_ids == nullptr) {
            return -1;
        }
        if (logits->ndim() != 2 || !logits->isContiguous()) {
            return -1;
        }
        if (sampled_ids->ndim() != 1 || sampled_ids->dtype() != LLAISYS_DTYPE_I64 || !sampled_ids->isContiguous()) {
            return -1;
        }
        if (sampled_ids->deviceType() != logits->deviceType() || sampled_ids->deviceId() != logits->deviceId()) {
            return -1;
        }

        const size_t n_outputs = logits->shape()[0];
        if (n_outputs == 0) {
            return 0;
        }
        if (sampled_ids->shape()[0] < n_outputs) {
            return -1;
        }

        if (sampled_ids->shape()[0] > n_outputs) {
            sampled_ids = sampled_ids->slice(0, 0, n_outputs);
        }

        llaisys::tensor_t max_idx = sampled_ids;
        llaisys::tensor_t max_val = llaisys::Tensor::create(
            {n_outputs}, logits->dtype(), logits->deviceType(), logits->deviceId());
        llaisys::ops::argmax_rows(max_idx, max_val, logits);
        return 0;
    } catch (...) {
        return -2;
    }
}

__export int llaisysRuntimeKvSeqCp(struct LlaisysRuntime *runtime, int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) {
    if (runtime == nullptr || runtime->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(runtime->impl->kv_seq_cp(dst_seq, src_seq, p0, p1));
}

__export int llaisysRuntimeKvSeqRm(struct LlaisysRuntime *runtime, int64_t seq_id, int64_t p0, int64_t p1) {
    if (runtime == nullptr || runtime->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(runtime->impl->kv_seq_rm(seq_id, p0, p1));
}

__export int llaisysRuntimeKvSeqAdd(struct LlaisysRuntime *runtime, int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    if (runtime == nullptr || runtime->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(runtime->impl->kv_seq_add(seq_id, p0, p1, delta));
}

__export int llaisysRuntimeKvSeqKeep(struct LlaisysRuntime *runtime, int64_t seq_id) {
    if (runtime == nullptr || runtime->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(runtime->impl->kv_seq_keep(seq_id));
}

__export int64_t llaisysRuntimeKvSeqPosMax(struct LlaisysRuntime *runtime, int64_t seq_id) {
    if (runtime == nullptr || runtime->impl == nullptr) {
        return -1;
    }
    return runtime->impl->kv_seq_pos_max(seq_id);
}

__export int llaisysRuntimeRequestFree(struct LlaisysRuntime *runtime, int64_t seq_id) {
    if (runtime == nullptr || runtime->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(runtime->impl->request_free(seq_id));
}

__export int llaisysRuntimeKvStats(struct LlaisysRuntime *runtime, struct LlaisysKvStats *out_stats) {
    if (runtime == nullptr || runtime->impl == nullptr || out_stats == nullptr) {
        return -1;
    }
    return runtime->impl->kv_stats(out_stats);
}

__export int llaisysRuntimeKvResetPrefixCache(struct LlaisysRuntime *runtime) {
    if (runtime == nullptr || runtime->impl == nullptr) {
        return to_kv_code(KvStatus::INTERNAL_ERROR);
    }
    return to_kv_code(runtime->impl->kv_reset_prefix_cache());
}

} // extern "C"
