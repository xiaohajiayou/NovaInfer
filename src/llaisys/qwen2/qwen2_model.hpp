#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/runtime/infer_types.h"

#include "../llaisys_tensor.hpp"
#include "../runtime/kv_cache/kv_cache.hpp"
#include "../runtime/output/output.hpp"
#include "../runtime/workspace/workspace.hpp"
#include "../runtime/weights/weights.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../../ops/self_attention/cuda/self_attention_cuda.hpp"
#endif
#include "../../tensor/tensor.hpp"
#include "../../utils/check.hpp"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ModelForwardInput;
struct ModelForwardOutput;
struct AttentionMetadata;

namespace llaisys::models::qwen2 {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device,
               int *device_ids,
               int ndevice);
    ~Qwen2Model();

    int configure_runtime(runtime::kv_cache::KvCacheLayout kv_layout,
                          size_t kv_block_size,
                          size_t kv_cache_capacity_tokens,
                          int64_t max_model_len);

    LlaisysQwen2Weights *weights() noexcept { return &weights_; }
    size_t nlayer() const noexcept { return meta_.nlayer; }
    size_t vocab_size() const noexcept { return meta_.voc; }
    int32_t forward(const ::ModelForwardInput &input, ::ModelForwardOutput *output);
    tensor_t step_logits() const noexcept { return step_logits_; }
    runtime::kv_cache::KvCacheBase *kv_cache() noexcept { return runtime_.kv_cache.get(); }
    const runtime::kv_cache::KvCacheBase *kv_cache() const noexcept { return runtime_.kv_cache.get(); }
    runtime::kv_cache::KvCacheLayout kv_layout() const noexcept { return runtime_.kv_layout; }
    size_t kv_cache_capacity_tokens() const noexcept { return runtime_.kv_cache_capacity_tokens; }
    int64_t kv_peak_used_tokens() const noexcept { return runtime_.kv_peak_used_tokens; }
    void set_kv_peak_used_tokens(int64_t value) noexcept { runtime_.kv_peak_used_tokens = value; }
    size_t nkvh() const noexcept { return meta_.nkvh; }
    size_t dh() const noexcept { return meta_.dh; }
    llaisysDataType_t dtype() const noexcept { return meta_.dtype; }
    tensor_t kv_layer_k(size_t layer) const;
    tensor_t kv_layer_v(size_t layer) const;

private:
    struct RuntimeState {
        runtime::kv_cache::KvCacheLayout kv_layout{runtime::kv_cache::KvCacheLayout::BLOCK};
        size_t kv_block_size{16};
        size_t max_model_len{0};
        size_t kv_cache_capacity_tokens{0};
        std::unique_ptr<runtime::kv_cache::KvCacheBase> kv_cache{};
        std::unique_ptr<runtime::output::OutputBuffer> output{};
        mutable int64_t kv_peak_used_tokens{0};
    };

    LlaisysQwen2Meta meta_{};
    llaisysDeviceType_t device_type_{LLAISYS_DEVICE_CPU};
    int device_id_{0};

    LlaisysQwen2Weights weights_{};
    bool validated_{false};

    RuntimeState runtime_{};
    runtime::workspace::qwen2_workspace_t workspace_{};
    tensor_t step_logits_{};

    // Zero biases used when the source weights do not provide a bias tensor.
    tensor_t zero_bias_attn_o_{};
    tensor_t zero_bias_attn_q_{};
    tensor_t zero_bias_attn_k_{};
    tensor_t zero_bias_attn_v_{};
    tensor_t zero_bias_mlp_gate_{};
    tensor_t zero_bias_mlp_up_{};
    tensor_t zero_bias_mlp_down_{};
    tensor_t zero_bias_logits_{};
#ifdef ENABLE_NVIDIA_API
    // Switch point for staged FlashInfer migration (NATIVE by default).
    ops::cuda::PagedAttentionBackend paged_attn_backend_{ops::cuda::PagedAttentionBackend::NATIVE};
#endif
    struct AttentionExecState {
        bool paged_attention{false};
        int32_t block_table_width{0};
        tensor_t attn_mask{};
        tensor_t q_seq_rows{};
        tensor_t q_pos{};
        tensor_t slot_mapping{};
        tensor_t seq_lens{};
        tensor_t block_tables{};
        std::vector<int32_t> slot_idxs{};
        std::vector<int32_t> used_slots{};
    };

    void init_weight_slots_();
    void init_runtime_state_();
    void validate_or_die_();
    void ensure_workspace_(size_t ntoken);

    tensor_t slice_tokens_(const tensor_t &t, size_t len) const;
    tensor_t view_2d_to_3d_(const tensor_t &t, size_t len, size_t nhead, size_t dim) const;

    void fill_pos_ids_from_values_(const tensor_t &pos_ids, const std::vector<int64_t> &pos_values);
    void build_hidden_and_pos_(const std::vector<int64_t> &tokens,
                               const std::vector<int64_t> &pos_values,
                               tensor_t *hidden,
                               tensor_t *pos_ids);
    int32_t prepare_slot_attention_state_(size_t ntoken,
                                          const tensor_t &seq_ids_t,
                                          const tensor_t &pos_ids_host_t,
                                          AttentionExecState *state);
    int32_t validate_and_bind_block_attention_state_(const ::AttentionMetadata &attn,
                                                     size_t ntoken,
                                                     AttentionExecState *state);
    void copy_token_into_cache_(tensor_t &cache, int32_t slot, const tensor_t &src, size_t token_idx);
    tensor_t gather_cache_by_slots_(const tensor_t &cache, const std::vector<int32_t> &slots, size_t len, const tensor_t &buffer);
    tensor_t run_attention_layer_(size_t layer,
                                  size_t ntoken,
                                  const tensor_t &attn_normed,
                                  const tensor_t &pos_ids,
                                  const AttentionExecState &attn_state);

    tensor_t create_zero_tensor_(const std::vector<size_t> &shape, llaisysDataType_t dtype) const;

    void check_meta_invariants_() const;
    void check_tensor_(const llaisysTensor_t handle,
                       const std::vector<size_t> &shape,
                       const char *name,
                       bool required) const;
    tensor_t bias_or_zero_(llaisysTensor_t handle, const tensor_t &zero_bias) const;

    void destroy_weights_();
};

} // namespace llaisys::models::qwen2
