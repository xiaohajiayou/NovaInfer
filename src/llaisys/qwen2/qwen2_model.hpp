#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/runtime/infer_types.h"

#include "../llaisys_tensor.hpp"
#include "../runtime/kv_cache/kv_cache.hpp"
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
#include <cstdio>
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
    int bind_parallel_context(int32_t tp_size,
                              int32_t tp_rank,
                              int32_t local_rank,
                              const int *device_ids,
                              int32_t ndevice,
                              const char *distributed_backend,
                              const char *init_method,
                              int32_t use_single_process_tp);

    LlaisysQwen2Weights *weights() noexcept { return &weights_; }
    size_t nlayer() const noexcept { return meta_.nlayer; }
    size_t vocab_size() const noexcept { return meta_.voc; }
    llaisysDeviceType_t device_type() const noexcept { return device_type_; }
    int device_id() const noexcept { return device_id_; }
    bool bind_kv_state_handle(const void *kv_state_handle) noexcept;
    int32_t forward(const ::ModelForwardInput &input, ::ModelForwardOutput *output);
    tensor_t step_logits() const noexcept { return step_logits_; }
    runtime::kv_cache::KvCacheBase *kv_cache() noexcept { return runtime_.kv_cache.get(); }
    const runtime::kv_cache::KvCacheBase *kv_cache() const noexcept { return runtime_.kv_cache.get(); }
    runtime::kv_cache::KvCacheLayout kv_layout() const noexcept { return runtime_.kv_layout; }
    size_t kv_cache_capacity_tokens() const noexcept { return runtime_.kv_cache_capacity_tokens; }
    int64_t kv_peak_used_tokens() const noexcept { return runtime_.kv_peak_used_tokens; }
    void set_kv_peak_used_tokens(int64_t value) noexcept { runtime_.kv_peak_used_tokens = value; }
    size_t nkvh() const noexcept { return tp_nkvh_local_ > 0 ? tp_nkvh_local_ : meta_.nkvh; }
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
        mutable int64_t kv_peak_used_tokens{0};
    };

    LlaisysQwen2Meta meta_{};
    llaisysDeviceType_t device_type_{LLAISYS_DEVICE_CPU};
    int device_id_{0};
    int32_t tp_size_{1};
    int32_t tp_rank_{0};
    int32_t local_rank_{0};
    int32_t use_single_process_tp_{0};
    bool parallel_bound_{false};
    std::vector<int> tp_device_ids_{};
    std::string distributed_backend_{};
    std::string tp_init_method_{};
    size_t tp_nh_local_{0};
    size_t tp_nkvh_local_{0};
    size_t tp_di_local_{0};

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
    ops::cuda::CudnnRuntimeState *cudnn_runtime_state_{nullptr};
#endif
    const void *bound_kv_state_handle_{nullptr};
    struct AttentionExecState {
        bool paged_attention{false};
        bool use_cudnn_backend{false};
        int32_t attention_phase{0}; // AttentionPhase::PREFILL
        int32_t n_batch_seq{0};
        int32_t block_table_width{0};
        int32_t max_seqlen_q{0};
        int32_t max_seqlen_k{0};
        tensor_t attn_mask{};
        tensor_t cu_seqlens_q{};
        tensor_t cu_seqlens_k{};
        tensor_t slot_mapping{};
        tensor_t block_tables{};
        tensor_t cudnn_seq_lens_q{};
        tensor_t cudnn_seq_lens_kv{};
        tensor_t cudnn_page_table{};
        tensor_t cudnn_qo_ragged_offset{};
        int32_t cudnn_b_exec{0};
        std::vector<int32_t> slot_idxs{};
        std::vector<int32_t> used_slots{};
    };

    void init_weight_slots_();
    void init_runtime_state_();
    void validate_or_die_();
    void ensure_workspace_(size_t ntoken);

    tensor_t slice_tokens_(const tensor_t &t, size_t len) const;
    tensor_t view_2d_to_3d_(const tensor_t &t, size_t len, size_t nhead, size_t dim) const;

    int32_t prepare_slot_attention_state_(size_t ntoken,
                                          const tensor_t &seq_ids_t,
                                          const tensor_t &pos_ids_host_t,
                                          AttentionExecState *state);
    int32_t validate_and_bind_block_attention_state_(const ::AttentionMetadata &attn,
                                                     size_t ntoken,
                                                     AttentionExecState *state);
    void copy_token_into_cache_(tensor_t &cache, int32_t slot, const tensor_t &src, size_t token_idx);
    tensor_t gather_cache_by_slots_(const tensor_t &cache, const std::vector<int32_t> &slots, size_t len, const tensor_t &buffer);
    tensor_t run_slot_attention_layer_(size_t layer,
                                       size_t ntoken,
                                       const tensor_t &attn_normed,
                                       const tensor_t &pos_ids,
                                       const AttentionExecState &attn_state);
    tensor_t run_block_attention_layer_(size_t layer,
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
    int init_nccl_comm_();
    int tp_allreduce_sum_(const tensor_t &tensor) const;
#ifdef ENABLE_NCCL_API
    void *tp_nccl_comm_{nullptr};
#endif

    void destroy_weights_();
};

} // namespace llaisys::models::qwen2
