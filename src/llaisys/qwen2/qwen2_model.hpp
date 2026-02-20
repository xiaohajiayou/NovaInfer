#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/runtime/infer_types.h"

#include "../llaisys_tensor.hpp"
#include "../runtime/kv_cache/kv_cache.hpp"
#include "../runtime/output/output.hpp"
#include "../runtime/workspace/workspace.hpp"
#include "../runtime/weights/weights.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
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

namespace llaisys::models::qwen2 {

class Qwen2Model {
public:
    struct KvStatsSnapshot {
        int64_t capacity_tokens{0};
        int64_t used_tokens{0};
        int64_t free_tokens{0};
        int64_t peak_used_tokens{0};
    };

    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device,
               int *device_ids,
               int ndevice,
               runtime::kv_cache::KvCacheLayout kv_layout,
               size_t kv_block_size,
               size_t kv_cache_capacity_tokens);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights() noexcept { return &weights_; }
    size_t nlayer() const noexcept { return meta_.nlayer; }
    int32_t decode(const LlaisysBatch &batch);
    float *logits() noexcept;
    float *logits_ith(int32_t i) noexcept;
    int32_t n_outputs() const noexcept;
    const int32_t *output_ids() const noexcept;

    // KV management APIs exposed through C wrapper:
    // - SLOT layout: kv_seq_* are fully implemented.
    // - BLOCK layout: kv_seq_* are compatibility APIs and may return INTERNAL_ERROR
    //   if the underlying KV implementation does not support that operation.
    runtime::kv_cache::KvStatus kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1);
    runtime::kv_cache::KvStatus kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1);
    runtime::kv_cache::KvStatus kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);
    runtime::kv_cache::KvStatus kv_seq_keep(int64_t seq_id);
    int64_t kv_seq_pos_max(int64_t seq_id) const noexcept;
    runtime::kv_cache::KvStatus request_free(int64_t seq_id);
    runtime::kv_cache::KvStatus kv_reset_prefix_cache();
    bool kv_stats(KvStatsSnapshot *out) const noexcept;

private:
    LlaisysQwen2Meta meta_{};
    llaisysDeviceType_t device_type_{LLAISYS_DEVICE_CPU};
    int device_id_{0};
    runtime::kv_cache::KvCacheLayout kv_layout_{runtime::kv_cache::KvCacheLayout::BLOCK};
    size_t kv_block_size_{16};
    size_t kv_cache_capacity_tokens_{0};

    LlaisysQwen2Weights weights_{};
    bool validated_{false};

    std::unique_ptr<runtime::kv_cache::KvCacheBase> kv_cache_{};
    std::unique_ptr<runtime::output::OutputBuffer> output_{};
    runtime::workspace::qwen2_workspace_t workspace_{};

    // Zero biases used when the source weights do not provide a bias tensor.
    tensor_t zero_bias_attn_o_{};
    tensor_t zero_bias_attn_q_{};
    tensor_t zero_bias_attn_k_{};
    tensor_t zero_bias_attn_v_{};
    tensor_t zero_bias_mlp_gate_{};
    tensor_t zero_bias_mlp_up_{};
    tensor_t zero_bias_mlp_down_{};
    tensor_t zero_bias_logits_{};

    bool validate_decode_batch_(const LlaisysBatch &batch) const;

    void init_weight_slots_();
    void init_kv_cache_();
    void validate_or_die_();
    void ensure_workspace_(size_t ntoken);

    tensor_t slice_tokens_(const tensor_t &t, size_t len) const;
    tensor_t view_2d_to_3d_(const tensor_t &t, size_t len, size_t nhead, size_t dim) const;

    void fill_pos_ids_from_values_(const tensor_t &pos_ids, const std::vector<int64_t> &pos_values);
    void copy_token_into_cache_(tensor_t &cache, int32_t slot, const tensor_t &src, size_t token_idx);
    tensor_t gather_cache_by_slots_(const tensor_t &cache, const std::vector<int32_t> &slots, size_t len, const tensor_t &buffer);

    tensor_t create_zero_tensor_(const std::vector<size_t> &shape, llaisysDataType_t dtype) const;
    void run_layers_and_collect_(const LlaisysBatch &batch,
                                 size_t ntoken,
                                 tensor_t hidden,
                                 tensor_t pos_ids,
                                 const std::vector<int32_t> &slot_idxs,
                                 const std::vector<int32_t> &used_slots,
                                 tensor_t attn_mask,
                                 const std::vector<int32_t> &attn_row_ptr,
                                 const std::vector<int32_t> &attn_col_idx,
                                 bool paged_attention);
    int32_t decode_slot_path_(const LlaisysBatch &batch,
                              size_t ntoken,
                              const std::vector<std::vector<int64_t>> &seq_sets,
                              const std::vector<int64_t> &pos_values,
                              const std::vector<int32_t> &nseq_values,
                              tensor_t hidden,
                              tensor_t pos_ids);
    int32_t decode_block_path_(const LlaisysBatch &batch,
                               size_t ntoken,
                               const std::vector<int64_t> &seq_ids_flat,
                               const std::vector<int64_t> &pos_values,
                               tensor_t hidden,
                               tensor_t pos_ids);

    void check_meta_invariants_() const;
    void check_tensor_(const llaisysTensor_t handle,
                       const std::vector<size_t> &shape,
                       const char *name,
                       bool required) const;
    tensor_t bias_or_zero_(llaisysTensor_t handle, const tensor_t &zero_bias) const;

    void destroy_weights_();
    mutable int64_t kv_peak_used_tokens_{0};
};

} // namespace llaisys::models::qwen2
