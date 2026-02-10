#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/runtime/infer_types.h"

#include "../llaisys_tensor.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../tensor/tensor.hpp"
#include "../../utils/check.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llaisys::models::qwen2 {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device,
               int *device_ids,
               int ndevice);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights() noexcept { return &weights_; }
    int32_t decode(const LlaisysBatch &batch);
    float *logits() noexcept;
    float *logits_ith(int32_t i) noexcept;
    int32_t n_outputs() const noexcept;
    const int32_t *output_ids() const noexcept;

    int kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1);
    int kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1);
    int kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);
    int kv_seq_keep(int64_t seq_id);
    int64_t kv_seq_pos_max(int64_t seq_id) const noexcept;

    int64_t infer(int64_t *token_ids, size_t ntoken);

private:
    struct LayerCache {
        tensor_t k_cache;
        tensor_t v_cache;
    };

    struct Workspace {
        tensor_t hidden;
        tensor_t normed;
        tensor_t q_proj;
        tensor_t k_proj;
        tensor_t v_proj;
        tensor_t q_3d;
        tensor_t k_new_3d;
        tensor_t v_new_3d;
        tensor_t rope_q;
        tensor_t rope_k;
        tensor_t attn_out;
        tensor_t attn_out_2d;
        tensor_t attn_proj;
        tensor_t mlp_normed;
        tensor_t gate;
        tensor_t up;
        tensor_t swiglu;
        tensor_t down;
        tensor_t logits;
        tensor_t pos_ids;
        tensor_t argmax_idx;
        tensor_t argmax_val;
    };

    LlaisysQwen2Meta meta_{};
    llaisysDeviceType_t device_type_{LLAISYS_DEVICE_CPU};
    int device_id_{0};

    LlaisysQwen2Weights weights_{};
    size_t cur_len_{0};
    bool validated_{false};

    std::vector<LayerCache> caches_{};
    Workspace workspace_{};

    // Zero biases used when the source weights do not provide a bias tensor.
    tensor_t zero_bias_attn_o_{};
    tensor_t zero_bias_attn_q_{};
    tensor_t zero_bias_attn_k_{};
    tensor_t zero_bias_attn_v_{};
    tensor_t zero_bias_mlp_gate_{};
    tensor_t zero_bias_mlp_up_{};
    tensor_t zero_bias_mlp_down_{};
    tensor_t zero_bias_logits_{};

    size_t workspace_token_cap_{0};
    std::vector<float> output_logits_f32_{};
    std::vector<int32_t> output_ids_{};

    bool validate_decode_batch_(const LlaisysBatch &batch) const;
    void materialize_outputs_(size_t ntoken, const int8_t *logits_mask);

    void init_weight_slots_();
    void init_kv_cache_();
    void validate_or_die_();
    void ensure_workspace_(size_t ntoken);

    tensor_t slice_tokens_(const tensor_t &t, size_t len) const;
    tensor_t view_2d_to_3d_(const tensor_t &t, size_t len, size_t nhead, size_t dim) const;

    void fill_pos_ids_(const tensor_t &pos_ids, size_t start, size_t len);
    void copy_into_cache_(tensor_t &cache, size_t start, const tensor_t &src);

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
