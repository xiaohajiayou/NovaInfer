#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstdint>
#include <vector>

namespace llaisys::ops::cuda {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t k_cache,
                          tensor_t v_cache,
                          const std::vector<int32_t> &used_slots,
                          const std::vector<int32_t> &row_ptr,
                          const std::vector<int32_t> &col_idx,
                          float scale);

} // namespace llaisys::ops::cuda
