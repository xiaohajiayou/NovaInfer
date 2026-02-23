#pragma once

#include "../../tensor/tensor.hpp"
#include <cstdint>
#include <vector>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void self_attention_masked(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, tensor_t mask, float scale);
void self_attention_masked_csr(tensor_t attn_val,
                               tensor_t q,
                               tensor_t k,
                               tensor_t v,
                               const std::vector<int32_t> &row_ptr,
                               const std::vector<int32_t> &col_idx,
                               float scale);
void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t k_cache,
                          tensor_t v_cache,
                          const std::vector<int32_t> &q_seq_rows,
                          const std::vector<int32_t> &q_pos,
                          const std::vector<int32_t> &block_tables,
                          const std::vector<int32_t> &seq_lens,
                          int32_t block_table_width,
                          int32_t block_size,
                          float scale);
}
