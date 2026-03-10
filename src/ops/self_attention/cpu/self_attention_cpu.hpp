#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
    void self_attention(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        float scale);
    void self_attention_masked(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        tensor_t mask,
        float scale);
    void self_attention_masked_csr(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        const std::vector<int32_t>& row_ptr,
        const std::vector<int32_t>& col_idx,
        float scale);
    void self_attention_paged(tensor_t attn_val,
        tensor_t q,
        tensor_t k_cache,
        tensor_t v_cache,
        tensor_t cu_seqlens_q,
        tensor_t cu_seqlens_k,
        tensor_t block_tables,
        tensor_t slot_mapping,
        int32_t max_seqlen_q,
        int32_t max_seqlen_k,
        int32_t block_table_width,
        int32_t block_size,
        float scale);
}
