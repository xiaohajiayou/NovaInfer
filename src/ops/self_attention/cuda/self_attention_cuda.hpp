#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstdint>
#include <string>

namespace llaisys::ops::cuda {

enum class PagedAttentionBackend : int32_t {
    NATIVE = 0,
    FLASHINFER = 1,
    CUDNN = 2,
};

struct CommonAttentionMetadata {
    const int32_t *q_seq_rows{nullptr};
    const int32_t *q_pos{nullptr};
    const int32_t *block_tables{nullptr};
    const int32_t *seq_lens{nullptr};
    int32_t nseq{0};
};

void reshape_and_cache(tensor_t k_cache,
                       tensor_t v_cache,
                       tensor_t k_src,
                       tensor_t v_src,
                       tensor_t slot_mapping);

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void dispatch_attention_with_backend(tensor_t attn_val,
                                     tensor_t q,
                                     tensor_t k_cache,
                                     tensor_t v_cache,
                                     const CommonAttentionMetadata &metadata,
                                     PagedAttentionBackend backend,
                                     int32_t block_table_width,
                                     int32_t block_size,
                                     float scale);
void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t k_cache,
                          tensor_t v_cache,
                          tensor_t q_seq_rows,
                          tensor_t q_pos,
                          tensor_t block_tables,
                          tensor_t seq_lens,
                          int32_t block_table_width,
                          int32_t block_size,
                          float scale);

} // namespace llaisys::ops::cuda
