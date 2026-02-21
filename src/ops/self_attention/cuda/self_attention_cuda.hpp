#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstdint>
#include <vector>

namespace llaisys::ops::cuda {

struct PagedAttentionPrepared {
    int32_t *q_seq_rows{nullptr};
    int32_t *q_pos{nullptr};
    int32_t *block_tables{nullptr};
    int32_t *seq_lens{nullptr};
    size_t q_seq_rows_cap{0};
    size_t q_pos_cap{0};
    size_t block_tables_cap{0};
    size_t seq_lens_cap{0};
    int32_t nseq{0};

    PagedAttentionPrepared() = default;
    ~PagedAttentionPrepared();
    PagedAttentionPrepared(const PagedAttentionPrepared &) = delete;
    PagedAttentionPrepared &operator=(const PagedAttentionPrepared &) = delete;
};

void prepare_paged_attention(PagedAttentionPrepared &prepared,
                             const std::vector<int32_t> &q_seq_rows,
                             const std::vector<int32_t> &q_pos,
                             const std::vector<int32_t> &block_tables,
                             const std::vector<int32_t> &seq_lens);

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void self_attention_paged_prepared(tensor_t attn_val,
                                   tensor_t q,
                                   tensor_t k_cache,
                                   tensor_t v_cache,
                                   const PagedAttentionPrepared &prepared,
                                   int32_t block_table_width,
                                   int32_t block_size,
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

} // namespace llaisys::ops::cuda
