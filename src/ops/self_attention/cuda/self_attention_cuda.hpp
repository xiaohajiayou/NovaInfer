#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace llaisys::ops::cuda {

enum class PagedAttentionBackend : int32_t {
    NATIVE = 0,
    FLASHINFER = 1,
    CUDNN = 2,
};

struct PagedAttentionPrepared {
    int32_t *q_seq_rows{nullptr};
    int32_t *q_pos{nullptr};
    int32_t *block_tables{nullptr};
    int32_t *seq_lens{nullptr};
    int32_t *last_page_lens{nullptr};
    size_t q_seq_rows_cap{0};
    size_t q_pos_cap{0};
    size_t block_tables_cap{0};
    size_t seq_lens_cap{0};
    size_t last_page_lens_cap{0};
    int32_t nseq{0};

    // Host mirrors used by backend integrations that need to rebuild token-level
    // page tables/sequence metadata at runtime (e.g., cuDNN frontend SDPA).
    std::vector<int32_t> host_q_seq_rows;
    std::vector<int32_t> host_q_pos;
    std::vector<int32_t> host_block_tables;
    std::vector<int32_t> host_seq_lens;
    std::vector<int32_t> host_last_page_lens;
    uint64_t host_revision{0};

    PagedAttentionPrepared() = default;
    ~PagedAttentionPrepared();
    PagedAttentionPrepared(const PagedAttentionPrepared &) = delete;
    PagedAttentionPrepared &operator=(const PagedAttentionPrepared &) = delete;
};

void prepare_paged_attention(PagedAttentionPrepared &prepared,
                             const std::vector<int32_t> &q_seq_rows,
                             const std::vector<int32_t> &q_pos,
                             const std::vector<int32_t> &block_tables,
                             const std::vector<int32_t> &seq_lens,
                             int32_t block_size,
                             bool upload_device_metadata = true);

void scatter_kv_cache_by_slots(tensor_t k_cache,
                               tensor_t v_cache,
                               tensor_t k_src,
                               tensor_t v_src,
                               const std::vector<int32_t> &slot_idxs);

void scatter_kv_cache_by_slots_device_indices(tensor_t k_cache,
                                              tensor_t v_cache,
                                              tensor_t k_src,
                                              tensor_t v_src,
                                              tensor_t slot_idxs_i32);

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void self_attention_paged_prepared_with_backend(tensor_t attn_val,
                                                tensor_t q,
                                                tensor_t k_cache,
                                                tensor_t v_cache,
                                                const PagedAttentionPrepared &prepared,
                                                PagedAttentionBackend backend,
                                                int32_t block_table_width,
                                                int32_t block_size,
                                                float scale);
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
