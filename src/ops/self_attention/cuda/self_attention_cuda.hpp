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

struct CommonAttentionMetadata {
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

    uint64_t revision{0};

    CommonAttentionMetadata() = default;
    ~CommonAttentionMetadata();
    CommonAttentionMetadata(const CommonAttentionMetadata &) = delete;
    CommonAttentionMetadata &operator=(const CommonAttentionMetadata &) = delete;
};

void build_attn_metadata(CommonAttentionMetadata &metadata,
                         const std::vector<int32_t> &q_seq_rows,
                         const std::vector<int32_t> &q_pos,
                         const std::vector<int32_t> &block_tables,
                         const std::vector<int32_t> &seq_lens,
                         int32_t block_size,
                         bool upload_device_metadata = true);
void build_attn_metadata(CommonAttentionMetadata &metadata,
                         const std::vector<int32_t> &q_seq_rows,
                         const std::vector<int32_t> &q_pos,
                         const int32_t *block_tables,
                         size_t block_tables_len,
                         const int32_t *seq_lens,
                         size_t seq_lens_len,
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
void dispatch_attention_with_backend(tensor_t attn_val,
                                     tensor_t q,
                                     tensor_t k_cache,
                                     tensor_t v_cache,
                                     const CommonAttentionMetadata &metadata,
                                     PagedAttentionBackend backend,
                                     int32_t block_table_width,
                                     int32_t block_size,
                                     float scale);
void self_attention_paged_prepared(tensor_t attn_val,
                                   tensor_t q,
                                   tensor_t k_cache,
                                   tensor_t v_cache,
                                   const CommonAttentionMetadata &prepared,
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

// Backward-compatible aliases for existing call sites.
using PagedAttentionPrepared = CommonAttentionMetadata;
inline void prepare_paged_attention(CommonAttentionMetadata &prepared,
                                    const std::vector<int32_t> &q_seq_rows,
                                    const std::vector<int32_t> &q_pos,
                                    const std::vector<int32_t> &block_tables,
                                    const std::vector<int32_t> &seq_lens,
                                    int32_t block_size,
                                    bool upload_device_metadata = true) {
    build_attn_metadata(prepared, q_seq_rows, q_pos, block_tables, seq_lens, block_size, upload_device_metadata);
}
inline void self_attention_paged_prepared_with_backend(tensor_t attn_val,
                                                       tensor_t q,
                                                       tensor_t k_cache,
                                                       tensor_t v_cache,
                                                       const CommonAttentionMetadata &prepared,
                                                       PagedAttentionBackend backend,
                                                       int32_t block_table_width,
                                                       int32_t block_size,
                                                       float scale) {
    dispatch_attention_with_backend(
        attn_val, q, k_cache, v_cache, prepared, backend, block_table_width, block_size, scale);
}

} // namespace llaisys::ops::cuda
