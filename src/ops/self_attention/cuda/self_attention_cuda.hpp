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

enum class AttentionPhase : int32_t {
    PREFILL = 0,
    DECODE = 1,
};

struct CommonAttentionMetadata {
    const int32_t *cu_seqlens_q{nullptr};
    const int32_t *cu_seqlens_k{nullptr};
    const int32_t *block_tables{nullptr};
    const int32_t *slot_mapping{nullptr};
    // CUDNN-only metadata prepared by Python BLOCK builder.
    const int32_t *cudnn_seq_lens_q{nullptr};
    const int32_t *cudnn_seq_lens_kv{nullptr};
    const int32_t *cudnn_page_table{nullptr};
    const int32_t *cudnn_qo_ragged_offset{nullptr};
    int32_t cudnn_b_exec{0};
    int32_t nseq{0};
    int32_t max_seqlen_q{0};
    int32_t max_seqlen_k{0};
    AttentionPhase phase{AttentionPhase::PREFILL};
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
                          tensor_t cu_seqlens_q,
                          tensor_t cu_seqlens_k,
                          tensor_t block_tables,
                          tensor_t slot_mapping,
                          int32_t max_seqlen_q,
                          int32_t max_seqlen_k,
                          int32_t block_table_width,
                          int32_t block_size,
                          float scale);

void build_slot_mapping(tensor_t slot_mapping,
                        tensor_t query_start_loc,
                        tensor_t seq_lens,
                        tensor_t block_tables,
                        int32_t block_table_width,
                        int32_t block_size);

void build_query_start_loc_and_seq_lens(tensor_t query_start_loc,
                                        tensor_t seq_lens,
                                        tensor_t req_num_scheduled_tokens,
                                        tensor_t req_num_computed_tokens);

void build_block_positions(tensor_t pos_ids,
                           tensor_t query_start_loc,
                           tensor_t seq_lens);

void build_last_token_logits_indices(tensor_t logits_indices,
                                     tensor_t query_start_loc);

} // namespace llaisys::ops::cuda
