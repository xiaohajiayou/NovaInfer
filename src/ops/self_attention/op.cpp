#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/self_attention_cuda.hpp"
#endif
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3-D.");
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3-D.");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3-D.");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3-D.");
    ASSERT(attn_val->shape()[0] == q->shape()[0], "SelfAttention: batch size mismatch.");
    ASSERT(attn_val->shape()[1] == q->shape()[1], "SelfAttention: seqlen mismatch.");
    ASSERT(attn_val->shape()[2] == q->shape()[2], "SelfAttention: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == k->shape()[2], "SelfAttention: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == v->shape()[2], "SelfAttention: head dim mismatch.");
    CHECK_SAME_DTYPE(q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val, q, k, v, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::self_attention(attn_val, q, k, v, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void self_attention_masked(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, tensor_t mask, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v, mask);
    ASSERT(attn_val->ndim() == 3, "SelfAttentionMasked: attn_val must be 3-D.");
    ASSERT(q->ndim() == 3, "SelfAttentionMasked: q must be 3-D.");
    ASSERT(k->ndim() == 3, "SelfAttentionMasked: k must be 3-D.");
    ASSERT(v->ndim() == 3, "SelfAttentionMasked: v must be 3-D.");
    ASSERT(mask->ndim() == 2, "SelfAttentionMasked: mask must be 2-D.");
    ASSERT(attn_val->shape()[0] == q->shape()[0], "SelfAttentionMasked: batch size mismatch.");
    ASSERT(attn_val->shape()[1] == q->shape()[1], "SelfAttentionMasked: seqlen mismatch.");
    ASSERT(attn_val->shape()[2] == q->shape()[2], "SelfAttentionMasked: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == k->shape()[2], "SelfAttentionMasked: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == v->shape()[2], "SelfAttentionMasked: head dim mismatch.");
    ASSERT(mask->shape()[0] == q->shape()[0], "SelfAttentionMasked: mask seqlen mismatch.");
    ASSERT(mask->shape()[1] == k->shape()[0], "SelfAttentionMasked: mask kvlen mismatch.");
    ASSERT(mask->dtype() == LLAISYS_DTYPE_U8 || mask->dtype() == LLAISYS_DTYPE_BOOL,
           "SelfAttentionMasked: mask dtype must be U8/BOOL.");
    CHECK_SAME_DTYPE(q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous() &&
               mask->isContiguous(),
           "SelfAttentionMasked: all tensors must be contiguous.");
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention_masked(attn_val, q, k, v, mask, scale);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void self_attention_masked_csr(tensor_t attn_val,
                               tensor_t q,
                               tensor_t k,
                               tensor_t v,
                               const std::vector<int32_t> &row_ptr,
                               const std::vector<int32_t> &col_idx,
                               float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->ndim() == 3, "SelfAttentionMaskedCSR: attn_val must be 3-D.");
    ASSERT(q->ndim() == 3, "SelfAttentionMaskedCSR: q must be 3-D.");
    ASSERT(k->ndim() == 3, "SelfAttentionMaskedCSR: k must be 3-D.");
    ASSERT(v->ndim() == 3, "SelfAttentionMaskedCSR: v must be 3-D.");
    ASSERT(attn_val->shape()[0] == q->shape()[0], "SelfAttentionMaskedCSR: batch size mismatch.");
    ASSERT(attn_val->shape()[1] == q->shape()[1], "SelfAttentionMaskedCSR: seqlen mismatch.");
    ASSERT(attn_val->shape()[2] == q->shape()[2], "SelfAttentionMaskedCSR: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == k->shape()[2], "SelfAttentionMaskedCSR: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == v->shape()[2], "SelfAttentionMaskedCSR: head dim mismatch.");
    CHECK_SAME_DTYPE(q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttentionMaskedCSR: all tensors must be contiguous.");
    ASSERT(row_ptr.size() == q->shape()[0] + 1, "SelfAttentionMaskedCSR: row_ptr size mismatch.");
    ASSERT(!row_ptr.empty(), "SelfAttentionMaskedCSR: row_ptr must be non-empty.");
    ASSERT(row_ptr.front() == 0, "SelfAttentionMaskedCSR: row_ptr must start at 0.");
    ASSERT(row_ptr.back() == static_cast<int32_t>(col_idx.size()),
           "SelfAttentionMaskedCSR: row_ptr end must equal col_idx size.");
    for (size_t i = 1; i < row_ptr.size(); ++i) {
        ASSERT(row_ptr[i] >= row_ptr[i - 1], "SelfAttentionMaskedCSR: row_ptr must be non-decreasing.");
    }
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention_masked_csr(attn_val, q, k, v, row_ptr, col_idx, scale);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

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
                          float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k_cache, v_cache);
    ASSERT(attn_val->ndim() == 3, "SelfAttentionPaged: attn_val must be 3-D.");
    ASSERT(q->ndim() == 3, "SelfAttentionPaged: q must be 3-D.");
    ASSERT(k_cache->ndim() == 3, "SelfAttentionPaged: k_cache must be 3-D.");
    ASSERT(v_cache->ndim() == 3, "SelfAttentionPaged: v_cache must be 3-D.");
    ASSERT(attn_val->shape()[0] == q->shape()[0], "SelfAttentionPaged: batch size mismatch.");
    ASSERT(attn_val->shape()[1] == q->shape()[1], "SelfAttentionPaged: seqlen mismatch.");
    ASSERT(attn_val->shape()[2] == q->shape()[2], "SelfAttentionPaged: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == k_cache->shape()[2], "SelfAttentionPaged: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == v_cache->shape()[2], "SelfAttentionPaged: head dim mismatch.");
    CHECK_SAME_DTYPE(q->dtype(), k_cache->dtype(), v_cache->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k_cache->isContiguous() && v_cache->isContiguous(),
           "SelfAttentionPaged: all tensors must be contiguous.");
    ASSERT(block_table_width > 0, "SelfAttentionPaged: block_table_width must be > 0.");
    ASSERT(block_size > 0, "SelfAttentionPaged: block_size must be > 0.");
    ASSERT(q_seq_rows != nullptr && q_pos != nullptr && block_tables != nullptr && seq_lens != nullptr,
           "SelfAttentionPaged: metadata tensors must be non-null.");
    ASSERT(q_seq_rows->deviceType() == LLAISYS_DEVICE_CPU && q_pos->deviceType() == LLAISYS_DEVICE_CPU &&
               block_tables->deviceType() == LLAISYS_DEVICE_CPU && seq_lens->deviceType() == LLAISYS_DEVICE_CPU,
           "SelfAttentionPaged: metadata tensors must be CPU.");
    ASSERT(q_seq_rows->dtype() == LLAISYS_DTYPE_I32 && q_pos->dtype() == LLAISYS_DTYPE_I32 &&
               block_tables->dtype() == LLAISYS_DTYPE_I32 && seq_lens->dtype() == LLAISYS_DTYPE_I32,
           "SelfAttentionPaged: metadata dtype must be I32.");
    ASSERT(q_seq_rows->ndim() == 1 && q_pos->ndim() == 1 && block_tables->ndim() == 1 && seq_lens->ndim() == 1,
           "SelfAttentionPaged: metadata tensors must be 1-D.");
    ASSERT(q_seq_rows->isContiguous() && q_pos->isContiguous() && block_tables->isContiguous() && seq_lens->isContiguous(),
           "SelfAttentionPaged: metadata tensors must be contiguous.");
    ASSERT(q_seq_rows->shape()[0] == q->shape()[0], "SelfAttentionPaged: q_seq_rows size mismatch.");
    ASSERT(q_pos->shape()[0] == q->shape()[0], "SelfAttentionPaged: q_pos size mismatch.");
    ASSERT(seq_lens->shape()[0] > 0, "SelfAttentionPaged: seq_lens must be non-empty.");
    ASSERT(block_tables->shape()[0] == seq_lens->shape()[0] * static_cast<size_t>(block_table_width),
           "SelfAttentionPaged: block_tables size mismatch.");
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention_paged(
            attn_val,
            q,
            k_cache,
            v_cache,
            q_seq_rows,
            q_pos,
            block_tables,
            seq_lens,
            block_table_width,
            block_size,
            scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::self_attention_paged(
            attn_val,
            q,
            k_cache,
            v_cache,
            q_seq_rows,
            q_pos,
            block_tables,
            seq_lens,
            block_table_width,
            block_size,
            scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
