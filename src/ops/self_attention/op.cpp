#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"
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
} // namespace llaisys::ops
