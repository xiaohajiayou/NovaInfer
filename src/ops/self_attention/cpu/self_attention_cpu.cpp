#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops::cpu {

template<typename T>
void self_attention_(
    T* out,       // [seqlen, nhead, head_kv_dim]
    const T* q,   // [seqlen, nhead, head_dim]
    const T* k,   // [kvlen, nkvhead, head_kv_dim]
    const T* v,   // [kvlen, nkvhead, head_kv_dim]
    size_t seqlen, 
    size_t kvlen,
    size_t nhead,
    size_t nkvhead,
    size_t head_dim,
    float scale)
{
    size_t group_size = nhead / nkvhead;

    std::vector<float> scores(kvlen);  
    // 整体计算 QK^T 得分，维度为[seq_len * kv_len], 每轮计算一个 q head 的得分，维度为[kv_len]

    for (size_t t = 0; t < seqlen; ++t) {
        for (size_t q_h_idx = 0; q_h_idx < nhead; ++q_h_idx) {

            size_t kv_h_idx = q_h_idx / group_size;  // 计算每个 q head 对应的 kv head 索引

            const T* q_ptr = q + (t * nhead + q_h_idx) * head_dim;
            T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;

            // ---- 1. 计算 QK^T ----
            for (size_t i = 0; i < kvlen; ++i) {
                ptrdiff_t offset = static_cast<ptrdiff_t>(kvlen) - static_cast<ptrdiff_t>(seqlen);
                if (static_cast<ptrdiff_t>(i) > static_cast<ptrdiff_t>(t) + offset) {
                    scores[i] = -1e30f;
                    continue;
                }
                

                const T* k_ptr = k + (i * nkvhead + kv_h_idx) * head_dim;
                //第i个时序位置，对应kv头的偏移位置

                float dot = 0.f;
                for (size_t j = 0; j < head_dim; ++j) {
                    dot += llaisys::utils::cast<float>(q_ptr[j]) * llaisys::utils::cast<float>(k_ptr[j]);
                }
                scores[i] = dot * scale;
            }

            // ---- 2. softmax ----
            float maxv = -1e30f;
            for (size_t i = 0; i < kvlen; ++i)
                maxv = std::max(maxv, scores[i]);

            float sum = 0.f;
            for (size_t i = 0; i < kvlen; ++i) {
                scores[i] = std::exp(scores[i] - maxv);
                sum += scores[i];
            }

            for (size_t i = 0; i < kvlen; ++i)
                scores[i] /= sum;

            // ---- 3. 加权 V ----
            for (size_t i = 0; i < head_dim; ++i) {
                float acc = 0.f;
                for (size_t j = 0; j < kvlen; ++j) {
                    const T* v_ptr = v + (j * nkvhead + kv_h_idx) * head_dim;
                    acc += scores[j] * llaisys::utils::cast<float>(v_ptr[i]);
                }
                out_ptr[i] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}
    

} // namespace llaisys::ops::cpu


namespace llaisys::ops::cpu {
void self_attention(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
    k->isContiguous() && v->isContiguous(),
    "self_attention: tensors must be contiguous");

    size_t seqlen   = q->shape()[0];
    size_t nhead    = q->shape()[1];
    size_t head_dim        = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead   = k->shape()[1];
    size_t head_kv_dim        = k->shape()[2];

    ASSERT(head_dim == head_kv_dim, "head_dim must equal head_kv_dim");
    ASSERT(attn_val->shape()[0] == seqlen, "attn_val seqlen mismatch");
    ASSERT(attn_val->shape()[1] == nhead,  "attn_val head mismatch");
    ASSERT(attn_val->shape()[2] == head_dim,     "attn_val dv mismatch");

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float*>(attn_val->data()),
            reinterpret_cast<const float*>(q->data()),
            reinterpret_cast<const float*>(k->data()),
            reinterpret_cast<const float*>(v->data()),
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::bf16_t*>(q->data()),
            reinterpret_cast<const llaisys::bf16_t*>(k->data()),
            reinterpret_cast<const llaisys::bf16_t*>(v->data()),
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::fp16_t*>(q->data()),
            reinterpret_cast<const llaisys::fp16_t*>(k->data()),
            reinterpret_cast<const llaisys::fp16_t*>(v->data()),
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops::cpu