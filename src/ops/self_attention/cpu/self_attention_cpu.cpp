#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

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
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t q_h_i = 0; q_h_i < static_cast<int64_t>(nhead); ++q_h_i) {
            const size_t q_h_idx = static_cast<size_t>(q_h_i);

            size_t kv_h_idx = q_h_idx / group_size;  // 计算每个 q head 对应的 kv head 索引

            const T* q_ptr = q + (t * nhead + q_h_idx) * head_dim;
            T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;
            std::vector<float> scores(kvlen);

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

template<typename T>
void self_attention_masked_(
    T* out,       // [seqlen, nhead, head_kv_dim]
    const T* q,   // [seqlen, nhead, head_dim]
    const T* k,   // [kvlen, nkvhead, head_kv_dim]
    const T* v,   // [kvlen, nkvhead, head_kv_dim]
    const uint8_t* mask, // [seqlen, kvlen], 1 means visible
    size_t seqlen,
    size_t kvlen,
    size_t nhead,
    size_t nkvhead,
    size_t head_dim,
    float scale)
{
    size_t group_size = nhead / nkvhead;
    std::vector<size_t> visible_cols;
    visible_cols.reserve(kvlen);

    for (size_t t = 0; t < seqlen; ++t) {
        const uint8_t* mask_row = mask + t * kvlen;

        visible_cols.clear();
        for (size_t i = 0; i < kvlen; ++i) {
            if (mask_row[i] != 0) {
                visible_cols.push_back(i);
            }
        }
        const size_t n_visible = visible_cols.size();

        if (n_visible == 0) {
            for (size_t q_h_idx = 0; q_h_idx < nhead; ++q_h_idx) {
                T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;
                for (size_t i = 0; i < head_dim; ++i) {
                    out_ptr[i] = llaisys::utils::cast<T>(0.f);
                }
            }
            continue;
        }

        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t q_h_i = 0; q_h_i < static_cast<int64_t>(nhead); ++q_h_i) {
            const size_t q_h_idx = static_cast<size_t>(q_h_i);
            size_t kv_h_idx = q_h_idx / group_size;
            const T* q_ptr = q + (t * nhead + q_h_idx) * head_dim;
            T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;
            std::vector<float> scores(n_visible);

            for (size_t vi = 0; vi < n_visible; ++vi) {
                const size_t i = visible_cols[vi];
                const T* k_ptr = k + (i * nkvhead + kv_h_idx) * head_dim;
                float dot = 0.f;
                for (size_t j = 0; j < head_dim; ++j) {
                    dot += llaisys::utils::cast<float>(q_ptr[j]) * llaisys::utils::cast<float>(k_ptr[j]);
                }
                scores[vi] = dot * scale;
            }

            float maxv = -1e30f;
            for (size_t vi = 0; vi < n_visible; ++vi) {
                maxv = std::max(maxv, scores[vi]);
            }

            float sum = 0.f;
            for (size_t vi = 0; vi < n_visible; ++vi) {
                scores[vi] = std::exp(scores[vi] - maxv);
                sum += scores[vi];
            }
            for (size_t vi = 0; vi < n_visible; ++vi) {
                scores[vi] /= sum;
            }

            for (size_t i = 0; i < head_dim; ++i) {
                float acc = 0.f;
                for (size_t vi = 0; vi < n_visible; ++vi) {
                    const size_t j = visible_cols[vi];
                    const T* v_ptr = v + (j * nkvhead + kv_h_idx) * head_dim;
                    acc += scores[vi] * llaisys::utils::cast<float>(v_ptr[i]);
                }
                out_ptr[i] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

template<typename T>
void self_attention_masked_csr_(
    T* out,       // [seqlen, nhead, head_kv_dim]
    const T* q,   // [seqlen, nhead, head_dim]
    const T* k,   // [kvlen, nkvhead, head_kv_dim]
    const T* v,   // [kvlen, nkvhead, head_kv_dim]
    const int32_t* row_ptr, // [seqlen + 1]
    const int32_t* col_idx, // [nnz]
    size_t seqlen,
    size_t kvlen,
    size_t nhead,
    size_t nkvhead,
    size_t head_dim,
    float scale)
{
    size_t group_size = nhead / nkvhead;
    for (size_t t = 0; t < seqlen; ++t) {
        const int32_t rb = row_ptr[t];
        const int32_t re = row_ptr[t + 1];
        ASSERT(rb >= 0 && re >= rb, "self_attention_masked_csr: invalid row range");
        const size_t n_visible = static_cast<size_t>(re - rb);

        if (n_visible == 0) {
            for (size_t q_h_idx = 0; q_h_idx < nhead; ++q_h_idx) {
                T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;
                for (size_t i = 0; i < head_dim; ++i) {
                    out_ptr[i] = llaisys::utils::cast<T>(0.f);
                }
            }
            continue;
        }

        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t q_h_i = 0; q_h_i < static_cast<int64_t>(nhead); ++q_h_i) {
            const size_t q_h_idx = static_cast<size_t>(q_h_i);
            size_t kv_h_idx = q_h_idx / group_size;
            const T* q_ptr = q + (t * nhead + q_h_idx) * head_dim;
            T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;
            std::vector<float> scores(n_visible);

            for (size_t vi = 0; vi < n_visible; ++vi) {
                const int32_t col = col_idx[static_cast<size_t>(rb) + vi];
                ASSERT(col >= 0 && static_cast<size_t>(col) < kvlen,
                       "self_attention_masked_csr: column index out of range");
                const T* k_ptr = k + (static_cast<size_t>(col) * nkvhead + kv_h_idx) * head_dim;
                float dot = 0.f;
                for (size_t j = 0; j < head_dim; ++j) {
                    dot += llaisys::utils::cast<float>(q_ptr[j]) * llaisys::utils::cast<float>(k_ptr[j]);
                }
                scores[vi] = dot * scale;
            }

            float maxv = -1e30f;
            for (size_t vi = 0; vi < n_visible; ++vi) {
                maxv = std::max(maxv, scores[vi]);
            }

            float sum = 0.f;
            for (size_t vi = 0; vi < n_visible; ++vi) {
                scores[vi] = std::exp(scores[vi] - maxv);
                sum += scores[vi];
            }
            for (size_t vi = 0; vi < n_visible; ++vi) {
                scores[vi] /= sum;
            }

            for (size_t i = 0; i < head_dim; ++i) {
                float acc = 0.f;
                for (size_t vi = 0; vi < n_visible; ++vi) {
                    const int32_t col = col_idx[static_cast<size_t>(rb) + vi];
                    const T* v_ptr = v + (static_cast<size_t>(col) * nkvhead + kv_h_idx) * head_dim;
                    acc += scores[vi] * llaisys::utils::cast<float>(v_ptr[i]);
                }
                out_ptr[i] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

template<typename T>
void self_attention_paged_(
    T* out,       // [seqlen, nhead, head_dim]
    const T* q,   // [seqlen, nhead, head_dim]
    const T* k_cache, // [nslot, nkvhead, head_dim]
    const T* v_cache, // [nslot, nkvhead, head_dim]
    const int32_t* q_seq_rows, // [seqlen], query token -> seq row
    const int32_t* q_pos, // [seqlen], query position in sequence
    const int32_t* block_tables, // [nseq, block_table_width]
    const int32_t* seq_lens, // [nseq]
    int32_t nseq,
    int32_t block_table_width,
    int32_t block_size,
    size_t seqlen,
    size_t nslot,
    size_t nhead,
    size_t nkvhead,
    size_t head_dim,
    float scale)
{
    size_t group_size = nhead / nkvhead;
    for (size_t t = 0; t < seqlen; ++t) {
        const int32_t row = q_seq_rows[t];
        ASSERT(row >= 0 && row < nseq, "self_attention_paged: q_seq_rows out of range");
        const int32_t seq_len = seq_lens[row];
        const int32_t vmax = std::min<int32_t>(q_pos[t], seq_len - 1);
        const size_t n_visible = (vmax >= 0) ? (static_cast<size_t>(vmax) + 1) : 0;

        if (n_visible == 0) {
            for (size_t q_h_idx = 0; q_h_idx < nhead; ++q_h_idx) {
                T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;
                for (size_t i = 0; i < head_dim; ++i) {
                    out_ptr[i] = llaisys::utils::cast<T>(0.f);
                }
            }
            continue;
        }

        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t q_h_i = 0; q_h_i < static_cast<int64_t>(nhead); ++q_h_i) {
            const size_t q_h_idx = static_cast<size_t>(q_h_i);
            const size_t kv_h_idx = q_h_idx / group_size;
            const T* q_ptr = q + (t * nhead + q_h_idx) * head_dim;
            T* out_ptr = out + (t * nhead + q_h_idx) * head_dim;
            std::vector<float> scores(n_visible);
            std::vector<int32_t> visible_slots(n_visible, -1);

            for (int32_t p = 0; p <= vmax; ++p) {
                const int32_t bidx = p / block_size;
                const int32_t boff = p % block_size;
                ASSERT(bidx >= 0 && bidx < block_table_width, "self_attention_paged: block index out of range");
                const int32_t bid = block_tables[row * block_table_width + bidx];
                ASSERT(bid >= 0, "self_attention_paged: invalid block id");
                const int32_t slot = bid * block_size + boff;
                ASSERT(slot >= 0 && static_cast<size_t>(slot) < nslot, "self_attention_paged: slot out of range");
                visible_slots[static_cast<size_t>(p)] = slot;
                const T* k_ptr = k_cache + (static_cast<size_t>(slot) * nkvhead + kv_h_idx) * head_dim;
                float dot = 0.f;
                for (size_t j = 0; j < head_dim; ++j) {
                    dot += llaisys::utils::cast<float>(q_ptr[j]) * llaisys::utils::cast<float>(k_ptr[j]);
                }
                scores[static_cast<size_t>(p)] = dot * scale;
            }

            float maxv = -1e30f;
            for (size_t vi = 0; vi < n_visible; ++vi) {
                maxv = std::max(maxv, scores[vi]);
            }

            float sum = 0.f;
            for (size_t vi = 0; vi < n_visible; ++vi) {
                scores[vi] = std::exp(scores[vi] - maxv);
                sum += scores[vi];
            }
            for (size_t vi = 0; vi < n_visible; ++vi) {
                scores[vi] /= sum;
            }

            for (size_t i = 0; i < head_dim; ++i) {
                float acc = 0.f;
                for (size_t vi = 0; vi < n_visible; ++vi) {
                    const int32_t slot = visible_slots[vi];
                    const T* v_ptr = v_cache + (static_cast<size_t>(slot) * nkvhead + kv_h_idx) * head_dim;
                    acc += scores[vi] * llaisys::utils::cast<float>(v_ptr[i]);
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

void self_attention_masked(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        tensor_t mask,
        float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v, mask);

    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
    k->isContiguous() && v->isContiguous() && mask->isContiguous(),
    "self_attention_masked: tensors must be contiguous");

    size_t seqlen   = q->shape()[0];
    size_t nhead    = q->shape()[1];
    size_t head_dim = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead   = k->shape()[1];
    size_t head_kv_dim = k->shape()[2];

    ASSERT(mask->shape()[0] == seqlen, "self_attention_masked: mask seqlen mismatch");
    ASSERT(mask->shape()[1] == total_len, "self_attention_masked: mask kvlen mismatch");
    ASSERT(mask->dtype() == LLAISYS_DTYPE_U8 || mask->dtype() == LLAISYS_DTYPE_BOOL,
           "self_attention_masked: mask dtype must be U8/BOOL");

    ASSERT(head_dim == head_kv_dim, "head_dim must equal head_kv_dim");
    ASSERT(attn_val->shape()[0] == seqlen, "attn_val seqlen mismatch");
    ASSERT(attn_val->shape()[1] == nhead,  "attn_val head mismatch");
    ASSERT(attn_val->shape()[2] == head_dim, "attn_val dv mismatch");

    const uint8_t *mask_ptr = reinterpret_cast<const uint8_t *>(mask->data());
    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_masked_(
            reinterpret_cast<float*>(attn_val->data()),
            reinterpret_cast<const float*>(q->data()),
            reinterpret_cast<const float*>(k->data()),
            reinterpret_cast<const float*>(v->data()),
            mask_ptr,
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_masked_(
            reinterpret_cast<llaisys::bf16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::bf16_t*>(q->data()),
            reinterpret_cast<const llaisys::bf16_t*>(k->data()),
            reinterpret_cast<const llaisys::bf16_t*>(v->data()),
            mask_ptr,
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_masked_(
            reinterpret_cast<llaisys::fp16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::fp16_t*>(q->data()),
            reinterpret_cast<const llaisys::fp16_t*>(k->data()),
            reinterpret_cast<const llaisys::fp16_t*>(v->data()),
            mask_ptr,
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}

void self_attention_masked_csr(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        const std::vector<int32_t>& row_ptr,
        const std::vector<int32_t>& col_idx,
        float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
    k->isContiguous() && v->isContiguous(),
    "self_attention_masked_csr: tensors must be contiguous");

    size_t seqlen   = q->shape()[0];
    size_t nhead    = q->shape()[1];
    size_t head_dim = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead   = k->shape()[1];
    size_t head_kv_dim = k->shape()[2];

    ASSERT(head_dim == head_kv_dim, "head_dim must equal head_kv_dim");
    ASSERT(attn_val->shape()[0] == seqlen, "attn_val seqlen mismatch");
    ASSERT(attn_val->shape()[1] == nhead,  "attn_val head mismatch");
    ASSERT(attn_val->shape()[2] == head_dim, "attn_val dv mismatch");

    ASSERT(row_ptr.size() == seqlen + 1, "self_attention_masked_csr: row_ptr size mismatch");
    ASSERT(!row_ptr.empty(), "self_attention_masked_csr: row_ptr must be non-empty");
    ASSERT(row_ptr.front() == 0, "self_attention_masked_csr: row_ptr must start at 0");
    ASSERT(row_ptr.back() == static_cast<int32_t>(col_idx.size()),
           "self_attention_masked_csr: row_ptr end mismatch");

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_masked_csr_(
            reinterpret_cast<float*>(attn_val->data()),
            reinterpret_cast<const float*>(q->data()),
            reinterpret_cast<const float*>(k->data()),
            reinterpret_cast<const float*>(v->data()),
            row_ptr.data(),
            col_idx.data(),
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_masked_csr_(
            reinterpret_cast<llaisys::bf16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::bf16_t*>(q->data()),
            reinterpret_cast<const llaisys::bf16_t*>(k->data()),
            reinterpret_cast<const llaisys::bf16_t*>(v->data()),
            row_ptr.data(),
            col_idx.data(),
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_masked_csr_(
            reinterpret_cast<llaisys::fp16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::fp16_t*>(q->data()),
            reinterpret_cast<const llaisys::fp16_t*>(k->data()),
            reinterpret_cast<const llaisys::fp16_t*>(v->data()),
            row_ptr.data(),
            col_idx.data(),
            seqlen, total_len, nhead, nkvhead, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}

void self_attention_paged(tensor_t attn_val,
        tensor_t q,
        tensor_t k_cache,
        tensor_t v_cache,
        const std::vector<int32_t>& q_seq_rows,
        const std::vector<int32_t>& q_pos,
        const std::vector<int32_t>& block_tables,
        const std::vector<int32_t>& seq_lens,
        int32_t block_table_width,
        int32_t block_size,
        float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k_cache, v_cache);

    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
    k_cache->isContiguous() && v_cache->isContiguous(),
    "self_attention_paged: tensors must be contiguous");

    const size_t seqlen = q->shape()[0];
    const size_t nhead = q->shape()[1];
    const size_t head_dim = q->shape()[2];

    const size_t nslot = k_cache->shape()[0];
    const size_t nkvhead = k_cache->shape()[1];
    const size_t head_kv_dim = k_cache->shape()[2];

    ASSERT(head_dim == head_kv_dim, "self_attention_paged: head_dim mismatch");
    ASSERT(v_cache->shape() == k_cache->shape(), "self_attention_paged: v_cache shape mismatch");
    ASSERT(attn_val->shape()[0] == seqlen, "self_attention_paged: attn_val seqlen mismatch");
    ASSERT(attn_val->shape()[1] == nhead, "self_attention_paged: attn_val head mismatch");
    ASSERT(attn_val->shape()[2] == head_dim, "self_attention_paged: attn_val head_dim mismatch");
    ASSERT(nhead % nkvhead == 0, "self_attention_paged: nhead must be divisible by nkvhead");

    ASSERT(block_table_width > 0, "self_attention_paged: block_table_width must be > 0");
    ASSERT(block_size > 0, "self_attention_paged: block_size must be > 0");
    ASSERT(q_seq_rows.size() == seqlen, "self_attention_paged: q_seq_rows size mismatch");
    ASSERT(q_pos.size() == seqlen, "self_attention_paged: q_pos size mismatch");
    ASSERT(!seq_lens.empty(), "self_attention_paged: seq_lens must be non-empty");
    ASSERT(block_tables.size() == seq_lens.size() * static_cast<size_t>(block_table_width),
           "self_attention_paged: block_tables size mismatch");

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_paged_(
            reinterpret_cast<float*>(attn_val->data()),
            reinterpret_cast<const float*>(q->data()),
            reinterpret_cast<const float*>(k_cache->data()),
            reinterpret_cast<const float*>(v_cache->data()),
            q_seq_rows.data(),
            q_pos.data(),
            block_tables.data(),
            seq_lens.data(),
            static_cast<int32_t>(seq_lens.size()),
            block_table_width,
            block_size,
            seqlen, nslot, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_paged_(
            reinterpret_cast<llaisys::bf16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::bf16_t*>(q->data()),
            reinterpret_cast<const llaisys::bf16_t*>(k_cache->data()),
            reinterpret_cast<const llaisys::bf16_t*>(v_cache->data()),
            q_seq_rows.data(),
            q_pos.data(),
            block_tables.data(),
            seq_lens.data(),
            static_cast<int32_t>(seq_lens.size()),
            block_table_width,
            block_size,
            seqlen, nslot, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_paged_(
            reinterpret_cast<llaisys::fp16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::fp16_t*>(q->data()),
            reinterpret_cast<const llaisys::fp16_t*>(k_cache->data()),
            reinterpret_cast<const llaisys::fp16_t*>(v_cache->data()),
            q_seq_rows.data(),
            q_pos.data(),
            block_tables.data(),
            seq_lens.data(),
            static_cast<int32_t>(seq_lens.size()),
            block_table_width,
            block_size,
            seqlen, nslot, nhead, nkvhead, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops::cpu
