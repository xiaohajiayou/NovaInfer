#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t num_indices, size_t embedding_dim, size_t vocab_size) {
    for (size_t row = 0; row < num_indices; ++row) {
        int64_t idx = index[row];
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < vocab_size, "Embedding: index out of range.");
        const T *src = weight + static_cast<size_t>(idx) * embedding_dim;
        T *dst = out + row * embedding_dim;
        std::memcpy(dst, src, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    const size_t num_indices = index->shape()[0];
    const size_t embedding_dim = weight->shape()[1];
    const size_t vocab_size = weight->shape()[0];
    const int64_t *index_ptr = reinterpret_cast<const int64_t *>(index->data());

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out->data()), index_ptr,
                        reinterpret_cast<const float *>(weight->data()), num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out->data()), index_ptr,
                        reinterpret_cast<const llaisys::bf16_t *>(weight->data()), num_indices, embedding_dim, vocab_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out->data()), index_ptr,
                        reinterpret_cast<const llaisys::fp16_t *>(weight->data()), num_indices, embedding_dim, vocab_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::cpu