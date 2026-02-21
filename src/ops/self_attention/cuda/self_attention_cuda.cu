#include "self_attention_cuda.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace llaisys::device::nvidia {
void cuda_check(cudaError_t rc, const char *what, const char *file, int line);
} // namespace llaisys::device::nvidia

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace llaisys::ops::cuda {

namespace {

struct CachedDeviceI32Buffer {
    int32_t *ptr{nullptr};
    size_t cap{0};
    CachedDeviceI32Buffer() = default;

    ~CachedDeviceI32Buffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    void ensure_capacity(size_t n) {
        if (n <= cap) {
            return;
        }
        if (ptr != nullptr) {
            LLAISYS_CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            cap = 0;
        }
        LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(int32_t)));
        cap = n;
    }

    void upload(const std::vector<int32_t> &host) {
        const size_t n = host.size();
        if (n == 0) {
            return;
        }
        ensure_capacity(n);
        LLAISYS_CUDA_CHECK(cudaMemcpy(ptr, host.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    CachedDeviceI32Buffer(const CachedDeviceI32Buffer &) = delete;
    CachedDeviceI32Buffer &operator=(const CachedDeviceI32Buffer &) = delete;
};

struct PagedAttentionMetaCache {
    CachedDeviceI32Buffer used_slots;
    CachedDeviceI32Buffer row_ptr;
    CachedDeviceI32Buffer col_idx;
    PagedAttentionMetaCache() = default;
};

template <typename T>
__global__ void self_attention_kernel(T *out,
                                      const T *q,
                                      const T *k,
                                      const T *v,
                                      std::int32_t seqlen,
                                      std::int32_t kvlen,
                                      std::int32_t nhead,
                                      std::int32_t nkvhead,
                                      std::int32_t head_dim,
                                      float scale) {
    const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::int32_t total = seqlen * nhead;
    if (idx >= total) {
        return;
    }

    const std::int32_t t = idx / nhead;
    const std::int32_t qh = idx % nhead;
    const std::int32_t group_size = nhead / nkvhead;
    const std::int32_t kvh = qh / group_size;
    const std::int32_t offset = kvlen - seqlen;

    const std::size_t q_base = (static_cast<std::size_t>(t) * static_cast<std::size_t>(nhead) + static_cast<std::size_t>(qh))
                               * static_cast<std::size_t>(head_dim);
    const T *q_ptr = q + q_base;

    float maxv = -1.0e30f;
    for (std::int32_t i = 0; i < kvlen; ++i) {
        if (i > t + offset) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(i) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k + k_base;
        float dot = 0.0f;
        for (std::int32_t j = 0; j < head_dim; ++j) {
            dot += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                   llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        dot *= scale;
        if (dot > maxv) {
            maxv = dot;
        }
    }

    float sum = 0.0f;
    for (std::int32_t i = 0; i < kvlen; ++i) {
        if (i > t + offset) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(i) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k + k_base;
        float dot = 0.0f;
        for (std::int32_t j = 0; j < head_dim; ++j) {
            dot += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                   llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        dot *= scale;
        sum += expf(dot - maxv);
    }
    if (sum <= 0.0f) {
        sum = 1.0f;
    }

    const std::size_t out_base = q_base;
    T *out_ptr = out + out_base;
    for (std::int32_t d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (std::int32_t i = 0; i < kvlen; ++i) {
            if (i > t + offset) {
                continue;
            }
            const std::size_t k_base = (static_cast<std::size_t>(i) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                       * static_cast<std::size_t>(head_dim);
            const std::size_t v_base = (static_cast<std::size_t>(i) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                       * static_cast<std::size_t>(head_dim);
            const T *k_ptr = k + k_base;
            const T *v_ptr = v + v_base;
            float dot = 0.0f;
            for (std::int32_t j = 0; j < head_dim; ++j) {
                dot += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                       llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
            }
            dot *= scale;
            const float w = expf(dot - maxv) / sum;
            acc += w * llaisys::device::nvidia::dtype::to_float<T>(v_ptr[d]);
        }
        out_ptr[d] = llaisys::device::nvidia::dtype::from_float<T>(acc);
    }
}

template <typename T>
void launch_self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale,
                           std::int32_t seqlen, std::int32_t kvlen, std::int32_t nhead, std::int32_t nkvhead,
                           std::int32_t head_dim) {
    constexpr int kBlock = 128;
    const int total = seqlen * nhead;
    const int grid = (total + kBlock - 1) / kBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    self_attention_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(attn_val->data()),
        reinterpret_cast<const T *>(q->data()),
        reinterpret_cast<const T *>(k->data()),
        reinterpret_cast<const T *>(v->data()),
        seqlen,
        kvlen,
        nhead,
        nkvhead,
        head_dim,
        scale);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void paged_attention_kernel(T *out,
                                       const T *q,
                                       const T *k_cache,
                                       const T *v_cache,
                                       const int32_t *used_slots,
                                       const int32_t *row_ptr,
                                       const int32_t *col_idx,
                                       std::int32_t seqlen,
                                       std::int32_t nslot,
                                       std::int32_t nhead,
                                       std::int32_t nkvhead,
                                       std::int32_t head_dim,
                                       float scale) {
    const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::int32_t total = seqlen * nhead;
    if (idx >= total) {
        return;
    }

    const std::int32_t t = idx / nhead;
    const std::int32_t qh = idx % nhead;
    const std::int32_t group_size = nhead / nkvhead;
    const std::int32_t kvh = qh / group_size;
    const std::size_t q_base = (static_cast<std::size_t>(t) * static_cast<std::size_t>(nhead) + static_cast<std::size_t>(qh))
                               * static_cast<std::size_t>(head_dim);
    const T *q_ptr = q + q_base;

    const int32_t rb = row_ptr[t];
    const int32_t re = row_ptr[t + 1];
    if (rb >= re) {
        return;
    }

    float maxv = -1.0e30f;
    for (int32_t p = rb; p < re; ++p) {
        const int32_t col = col_idx[p];
        if (col < 0) {
            continue;
        }
        const int32_t slot = used_slots[col];
        if (slot < 0 || slot >= nslot) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k_cache + k_base;
        float dot = 0.0f;
        for (std::int32_t j = 0; j < head_dim; ++j) {
            dot += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                   llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        dot *= scale;
        if (dot > maxv) {
            maxv = dot;
        }
    }

    float sum = 0.0f;
    for (int32_t p = rb; p < re; ++p) {
        const int32_t col = col_idx[p];
        if (col < 0) {
            continue;
        }
        const int32_t slot = used_slots[col];
        if (slot < 0 || slot >= nslot) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k_cache + k_base;
        float dot = 0.0f;
        for (std::int32_t j = 0; j < head_dim; ++j) {
            dot += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                   llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        dot *= scale;
        sum += expf(dot - maxv);
    }
    if (sum <= 0.0f) {
        sum = 1.0f;
    }

    T *out_ptr = out + q_base;
    for (std::int32_t d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int32_t p = rb; p < re; ++p) {
            const int32_t col = col_idx[p];
            if (col < 0) {
                continue;
            }
            const int32_t slot = used_slots[col];
            if (slot < 0 || slot >= nslot) {
                continue;
            }
            const std::size_t k_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead)
                                        + static_cast<std::size_t>(kvh))
                                       * static_cast<std::size_t>(head_dim);
            const std::size_t v_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead)
                                        + static_cast<std::size_t>(kvh))
                                       * static_cast<std::size_t>(head_dim);
            const T *k_ptr = k_cache + k_base;
            const T *v_ptr = v_cache + v_base;
            float dot = 0.0f;
            for (std::int32_t j = 0; j < head_dim; ++j) {
                dot += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                       llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
            }
            dot *= scale;
            const float w = expf(dot - maxv) / sum;
            acc += w * llaisys::device::nvidia::dtype::to_float<T>(v_ptr[d]);
        }
        out_ptr[d] = llaisys::device::nvidia::dtype::from_float<T>(acc);
    }
}

template <typename T>
void launch_paged_attention(tensor_t attn_val,
                            tensor_t q,
                            tensor_t k_cache,
                            tensor_t v_cache,
                            const std::vector<int32_t> &used_slots,
                            const std::vector<int32_t> &row_ptr,
                            const std::vector<int32_t> &col_idx,
                            float scale,
                            std::int32_t seqlen,
                            std::int32_t nslot,
                            std::int32_t nhead,
                            std::int32_t nkvhead,
                            std::int32_t head_dim) {
    // Reuse device-side metadata buffers across decode steps to avoid
    // repeated cudaMalloc/cudaFree in paged attention hot path.
    static thread_local PagedAttentionMetaCache cache;
    cache.used_slots.upload(used_slots);
    cache.row_ptr.upload(row_ptr);
    cache.col_idx.upload(col_idx);
    constexpr int kBlock = 128;
    const int total = seqlen * nhead;
    const int grid = (total + kBlock - 1) / kBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    paged_attention_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(attn_val->data()),
        reinterpret_cast<const T *>(q->data()),
        reinterpret_cast<const T *>(k_cache->data()),
        reinterpret_cast<const T *>(v_cache->data()),
        cache.used_slots.ptr,
        cache.row_ptr.ptr,
        cache.col_idx.ptr,
        seqlen,
        nslot,
        nhead,
        nkvhead,
        head_dim,
        scale);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    const std::int32_t seqlen = static_cast<std::int32_t>(q->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(q->shape()[1]);
    const std::int32_t head_dim = static_cast<std::int32_t>(q->shape()[2]);
    const std::int32_t kvlen = static_cast<std::int32_t>(k->shape()[0]);
    const std::int32_t nkvhead = static_cast<std::int32_t>(k->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || kvlen <= 0 || nkvhead <= 0) {
        return;
    }

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_self_attention<float>(attn_val, q, k, v, scale, seqlen, kvlen, nhead, nkvhead, head_dim);
        return;
    case LLAISYS_DTYPE_F16:
        launch_self_attention<llaisys::fp16_t>(attn_val, q, k, v, scale, seqlen, kvlen, nhead, nkvhead, head_dim);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_self_attention<llaisys::bf16_t>(attn_val, q, k, v, scale, seqlen, kvlen, nhead, nkvhead, head_dim);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}

void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t k_cache,
                          tensor_t v_cache,
                          const std::vector<int32_t> &used_slots,
                          const std::vector<int32_t> &row_ptr,
                          const std::vector<int32_t> &col_idx,
                          float scale) {
    const std::int32_t seqlen = static_cast<std::int32_t>(q->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(q->shape()[1]);
    const std::int32_t head_dim = static_cast<std::int32_t>(q->shape()[2]);
    const std::int32_t nslot = static_cast<std::int32_t>(k_cache->shape()[0]);
    const std::int32_t nkvhead = static_cast<std::int32_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0) {
        return;
    }
    if (used_slots.empty() || row_ptr.empty()) {
        return;
    }

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_paged_attention<float>(
            attn_val, q, k_cache, v_cache, used_slots, row_ptr, col_idx, scale, seqlen, nslot, nhead, nkvhead, head_dim);
        return;
    case LLAISYS_DTYPE_F16:
        launch_paged_attention<llaisys::fp16_t>(
            attn_val, q, k_cache, v_cache, used_slots, row_ptr, col_idx, scale, seqlen, nslot, nhead, nkvhead, head_dim);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_paged_attention<llaisys::bf16_t>(
            attn_val, q, k_cache, v_cache, used_slots, row_ptr, col_idx, scale, seqlen, nslot, nhead, nkvhead, head_dim);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}

} // namespace llaisys::ops::cuda
