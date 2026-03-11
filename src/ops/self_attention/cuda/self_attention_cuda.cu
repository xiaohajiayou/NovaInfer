#include "self_attention_cuda.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime_api.h>
#ifdef ENABLE_CUDNN_API
#include <cudnn.h>
#ifdef ENABLE_CUDNN_FRONTEND
#include <cudnn_frontend.h>
#endif
#endif

#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llaisys::device::nvidia {
void cuda_check(cudaError_t rc, const char *what, const char *file, int line);
} // namespace llaisys::device::nvidia

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace llaisys::ops::cuda {

namespace {

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
__global__ void scatter_cache_by_slots_kernel(T *dst,
                                              const T *src,
                                              const int32_t *slot_idxs,
                                              int32_t ntoken,
                                              int32_t nhead,
                                              int32_t head_dim,
                                              int32_t nslot) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    const int64_t per_token = static_cast<int64_t>(nhead) * static_cast<int64_t>(head_dim);
    const int64_t total = static_cast<int64_t>(ntoken) * per_token;
    if (idx >= total) {
        return;
    }

    const int32_t tok = static_cast<int32_t>(idx / per_token);
    const int64_t rem = idx - static_cast<int64_t>(tok) * per_token;
    const int32_t h = static_cast<int32_t>(rem / head_dim);
    const int32_t d = static_cast<int32_t>(rem - static_cast<int64_t>(h) * head_dim);
    const int32_t slot = slot_idxs[tok];
    if (slot < 0 || slot >= nslot) {
        return;
    }

    const size_t src_off =
        (static_cast<size_t>(tok) * static_cast<size_t>(nhead) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim) +
        static_cast<size_t>(d);
    const size_t dst_off =
        (static_cast<size_t>(slot) * static_cast<size_t>(nhead) + static_cast<size_t>(h)) * static_cast<size_t>(head_dim) +
        static_cast<size_t>(d);
    dst[dst_off] = src[src_off];
}

template <typename T>
void launch_scatter_cache_by_slots(tensor_t dst_cache, tensor_t src_tokens, const int32_t *slot_idxs_dev) {
    const int32_t ntoken = static_cast<int32_t>(src_tokens->shape()[0]);
    const int32_t nhead = static_cast<int32_t>(src_tokens->shape()[1]);
    const int32_t head_dim = static_cast<int32_t>(src_tokens->shape()[2]);
    const int32_t nslot = static_cast<int32_t>(dst_cache->shape()[0]);
    if (ntoken <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0) {
        return;
    }
    const int64_t total = static_cast<int64_t>(ntoken) * static_cast<int64_t>(nhead) * static_cast<int64_t>(head_dim);
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    scatter_cache_by_slots_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(dst_cache->data()),
        reinterpret_cast<const T *>(src_tokens->data()),
        slot_idxs_dev,
        ntoken,
        nhead,
        head_dim,
        nslot);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void compact_prefill_output_kernel(T *dst_packed,
                                              const T *src_dense,
                                              const int32_t *qo_ragged_offset,
                                              int32_t seqlen,
                                              int32_t b_exec,
                                              int32_t s_q_exec,
                                              int32_t nhead,
                                              int32_t head_dim) {
    const int64_t per_token = static_cast<int64_t>(nhead) * static_cast<int64_t>(head_dim);
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    const int64_t total = static_cast<int64_t>(seqlen) * per_token;
    if (idx >= total || qo_ragged_offset == nullptr || b_exec <= 0 || s_q_exec <= 0 || nhead <= 0 || head_dim <= 0) {
        return;
    }

    const int64_t tok = idx / per_token;
    const int64_t rem = idx - tok * per_token;
    const int32_t h = static_cast<int32_t>(rem / static_cast<int64_t>(head_dim));
    const int32_t d = static_cast<int32_t>(rem - static_cast<int64_t>(h) * static_cast<int64_t>(head_dim));

    int32_t row = 0;
    for (; row < b_exec; ++row) {
        const int64_t row_tok_end =
            static_cast<int64_t>(qo_ragged_offset[row + 1]) / per_token;
        if (tok < row_tok_end) {
            break;
        }
    }
    if (row >= b_exec) {
        return;
    }
    const int64_t row_tok_begin =
        static_cast<int64_t>(qo_ragged_offset[row]) / per_token;
    const int64_t local = tok - row_tok_begin;
    if (local < 0 || local >= static_cast<int64_t>(s_q_exec)) {
        return;
    }
    const int64_t src_tok = static_cast<int64_t>(row) * static_cast<int64_t>(s_q_exec) + local;
    const int64_t src_off = (src_tok * static_cast<int64_t>(nhead) + static_cast<int64_t>(h)) * static_cast<int64_t>(head_dim)
                            + static_cast<int64_t>(d);
    dst_packed[idx] = src_dense[src_off];
}

template <typename T>
void launch_compact_prefill_output(tensor_t dst_packed,
                                   tensor_t src_dense,
                                   const int32_t *qo_ragged_offset,
                                   int32_t b_exec,
                                   int32_t s_q_exec,
                                   int32_t nhead,
                                   int32_t head_dim,
                                   cudaStream_t stream) {
    const int32_t seqlen = static_cast<int32_t>(dst_packed->shape()[0]);
    if (seqlen <= 0 || b_exec <= 0 || s_q_exec <= 0 || nhead <= 0 || head_dim <= 0 || qo_ragged_offset == nullptr) {
        return;
    }
    const int64_t total =
        static_cast<int64_t>(seqlen) * static_cast<int64_t>(nhead) * static_cast<int64_t>(head_dim);
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    compact_prefill_output_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(dst_packed->data()),
        reinterpret_cast<const T *>(src_dense->data()),
        qo_ragged_offset,
        seqlen,
        b_exec,
        s_q_exec,
        nhead,
        head_dim);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return x;
}

template <typename T, int WARPS_PER_BLOCK>
__global__ void paged_attention_warp_kernel(T *out,
                                            const T *q,
                                            const T *k_cache,
                                            const T *v_cache,
                                            const int32_t *cu_seqlens_q,
                                            const int32_t *block_tables,
                                            const int32_t *cu_seqlens_k,
                                            std::int32_t seqlen,
                                            std::int32_t nslot,
                                            std::int32_t nseq,
                                            std::int32_t block_table_width,
                                            std::int32_t block_size,
                                            std::int32_t nhead,
                                            std::int32_t nkvhead,
                                            std::int32_t head_dim,
                                            float scale) {
    constexpr int WARP = 32;
    const int warp_id = static_cast<int>(threadIdx.x) / WARP;
    const int lane = static_cast<int>(threadIdx.x) % WARP;
    const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * WARPS_PER_BLOCK + warp_id);
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

    int32_t row = 0;
    {
        int32_t lo = 0;
        int32_t hi = nseq;
        while (lo < hi) {
            const int32_t mid = lo + ((hi - lo) >> 1);
            if (cu_seqlens_q[mid + 1] <= t) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        row = lo;
    }
    if (row < 0 || row >= nseq) {
        return;
    }
    const int32_t row_start = cu_seqlens_q[row];
    const int32_t row_end = cu_seqlens_q[row + 1];
    const int32_t row_scheduled = row_end - row_start;
    if (row_scheduled <= 0) {
        return;
    }
    const int32_t local = t - row_start;
    if (local < 0 || local >= row_scheduled) {
        return;
    }
    const int32_t seq_len = cu_seqlens_k[row + 1] - cu_seqlens_k[row];
    if (seq_len <= 0) {
        return;
    }
    const int32_t qpos = (seq_len - row_scheduled) + local;
    const int32_t vmax = min(qpos, seq_len - 1);
    if (vmax < 0) {
        return;
    }

    float maxv = -FLT_MAX;
    for (int32_t p = 0; p <= vmax; ++p) {
        const int32_t bidx = p / block_size;
        const int32_t boff = p % block_size;
        if (bidx < 0 || bidx >= block_table_width) {
            continue;
        }
        const int32_t bid = block_tables[row * block_table_width + bidx];
        if (bid < 0) {
            continue;
        }
        const int32_t slot = bid * block_size + boff;
        if (slot < 0 || slot >= nslot) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k_cache + k_base;
        float dot_local = 0.0f;
        for (std::int32_t j = lane; j < head_dim; j += WARP) {
            dot_local += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                         llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        float dot = warp_sum(dot_local) * scale;
        if (dot > maxv) {
            maxv = dot;
        }
    }

    float sum = 0.0f;
    for (int32_t p = 0; p <= vmax; ++p) {
        const int32_t bidx = p / block_size;
        const int32_t boff = p % block_size;
        if (bidx < 0 || bidx >= block_table_width) {
            continue;
        }
        const int32_t bid = block_tables[row * block_table_width + bidx];
        if (bid < 0) {
            continue;
        }
        const int32_t slot = bid * block_size + boff;
        if (slot < 0 || slot >= nslot) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k_cache + k_base;
        float dot_local = 0.0f;
        for (std::int32_t j = lane; j < head_dim; j += WARP) {
            dot_local += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                         llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        const float dot = warp_sum(dot_local) * scale;
        sum += expf(dot - maxv);
    }

    T *out_ptr = out + q_base;
    if (sum <= 0.0f) {
        sum = 1.0f;
    }
    for (std::int32_t d = lane; d < head_dim; d += WARP) {
        float acc = 0.0f;
        for (int32_t p = 0; p <= vmax; ++p) {
            const int32_t bidx = p / block_size;
            const int32_t boff = p % block_size;
            if (bidx < 0 || bidx >= block_table_width) {
                continue;
            }
            const int32_t bid = block_tables[row * block_table_width + bidx];
            if (bid < 0) {
                continue;
            }
            const int32_t slot = bid * block_size + boff;
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
            float dot_local = 0.0f;
            for (std::int32_t j = lane; j < head_dim; j += WARP) {
                dot_local += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                             llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
            }
            const float dot = warp_sum(dot_local) * scale;
            const float w = expf(dot - maxv) / sum;
            acc += w * llaisys::device::nvidia::dtype::to_float<T>(v_ptr[d]);
        }
        out_ptr[d] = llaisys::device::nvidia::dtype::from_float<T>(acc);
    }
}

template <typename T, int WARPS_PER_BLOCK>
__global__ void paged_attention_decode_warp_kernel(T *out,
                                                   const T *q,
                                                   const T *k_cache,
                                                   const T *v_cache,
                                                   const int32_t *cu_seqlens_k,
                                                   const int32_t *block_tables,
                                                   std::int32_t seqlen,
                                                   std::int32_t nslot,
                                                   std::int32_t nseq,
                                                   std::int32_t block_table_width,
                                                   std::int32_t block_size,
                                                   std::int32_t nhead,
                                                   std::int32_t nkvhead,
                                                   std::int32_t head_dim,
                                                   float scale) {
    constexpr int WARP = 32;
    const int warp_id = static_cast<int>(threadIdx.x) / WARP;
    const int lane = static_cast<int>(threadIdx.x) % WARP;
    const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * WARPS_PER_BLOCK + warp_id);
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

    // Decode invariant: one scheduled token per sequence, and tokens are ordered by sequence row.
    const int32_t row = t;
    if (row < 0 || row >= nseq) {
        return;
    }
    const int32_t seq_len = cu_seqlens_k[row + 1] - cu_seqlens_k[row];
    if (seq_len <= 0) {
        return;
    }
    const int32_t vmax = seq_len - 1;

    float maxv = -FLT_MAX;
    for (int32_t p = 0; p <= vmax; ++p) {
        const int32_t bidx = p / block_size;
        const int32_t boff = p % block_size;
        if (bidx < 0 || bidx >= block_table_width) {
            continue;
        }
        const int32_t bid = block_tables[row * block_table_width + bidx];
        if (bid < 0) {
            continue;
        }
        const int32_t slot = bid * block_size + boff;
        if (slot < 0 || slot >= nslot) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k_cache + k_base;
        float dot_local = 0.0f;
        for (std::int32_t j = lane; j < head_dim; j += WARP) {
            dot_local += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                         llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        float dot = warp_sum(dot_local) * scale;
        if (dot > maxv) {
            maxv = dot;
        }
    }

    float sum = 0.0f;
    for (int32_t p = 0; p <= vmax; ++p) {
        const int32_t bidx = p / block_size;
        const int32_t boff = p % block_size;
        if (bidx < 0 || bidx >= block_table_width) {
            continue;
        }
        const int32_t bid = block_tables[row * block_table_width + bidx];
        if (bid < 0) {
            continue;
        }
        const int32_t slot = bid * block_size + boff;
        if (slot < 0 || slot >= nslot) {
            continue;
        }
        const std::size_t k_base = (static_cast<std::size_t>(slot) * static_cast<std::size_t>(nkvhead) + static_cast<std::size_t>(kvh))
                                   * static_cast<std::size_t>(head_dim);
        const T *k_ptr = k_cache + k_base;
        float dot_local = 0.0f;
        for (std::int32_t j = lane; j < head_dim; j += WARP) {
            dot_local += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                         llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
        }
        const float dot = warp_sum(dot_local) * scale;
        sum += expf(dot - maxv);
    }

    T *out_ptr = out + q_base;
    if (sum <= 0.0f) {
        sum = 1.0f;
    }
    for (std::int32_t d = lane; d < head_dim; d += WARP) {
        float acc = 0.0f;
        for (int32_t p = 0; p <= vmax; ++p) {
            const int32_t bidx = p / block_size;
            const int32_t boff = p % block_size;
            if (bidx < 0 || bidx >= block_table_width) {
                continue;
            }
            const int32_t bid = block_tables[row * block_table_width + bidx];
            if (bid < 0) {
                continue;
            }
            const int32_t slot = bid * block_size + boff;
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
            float dot_local = 0.0f;
            for (std::int32_t j = lane; j < head_dim; j += WARP) {
                dot_local += llaisys::device::nvidia::dtype::to_float<T>(q_ptr[j]) *
                             llaisys::device::nvidia::dtype::to_float<T>(k_ptr[j]);
            }
            const float dot = warp_sum(dot_local) * scale;
            const float w = expf(dot - maxv) / sum;
            acc += w * llaisys::device::nvidia::dtype::to_float<T>(v_ptr[d]);
        }
        out_ptr[d] = llaisys::device::nvidia::dtype::from_float<T>(acc);
    }
}

template <typename T>
void launch_paged_attention_prepared(tensor_t attn_val,
                                     tensor_t q,
                                     tensor_t k_cache,
                                     tensor_t v_cache,
                                     const CommonAttentionMetadata &prepared,
                                     std::int32_t block_table_width,
                                     std::int32_t block_size,
                                     float scale,
                                     std::int32_t seqlen,
                                     std::int32_t nslot,
                                     std::int32_t nseq,
                                     std::int32_t nhead,
                                     std::int32_t nkvhead,
                                     std::int32_t head_dim) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlock = kWarpsPerBlock * 32;
    const int total = seqlen * nhead;
    const int grid = (total + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    paged_attention_warp_kernel<T, kWarpsPerBlock><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(attn_val->data()),
        reinterpret_cast<const T *>(q->data()),
        reinterpret_cast<const T *>(k_cache->data()),
        reinterpret_cast<const T *>(v_cache->data()),
        prepared.cu_seqlens_q,
        prepared.block_tables,
        prepared.cu_seqlens_k,
        seqlen,
        nslot,
        nseq,
        block_table_width,
        block_size,
        nhead,
        nkvhead,
        head_dim,
        scale);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_paged_attention_decode_prepared(tensor_t attn_val,
                                            tensor_t q,
                                            tensor_t k_cache,
                                            tensor_t v_cache,
                                            const CommonAttentionMetadata &prepared,
                                            std::int32_t block_table_width,
                                            std::int32_t block_size,
                                            float scale,
                                            std::int32_t seqlen,
                                            std::int32_t nslot,
                                            std::int32_t nseq,
                                            std::int32_t nhead,
                                            std::int32_t nkvhead,
                                            std::int32_t head_dim) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlock = kWarpsPerBlock * 32;
    const int total = seqlen * nhead;
    const int grid = (total + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    paged_attention_decode_warp_kernel<T, kWarpsPerBlock><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(attn_val->data()),
        reinterpret_cast<const T *>(q->data()),
        reinterpret_cast<const T *>(k_cache->data()),
        reinterpret_cast<const T *>(v_cache->data()),
        prepared.cu_seqlens_k,
        prepared.block_tables,
        seqlen,
        nslot,
        nseq,
        block_table_width,
        block_size,
        nhead,
        nkvhead,
        head_dim,
        scale);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

bool is_decode_phase(const CommonAttentionMetadata &prepared) {
    return prepared.phase == AttentionPhase::DECODE;
}

bool is_prefill_phase(const CommonAttentionMetadata &prepared) {
    return prepared.phase == AttentionPhase::PREFILL;
}

#ifdef ENABLE_CUDNN_API
struct CudnnPagedPlan {
#ifdef ENABLE_CUDNN_FRONTEND
    std::shared_ptr<cudnn_frontend::graph::Graph> graph{};
    bool graph_ready{false};
    int64_t built_b{0};
    int64_t built_s_q{0};
    int32_t built_table_size{0};
    int64_t built_max_seq_len_kv{0};
    int64_t built_num_blocks{0};
    int32_t built_block_size{0};

    void *workspace{nullptr};
    size_t workspace_cap{0};

#endif

    ~CudnnPagedPlan() {
        if (workspace != nullptr) {
            cudaFree(workspace);
            workspace = nullptr;
        }
    }
};

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t SEQ_LEN_Q_UID = 7;
constexpr int64_t SEQ_LEN_KV_UID = 8;
constexpr int64_t PAGE_TABLE_K_UID = 9;
constexpr int64_t PAGE_TABLE_V_UID = 10;
constexpr int64_t QO_RAGGED_OFFSET_UID = 11;

struct CudnnRuntimeState {
    cudnnHandle_t handle{nullptr};
    CudnnPagedPlan decode_plan{};
    CudnnPagedPlan prefill_plan{};
};

static CudnnRuntimeState &cudnn_runtime_state() {
    static thread_local CudnnRuntimeState state{};
    return state;
}

static CudnnPagedPlan &select_cudnn_plan(CudnnRuntimeState &state, bool is_prefill) {
    return is_prefill ? state.prefill_plan : state.decode_plan;
}

static bool cudnn_set_stream(cudnnHandle_t &handle, cudaStream_t stream) {
    if (handle == nullptr) {
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
            return false;
        }
    }
    return cudnnSetStream(handle, stream) == CUDNN_STATUS_SUCCESS;
}

#ifdef ENABLE_CUDNN_FRONTEND
static bool cudnn_debug_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char *raw = std::getenv("LLAISYS_CUDNN_DEBUG");
    if (raw == nullptr || raw[0] == '\0' || (raw[0] == '0' && raw[1] == '\0')) {
        enabled = 0;
    } else {
        enabled = 1;
    }
    return enabled != 0;
}

static void cudnn_debug_dump_i32(const char *tag, const int32_t *dev_ptr, int64_t n, int64_t limit = 16) {
    if (!cudnn_debug_enabled()) {
        return;
    }
    if (dev_ptr == nullptr || n <= 0) {
        printf("[cudnn][dbg] %s ptr=%p n=%lld\n", tag, static_cast<const void *>(dev_ptr), static_cast<long long>(n));
        return;
    }
    const int64_t m = std::max<int64_t>(0, std::min<int64_t>(n, limit));
    std::vector<int32_t> host(static_cast<size_t>(m), 0);
    const cudaError_t rc = cudaMemcpy(host.data(), dev_ptr, static_cast<size_t>(m) * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
        printf("[cudnn][dbg] %s memcpy failed: %s\n", tag, cudaGetErrorString(rc));
        return;
    }
    printf("[cudnn][dbg] %s n=%lld first=%lld vals=", tag, static_cast<long long>(n), static_cast<long long>(m));
    for (int64_t i = 0; i < m; ++i) {
        printf("%d%s", host[static_cast<size_t>(i)], (i + 1 == m ? "" : ","));
    }
    printf("\n");
}

static void cudnn_debug_dump_override(const std::vector<int64_t> &override_uids,
                                      const std::vector<std::vector<int64_t>> &override_shapes,
                                      const std::vector<std::vector<int64_t>> &override_strides) {
    if (!cudnn_debug_enabled()) {
        return;
    }
    const size_t n = std::min(override_uids.size(), std::min(override_shapes.size(), override_strides.size()));
    printf("[cudnn][dbg] override_count=%lld\n", static_cast<long long>(n));
    for (size_t i = 0; i < n; ++i) {
        auto print_vec = [](const std::vector<int64_t> &v) {
            printf("[");
            for (size_t j = 0; j < v.size(); ++j) {
                printf("%lld%s", static_cast<long long>(v[j]), (j + 1 == v.size() ? "" : ","));
            }
            printf("]");
        };
        printf("[cudnn][dbg] uid=%lld shape=", static_cast<long long>(override_uids[i]));
        print_vec(override_shapes[i]);
        printf(" stride=");
        print_vec(override_strides[i]);
        printf("\n");
    }
}

static void build_cudnn_override_tensors(bool is_prefill,
                                         int64_t b_exec,
                                         int64_t s_q_exec,
                                         int64_t nhead,
                                         int64_t head_dim,
                                         int32_t table_size,
                                         std::vector<int64_t> &override_uids,
                                         std::vector<std::vector<int64_t>> &override_shapes,
                                         std::vector<std::vector<int64_t>> &override_strides) {
    override_uids.clear();
    override_shapes.clear();
    override_strides.clear();

    auto push = [&](int64_t uid, std::vector<int64_t> shape, std::vector<int64_t> stride) {
        override_uids.push_back(uid);
        override_shapes.push_back(std::move(shape));
        override_strides.push_back(std::move(stride));
    };

    // Runtime q/attn_val buffers are token-major contiguous: [sum_q, nhead, head_dim].
    // Keep BSHD logical strides aligned with cuDNN THD/ragged samples:
    //   stride_b = s_q * h * d, stride_s = h * d, stride_h = d.
    // Ragged offsets carry per-request packed starts.
    const int64_t q_stride_token = nhead * head_dim;
    const int64_t q_stride_b = std::max<int64_t>(1, s_q_exec) * q_stride_token;
    const int64_t q_stride_h = head_dim;
    const int64_t q_stride_s = q_stride_token;
    push(Q_UID, {b_exec, nhead, s_q_exec, head_dim}, {q_stride_b, q_stride_h, q_stride_s, 1});
    push(O_UID, {b_exec, nhead, s_q_exec, head_dim}, {q_stride_b, q_stride_h, q_stride_s, 1});

    push(SEQ_LEN_Q_UID, {b_exec, 1, 1, 1}, {1, 1, 1, 1});
    push(SEQ_LEN_KV_UID, {b_exec, 1, 1, 1}, {1, 1, 1, 1});

    const int64_t tbl = static_cast<int64_t>(table_size);
    push(PAGE_TABLE_K_UID, {b_exec, 1, tbl, 1}, {tbl, tbl, 1, 1});
    push(PAGE_TABLE_V_UID, {b_exec, 1, tbl, 1}, {tbl, tbl, 1, 1});

    if (is_prefill) {
        push(QO_RAGGED_OFFSET_UID, {b_exec + 1, 1, 1, 1}, {1, 1, 1, 1});
    }
}

static bool ensure_cudnn_plan_ready(CudnnPagedPlan &plan,
                                    cudnnHandle_t handle,
                                    llaisysDataType_t dtype,
                                    bool is_prefill,
                                    int64_t b_exec,
                                    int64_t max_seq_len_q,
                                    int64_t nhead,
                                    int64_t nkvhead,
                                    int64_t head_dim,
                                    int64_t num_blocks,
                                    int32_t table_size,
                                    int64_t max_seq_len_kv,
                                    int32_t block_size,
                                    float scale,
                                    bool force_rebuild = false) {
    LLAISYS_NVTX_SCOPE(is_prefill ? "attn/cudnn/prefill/ensure_plan" : "attn/cudnn/decode/ensure_plan");
    namespace fe = cudnn_frontend;
    const int64_t b_exec_safe = std::max<int64_t>(1, b_exec);
    const int64_t s_q_exec = is_prefill ? std::max<int64_t>(1, max_seq_len_q) : 1;
    // Dynamic-shape mode with bounded templates: runtime shapes can shrink via override_*.
    // Rebuild when any runtime bound grows beyond built template, or cache geometry changes.
    bool need_rebuild = force_rebuild || !plan.graph_ready;
    if (!need_rebuild) {
        if (plan.built_num_blocks != num_blocks || plan.built_block_size != block_size) {
            need_rebuild = true;
        }
    }
    // For dynamic-shape graphs (cuDNN >= 9.18), runtime b/s_q growth can be
    // handled via override shapes without mandatory rebuild. Keep rebuild checks
    // for cache geometry and paged-KV capacity semantics.
    if (!need_rebuild) {
        if (table_size > plan.built_table_size || max_seq_len_kv > plan.built_max_seq_len_kv) {
            need_rebuild = true;
        }
    }
    // printf(
    //     "[cudnn] ensure_plan phase=%s need_rebuild=%d dynamic=%d dtype=%d b_exec=%lld s_q_exec=%lld nhead=%lld nkvhead=%lld "
    //     "head_dim=%lld num_blocks=%lld table_size=%d max_seq_len_kv=%lld block_size=%d built_b=%lld built_s_q=%lld cudnn=%lld\n",
    //     is_prefill ? "prefill" : "decode",
    //     int(need_rebuild),
    //     int(use_dynamic_shape),
    //     int(dtype),
    //     static_cast<long long>(b_exec),
    //     static_cast<long long>(s_q_exec),
    //     static_cast<long long>(nhead),
    //     static_cast<long long>(nkvhead),
    //     static_cast<long long>(head_dim),
    //     static_cast<long long>(num_blocks),
    //     int(table_size),
    //     static_cast<long long>(max_seq_len_kv),
    //     int(block_size),
    //     static_cast<long long>(plan.built_b),
    //     static_cast<long long>(plan.built_s_q),
    //     static_cast<long long>(cudnnGetVersion()));
    if (!need_rebuild) {
        return true;
    }

    auto io_dtype = dtype == LLAISYS_DTYPE_BF16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
    const int64_t b_plan = b_exec_safe;
    const int64_t s_q_plan = s_q_exec;
    const int32_t table_size_plan = table_size;
    const int64_t max_seq_len_kv_plan = max_seq_len_kv;
    if (is_prefill && cudnnGetVersion() < 90700) {
        return false;
    }
    plan.graph = std::make_shared<fe::graph::Graph>();
    plan.graph->set_io_data_type(io_dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(true);

    std::shared_ptr<fe::graph::Tensor_attributes> qo_ragged_offset = nullptr;
    if (is_prefill) {
        qo_ragged_offset = plan.graph->tensor(fe::graph::Tensor_attributes()
                                                  .set_name("QO_ragged_offset")
                                                  .set_uid(QO_RAGGED_OFFSET_UID)
                                                  .set_dim({b_plan + 1, 1, 1, 1})
                                                  .set_stride({1, 1, 1, 1})
                                                  .set_data_type(fe::DataType_t::INT32));
    }

    const int64_t q_stride_b = s_q_plan * nhead * head_dim;
    const int64_t q_stride_s = nhead * head_dim;
    auto q_attrs = fe::graph::Tensor_attributes()
                       .set_name("Q")
                       .set_uid(Q_UID)
                       .set_dim({b_plan, nhead, s_q_plan, head_dim})
                       .set_stride({q_stride_b, head_dim, q_stride_s, 1});
    if (qo_ragged_offset != nullptr) {
        q_attrs.set_ragged_offset(qo_ragged_offset);
    }

    auto Q = plan.graph->tensor(q_attrs);

    auto K = plan.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("container_K")
                                     .set_uid(K_UID)
                                     .set_dim({num_blocks, nkvhead, static_cast<int64_t>(block_size), head_dim})
                                     .set_stride({static_cast<int64_t>(block_size) * nkvhead * head_dim,
                                                  head_dim,
                                                  nkvhead * head_dim,
                                                  1}));

    auto V = plan.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("container_V")
                                     .set_uid(V_UID)
                                     .set_dim({num_blocks, nkvhead, static_cast<int64_t>(block_size), head_dim})
                                     .set_stride({static_cast<int64_t>(block_size) * nkvhead * head_dim,
                                                  head_dim,
                                                  nkvhead * head_dim,
                                                  1}));

    auto seq_q = plan.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("seq_q")
                                         .set_uid(SEQ_LEN_Q_UID)
                                         .set_dim({b_plan, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = plan.graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("seq_kv")
                                          .set_uid(SEQ_LEN_KV_UID)
                                          .set_dim({b_plan, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::INT32));
    auto page_table_k = plan.graph->tensor(fe::graph::Tensor_attributes()
                                                .set_name("page_table_k")
                                                .set_uid(PAGE_TABLE_K_UID)
                                                .set_dim({b_plan, 1, static_cast<int64_t>(table_size_plan), 1})
                                                .set_stride({static_cast<int64_t>(table_size_plan), static_cast<int64_t>(table_size_plan), 1, 1})
                                                .set_data_type(fe::DataType_t::INT32));
    auto page_table_v = plan.graph->tensor(fe::graph::Tensor_attributes()
                                                .set_name("page_table_v")
                                                .set_uid(PAGE_TABLE_V_UID)
                                                .set_dim({b_plan, 1, static_cast<int64_t>(table_size_plan), 1})
                                                .set_stride({static_cast<int64_t>(table_size_plan), static_cast<int64_t>(table_size_plan), 1, 1})
                                                .set_data_type(fe::DataType_t::INT32));

    auto sdpa_options =
        fe::graph::SDPA_attributes().set_name("novainfer_paged_sdpa").set_generate_stats(false).set_attn_scale(scale);
    sdpa_options.set_implementation(fe::AttentionImplementation_t::COMPOSITE);
    if (is_prefill) {
        // Paged prefill uses Q=suffix, KV=full context (q_len <= kv_len).
        // Causal masking must align query's right edge to kv right edge.
        sdpa_options.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT).set_diagonal_band_right_bound(0);
    }
    sdpa_options.set_padding_mask(true).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    sdpa_options.set_paged_attention_k_table(page_table_k);
    sdpa_options.set_paged_attention_v_table(page_table_v);
    sdpa_options.set_paged_attention_max_seq_len_kv(static_cast<int>(max_seq_len_kv_plan));

    auto [O, Stats] = plan.graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;
    O->set_output(true)
        .set_uid(O_UID)
        .set_dim({b_plan, nhead, s_q_plan, head_dim})
        .set_stride({q_stride_b, head_dim, q_stride_s, 1});
    if (qo_ragged_offset != nullptr) {
        // Emit packed prefill output directly so runtime output layout stays
        // consistent with native path ([sum_q, nhead, head_dim]).
        O->set_ragged_offset(qo_ragged_offset);
    }

    auto build_status = [&]() {
        LLAISYS_NVTX_SCOPE(is_prefill ? "attn/cudnn/prefill/graph_build" : "attn/cudnn/decode/graph_build");
        return plan.graph->build(handle, {fe::HeurMode_t::A});
    }();
    if (build_status.is_bad()) {
        printf(
            "[cudnn] build_failed phase=%s b_plan=%lld s_q_plan=%lld nhead=%lld nkvhead=%lld head_dim=%lld "
            "num_blocks=%lld table_size=%d max_seq_len_kv=%lld block_size=%d scale=%f msg=%s\n",
            is_prefill ? "prefill" : "decode",
            static_cast<long long>(b_plan),
            static_cast<long long>(s_q_plan),
            static_cast<long long>(nhead),
            static_cast<long long>(nkvhead),
            static_cast<long long>(head_dim),
            static_cast<long long>(num_blocks),
            int(table_size),
            static_cast<long long>(max_seq_len_kv),
            int(block_size),
            static_cast<double>(scale),
            build_status.get_message().c_str());
        static bool warned_build = false;
        if (!warned_build) {
            warned_build = true;
            printf("[warn] self_attention_paged: CUDNN build failed: %s\n",
                   build_status.get_message().c_str());
        }
        plan.graph_ready = false;
        return false;
    }
    plan.graph_ready = true;
    plan.built_b = b_plan;
    plan.built_s_q = s_q_plan;
    plan.built_table_size = table_size_plan;
    plan.built_max_seq_len_kv = max_seq_len_kv_plan;
    plan.built_num_blocks = num_blocks;
    plan.built_block_size = block_size;
    return true;
}

static bool ensure_cudnn_workspace(CudnnPagedPlan &plan) {
    LLAISYS_NVTX_SCOPE("attn/cudnn/ensure_workspace");
    int64_t workspace_size = 0;
    auto ws_status = plan.graph->get_workspace_size(workspace_size);
    if (ws_status.is_bad()) {
        return false;
    }
    if (static_cast<size_t>(workspace_size) > plan.workspace_cap) {
        if (plan.workspace != nullptr) {
            cudaFree(plan.workspace);
            plan.workspace = nullptr;
        }
        if (workspace_size > 0) {
            LLAISYS_CUDA_CHECK(cudaMalloc(&plan.workspace, static_cast<size_t>(workspace_size)));
        }
        plan.workspace_cap = static_cast<size_t>(workspace_size);
    }
    return true;
}
#endif

bool cudnn_try_paged_attention_decode(tensor_t attn_val,
                                      tensor_t q,
                                      tensor_t k_cache,
                                      tensor_t v_cache,
                                      const CommonAttentionMetadata &prepared,
                                      int32_t block_table_width,
                                      int32_t block_size,
                                      float scale) {
#ifdef ENABLE_CUDNN_FRONTEND
    LLAISYS_NVTX_SCOPE("attn/cudnn/decode");
    if (cudnnGetVersion() < 90500) {
        static bool warned_version = false;
        if (!warned_version) {
            warned_version = true;
            printf("[warn] self_attention_paged: CUDNN paged SDPA requires cuDNN>=9.5.0\n");
        }
        return false;
    }
    if (q->dtype() != LLAISYS_DTYPE_F16 && q->dtype() != LLAISYS_DTYPE_BF16) {
        return false;
    }
    if (q->dtype() != k_cache->dtype() || q->dtype() != v_cache->dtype()) {
        return false;
    }
    const int64_t seqlen = static_cast<int64_t>(q->shape()[0]);
    const int64_t nhead = static_cast<int64_t>(q->shape()[1]);
    const int64_t head_dim = static_cast<int64_t>(q->shape()[2]);
    const int64_t nslot = static_cast<int64_t>(k_cache->shape()[0]);
    const int64_t nkvhead = static_cast<int64_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0) {
        return false;
    }
    if (block_size <= 0 || block_table_width <= 0 || (nslot % block_size) != 0) {
        return false;
    }
    if (prepared.cudnn_seq_lens_q == nullptr || prepared.cudnn_seq_lens_kv == nullptr ||
        prepared.cudnn_page_table == nullptr || prepared.cudnn_b_exec <= 0) {
        return false;
    }

    const int64_t b_exec = static_cast<int64_t>(prepared.cudnn_b_exec);
    const int64_t max_seq_len_q = static_cast<int64_t>(prepared.max_seqlen_q);
    if (max_seq_len_q != 1) {
        return false;
    }
    if (b_exec < seqlen) {
        return false;
    }
    const int32_t table_size = block_table_width;
    const int64_t num_blocks = nslot / block_size;
    const int64_t max_seq_len_kv = static_cast<int64_t>(table_size) * static_cast<int64_t>(block_size);
    if (max_seq_len_kv <= 0) {
        return false;
    }
    if (prepared.max_seqlen_k <= 0 || static_cast<int64_t>(prepared.max_seqlen_k) > max_seq_len_kv) {
        return false;
    }
    if (cudnn_debug_enabled()) {
        printf(
            "[cudnn][dbg] decode_input q=[%lld,%lld,%lld] k=[%lld,%lld,%lld] b_exec=%lld max_seqlen_q=%lld "
            "max_seqlen_k=%d table_size=%d block_size=%d\n",
            static_cast<long long>(seqlen),
            static_cast<long long>(nhead),
            static_cast<long long>(head_dim),
            static_cast<long long>(nslot),
            static_cast<long long>(nkvhead),
            static_cast<long long>(head_dim),
            static_cast<long long>(b_exec),
            static_cast<long long>(max_seq_len_q),
            int(prepared.max_seqlen_k),
            int(table_size),
            int(block_size));
    }

    CudnnRuntimeState &state = cudnn_runtime_state();
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    if (!cudnn_set_stream(state.handle, stream)) {
        return false;
    }

    auto &plan = select_cudnn_plan(state, /*is_prefill=*/false);
    if (!ensure_cudnn_plan_ready(
            plan, state.handle, q->dtype(), false, b_exec, max_seq_len_q, nhead, nkvhead, head_dim, num_blocks,
            table_size, max_seq_len_kv, block_size, scale)) {
        return false;
    }
    if (!ensure_cudnn_workspace(plan)) {
        return false;
    }

    std::unordered_map<int64_t, void *> variant_pack;
    std::vector<int64_t> override_uids;
    std::vector<std::vector<int64_t>> override_shapes;
    std::vector<std::vector<int64_t>> override_strides;
    {
        LLAISYS_NVTX_SCOPE("attn/cudnn/decode/variant_pack");
        variant_pack = {
            {Q_UID, q->data()},
            {K_UID, k_cache->data()},
            {V_UID, v_cache->data()},
            {O_UID, attn_val->data()},
            {SEQ_LEN_Q_UID, const_cast<int32_t *>(prepared.cudnn_seq_lens_q)},
            {SEQ_LEN_KV_UID, const_cast<int32_t *>(prepared.cudnn_seq_lens_kv)},
            {PAGE_TABLE_K_UID, const_cast<int32_t *>(prepared.cudnn_page_table)},
            {PAGE_TABLE_V_UID, const_cast<int32_t *>(prepared.cudnn_page_table)},
        };
        build_cudnn_override_tensors(
            /*is_prefill=*/false,
            b_exec,
            /*s_q_exec=*/1,
            nhead,
            head_dim,
            table_size,
            override_uids,
            override_shapes,
            override_strides);
        cudnn_debug_dump_override(override_uids, override_shapes, override_strides);
        cudnn_debug_dump_i32("decode.seq_lens_q", prepared.cudnn_seq_lens_q, b_exec, 32);
        cudnn_debug_dump_i32("decode.seq_lens_kv", prepared.cudnn_seq_lens_kv, b_exec, 32);
        cudnn_debug_dump_i32("decode.page_table", prepared.cudnn_page_table, b_exec * table_size, 64);
    }

    auto run_decode_execute = [&]() {
        LLAISYS_NVTX_SCOPE("attn/cudnn/decode/execute");
        return plan.graph->execute(state.handle, variant_pack, plan.workspace, override_uids, override_shapes,
                                   override_strides);
    };
    auto exec_status = run_decode_execute();
    if (exec_status.is_bad()) {
        // Dynamic-shape execution may fail for a plan that was built with a
        // smaller template. Force a one-shot rebuild and retry once.
        if (ensure_cudnn_plan_ready(plan,
                                    state.handle,
                                    q->dtype(),
                                    false,
                                    b_exec,
                                    max_seq_len_q,
                                    nhead,
                                    nkvhead,
                                    head_dim,
                                    num_blocks,
                                    table_size,
                                    max_seq_len_kv,
                                    block_size,
                                    scale,
                                    /*force_rebuild=*/true) &&
            ensure_cudnn_workspace(plan)) {
            exec_status = run_decode_execute();
        }
    }
    if (exec_status.is_bad()) {
        static bool warned_exec = false;
        if (!warned_exec) {
            warned_exec = true;
            printf("[warn] self_attention_paged: CUDNN execute failed: %s\n",
                   exec_status.get_message().c_str());
        }
        return false;
    }
    return true;
#else
    static bool warned_no_frontend = false;
    if (!warned_no_frontend) {
        warned_no_frontend = true;
        printf("[warn] self_attention_paged: CUDNN backend requested but cudnn_frontend headers are missing (expected third_party/cudnn_frontend/include)\n");
    }
#endif
    return false;
}

bool cudnn_try_paged_attention_prefill(tensor_t attn_val,
                                       tensor_t q,
                                       tensor_t k_cache,
                                       tensor_t v_cache,
                                       const CommonAttentionMetadata &prepared,
                                       int32_t block_table_width,
                                       int32_t block_size,
                                       float scale) {
#ifdef ENABLE_CUDNN_FRONTEND
    LLAISYS_NVTX_SCOPE("attn/cudnn/prefill");
    if (cudnnGetVersion() < 90500) {
        static bool warned_version = false;
        if (!warned_version) {
            warned_version = true;
            printf("[warn] self_attention_paged: CUDNN paged SDPA requires cuDNN>=9.5.0\n");
        }
        return false;
    }
    if (q->dtype() != LLAISYS_DTYPE_F16 && q->dtype() != LLAISYS_DTYPE_BF16) {
        return false;
    }
    if (q->dtype() != k_cache->dtype() || q->dtype() != v_cache->dtype()) {
        return false;
    }
    const int64_t seqlen = static_cast<int64_t>(q->shape()[0]);
    const int64_t nhead = static_cast<int64_t>(q->shape()[1]);
    const int64_t head_dim = static_cast<int64_t>(q->shape()[2]);
    const int64_t nslot = static_cast<int64_t>(k_cache->shape()[0]);
    const int64_t nkvhead = static_cast<int64_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0) {
        return false;
    }
    if (block_size <= 0 || block_table_width <= 0 || (nslot % block_size) != 0) {
        return false;
    }
    if (prepared.cudnn_seq_lens_q == nullptr || prepared.cudnn_seq_lens_kv == nullptr ||
        prepared.cudnn_page_table == nullptr || prepared.cudnn_qo_ragged_offset == nullptr || prepared.cudnn_b_exec <= 0) {
        return false;
    }
    if (!is_prefill_phase(prepared)) {
        return false;
    }
    const int64_t b_exec = static_cast<int64_t>(prepared.cudnn_b_exec);
    const int64_t max_seq_len_q = static_cast<int64_t>(prepared.max_seqlen_q);
    if (max_seq_len_q <= 0) {
        return false;
    }
    if (b_exec > seqlen) {
        return false;
    }
    const int32_t table_size = block_table_width;
    const int64_t num_blocks = nslot / block_size;
    const int64_t max_seq_len_kv = static_cast<int64_t>(table_size) * static_cast<int64_t>(block_size);
    if (max_seq_len_kv <= 0) {
        return false;
    }
    if (prepared.max_seqlen_k <= 0 || static_cast<int64_t>(prepared.max_seqlen_k) > max_seq_len_kv) {
        return false;
    }
    if (cudnn_debug_enabled()) {
        printf(
            "[cudnn][dbg] prefill_input q=[%lld,%lld,%lld] k=[%lld,%lld,%lld] b_exec=%lld max_seqlen_q=%lld "
            "max_seqlen_k=%d table_size=%d block_size=%d\n",
            static_cast<long long>(seqlen),
            static_cast<long long>(nhead),
            static_cast<long long>(head_dim),
            static_cast<long long>(nslot),
            static_cast<long long>(nkvhead),
            static_cast<long long>(head_dim),
            static_cast<long long>(b_exec),
            static_cast<long long>(max_seq_len_q),
            int(prepared.max_seqlen_k),
            int(table_size),
            int(block_size));
    }

    CudnnRuntimeState &state = cudnn_runtime_state();
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    if (!cudnn_set_stream(state.handle, stream)) {
        return false;
    }

    auto &plan = select_cudnn_plan(state, /*is_prefill=*/true);
    if (!ensure_cudnn_plan_ready(
            plan, state.handle, q->dtype(), true, b_exec, max_seq_len_q, nhead, nkvhead, head_dim, num_blocks,
            table_size, max_seq_len_kv, block_size, scale)) {
        return false;
    }
    if (!ensure_cudnn_workspace(plan)) {
        return false;
    }

    std::unordered_map<int64_t, void *> variant_pack;
    std::vector<int64_t> override_uids;
    std::vector<std::vector<int64_t>> override_shapes;
    std::vector<std::vector<int64_t>> override_strides;
    {
        LLAISYS_NVTX_SCOPE("attn/cudnn/prefill/variant_pack");
        variant_pack = {
            {Q_UID, q->data()},
            {K_UID, k_cache->data()},
            {V_UID, v_cache->data()},
            {O_UID, attn_val->data()},
            {SEQ_LEN_Q_UID, const_cast<int32_t *>(prepared.cudnn_seq_lens_q)},
            {SEQ_LEN_KV_UID, const_cast<int32_t *>(prepared.cudnn_seq_lens_kv)},
            {PAGE_TABLE_K_UID, const_cast<int32_t *>(prepared.cudnn_page_table)},
            {PAGE_TABLE_V_UID, const_cast<int32_t *>(prepared.cudnn_page_table)},
            {QO_RAGGED_OFFSET_UID, const_cast<int32_t *>(prepared.cudnn_qo_ragged_offset)},
        };
        build_cudnn_override_tensors(
            /*is_prefill=*/true,
            b_exec,
            std::max<int64_t>(1, max_seq_len_q),
            nhead,
            head_dim,
            table_size,
            override_uids,
            override_shapes,
            override_strides);
        cudnn_debug_dump_override(override_uids, override_shapes, override_strides);
        cudnn_debug_dump_i32("prefill.seq_lens_q", prepared.cudnn_seq_lens_q, b_exec, 32);
        cudnn_debug_dump_i32("prefill.seq_lens_kv", prepared.cudnn_seq_lens_kv, b_exec, 32);
        cudnn_debug_dump_i32("prefill.ragged_qo", prepared.cudnn_qo_ragged_offset, b_exec + 1, 33);
        cudnn_debug_dump_i32("prefill.page_table", prepared.cudnn_page_table, b_exec * table_size, 64);
    }

    auto run_prefill_execute = [&]() {
        LLAISYS_NVTX_SCOPE("attn/cudnn/prefill/execute");
        return plan.graph->execute(state.handle, variant_pack, plan.workspace, override_uids, override_shapes,
                                   override_strides);
    };
    auto exec_status = run_prefill_execute();
    if (exec_status.is_bad()) {
        // Dynamic-shape execution may fail for a plan that was built with a
        // smaller template. Force a one-shot rebuild and retry once.
        if (ensure_cudnn_plan_ready(plan,
                                    state.handle,
                                    q->dtype(),
                                    true,
                                    b_exec,
                                    max_seq_len_q,
                                    nhead,
                                    nkvhead,
                                    head_dim,
                                    num_blocks,
                                    table_size,
                                    max_seq_len_kv,
                                    block_size,
                                    scale,
                                    /*force_rebuild=*/true) &&
            ensure_cudnn_workspace(plan)) {
            exec_status = run_prefill_execute();
        }
    }
    if (exec_status.is_bad()) {
        static bool warned_exec = false;
        if (!warned_exec) {
            warned_exec = true;
            printf("[warn] self_attention_paged: CUDNN prefill execute failed: %s\n",
                   exec_status.get_message().c_str());
        }
        return false;
    }
    return true;
#else
    static bool warned_no_frontend = false;
    if (!warned_no_frontend) {
        warned_no_frontend = true;
        printf("[warn] self_attention_paged: CUDNN backend requested but cudnn_frontend headers are missing (expected third_party/cudnn_frontend/include)\n");
    }
#endif
    return false;
}
#endif

} // namespace

void reshape_and_cache(tensor_t k_cache,
                       tensor_t v_cache,
                       tensor_t k_src,
                       tensor_t v_src,
                       tensor_t slot_idxs_i32) {
    LLAISYS_NVTX_SCOPE("attn/reshape_and_cache");
    CHECK_ARGUMENT(k_cache->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "reshape_and_cache: k_cache must be CUDA");
    CHECK_ARGUMENT(v_cache->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "reshape_and_cache: v_cache must be CUDA");
    CHECK_ARGUMENT(k_src->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "reshape_and_cache: k_src must be CUDA");
    CHECK_ARGUMENT(v_src->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "reshape_and_cache: v_src must be CUDA");
    CHECK_ARGUMENT(k_cache->dtype() == k_src->dtype(),
                   "reshape_and_cache: k dtype mismatch");
    CHECK_ARGUMENT(v_cache->dtype() == v_src->dtype(),
                   "reshape_and_cache: v dtype mismatch");
    CHECK_ARGUMENT(k_cache->shape()[1] == k_src->shape()[1] && k_cache->shape()[2] == k_src->shape()[2],
                   "reshape_and_cache: k shape mismatch");
    CHECK_ARGUMENT(v_cache->shape()[1] == v_src->shape()[1] && v_cache->shape()[2] == v_src->shape()[2],
                   "reshape_and_cache: v shape mismatch");
    CHECK_ARGUMENT(slot_idxs_i32 != nullptr, "reshape_and_cache: slot idx tensor is null");
    CHECK_ARGUMENT(slot_idxs_i32->dtype() == LLAISYS_DTYPE_I32,
                   "reshape_and_cache: slot idx dtype must be I32");
    CHECK_ARGUMENT(slot_idxs_i32->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "reshape_and_cache: slot idx must be CUDA");
    CHECK_ARGUMENT(slot_idxs_i32->shape().size() == 1,
                   "reshape_and_cache: slot idx must be 1D");
    CHECK_ARGUMENT(slot_idxs_i32->shape()[0] == k_src->shape()[0],
                   "reshape_and_cache: slot idx length mismatch");

    const int32_t *slot_ptr = reinterpret_cast<const int32_t *>(slot_idxs_i32->data());
    switch (k_src->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_scatter_cache_by_slots<float>(k_cache, k_src, slot_ptr);
        launch_scatter_cache_by_slots<float>(v_cache, v_src, slot_ptr);
        break;
    case LLAISYS_DTYPE_F16:
        launch_scatter_cache_by_slots<llaisys::fp16_t>(k_cache, k_src, slot_ptr);
        launch_scatter_cache_by_slots<llaisys::fp16_t>(v_cache, v_src, slot_ptr);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_scatter_cache_by_slots<llaisys::bf16_t>(k_cache, k_src, slot_ptr);
        launch_scatter_cache_by_slots<llaisys::bf16_t>(v_cache, v_src, slot_ptr);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(k_src->dtype());
    }
}

__global__ void build_slot_mapping_kernel(int32_t *slot_mapping_out,
                                          int32_t ntoken,
                                          int32_t nseq,
                                          const int32_t *query_start_loc,
                                          const int32_t *seq_lens,
                                          const int32_t *block_tables,
                                          int32_t block_table_width,
                                          int32_t block_size) {
    const int32_t t = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t < 0 || t >= ntoken) {
        return;
    }

    int32_t lo = 0;
    int32_t hi = nseq;
    while (lo < hi) {
        const int32_t mid = lo + ((hi - lo) >> 1);
        if (query_start_loc[mid + 1] <= t) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int32_t row = lo;
    if (row < 0 || row >= nseq) {
        slot_mapping_out[t] = -1;
        return;
    }

    const int32_t row_start = query_start_loc[row];
    const int32_t row_end = query_start_loc[row + 1];
    const int32_t row_scheduled = row_end - row_start;
    if (row_scheduled <= 0) {
        slot_mapping_out[t] = -1;
        return;
    }
    const int32_t local = t - row_start;
    if (local < 0 || local >= row_scheduled) {
        slot_mapping_out[t] = -1;
        return;
    }

    const int32_t seq_len = seq_lens[row];
    const int32_t qpos = (seq_len - row_scheduled) + local;
    if (qpos < 0) {
        slot_mapping_out[t] = -1;
        return;
    }
    const int32_t bidx = qpos / block_size;
    const int32_t boff = qpos % block_size;
    if (bidx < 0 || bidx >= block_table_width) {
        slot_mapping_out[t] = -1;
        return;
    }
    const int32_t bid = block_tables[row * block_table_width + bidx];
    if (bid < 0) {
        slot_mapping_out[t] = -1;
        return;
    }
    slot_mapping_out[t] = bid * block_size + boff;
}

__global__ void build_query_start_loc_and_seq_lens_kernel(int32_t *query_start_loc_out,
                                                           int32_t *seq_lens_out,
                                                           int32_t nseq,
                                                           const int32_t *req_num_scheduled_tokens,
                                                           const int32_t *req_num_computed_tokens) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int32_t acc = 0;
    query_start_loc_out[0] = 0;
    for (int32_t i = 0; i < nseq; ++i) {
        const int32_t n_scheduled = req_num_scheduled_tokens[i];
        const int32_t n_computed = req_num_computed_tokens[i];
        acc += n_scheduled;
        query_start_loc_out[i + 1] = acc;
        seq_lens_out[i] = n_scheduled + n_computed;
    }
}

void build_query_start_loc_and_seq_lens(tensor_t query_start_loc,
                                        tensor_t seq_lens,
                                        tensor_t req_num_scheduled_tokens,
                                        tensor_t req_num_computed_tokens) {
    CHECK_ARGUMENT(query_start_loc != nullptr && seq_lens != nullptr && req_num_scheduled_tokens != nullptr &&
                       req_num_computed_tokens != nullptr,
                   "build_query_start_loc_and_seq_lens: tensors must be non-null");
    CHECK_ARGUMENT(query_start_loc->deviceType() == LLAISYS_DEVICE_NVIDIA && seq_lens->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       req_num_scheduled_tokens->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       req_num_computed_tokens->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "build_query_start_loc_and_seq_lens: tensors must be CUDA");
    CHECK_ARGUMENT(query_start_loc->dtype() == LLAISYS_DTYPE_I32 && seq_lens->dtype() == LLAISYS_DTYPE_I32 &&
                       req_num_scheduled_tokens->dtype() == LLAISYS_DTYPE_I32 &&
                       req_num_computed_tokens->dtype() == LLAISYS_DTYPE_I32,
                   "build_query_start_loc_and_seq_lens: tensors must be I32");
    CHECK_ARGUMENT(query_start_loc->ndim() == 1 && seq_lens->ndim() == 1 && req_num_scheduled_tokens->ndim() == 1 &&
                       req_num_computed_tokens->ndim() == 1,
                   "build_query_start_loc_and_seq_lens: tensors must be 1D");
    CHECK_ARGUMENT(query_start_loc->isContiguous() && seq_lens->isContiguous() && req_num_scheduled_tokens->isContiguous() &&
                       req_num_computed_tokens->isContiguous(),
                   "build_query_start_loc_and_seq_lens: tensors must be contiguous");

    const int32_t nseq = static_cast<int32_t>(req_num_scheduled_tokens->shape()[0]);
    CHECK_ARGUMENT(nseq >= 0, "build_query_start_loc_and_seq_lens: invalid nseq");
    CHECK_ARGUMENT(req_num_computed_tokens->shape()[0] == static_cast<size_t>(nseq),
                   "build_query_start_loc_and_seq_lens: req_num_computed_tokens size mismatch");
    CHECK_ARGUMENT(seq_lens->shape()[0] == static_cast<size_t>(nseq),
                   "build_query_start_loc_and_seq_lens: seq_lens size mismatch");
    CHECK_ARGUMENT(query_start_loc->shape()[0] == static_cast<size_t>(nseq + 1),
                   "build_query_start_loc_and_seq_lens: query_start_loc size mismatch");

    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    build_query_start_loc_and_seq_lens_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<int32_t *>(query_start_loc->data()),
                                                                    reinterpret_cast<int32_t *>(seq_lens->data()),
                                                                    nseq,
                                                                    reinterpret_cast<const int32_t *>(
                                                                        req_num_scheduled_tokens->data()),
                                                                    reinterpret_cast<const int32_t *>(
                                                                        req_num_computed_tokens->data()));
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

__global__ void build_block_positions_kernel(int64_t *pos_ids_out,
                                             int32_t ntoken,
                                             int32_t nseq,
                                             const int32_t *query_start_loc,
                                             const int32_t *seq_lens) {
    const int32_t t = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t < 0 || t >= ntoken) {
        return;
    }

    int32_t lo = 0;
    int32_t hi = nseq;
    while (lo < hi) {
        const int32_t mid = lo + ((hi - lo) >> 1);
        if (query_start_loc[mid + 1] <= t) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int32_t row = lo;
    if (row < 0 || row >= nseq) {
        pos_ids_out[t] = -1;
        return;
    }
    const int32_t row_start = query_start_loc[row];
    const int32_t row_end = query_start_loc[row + 1];
    const int32_t row_scheduled = row_end - row_start;
    if (row_scheduled <= 0) {
        pos_ids_out[t] = -1;
        return;
    }
    const int32_t local = t - row_start;
    if (local < 0 || local >= row_scheduled) {
        pos_ids_out[t] = -1;
        return;
    }
    const int32_t qpos = (seq_lens[row] - row_scheduled) + local;
    pos_ids_out[t] = static_cast<int64_t>(qpos);
}

void build_block_positions(tensor_t pos_ids,
                           tensor_t query_start_loc,
                           tensor_t seq_lens) {
    CHECK_ARGUMENT(pos_ids != nullptr && query_start_loc != nullptr && seq_lens != nullptr,
                   "build_block_positions: tensors must be non-null");
    CHECK_ARGUMENT(pos_ids->deviceType() == LLAISYS_DEVICE_NVIDIA && query_start_loc->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       seq_lens->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "build_block_positions: tensors must be CUDA");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64 && query_start_loc->dtype() == LLAISYS_DTYPE_I32 &&
                       seq_lens->dtype() == LLAISYS_DTYPE_I32,
                   "build_block_positions: dtype mismatch");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && query_start_loc->ndim() == 1 && seq_lens->ndim() == 1,
                   "build_block_positions: tensors must be 1D");
    CHECK_ARGUMENT(pos_ids->isContiguous() && query_start_loc->isContiguous() && seq_lens->isContiguous(),
                   "build_block_positions: tensors must be contiguous");
    CHECK_ARGUMENT(query_start_loc->shape()[0] == seq_lens->shape()[0] + 1,
                   "build_block_positions: query_start_loc size mismatch");
    const int32_t ntoken = static_cast<int32_t>(pos_ids->shape()[0]);
    const int32_t nseq = static_cast<int32_t>(seq_lens->shape()[0]);
    if (ntoken <= 0 || nseq <= 0) {
        return;
    }
    constexpr int32_t kBlock = 256;
    const int32_t grid = (ntoken + kBlock - 1) / kBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    build_block_positions_kernel<<<grid, kBlock, 0, stream>>>(reinterpret_cast<int64_t *>(pos_ids->data()),
                                                               ntoken,
                                                               nseq,
                                                               reinterpret_cast<const int32_t *>(query_start_loc->data()),
                                                               reinterpret_cast<const int32_t *>(seq_lens->data()));
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

__global__ void build_last_token_logits_indices_kernel(int64_t *logits_indices_out,
                                                       int32_t nseq,
                                                       const int32_t *query_start_loc) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= nseq) {
        return;
    }
    logits_indices_out[i] = static_cast<int64_t>(query_start_loc[i + 1] - 1);
}

void build_last_token_logits_indices(tensor_t logits_indices,
                                     tensor_t query_start_loc) {
    CHECK_ARGUMENT(logits_indices != nullptr && query_start_loc != nullptr,
                   "build_last_token_logits_indices: tensors must be non-null");
    CHECK_ARGUMENT(logits_indices->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       query_start_loc->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "build_last_token_logits_indices: tensors must be CUDA");
    CHECK_ARGUMENT(logits_indices->dtype() == LLAISYS_DTYPE_I64 && query_start_loc->dtype() == LLAISYS_DTYPE_I32,
                   "build_last_token_logits_indices: dtype mismatch");
    CHECK_ARGUMENT(logits_indices->ndim() == 1 && query_start_loc->ndim() == 1,
                   "build_last_token_logits_indices: tensors must be 1D");
    CHECK_ARGUMENT(logits_indices->isContiguous() && query_start_loc->isContiguous(),
                   "build_last_token_logits_indices: tensors must be contiguous");
    const int32_t nseq = static_cast<int32_t>(logits_indices->shape()[0]);
    CHECK_ARGUMENT(query_start_loc->shape()[0] == static_cast<size_t>(nseq + 1),
                   "build_last_token_logits_indices: query_start_loc size mismatch");
    if (nseq <= 0) {
        return;
    }
    constexpr int32_t kBlock = 256;
    const int32_t grid = (nseq + kBlock - 1) / kBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    build_last_token_logits_indices_kernel<<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<int64_t *>(logits_indices->data()),
        nseq,
        reinterpret_cast<const int32_t *>(query_start_loc->data()));
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

void build_slot_mapping(tensor_t slot_mapping,
                        tensor_t query_start_loc,
                        tensor_t seq_lens,
                        tensor_t block_tables,
                        int32_t block_table_width,
                        int32_t block_size) {
    CHECK_ARGUMENT(slot_mapping != nullptr && query_start_loc != nullptr && seq_lens != nullptr && block_tables != nullptr,
                   "build_slot_mapping: metadata tensors must be non-null");
    CHECK_ARGUMENT(slot_mapping->deviceType() == LLAISYS_DEVICE_NVIDIA && query_start_loc->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       seq_lens->deviceType() == LLAISYS_DEVICE_NVIDIA && block_tables->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "build_slot_mapping: tensors must be CUDA");
    CHECK_ARGUMENT(slot_mapping->dtype() == LLAISYS_DTYPE_I32 && query_start_loc->dtype() == LLAISYS_DTYPE_I32 &&
                       seq_lens->dtype() == LLAISYS_DTYPE_I32 && block_tables->dtype() == LLAISYS_DTYPE_I32,
                   "build_slot_mapping: tensors must be I32");
    CHECK_ARGUMENT(slot_mapping->ndim() == 1 && query_start_loc->ndim() == 1 && seq_lens->ndim() == 1 && block_tables->ndim() == 1,
                   "build_slot_mapping: tensors must be 1D");
    CHECK_ARGUMENT(slot_mapping->isContiguous() && query_start_loc->isContiguous() && seq_lens->isContiguous() &&
                       block_tables->isContiguous(),
                   "build_slot_mapping: tensors must be contiguous");
    CHECK_ARGUMENT(block_table_width > 0 && block_size > 0, "build_slot_mapping: block params must be > 0");
    CHECK_ARGUMENT(query_start_loc->shape()[0] == seq_lens->shape()[0] + 1,
                   "build_slot_mapping: query_start_loc size mismatch");
    CHECK_ARGUMENT(block_tables->shape()[0] == seq_lens->shape()[0] * static_cast<size_t>(block_table_width),
                   "build_slot_mapping: block_tables size mismatch");

    const int32_t ntoken = static_cast<int32_t>(slot_mapping->shape()[0]);
    const int32_t nseq = static_cast<int32_t>(seq_lens->shape()[0]);
    if (ntoken <= 0 || nseq <= 0) {
        return;
    }
    constexpr int32_t kBlock = 256;
    const int32_t grid = (ntoken + kBlock - 1) / kBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    build_slot_mapping_kernel<<<grid, kBlock, 0, stream>>>(reinterpret_cast<int32_t *>(slot_mapping->data()),
                                                            ntoken,
                                                            nseq,
                                                            reinterpret_cast<const int32_t *>(query_start_loc->data()),
                                                            reinterpret_cast<const int32_t *>(seq_lens->data()),
                                                            reinterpret_cast<const int32_t *>(block_tables->data()),
                                                            block_table_width,
                                                            block_size);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    LLAISYS_NVTX_SCOPE("attn/self_attention");
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

void self_attention_paged_prepared(tensor_t attn_val,
                                   tensor_t q,
                                   tensor_t k_cache,
                                   tensor_t v_cache,
                                   const CommonAttentionMetadata &prepared,
                                   int32_t block_table_width,
                                   int32_t block_size,
                                   float scale) {
    LLAISYS_NVTX_SCOPE("attn/self_attention_paged_prepared");
    const std::int32_t seqlen = static_cast<std::int32_t>(q->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(q->shape()[1]);
    const std::int32_t head_dim = static_cast<std::int32_t>(q->shape()[2]);
    const std::int32_t nslot = static_cast<std::int32_t>(k_cache->shape()[0]);
    const std::int32_t nkvhead = static_cast<std::int32_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0 || prepared.nseq <= 0) {
        return;
    }
    if (prepared.cu_seqlens_q == nullptr || prepared.block_tables == nullptr ||
        prepared.cu_seqlens_k == nullptr) {
        return;
    }
    const bool is_decode = is_decode_phase(prepared);

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        if (is_decode) {
            launch_paged_attention_decode_prepared<float>(
                attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
                nhead, nkvhead, head_dim);
        } else {
            launch_paged_attention_prepared<float>(
                attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
                nhead, nkvhead, head_dim);
        }
        return;
    case LLAISYS_DTYPE_F16:
        if (is_decode) {
            launch_paged_attention_decode_prepared<llaisys::fp16_t>(
                attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
                nhead, nkvhead, head_dim);
        } else {
            launch_paged_attention_prepared<llaisys::fp16_t>(
                attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
                nhead, nkvhead, head_dim);
        }
        return;
    case LLAISYS_DTYPE_BF16:
        if (is_decode) {
            launch_paged_attention_decode_prepared<llaisys::bf16_t>(
                attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
                nhead, nkvhead, head_dim);
        } else {
            launch_paged_attention_prepared<llaisys::bf16_t>(
                attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
                nhead, nkvhead, head_dim);
        }
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}

void dispatch_attention_with_backend(tensor_t attn_val,
                                     tensor_t q,
                                     tensor_t k_cache,
                                     tensor_t v_cache,
                                     const CommonAttentionMetadata &metadata,
                                     PagedAttentionBackend backend,
                                     int32_t block_table_width,
                                     int32_t block_size,
                                     float scale) {
    const CommonAttentionMetadata &prepared = metadata;
    const bool is_decode = is_decode_phase(prepared);
    const bool is_prefill = is_prefill_phase(prepared);
    CHECK_ARGUMENT(is_decode || is_prefill, "dispatch_attention_with_backend: invalid attention phase");
    if (backend == PagedAttentionBackend::CUDNN) {
        CHECK_ARGUMENT(prepared.cudnn_seq_lens_q != nullptr && prepared.cudnn_seq_lens_kv != nullptr &&
                           prepared.cudnn_page_table != nullptr && prepared.cudnn_b_exec > 0,
                       "dispatch_attention_with_backend: missing CUDNN metadata");
        if (is_prefill) {
            CHECK_ARGUMENT(prepared.cudnn_qo_ragged_offset != nullptr,
                           "dispatch_attention_with_backend: missing CUDNN prefill ragged metadata");
        }
#ifdef ENABLE_CUDNN_API
        const bool ok = is_prefill
                            ? cudnn_try_paged_attention_prefill(
                                  attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale)
                            : cudnn_try_paged_attention_decode(
                                  attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale);
        CHECK_ARGUMENT(ok, "self_attention_paged: CUDNN backend execution failed");
        return;
#else
        CHECK_ARGUMENT(false, "self_attention_paged: CUDNN backend requested but ENABLE_CUDNN_API is off");
#endif
    }
    CHECK_ARGUMENT(prepared.cu_seqlens_q != nullptr && prepared.block_tables != nullptr &&
                       prepared.cu_seqlens_k != nullptr,
                   "dispatch_attention_with_backend: missing native BLOCK metadata");
    self_attention_paged_prepared(attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale);
}

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
                          float scale) {
    const std::int32_t seqlen = static_cast<std::int32_t>(q->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(q->shape()[1]);
    const std::int32_t head_dim = static_cast<std::int32_t>(q->shape()[2]);
    const std::int32_t nslot = static_cast<std::int32_t>(k_cache->shape()[0]);
    const std::int32_t nkvhead = static_cast<std::int32_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0) {
        return;
    }
    if (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr || block_tables == nullptr || slot_mapping == nullptr) {
        return;
    }

    CHECK_ARGUMENT(cu_seqlens_q->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       block_tables->deviceType() == LLAISYS_DEVICE_NVIDIA && cu_seqlens_k->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       slot_mapping->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "self_attention_paged: metadata tensors must be CUDA");
    CHECK_ARGUMENT(cu_seqlens_q->shape()[0] == cu_seqlens_k->shape()[0], "self_attention_paged: cu_seqlens size mismatch");
    CHECK_ARGUMENT(slot_mapping->shape()[0] == static_cast<size_t>(seqlen), "self_attention_paged: slot_mapping size mismatch");
    CHECK_ARGUMENT(max_seqlen_q > 0 && max_seqlen_k > 0, "self_attention_paged: invalid max_seqlen");
    CommonAttentionMetadata prepared{
        reinterpret_cast<const int32_t *>(cu_seqlens_q->data()),
        reinterpret_cast<const int32_t *>(cu_seqlens_k->data()),
        reinterpret_cast<const int32_t *>(block_tables->data()),
        reinterpret_cast<const int32_t *>(slot_mapping->data()),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        static_cast<int32_t>(cu_seqlens_q->shape()[0]) - 1,
        max_seqlen_q,
        max_seqlen_k,
        (max_seqlen_q == 1 ? AttentionPhase::DECODE : AttentionPhase::PREFILL)};
    self_attention_paged_prepared(attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale);
}

} // namespace llaisys::ops::cuda
