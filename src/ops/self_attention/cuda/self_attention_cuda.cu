#include "self_attention_cuda.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime_api.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace llaisys::device::nvidia {
void cuda_check(cudaError_t rc, const char *what, const char *file, int line);
} // namespace llaisys::device::nvidia

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace llaisys::ops::cuda {

namespace {

void release_device_i32(int32_t *&ptr) {
    if (ptr != nullptr) {
        LLAISYS_CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
}

void ensure_device_i32_capacity(int32_t *&ptr, size_t &cap, size_t n) {
    if (n <= cap) {
        return;
    }
    release_device_i32(ptr);
    LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(int32_t)));
    cap = n;
}

void upload_device_i32(int32_t *&ptr, size_t &cap, const std::vector<int32_t> &host) {
    const size_t n = host.size();
    if (n == 0) {
        return;
    }
    ensure_device_i32_capacity(ptr, cap, n);
    LLAISYS_CUDA_CHECK(cudaMemcpy(ptr, host.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
}

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
                                            const int32_t *q_seq_rows,
                                            const int32_t *q_pos,
                                            const int32_t *block_tables,
                                            const int32_t *seq_lens,
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

    const int32_t row = q_seq_rows[t];
    if (row < 0 || row >= nseq) {
        return;
    }
    const int32_t qpos = q_pos[t];
    const int32_t seq_len = seq_lens[row];
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

template <typename T>
void launch_paged_attention_prepared(tensor_t attn_val,
                                     tensor_t q,
                                     tensor_t k_cache,
                                     tensor_t v_cache,
                                     const PagedAttentionPrepared &prepared,
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
        prepared.q_seq_rows,
        prepared.q_pos,
        prepared.block_tables,
        prepared.seq_lens,
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

} // namespace

PagedAttentionPrepared::~PagedAttentionPrepared() {
    release_device_i32(q_seq_rows);
    release_device_i32(q_pos);
    release_device_i32(block_tables);
    release_device_i32(seq_lens);
    q_seq_rows_cap = 0;
    q_pos_cap = 0;
    block_tables_cap = 0;
    seq_lens_cap = 0;
    nseq = 0;
}

void prepare_paged_attention(PagedAttentionPrepared &prepared,
                             const std::vector<int32_t> &q_seq_rows,
                             const std::vector<int32_t> &q_pos,
                             const std::vector<int32_t> &block_tables,
                             const std::vector<int32_t> &seq_lens) {
    upload_device_i32(prepared.q_seq_rows, prepared.q_seq_rows_cap, q_seq_rows);
    upload_device_i32(prepared.q_pos, prepared.q_pos_cap, q_pos);
    upload_device_i32(prepared.block_tables, prepared.block_tables_cap, block_tables);
    upload_device_i32(prepared.seq_lens, prepared.seq_lens_cap, seq_lens);
    prepared.nseq = static_cast<int32_t>(seq_lens.size());
}

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

void self_attention_paged_prepared(tensor_t attn_val,
                                   tensor_t q,
                                   tensor_t k_cache,
                                   tensor_t v_cache,
                                   const PagedAttentionPrepared &prepared,
                                   int32_t block_table_width,
                                   int32_t block_size,
                                   float scale) {
    const std::int32_t seqlen = static_cast<std::int32_t>(q->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(q->shape()[1]);
    const std::int32_t head_dim = static_cast<std::int32_t>(q->shape()[2]);
    const std::int32_t nslot = static_cast<std::int32_t>(k_cache->shape()[0]);
    const std::int32_t nkvhead = static_cast<std::int32_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0 || prepared.nseq <= 0) {
        return;
    }
    if (prepared.q_seq_rows == nullptr || prepared.q_pos == nullptr || prepared.block_tables == nullptr ||
        prepared.seq_lens == nullptr) {
        return;
    }

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_paged_attention_prepared<float>(
            attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
            nhead, nkvhead, head_dim);
        return;
    case LLAISYS_DTYPE_F16:
        launch_paged_attention_prepared<llaisys::fp16_t>(
            attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
            nhead, nkvhead, head_dim);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_paged_attention_prepared<llaisys::bf16_t>(
            attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale, seqlen, nslot, prepared.nseq,
            nhead, nkvhead, head_dim);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}

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
                          float scale) {
    const std::int32_t seqlen = static_cast<std::int32_t>(q->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(q->shape()[1]);
    const std::int32_t head_dim = static_cast<std::int32_t>(q->shape()[2]);
    const std::int32_t nslot = static_cast<std::int32_t>(k_cache->shape()[0]);
    const std::int32_t nkvhead = static_cast<std::int32_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0) {
        return;
    }
    if (q_seq_rows.empty() || q_pos.empty() || block_tables.empty() || seq_lens.empty()) {
        return;
    }

    // Compatibility path: keep one-shot API by preparing metadata internally.
    static thread_local PagedAttentionPrepared prepared{};
    prepare_paged_attention(prepared, q_seq_rows, q_pos, block_tables, seq_lens);
    self_attention_paged_prepared(attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale);
}

} // namespace llaisys::ops::cuda
