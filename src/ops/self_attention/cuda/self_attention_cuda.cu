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

#ifdef ENABLE_CUDNN_API
struct CudnnPlanKey {
    llaisysDataType_t dtype{LLAISYS_DTYPE_F32};
    int64_t b{0};
    int64_t nhead{0};
    int64_t nkvhead{0};
    int64_t head_dim{0};
    int64_t nslot{0};
    int32_t block_size{0};
    int32_t block_table_capacity{0};

    bool operator==(const CudnnPlanKey &o) const {
        return dtype == o.dtype && b == o.b && nhead == o.nhead && nkvhead == o.nkvhead &&
               head_dim == o.head_dim && nslot == o.nslot && block_size == o.block_size &&
               block_table_capacity == o.block_table_capacity;
    }
};

struct CudnnPlanKeyHash {
    size_t operator()(const CudnnPlanKey &k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) {
            h ^= static_cast<size_t>(v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        };
        mix(static_cast<uint64_t>(k.dtype));
        mix(static_cast<uint64_t>(k.b));
        mix(static_cast<uint64_t>(k.nhead));
        mix(static_cast<uint64_t>(k.nkvhead));
        mix(static_cast<uint64_t>(k.head_dim));
        mix(static_cast<uint64_t>(k.nslot));
        mix(static_cast<uint64_t>(k.block_size));
        mix(static_cast<uint64_t>(k.block_table_capacity));
        return h;
    }
};

struct CudnnPagedPlan {
#ifdef ENABLE_CUDNN_FRONTEND
    std::shared_ptr<cudnn_frontend::graph::Graph> graph{};
    bool graph_ready{false};

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

struct CudnnMetaBuffer {
    int32_t *d_meta{nullptr};
    int32_t *h_meta{nullptr};
    size_t cap{0};

    ~CudnnMetaBuffer() {
        if (d_meta != nullptr) {
            cudaFree(d_meta);
            d_meta = nullptr;
        }
        if (h_meta != nullptr) {
            cudaFreeHost(h_meta);
            h_meta = nullptr;
        }
        cap = 0;
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

struct CudnnRuntimeState {
    cudnnHandle_t handle{nullptr};
    std::unordered_map<CudnnPlanKey, std::unique_ptr<CudnnPagedPlan>, CudnnPlanKeyHash> plan_cache;
    std::unordered_map<size_t, int64_t> prebuilt_max_bucket;
    CudnnMetaBuffer meta_buf;
};

struct CudnnProblemShape {
    int64_t seqlen{0};
    int64_t nhead{0};
    int64_t head_dim{0};
    int64_t nslot{0};
    int64_t nkvhead{0};
    int64_t b_exec{0};
    int32_t block_table_capacity{0};
    int64_t table_size{0};
    int64_t num_blocks{0};
    int64_t max_seq_len_kv{0};
};

struct CudnnMetaPtrs {
    int32_t *d_seq_q{nullptr};
    int32_t *d_seq_kv{nullptr};
    int32_t *d_page{nullptr};
    size_t bsz{0};
    size_t page_n{0};
    size_t meta_n{0};
};

static CudnnRuntimeState &cudnn_runtime_state() {
    static thread_local CudnnRuntimeState state{};
    return state;
}

static bool cudnn_validate_inputs(tensor_t q,
                                  tensor_t k_cache,
                                  tensor_t v_cache,
                                  const CommonAttentionMetadata &prepared,
                                  int32_t block_table_width,
                                  int32_t block_size,
                                  CudnnProblemShape &shape) {
    if (cudnnGetVersion() < 90500) {
        static bool warned_version = false;
        if (!warned_version) {
            warned_version = true;
            printf("[warn] self_attention_paged: CUDNN paged SDPA requires cuDNN>=9.5.0, fallback to NATIVE\n");
        }
        return false;
    }
    if (q->dtype() != LLAISYS_DTYPE_F16 && q->dtype() != LLAISYS_DTYPE_BF16) {
        return false;
    }
    if (q->dtype() != k_cache->dtype() || q->dtype() != v_cache->dtype()) {
        return false;
    }
    shape.seqlen = static_cast<int64_t>(q->shape()[0]);
    shape.nhead = static_cast<int64_t>(q->shape()[1]);
    shape.head_dim = static_cast<int64_t>(q->shape()[2]);
    shape.nslot = static_cast<int64_t>(k_cache->shape()[0]);
    shape.nkvhead = static_cast<int64_t>(k_cache->shape()[1]);
    if (shape.seqlen <= 0 || shape.nhead <= 0 || shape.head_dim <= 0 || shape.nslot <= 0 || shape.nkvhead <= 0) {
        return false;
    }
    if (block_size <= 0 || block_table_width <= 0 || (shape.nslot % block_size) != 0) {
        return false;
    }
    if (prepared.q_seq_rows == nullptr || prepared.q_pos == nullptr || prepared.seq_lens == nullptr ||
        prepared.block_tables == nullptr || prepared.nseq <= 0) {
        return false;
    }

    shape.b_exec = 1;
    while (shape.b_exec < shape.seqlen) {
        shape.b_exec <<= 1;
    }
    int32_t width_p2 = 1;
    while (width_p2 < block_table_width) {
        width_p2 <<= 1;
    }
    shape.block_table_capacity = width_p2;
    if (const char *raw = std::getenv("LLAISYS_CUDNN_BLOCK_TABLE_CAP"); raw != nullptr) {
        int32_t cfg_cap = static_cast<int32_t>(std::atoi(raw));
        if (cfg_cap > 0) {
            int32_t cfg_p2 = 1;
            while (cfg_p2 < cfg_cap) {
                cfg_p2 <<= 1;
            }
            shape.block_table_capacity = std::max<int32_t>(shape.block_table_capacity, cfg_p2);
        }
    }
    shape.table_size = shape.block_table_capacity;
    shape.num_blocks = shape.nslot / block_size;
    shape.max_seq_len_kv = static_cast<int64_t>(shape.block_table_capacity) * static_cast<int64_t>(block_size);
    return true;
}

static CudnnPlanKey make_cudnn_plan_key(llaisysDataType_t dtype,
                                        const CudnnProblemShape &shape,
                                        int32_t block_size,
                                        int64_t b_key) {
    return CudnnPlanKey{
        dtype, b_key, shape.nhead, shape.nkvhead, shape.head_dim, shape.nslot, block_size, shape.block_table_capacity};
}

static size_t make_cudnn_bucket_signature(llaisysDataType_t dtype, const CudnnProblemShape &shape, int32_t block_size) {
    size_t sig = 1469598103934665603ull;
    auto mix_sig = [&](uint64_t v) {
        sig ^= static_cast<size_t>(v + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2));
    };
    mix_sig(static_cast<uint64_t>(dtype));
    mix_sig(static_cast<uint64_t>(shape.nhead));
    mix_sig(static_cast<uint64_t>(shape.nkvhead));
    mix_sig(static_cast<uint64_t>(shape.head_dim));
    mix_sig(static_cast<uint64_t>(shape.nslot));
    mix_sig(static_cast<uint64_t>(block_size));
    mix_sig(static_cast<uint64_t>(shape.block_table_capacity));
    return sig;
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
__global__ void build_cudnn_meta_kernel(int64_t b_exec,
                                        int64_t b_valid,
                                        int32_t table_size,
                                        int32_t block_table_width,
                                        int32_t nseq,
                                        const int32_t *q_seq_rows,
                                        const int32_t *q_pos,
                                        const int32_t *seq_lens,
                                        const int32_t *block_tables,
                                        int32_t *seq_q_out,
                                        int32_t *seq_kv_out,
                                        int32_t *page_out) {
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x);
    if (row_idx >= b_exec) {
        return;
    }

    int32_t row = -1;
    int32_t visible = 0;
    if (row_idx < b_valid) {
        row = q_seq_rows[row_idx];
        if (row >= 0 && row < nseq) {
            const int32_t seq_len = seq_lens[row];
            const int32_t qpos = q_pos[row_idx];
            const int32_t q_visible = qpos + 1;
            visible = q_visible < 0 ? 0 : (q_visible > seq_len ? seq_len : q_visible);
        }
    }

    seq_q_out[row_idx] = 1;
    seq_kv_out[row_idx] = visible;

    const int64_t dst_base = row_idx * static_cast<int64_t>(table_size);
    for (int32_t c = static_cast<int32_t>(threadIdx.x); c < table_size; c += static_cast<int32_t>(blockDim.x)) {
        int32_t v = -1;
        if (row >= 0 && row < nseq && c < block_table_width) {
            const int64_t src_idx = static_cast<int64_t>(row) * static_cast<int64_t>(block_table_width) + c;
            v = block_tables[src_idx];
        }
        page_out[dst_base + static_cast<int64_t>(c)] = v;
    }
}

static bool ensure_cudnn_plan_ready(CudnnPagedPlan &plan,
                                    cudnnHandle_t handle,
                                    llaisysDataType_t dtype,
                                    const CudnnProblemShape &shape,
                                    int32_t block_size,
                                    int64_t b_plan,
                                    float scale) {
    namespace fe = cudnn_frontend;
    if (plan.graph_ready) {
        return true;
    }

    auto io_dtype = dtype == LLAISYS_DTYPE_BF16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
    constexpr int64_t s_q = 1;
    plan.graph = std::make_shared<fe::graph::Graph>();
    plan.graph->set_io_data_type(io_dtype).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(
        fe::DataType_t::FLOAT);

    auto Q = plan.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Q")
                                     .set_uid(Q_UID)
                                     .set_dim({b_plan, shape.nhead, s_q, shape.head_dim})
                                     .set_stride({shape.nhead * s_q * shape.head_dim, s_q * shape.head_dim, shape.head_dim, 1}));

    auto K = plan.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("container_K")
                                     .set_uid(K_UID)
                                     .set_dim({shape.num_blocks, shape.nkvhead, static_cast<int64_t>(block_size), shape.head_dim})
                                     .set_stride({static_cast<int64_t>(block_size) * shape.nkvhead * shape.head_dim,
                                                  shape.head_dim,
                                                  shape.nkvhead * shape.head_dim,
                                                  1}));

    auto V = plan.graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("container_V")
                                     .set_uid(V_UID)
                                     .set_dim({shape.num_blocks, shape.nkvhead, static_cast<int64_t>(block_size), shape.head_dim})
                                     .set_stride({static_cast<int64_t>(block_size) * shape.nkvhead * shape.head_dim,
                                                  shape.head_dim,
                                                  shape.nkvhead * shape.head_dim,
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
                                                .set_dim({b_plan, 1, shape.table_size, 1})
                                                .set_stride({shape.table_size, shape.table_size, 1, 1})
                                                .set_data_type(fe::DataType_t::INT32));
    auto page_table_v = plan.graph->tensor(fe::graph::Tensor_attributes()
                                                .set_name("page_table_v")
                                                .set_uid(PAGE_TABLE_V_UID)
                                                .set_dim({b_plan, 1, shape.table_size, 1})
                                                .set_stride({shape.table_size, shape.table_size, 1, 1})
                                                .set_data_type(fe::DataType_t::INT32));

    auto sdpa_options =
        fe::graph::SDPA_attributes().set_name("novainfer_paged_sdpa").set_generate_stats(false).set_attn_scale(scale);
    sdpa_options.set_padding_mask(true).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    sdpa_options.set_paged_attention_k_table(page_table_k);
    sdpa_options.set_paged_attention_v_table(page_table_v);
    sdpa_options.set_paged_attention_max_seq_len_kv(static_cast<int>(shape.max_seq_len_kv));

    auto [O, Stats] = plan.graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;
    O->set_output(true).set_uid(O_UID).set_dim({b_plan, shape.nhead, s_q, shape.head_dim}).set_stride(
        {shape.nhead * s_q * shape.head_dim, s_q * shape.head_dim, shape.head_dim, 1});

    auto build_status = plan.graph->build(handle, {fe::HeurMode_t::A});
    if (build_status.is_bad()) {
        static bool warned_build = false;
        if (!warned_build) {
            warned_build = true;
            printf("[warn] self_attention_paged: CUDNN build failed: %s; fallback to NATIVE\n",
                   build_status.get_message().c_str());
        }
        plan.graph_ready = false;
        return false;
    }
    plan.graph_ready = true;
    return true;
}

static bool prebuild_cudnn_buckets(CudnnRuntimeState &state,
                                   llaisysDataType_t dtype,
                                   const CudnnProblemShape &shape,
                                   int32_t block_size,
                                   float scale) {
    const size_t sig = make_cudnn_bucket_signature(dtype, shape, block_size);
    int64_t warmup_target = std::max<int64_t>(shape.b_exec, 64);
    if (const char *raw = std::getenv("LLAISYS_CUDNN_PREBUILD_MAX_B"); raw != nullptr) {
        int64_t parsed = std::atoll(raw);
        if (parsed > 0) {
            int64_t p2 = 1;
            while (p2 < parsed) {
                p2 <<= 1;
            }
            warmup_target = std::max<int64_t>(warmup_target, p2);
        }
    }
    int64_t &built_max = state.prebuilt_max_bucket[sig];
    if (built_max < warmup_target) {
        int64_t bb = built_max > 0 ? (built_max << 1) : 1;
        while (bb <= warmup_target) {
            auto &pp = state.plan_cache[make_cudnn_plan_key(dtype, shape, block_size, bb)];
            if (!pp) {
                pp = std::make_unique<CudnnPagedPlan>();
            }
            if (!ensure_cudnn_plan_ready(*pp, state.handle, dtype, shape, block_size, bb, scale)) {
                return false;
            }
            bb <<= 1;
        }
        built_max = warmup_target;
    }
    return true;
}

static CudnnMetaPtrs ensure_cudnn_meta_buffer(CudnnRuntimeState &state, const CudnnProblemShape &shape) {
    CudnnMetaPtrs ptrs{};
    ptrs.bsz = static_cast<size_t>(shape.b_exec);
    ptrs.page_n = ptrs.bsz * static_cast<size_t>(shape.table_size);
    ptrs.meta_n = ptrs.bsz * 2 + ptrs.page_n;
    if (state.meta_buf.cap < ptrs.meta_n) {
        if (state.meta_buf.d_meta != nullptr) {
            cudaFree(state.meta_buf.d_meta);
            state.meta_buf.d_meta = nullptr;
        }
        LLAISYS_CUDA_CHECK(cudaMalloc(&state.meta_buf.d_meta, ptrs.meta_n * sizeof(int32_t)));
        state.meta_buf.cap = ptrs.meta_n;
    }
    ptrs.d_seq_q = state.meta_buf.d_meta;
    ptrs.d_seq_kv = state.meta_buf.d_meta + static_cast<std::ptrdiff_t>(ptrs.bsz);
    ptrs.d_page = state.meta_buf.d_meta + static_cast<std::ptrdiff_t>(ptrs.bsz * 2);
    return ptrs;
}

static void refresh_cudnn_meta(const CommonAttentionMetadata &prepared,
                               const CudnnProblemShape &shape,
                               int32_t block_table_width,
                               const CudnnMetaPtrs &ptrs,
                               cudaStream_t stream) {
    // Step metadata is dynamic and must be rebuilt every step.

    const int32_t *q_seq_rows_dev = prepared.q_seq_rows;
    const int32_t *q_pos_dev = prepared.q_pos;
    const int32_t *seq_lens_dev = prepared.seq_lens;
    const int32_t *block_tables_dev = prepared.block_tables;
    if (q_seq_rows_dev == nullptr || q_pos_dev == nullptr || seq_lens_dev == nullptr || block_tables_dev == nullptr) {
        return;
    }
    const dim3 grid(static_cast<unsigned int>(shape.b_exec));
    const dim3 block(256);
    build_cudnn_meta_kernel<<<grid, block, 0, stream>>>(shape.b_exec,
                                                         shape.seqlen,
                                                         shape.block_table_capacity,
                                                         block_table_width,
                                                         prepared.nseq,
                                                         q_seq_rows_dev,
                                                         q_pos_dev,
                                                         seq_lens_dev,
                                                         block_tables_dev,
                                                         ptrs.d_seq_q,
                                                         ptrs.d_seq_kv,
                                                         ptrs.d_page);
    LLAISYS_CUDA_CHECK(cudaGetLastError());

}

static bool ensure_cudnn_workspace(CudnnPagedPlan &plan) {
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

bool cudnn_try_paged_attention(tensor_t attn_val,
                               tensor_t q,
                               tensor_t k_cache,
                               tensor_t v_cache,
                               const CommonAttentionMetadata &prepared,
                               int32_t block_table_width,
                               int32_t block_size,
                               float scale) {
#ifdef ENABLE_CUDNN_FRONTEND
    CudnnProblemShape shape{};
    if (!cudnn_validate_inputs(q, k_cache, v_cache, prepared, block_table_width, block_size, shape)) {
        return false;
    }
    CudnnRuntimeState &state = cudnn_runtime_state();
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    if (!cudnn_set_stream(state.handle, stream)) {
        return false;
    }
    if (!prebuild_cudnn_buckets(state, q->dtype(), shape, block_size, scale)) {
        return false;
    }

    auto &plan_ptr = state.plan_cache[make_cudnn_plan_key(q->dtype(), shape, block_size, shape.b_exec)];
    if (!plan_ptr) {
        plan_ptr = std::make_unique<CudnnPagedPlan>();
    }
    auto &plan = *plan_ptr;
    if (!ensure_cudnn_plan_ready(plan, state.handle, q->dtype(), shape, block_size, shape.b_exec, scale)) {
        return false;
    }
    const CudnnMetaPtrs ptrs = ensure_cudnn_meta_buffer(state, shape);
    refresh_cudnn_meta(prepared, shape, block_table_width, ptrs, stream);
    if (!ensure_cudnn_workspace(plan)) {
        return false;
    }

    std::unordered_map<int64_t, void *> variant_pack = {
        {Q_UID, q->data()},
        {K_UID, k_cache->data()},
        {V_UID, v_cache->data()},
        {O_UID, attn_val->data()},
        {SEQ_LEN_Q_UID, ptrs.d_seq_q},
        {SEQ_LEN_KV_UID, ptrs.d_seq_kv},
        {PAGE_TABLE_K_UID, ptrs.d_page},
        {PAGE_TABLE_V_UID, ptrs.d_page},
    };

    auto exec_status = plan.graph->execute(state.handle, variant_pack, plan.workspace);
    if (exec_status.is_bad()) {
        static bool warned_exec = false;
        if (!warned_exec) {
            warned_exec = true;
            printf("[warn] self_attention_paged: CUDNN execute failed: %s; fallback to NATIVE\n",
                   exec_status.get_message().c_str());
        }
        return false;
    }
    return true;
#else
    static bool warned_no_frontend = false;
    if (!warned_no_frontend) {
        warned_no_frontend = true;
        printf("[warn] self_attention_paged: CUDNN backend requested but cudnn_frontend headers are missing (expected third_party/cudnn_frontend/include); fallback to NATIVE\n");
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

void dispatch_attention_with_backend(tensor_t attn_val,
                                     tensor_t q,
                                     tensor_t k_cache,
                                     tensor_t v_cache,
                                     const CommonAttentionMetadata &metadata,
                                     PagedAttentionBackend backend,
                                     int32_t block_table_width,
                                     int32_t block_size,
                                     float scale) {
    CHECK_ARGUMENT(metadata.q_seq_rows != nullptr && metadata.q_pos != nullptr && metadata.block_tables != nullptr &&
                       metadata.seq_lens != nullptr,
                   "dispatch_attention_with_backend: metadata tensors must be non-null");
    const CommonAttentionMetadata &prepared = metadata;
    if (backend == PagedAttentionBackend::FLASHINFER) {
        // Stage-1 FlashInfer migration scaffold: keep behavior stable via native fallback
        // while preserving a dedicated backend switch point for later real integration.
        static bool warned_flashinfer = false;
        if (!warned_flashinfer) {
            warned_flashinfer = true;
            printf("[warn] self_attention_paged: FLASHINFER backend requested, fallback to NATIVE\n");
        }
    } else if (backend == PagedAttentionBackend::CUDNN) {
#ifdef ENABLE_CUDNN_API
        if (cudnn_try_paged_attention(attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale)) {
            return;
        }
#else
        static bool warned_cudnn_disabled = false;
        if (!warned_cudnn_disabled) {
            warned_cudnn_disabled = true;
            printf("[warn] self_attention_paged: CUDNN backend requested but ENABLE_CUDNN_API is off; fallback to NATIVE\n");
        }
#endif
    }
    self_attention_paged_prepared(attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale);
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
    const std::int32_t seqlen = static_cast<std::int32_t>(q->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(q->shape()[1]);
    const std::int32_t head_dim = static_cast<std::int32_t>(q->shape()[2]);
    const std::int32_t nslot = static_cast<std::int32_t>(k_cache->shape()[0]);
    const std::int32_t nkvhead = static_cast<std::int32_t>(k_cache->shape()[1]);
    if (seqlen <= 0 || nhead <= 0 || head_dim <= 0 || nslot <= 0 || nkvhead <= 0) {
        return;
    }
    if (q_seq_rows == nullptr || q_pos == nullptr || block_tables == nullptr || seq_lens == nullptr) {
        return;
    }

    CHECK_ARGUMENT(q_seq_rows->deviceType() == LLAISYS_DEVICE_NVIDIA && q_pos->deviceType() == LLAISYS_DEVICE_NVIDIA &&
                       block_tables->deviceType() == LLAISYS_DEVICE_NVIDIA && seq_lens->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "self_attention_paged: metadata tensors must be CUDA");
    CommonAttentionMetadata prepared{
        reinterpret_cast<const int32_t *>(q_seq_rows->data()),
        reinterpret_cast<const int32_t *>(q_pos->data()),
        reinterpret_cast<const int32_t *>(block_tables->data()),
        reinterpret_cast<const int32_t *>(seq_lens->data()),
        static_cast<int32_t>(seq_lens->shape()[0])};
    self_attention_paged_prepared(attn_val, q, k_cache, v_cache, prepared, block_table_width, block_size, scale);
}

} // namespace llaisys::ops::cuda
