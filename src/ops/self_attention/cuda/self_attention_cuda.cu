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
    bool meta_initialized{false};
    int64_t meta_b{0};

    void *workspace{nullptr};
    size_t workspace_cap{0};

    const PagedAttentionPrepared *last_prepared{nullptr};
    uint64_t last_host_revision{0};
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

bool cudnn_try_paged_attention(tensor_t attn_val,
                               tensor_t q,
                               tensor_t k_cache,
                               tensor_t v_cache,
                               const PagedAttentionPrepared &prepared,
                               int32_t block_table_width,
                               int32_t block_size,
                               float scale) {
#ifdef ENABLE_CUDNN_FRONTEND
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
    if (prepared.host_q_seq_rows.size() != static_cast<size_t>(seqlen) ||
        prepared.host_q_pos.size() != static_cast<size_t>(seqlen) ||
        prepared.host_seq_lens.empty() ||
        prepared.host_block_tables.size() != static_cast<size_t>(prepared.nseq) * static_cast<size_t>(block_table_width)) {
        return false;
    }

    namespace fe = cudnn_frontend;

    constexpr int64_t Q_UID = 1;
    constexpr int64_t K_UID = 2;
    constexpr int64_t V_UID = 3;
    constexpr int64_t O_UID = 4;
    constexpr int64_t SEQ_LEN_Q_UID = 7;
    constexpr int64_t SEQ_LEN_KV_UID = 8;
    constexpr int64_t PAGE_TABLE_K_UID = 9;
    constexpr int64_t PAGE_TABLE_V_UID = 10;

    auto io_dtype = q->dtype() == LLAISYS_DTYPE_BF16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;

    const int64_t b = seqlen;
    int64_t b_exec = 1;
    while (b_exec < b) {
        b_exec <<= 1;
    }
    const int64_t s_q = 1;
    int32_t width_p2 = 1;
    while (width_p2 < block_table_width) {
        width_p2 <<= 1;
    }
    int32_t block_table_capacity = width_p2;
    if (const char *raw = std::getenv("LLAISYS_CUDNN_BLOCK_TABLE_CAP"); raw != nullptr) {
        int32_t cfg_cap = static_cast<int32_t>(std::atoi(raw));
        if (cfg_cap > 0) {
            int32_t cfg_p2 = 1;
            while (cfg_p2 < cfg_cap) {
                cfg_p2 <<= 1;
            }
            block_table_capacity = std::max<int32_t>(block_table_capacity, cfg_p2);
        }
    }
    const int64_t table_size = block_table_capacity;
    const int64_t num_blocks = nslot / block_size;
    const int64_t max_seq_len_kv = static_cast<int64_t>(block_table_capacity) * static_cast<int64_t>(block_size);

    static thread_local cudnnHandle_t handle = nullptr;
    static thread_local std::unordered_map<CudnnPlanKey, std::unique_ptr<CudnnPagedPlan>, CudnnPlanKeyHash> plan_cache;
    static thread_local std::unordered_map<size_t, int64_t> prebuilt_max_bucket;
    static thread_local CudnnMetaBuffer meta_buf;
    static thread_local CudnnPagedPlan *meta_owner_plan = nullptr;

    auto make_key = [&](int64_t b_key) {
        return CudnnPlanKey{q->dtype(), b_key, nhead, nkvhead, head_dim, nslot, block_size, block_table_capacity};
    };

    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    if (handle == nullptr) {
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
            return false;
        }
    }
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) {
        return false;
    }

    auto ensure_plan_ready = [&](CudnnPagedPlan &plan, int64_t b_plan) -> bool {
        if (plan.graph_ready) {
            return true;
        }
        plan.graph = std::make_shared<fe::graph::Graph>();
        plan.graph->set_io_data_type(io_dtype)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto Q = plan.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("Q")
                                         .set_uid(Q_UID)
                                         .set_dim({b_plan, nhead, s_q, head_dim})
                                         .set_stride({nhead * s_q * head_dim, s_q * head_dim, head_dim, 1}));

        // cuDNN paged SDPA expects K/V head axis to be dim-1 (B, Hk, S, D),
        // but NovaInfer physical cache is [nslot, Hk, D] with
        // nslot = num_blocks * block_size and slot = block * block_size + offset.
        // Use custom strides so [block, head, offset, d] maps to
        // ((block * block_size + offset) * Hk + head) * D + d.
        auto K = plan.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("container_K")
                                         .set_uid(K_UID)
                                         .set_dim({num_blocks, nkvhead, block_size, head_dim})
                                         .set_stride({block_size * nkvhead * head_dim, head_dim, nkvhead * head_dim, 1}));

        auto V = plan.graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("container_V")
                                         .set_uid(V_UID)
                                         .set_dim({num_blocks, nkvhead, block_size, head_dim})
                                         .set_stride({block_size * nkvhead * head_dim, head_dim, nkvhead * head_dim, 1}));

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
                                                    .set_dim({b_plan, 1, table_size, 1})
                                                    .set_stride({table_size, table_size, 1, 1})
                                                    .set_data_type(fe::DataType_t::INT32));
        auto page_table_v = plan.graph->tensor(fe::graph::Tensor_attributes()
                                                    .set_name("page_table_v")
                                                    .set_uid(PAGE_TABLE_V_UID)
                                                    .set_dim({b_plan, 1, table_size, 1})
                                                    .set_stride({table_size, table_size, 1, 1})
                                                    .set_data_type(fe::DataType_t::INT32));

        auto sdpa_options = fe::graph::SDPA_attributes()
                                .set_name("novainfer_paged_sdpa")
                                .set_generate_stats(false)
                                .set_attn_scale(scale);
        sdpa_options.set_padding_mask(true).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
        sdpa_options.set_paged_attention_k_table(page_table_k);
        sdpa_options.set_paged_attention_v_table(page_table_v);
        sdpa_options.set_paged_attention_max_seq_len_kv(static_cast<int>(max_seq_len_kv));

        auto [O, Stats] = plan.graph->sdpa(Q, K, V, sdpa_options);
        (void)Stats;
        O->set_output(true).set_uid(O_UID).set_dim({b_plan, nhead, s_q, head_dim}).set_stride(
            {nhead * s_q * head_dim, s_q * head_dim, head_dim, 1});

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
        plan.last_prepared = nullptr;
        plan.last_host_revision = 0;
        return true;
    };

    size_t sig = 1469598103934665603ull;
    auto mix_sig = [&](uint64_t v) {
        sig ^= static_cast<size_t>(v + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2));
    };
    mix_sig(static_cast<uint64_t>(q->dtype()));
    mix_sig(static_cast<uint64_t>(nhead));
    mix_sig(static_cast<uint64_t>(nkvhead));
    mix_sig(static_cast<uint64_t>(head_dim));
    mix_sig(static_cast<uint64_t>(nslot));
    mix_sig(static_cast<uint64_t>(block_size));
    mix_sig(static_cast<uint64_t>(block_table_capacity));

    // Default to a stable prebuild floor to avoid run-time graph builds when
    // decode micro-batch size jitters across small buckets.
    int64_t warmup_target = std::max<int64_t>(b_exec, 64);
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
    int64_t &built_max = prebuilt_max_bucket[sig];
    if (built_max < warmup_target) {
        int64_t bb = built_max > 0 ? (built_max << 1) : 1;
        while (bb <= warmup_target) {
            auto &pp = plan_cache[make_key(bb)];
            if (!pp) {
                pp = std::make_unique<CudnnPagedPlan>();
            }
            if (!ensure_plan_ready(*pp, bb)) {
                return false;
            }
            bb <<= 1;
        }
        built_max = warmup_target;
    }

    auto &plan_ptr = plan_cache[make_key(b_exec)];
    if (!plan_ptr) {
        plan_ptr = std::make_unique<CudnnPagedPlan>();
    }
    auto &plan = *plan_ptr;
    if (!ensure_plan_ready(plan, b_exec)) {
        return false;
    }

    const size_t bsz = static_cast<size_t>(b_exec);
    const size_t page_n = bsz * static_cast<size_t>(table_size);
    const size_t meta_n = bsz * 2 + page_n;
    if (meta_buf.cap < meta_n) {
        if (meta_buf.d_meta != nullptr) {
            cudaFree(meta_buf.d_meta);
            meta_buf.d_meta = nullptr;
        }
        if (meta_buf.h_meta != nullptr) {
            cudaFreeHost(meta_buf.h_meta);
            meta_buf.h_meta = nullptr;
        }
        LLAISYS_CUDA_CHECK(cudaMalloc(&meta_buf.d_meta, meta_n * sizeof(int32_t)));
        LLAISYS_CUDA_CHECK(cudaHostAlloc(&meta_buf.h_meta, meta_n * sizeof(int32_t), cudaHostAllocDefault));
        meta_buf.cap = meta_n;
        plan.meta_initialized = false;
        meta_owner_plan = nullptr;
    }
    int32_t *d_seq_q = meta_buf.d_meta;
    int32_t *d_seq_kv = meta_buf.d_meta + static_cast<std::ptrdiff_t>(bsz);
    int32_t *d_page = meta_buf.d_meta + static_cast<std::ptrdiff_t>(bsz * 2);
    int32_t *h_seq_q = meta_buf.h_meta;
    int32_t *h_seq_kv = meta_buf.h_meta + static_cast<std::ptrdiff_t>(bsz);
    int32_t *h_page = meta_buf.h_meta + static_cast<std::ptrdiff_t>(bsz * 2);

    const bool meta_changed =
        (plan.last_prepared != &prepared) || (plan.last_host_revision != prepared.host_revision);
    if (!plan.meta_initialized || plan.meta_b != b_exec || meta_owner_plan != &plan) {
        std::fill_n(h_seq_q, static_cast<std::ptrdiff_t>(bsz), 1);
        std::fill_n(h_seq_kv, static_cast<std::ptrdiff_t>(bsz), 0);
        std::fill_n(h_page, static_cast<std::ptrdiff_t>(page_n), -1);
        plan.meta_initialized = true;
        plan.meta_b = b_exec;
        meta_owner_plan = &plan;
        LLAISYS_CUDA_CHECK(cudaMemcpyAsync(meta_buf.d_meta, meta_buf.h_meta, meta_n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    if (meta_changed) {
        int64_t seq_min = b_exec;
        int64_t seq_max = -1;
        int64_t page_min = static_cast<int64_t>(page_n);
        int64_t page_max = -1;

        for (int64_t i = 0; i < b_exec; ++i) {
            int32_t visible = 0;
            int32_t row = -1;
            if (i < b) {
                row = prepared.host_q_seq_rows[static_cast<size_t>(i)];
                if (row >= 0 && row < prepared.nseq) {
                    const int32_t seq_len = prepared.host_seq_lens[static_cast<size_t>(row)];
                    const int32_t qpos = prepared.host_q_pos[static_cast<size_t>(i)];
                    visible = std::max<int32_t>(0, std::min<int32_t>(seq_len, qpos + 1));
                }
            }
            if (h_seq_kv[static_cast<size_t>(i)] != visible) {
                h_seq_kv[static_cast<size_t>(i)] = visible;
                seq_min = std::min(seq_min, i);
                seq_max = std::max(seq_max, i);
            }

            const size_t dst_off = static_cast<size_t>(i) * static_cast<size_t>(table_size);
            if (row < 0 || row >= prepared.nseq) {
                for (int32_t c = 0; c < block_table_capacity; ++c) {
                    const size_t idx = dst_off + static_cast<size_t>(c);
                    if (h_page[idx] != -1) {
                        h_page[idx] = -1;
                        page_min = std::min(page_min, static_cast<int64_t>(idx));
                        page_max = std::max(page_max, static_cast<int64_t>(idx));
                    }
                }
                continue;
            }

            const size_t src_off = static_cast<size_t>(row) * static_cast<size_t>(block_table_width);
            for (int32_t c = 0; c < block_table_capacity; ++c) {
                int32_t next = -1;
                if (c < block_table_width) {
                    next = prepared.host_block_tables[src_off + static_cast<size_t>(c)];
                }
                const size_t idx = dst_off + static_cast<size_t>(c);
                if (h_page[idx] != next) {
                    h_page[idx] = next;
                    page_min = std::min(page_min, static_cast<int64_t>(idx));
                    page_max = std::max(page_max, static_cast<int64_t>(idx));
                }
            }
        }

        if (seq_max >= seq_min) {
            const size_t off = static_cast<size_t>(seq_min);
            const size_t len = static_cast<size_t>(seq_max - seq_min + 1);
            LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
                d_seq_kv + static_cast<std::ptrdiff_t>(off),
                h_seq_kv + static_cast<std::ptrdiff_t>(off),
                len * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                stream));
        }
        if (page_max >= page_min) {
            const size_t off = static_cast<size_t>(page_min);
            const size_t len = static_cast<size_t>(page_max - page_min + 1);
            LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
                d_page + static_cast<std::ptrdiff_t>(off),
                h_page + static_cast<std::ptrdiff_t>(off),
                len * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                stream));
        }

        plan.last_prepared = &prepared;
        plan.last_host_revision = prepared.host_revision;
    }

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

    std::unordered_map<int64_t, void *> variant_pack = {
        {Q_UID, q->data()},
        {K_UID, k_cache->data()},
        {V_UID, v_cache->data()},
        {O_UID, attn_val->data()},
        {SEQ_LEN_Q_UID, d_seq_q},
        {SEQ_LEN_KV_UID, d_seq_kv},
        {PAGE_TABLE_K_UID, d_page},
        {PAGE_TABLE_V_UID, d_page},
    };

    auto exec_status = plan.graph->execute(handle, variant_pack, plan.workspace);
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

PagedAttentionPrepared::~PagedAttentionPrepared() {
    release_device_i32(q_seq_rows);
    release_device_i32(q_pos);
    release_device_i32(block_tables);
    release_device_i32(seq_lens);
    release_device_i32(last_page_lens);
    q_seq_rows_cap = 0;
    q_pos_cap = 0;
    block_tables_cap = 0;
    seq_lens_cap = 0;
    last_page_lens_cap = 0;
    nseq = 0;
}

void prepare_paged_attention(PagedAttentionPrepared &prepared,
                             const std::vector<int32_t> &q_seq_rows,
                             const std::vector<int32_t> &q_pos,
                             const std::vector<int32_t> &block_tables,
                             const std::vector<int32_t> &seq_lens,
                             int32_t block_size,
                             bool upload_device_metadata) {
    prepared.host_q_seq_rows = q_seq_rows;
    prepared.host_q_pos = q_pos;
    prepared.host_block_tables = block_tables;
    prepared.host_seq_lens = seq_lens;

    if (upload_device_metadata) {
        upload_device_i32(prepared.q_seq_rows, prepared.q_seq_rows_cap, q_seq_rows);
        upload_device_i32(prepared.q_pos, prepared.q_pos_cap, q_pos);
        upload_device_i32(prepared.block_tables, prepared.block_tables_cap, block_tables);
        upload_device_i32(prepared.seq_lens, prepared.seq_lens_cap, seq_lens);
    }
    std::vector<int32_t> last_page_lens(seq_lens.size(), 0);
    if (block_size > 0) {
        for (size_t i = 0; i < seq_lens.size(); ++i) {
            const int32_t len = seq_lens[i];
            if (len > 0) {
                last_page_lens[i] = ((len - 1) % block_size) + 1;
            }
        }
    }
    prepared.host_last_page_lens = last_page_lens;
    prepared.host_revision += 1;
    if (upload_device_metadata) {
        upload_device_i32(prepared.last_page_lens, prepared.last_page_lens_cap, last_page_lens);
    }
    prepared.nseq = static_cast<int32_t>(seq_lens.size());
}

void scatter_kv_cache_by_slots(tensor_t k_cache,
                               tensor_t v_cache,
                               tensor_t k_src,
                               tensor_t v_src,
                               const std::vector<int32_t> &slot_idxs) {
    if (slot_idxs.empty()) {
        return;
    }
    CHECK_ARGUMENT(k_cache->deviceType() == LLAISYS_DEVICE_NVIDIA, "scatter_kv_cache_by_slots: k_cache must be CUDA");
    CHECK_ARGUMENT(v_cache->deviceType() == LLAISYS_DEVICE_NVIDIA, "scatter_kv_cache_by_slots: v_cache must be CUDA");
    CHECK_ARGUMENT(k_src->deviceType() == LLAISYS_DEVICE_NVIDIA, "scatter_kv_cache_by_slots: k_src must be CUDA");
    CHECK_ARGUMENT(v_src->deviceType() == LLAISYS_DEVICE_NVIDIA, "scatter_kv_cache_by_slots: v_src must be CUDA");
    CHECK_ARGUMENT(k_cache->dtype() == k_src->dtype(), "scatter_kv_cache_by_slots: k dtype mismatch");
    CHECK_ARGUMENT(v_cache->dtype() == v_src->dtype(), "scatter_kv_cache_by_slots: v dtype mismatch");
    CHECK_ARGUMENT(k_src->shape()[0] == slot_idxs.size(), "scatter_kv_cache_by_slots: k slot size mismatch");
    CHECK_ARGUMENT(v_src->shape()[0] == slot_idxs.size(), "scatter_kv_cache_by_slots: v slot size mismatch");
    CHECK_ARGUMENT(k_cache->shape()[1] == k_src->shape()[1] && k_cache->shape()[2] == k_src->shape()[2],
                   "scatter_kv_cache_by_slots: k shape mismatch");
    CHECK_ARGUMENT(v_cache->shape()[1] == v_src->shape()[1] && v_cache->shape()[2] == v_src->shape()[2],
                   "scatter_kv_cache_by_slots: v shape mismatch");

    static thread_local int32_t *d_slot_idxs = nullptr;
    static thread_local size_t d_slot_cap = 0;
    ensure_device_i32_capacity(d_slot_idxs, d_slot_cap, slot_idxs.size());
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
        d_slot_idxs, slot_idxs.data(), slot_idxs.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    switch (k_src->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_scatter_cache_by_slots<float>(k_cache, k_src, d_slot_idxs);
        launch_scatter_cache_by_slots<float>(v_cache, v_src, d_slot_idxs);
        break;
    case LLAISYS_DTYPE_F16:
        launch_scatter_cache_by_slots<llaisys::fp16_t>(k_cache, k_src, d_slot_idxs);
        launch_scatter_cache_by_slots<llaisys::fp16_t>(v_cache, v_src, d_slot_idxs);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_scatter_cache_by_slots<llaisys::bf16_t>(k_cache, k_src, d_slot_idxs);
        launch_scatter_cache_by_slots<llaisys::bf16_t>(v_cache, v_src, d_slot_idxs);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(k_src->dtype());
    }
}

void scatter_kv_cache_by_slots_device_indices(tensor_t k_cache,
                                              tensor_t v_cache,
                                              tensor_t k_src,
                                              tensor_t v_src,
                                              tensor_t slot_idxs_i32) {
    CHECK_ARGUMENT(k_cache->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "scatter_kv_cache_by_slots_device_indices: k_cache must be CUDA");
    CHECK_ARGUMENT(v_cache->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "scatter_kv_cache_by_slots_device_indices: v_cache must be CUDA");
    CHECK_ARGUMENT(k_src->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "scatter_kv_cache_by_slots_device_indices: k_src must be CUDA");
    CHECK_ARGUMENT(v_src->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "scatter_kv_cache_by_slots_device_indices: v_src must be CUDA");
    CHECK_ARGUMENT(k_cache->dtype() == k_src->dtype(),
                   "scatter_kv_cache_by_slots_device_indices: k dtype mismatch");
    CHECK_ARGUMENT(v_cache->dtype() == v_src->dtype(),
                   "scatter_kv_cache_by_slots_device_indices: v dtype mismatch");
    CHECK_ARGUMENT(k_cache->shape()[1] == k_src->shape()[1] && k_cache->shape()[2] == k_src->shape()[2],
                   "scatter_kv_cache_by_slots_device_indices: k shape mismatch");
    CHECK_ARGUMENT(v_cache->shape()[1] == v_src->shape()[1] && v_cache->shape()[2] == v_src->shape()[2],
                   "scatter_kv_cache_by_slots_device_indices: v shape mismatch");
    CHECK_ARGUMENT(slot_idxs_i32 != nullptr, "scatter_kv_cache_by_slots_device_indices: slot idx tensor is null");
    CHECK_ARGUMENT(slot_idxs_i32->dtype() == LLAISYS_DTYPE_I32,
                   "scatter_kv_cache_by_slots_device_indices: slot idx dtype must be I32");
    CHECK_ARGUMENT(slot_idxs_i32->deviceType() == LLAISYS_DEVICE_NVIDIA,
                   "scatter_kv_cache_by_slots_device_indices: slot idx must be CUDA");
    CHECK_ARGUMENT(slot_idxs_i32->shape().size() == 1,
                   "scatter_kv_cache_by_slots_device_indices: slot idx must be 1D");
    CHECK_ARGUMENT(slot_idxs_i32->shape()[0] == k_src->shape()[0],
                   "scatter_kv_cache_by_slots_device_indices: slot idx length mismatch");

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

void self_attention_paged_prepared_with_backend(tensor_t attn_val,
                                                tensor_t q,
                                                tensor_t k_cache,
                                                tensor_t v_cache,
                                                const PagedAttentionPrepared &prepared,
                                                PagedAttentionBackend backend,
                                                int32_t block_table_width,
                                                int32_t block_size,
                                                float scale) {
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
    prepare_paged_attention(prepared, q_seq_rows, q_pos, block_tables, seq_lens, block_size);
    self_attention_paged_prepared_with_backend(
        attn_val, q, k_cache, v_cache, prepared, PagedAttentionBackend::NATIVE, block_table_width, block_size, scale);
}

} // namespace llaisys::ops::cuda
