#include "sampler_cuda.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_dtype.cuh"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

namespace llaisys::device::nvidia {
void cuda_check(cudaError_t rc, const char *what, const char *file, int line);
} // namespace llaisys::device::nvidia

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace llaisys::ops::cuda {

namespace {

constexpr std::uint64_t kDefaultSeed = 0x4c4c41495359534full;

__device__ __forceinline__ bool is_greedy_row(float temperature, float top_p, std::int32_t top_k, std::int32_t has_seed) {
    if (!isfinite(temperature) || temperature <= 0.0f) {
        return true;
    }
    if (top_k == 1) {
        return true;
    }
    return top_k <= 0 && top_p >= 1.0f && fabsf(temperature - 1.0f) <= 1e-6f && has_seed == 0;
}

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform01(std::uint64_t seed) {
    const std::uint64_t bits = splitmix64(seed);
    return static_cast<float>((bits >> 11) * (1.0 / 9007199254740992.0));
}

template <typename T, int Cap>
__global__ void sample_rows_kernel(std::int64_t *out_ids,
                                   const T *logits,
                                   const float *temperatures,
                                   const float *top_ps,
                                   const std::int32_t *top_ks,
                                   const std::int64_t *seeds,
                                   const std::int32_t *has_seeds,
                                   std::int32_t ncol,
                                   std::int32_t candidate_cap) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    if (threadIdx.x != 0) {
        return;
    }
    const T *row_vals = logits + static_cast<std::size_t>(row) * static_cast<std::size_t>(ncol);
    const float temperature = temperatures[row];
    const float top_p = top_ps[row];
    const std::int32_t top_k = top_ks[row];
    const std::int32_t has_seed = has_seeds[row];
    const std::int64_t seed = seeds[row];

    if (is_greedy_row(temperature, top_p, top_k, has_seed)) {
        std::int64_t best_idx = 0;
        float best_val = llaisys::device::nvidia::dtype::to_float<T>(row_vals[0]);
        for (std::int32_t i = 1; i < ncol; ++i) {
            const float v = llaisys::device::nvidia::dtype::to_float<T>(row_vals[i]);
            if (v > best_val) {
                best_val = v;
                best_idx = static_cast<std::int64_t>(i);
            }
        }
        out_ids[row] = best_idx;
        return;
    }

    const int max_candidates = min(candidate_cap, Cap);
    int eff_k = (top_k > 0) ? min(top_k, max_candidates) : max_candidates;
    eff_k = min(eff_k, static_cast<int>(ncol));
    if (eff_k <= 1) {
        std::int64_t best_idx = 0;
        float best_val = llaisys::device::nvidia::dtype::to_float<T>(row_vals[0]);
        for (std::int32_t i = 1; i < ncol; ++i) {
            const float v = llaisys::device::nvidia::dtype::to_float<T>(row_vals[i]);
            if (v > best_val) {
                best_val = v;
                best_idx = static_cast<std::int64_t>(i);
            }
        }
        out_ids[row] = best_idx;
        return;
    }

    float cand_logits[Cap];
    std::int64_t cand_ids[Cap];
    int count = 0;
    for (std::int32_t i = 0; i < ncol; ++i) {
        const float v = llaisys::device::nvidia::dtype::to_float<T>(row_vals[i]);
        if (count < eff_k) {
            int pos = count++;
            while (pos > 0 && v > cand_logits[pos - 1]) {
                cand_logits[pos] = cand_logits[pos - 1];
                cand_ids[pos] = cand_ids[pos - 1];
                --pos;
            }
            cand_logits[pos] = v;
            cand_ids[pos] = static_cast<std::int64_t>(i);
        } else if (v > cand_logits[eff_k - 1]) {
            int pos = eff_k - 1;
            while (pos > 0 && v > cand_logits[pos - 1]) {
                cand_logits[pos] = cand_logits[pos - 1];
                cand_ids[pos] = cand_ids[pos - 1];
                --pos;
            }
            cand_logits[pos] = v;
            cand_ids[pos] = static_cast<std::int64_t>(i);
        }
    }

    const float eff_temp = (isfinite(temperature) && temperature > 1e-6f) ? temperature : 1.0f;
    float max_scaled = cand_logits[0] / eff_temp;
    for (int i = 1; i < count; ++i) {
        max_scaled = fmaxf(max_scaled, cand_logits[i] / eff_temp);
    }

    float probs[Cap];
    float prob_sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float scaled = cand_logits[i] / eff_temp;
        const float p = expf(scaled - max_scaled);
        probs[i] = p;
        prob_sum += p;
    }
    if (!(prob_sum > 0.0f) || !isfinite(prob_sum)) {
        out_ids[row] = cand_ids[0];
        return;
    }

    int kept = count;
    float cumulative = 0.0f;
    if (top_p > 0.0f && top_p < 1.0f) {
        for (int i = 0; i < count; ++i) {
            cumulative += probs[i] / prob_sum;
            if (cumulative >= top_p) {
                kept = i + 1;
                break;
            }
        }
    }

    float kept_sum = 0.0f;
    for (int i = 0; i < kept; ++i) {
        kept_sum += probs[i];
    }
    if (!(kept_sum > 0.0f) || !isfinite(kept_sum)) {
        out_ids[row] = cand_ids[0];
        return;
    }

    const std::uint64_t base_seed =
        (has_seed != 0) ? static_cast<std::uint64_t>(seed) : (kDefaultSeed ^ (static_cast<std::uint64_t>(row) * 0x9e3779b97f4a7c15ull));
    const float draw = uniform01(base_seed ^ (static_cast<std::uint64_t>(row) * 0xbf58476d1ce4e5b9ull));
    const float threshold = draw * kept_sum;
    float prefix = 0.0f;
    for (int i = 0; i < kept; ++i) {
        prefix += probs[i];
        if (threshold <= prefix || i == kept - 1) {
            out_ids[row] = cand_ids[i];
            return;
        }
    }
    out_ids[row] = cand_ids[kept - 1];
}

template <typename T>
void launch_sample_rows(std::byte *sampled_ids,
                        const std::byte *logits,
                        const std::byte *temperatures,
                        const std::byte *top_ps,
                        const std::byte *top_ks,
                        const std::byte *seeds,
                        const std::byte *has_seeds,
                        std::int32_t nrow,
                        std::int32_t ncol,
                        std::int32_t candidate_cap) {
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    sample_rows_kernel<T, 256><<<nrow, 1, 0, stream>>>(reinterpret_cast<std::int64_t *>(sampled_ids),
                                                        reinterpret_cast<const T *>(logits),
                                                        reinterpret_cast<const float *>(temperatures),
                                                        reinterpret_cast<const float *>(top_ps),
                                                        reinterpret_cast<const std::int32_t *>(top_ks),
                                                        reinterpret_cast<const std::int64_t *>(seeds),
                                                        reinterpret_cast<const std::int32_t *>(has_seeds),
                                                        ncol,
                                                        candidate_cap);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void sample_rows(std::byte *sampled_ids,
                 const std::byte *logits,
                 llaisysDataType_t logits_type,
                 const std::byte *temperatures,
                 const std::byte *top_ps,
                 const std::byte *top_ks,
                 const std::byte *seeds,
                 const std::byte *has_seeds,
                 size_t nrow,
                 size_t ncol,
                 int candidate_cap) {
    const std::int32_t row = static_cast<std::int32_t>(nrow);
    const std::int32_t col = static_cast<std::int32_t>(ncol);
    if (row <= 0 || col <= 0) {
        return;
    }
    switch (logits_type) {
    case LLAISYS_DTYPE_F32:
        return launch_sample_rows<float>(sampled_ids, logits, temperatures, top_ps, top_ks, seeds, has_seeds, row, col, candidate_cap);
    case LLAISYS_DTYPE_F16:
        return launch_sample_rows<llaisys::fp16_t>(sampled_ids, logits, temperatures, top_ps, top_ks, seeds, has_seeds, row, col, candidate_cap);
    case LLAISYS_DTYPE_BF16:
        return launch_sample_rows<llaisys::bf16_t>(sampled_ids, logits, temperatures, top_ps, top_ks, seeds, has_seeds, row, col, candidate_cap);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits_type);
    }
}

} // namespace llaisys::ops::cuda
