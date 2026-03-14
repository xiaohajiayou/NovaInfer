#include "sampler_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {

constexpr std::uint64_t kDefaultSeed = 0x4c4c41495359534full;

inline bool is_greedy_row(float temperature, float top_p, std::int32_t top_k, std::int32_t has_seed) {
    if (!std::isfinite(temperature) || temperature <= 0.0f) {
        return true;
    }
    if (top_k == 1) {
        return true;
    }
    return top_k <= 0 && top_p >= 1.0f && std::fabs(temperature - 1.0f) <= 1e-6f && has_seed == 0;
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

inline float uniform01(std::uint64_t seed) {
    const std::uint64_t bits = splitmix64(seed);
    return static_cast<float>((bits >> 11) * (1.0 / 9007199254740992.0));
}

template <typename T, int Cap>
std::int64_t sample_row(const T *row,
                        size_t row_idx,
                        size_t ncol,
                        float temperature,
                        float top_p,
                        std::int32_t top_k,
                        std::int64_t seed,
                        std::int32_t has_seed,
                        int candidate_cap) {
    if (is_greedy_row(temperature, top_p, top_k, has_seed)) {
        std::int64_t best_idx = 0;
        float best_val = llaisys::utils::cast<float>(row[0]);
        for (size_t i = 1; i < ncol; ++i) {
            const float v = llaisys::utils::cast<float>(row[i]);
            if (v > best_val) {
                best_val = v;
                best_idx = static_cast<std::int64_t>(i);
            }
        }
        return best_idx;
    }

    const int max_candidates = std::min<int>(candidate_cap, Cap);
    int eff_k = (top_k > 0) ? std::min<int>(top_k, max_candidates) : max_candidates;
    eff_k = std::min<int>(eff_k, static_cast<int>(ncol));
    if (eff_k <= 1) {
        std::int64_t best_idx = 0;
        float best_val = llaisys::utils::cast<float>(row[0]);
        for (size_t i = 1; i < ncol; ++i) {
            const float v = llaisys::utils::cast<float>(row[i]);
            if (v > best_val) {
                best_val = v;
                best_idx = static_cast<std::int64_t>(i);
            }
        }
        return best_idx;
    }

    std::array<float, Cap> cand_logits{};
    std::array<std::int64_t, Cap> cand_ids{};
    int count = 0;
    for (size_t i = 0; i < ncol; ++i) {
        const float v = llaisys::utils::cast<float>(row[i]);
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

    const float eff_temp = (std::isfinite(temperature) && temperature > 1e-6f) ? temperature : 1.0f;
    float max_scaled = cand_logits[0] / eff_temp;
    for (int i = 1; i < count; ++i) {
        max_scaled = std::max(max_scaled, cand_logits[i] / eff_temp);
    }

    std::array<float, Cap> probs{};
    float prob_sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float scaled = cand_logits[i] / eff_temp;
        const float p = std::exp(scaled - max_scaled);
        probs[i] = p;
        prob_sum += p;
    }
    if (!(prob_sum > 0.0f) || !std::isfinite(prob_sum)) {
        return cand_ids[0];
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
    if (!(kept_sum > 0.0f) || !std::isfinite(kept_sum)) {
        return cand_ids[0];
    }

    const std::uint64_t base_seed =
        (has_seed != 0) ? static_cast<std::uint64_t>(seed) : (kDefaultSeed ^ (static_cast<std::uint64_t>(row_idx) * 0x9e3779b97f4a7c15ull));
    const float draw = uniform01(base_seed ^ (static_cast<std::uint64_t>(row_idx) * 0xbf58476d1ce4e5b9ull));
    float threshold = draw * kept_sum;
    float prefix = 0.0f;
    for (int i = 0; i < kept; ++i) {
        prefix += probs[i];
        if (threshold <= prefix || i == kept - 1) {
            return cand_ids[i];
        }
    }
    return cand_ids[kept - 1];
}

template <typename T>
void sample_rows_t(std::byte *sampled_ids,
                   const std::byte *logits,
                   const std::byte *temperatures,
                   const std::byte *top_ps,
                   const std::byte *top_ks,
                   const std::byte *seeds,
                   const std::byte *has_seeds,
                   size_t nrow,
                   size_t ncol,
                   int candidate_cap) {
    auto *out = reinterpret_cast<std::int64_t *>(sampled_ids);
    const auto *row_ptr = reinterpret_cast<const T *>(logits);
    const auto *temps = reinterpret_cast<const float *>(temperatures);
    const auto *ps = reinterpret_cast<const float *>(top_ps);
    const auto *ks = reinterpret_cast<const std::int32_t *>(top_ks);
    const auto *seed_ptr = reinterpret_cast<const std::int64_t *>(seeds);
    const auto *seed_mask = reinterpret_cast<const std::int32_t *>(has_seeds);
    for (size_t r = 0; r < nrow; ++r) {
        out[r] = sample_row<T, 256>(row_ptr + r * ncol,
                                    r,
                                    ncol,
                                    temps[r],
                                    ps[r],
                                    ks[r],
                                    seed_ptr[r],
                                    seed_mask[r],
                                    candidate_cap);
    }
}

} // namespace

namespace llaisys::ops::cpu {

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
    switch (logits_type) {
    case LLAISYS_DTYPE_F32:
        return sample_rows_t<float>(sampled_ids, logits, temperatures, top_ps, top_ks, seeds, has_seeds, nrow, ncol, candidate_cap);
    case LLAISYS_DTYPE_F16:
        return sample_rows_t<llaisys::fp16_t>(sampled_ids, logits, temperatures, top_ps, top_ks, seeds, has_seeds, nrow, ncol, candidate_cap);
    case LLAISYS_DTYPE_BF16:
        return sample_rows_t<llaisys::bf16_t>(sampled_ids, logits, temperatures, top_ps, top_ks, seeds, has_seeds, nrow, ncol, candidate_cap);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits_type);
    }
}

} // namespace llaisys::ops::cpu
