#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <cstdint>

template <typename T>
void linear_(T *out,
             const T *in,
             const T *weight,
             const T *bias,
             size_t M,
             size_t K,
             size_t N) {
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t m_i = 0; m_i < static_cast<int64_t>(M); ++m_i) {
        for (int64_t n_i = 0; n_i < static_cast<int64_t>(N); ++n_i) {
            const size_t m = static_cast<size_t>(m_i);
            const size_t n = static_cast<size_t>(n_i);

            float acc = 0.0f;   // ✅ 用 float 累加

            const T *w_row = weight + n * K;
            const T *x_row = in + m * K;

            if (bias) {
                acc = llaisys::utils::cast<float>(bias[n]);
            }

            for (size_t k = 0; k < K; ++k) {
                acc += llaisys::utils::cast<float>(x_row[k]) *
                       llaisys::utils::cast<float>(w_row[k]);
            }

            out[m * N + n] = llaisys::utils::cast<T>(acc);  // ✅ 最后再转回 T
        }
    }
}


// template <typename T>
// void linear_(T *out,
//                     const T *in,
//                     const T *weight,
//                     const T *bias,
//                     size_t M,
//                     size_t K,
//                     size_t N) {

//     // 🔹 Block 大小（可调优）
//     constexpr size_t BM = 64;   // 行块
//     constexpr size_t BN = 64;   // 列块
//     constexpr size_t BK = 64;   // K 维块

//     for (size_t m0 = 0; m0 < M; m0 += BM) {
//         for (size_t n0 = 0; n0 < N; n0 += BN) {
//             for (size_t k0 = 0; k0 < K; k0 += BK) {

//                 size_t m_max = std::min(m0 + BM, M);
//                 size_t n_max = std::min(n0 + BN, N);
//                 size_t k_max = std::min(k0 + BK, K);

//                 for (size_t m = m0; m < m_max; ++m) {
//                     const T* x_row = in + m * K;
//                     T* y_row = out + m * N;

//                     for (size_t n = n0; n < n_max; ++n) {

//                         float acc;

//                         // 只有第一次 k-block 时才加 bias
//                         if (k0 == 0) {
//                             acc = bias ? llaisys::utils::cast<float>(bias[n]) : 0.0f;
//                         } else {
//                             acc = llaisys::utils::cast<float>(y_row[n]);
//                         }

//                         const T* w_row = weight + n * K;

//                         // K-block 累加
//                         for (size_t k = k0; k < k_max; ++k) {
//                             acc += llaisys::utils::cast<float>(x_row[k]) *
//                                    llaisys::utils::cast<float>(w_row[k]);
//                         }

//                         y_row[n] = llaisys::utils::cast<T>(acc);
//                     }
//                 }
//             }
//         }
//     }
// }


namespace llaisys::ops::cpu {

    void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
        // 1️⃣ 取 shape
        const size_t M = in->shape()[0];
        const size_t K = in->shape()[1];
        const size_t N = weight->shape()[0];
    
        // 2️⃣ 基本 shape 校验（测试里一般会要求）
        ASSERT(weight->shape()[1] == K, "Linear: weight shape mismatch.");
        ASSERT(out->shape()[0] == M && out->shape()[1] == N,
               "Linear: output shape mismatch.");
    
        const bool has_bias = (bias != nullptr);
    
        // 3️⃣ 数据指针
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return linear_(
                reinterpret_cast<float *>(out->data()),
                reinterpret_cast<const float *>(in->data()),
                reinterpret_cast<const float *>(weight->data()),
                has_bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                M, K, N);
    
        case LLAISYS_DTYPE_BF16:
            return linear_(
                reinterpret_cast<llaisys::bf16_t *>(out->data()),
                reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                has_bias ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr,
                M, K, N);
    
        case LLAISYS_DTYPE_F16:
            return linear_(
                reinterpret_cast<llaisys::fp16_t *>(out->data()),
                reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                has_bias ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr,
                M, K, N);
    
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    void linear_indexed(tensor_t out, tensor_t in, tensor_t row_indices, tensor_t weight, tensor_t bias) {
        const size_t B = row_indices->shape()[0];
        const size_t M = in->shape()[0];
        const size_t K = in->shape()[1];
        const size_t N = weight->shape()[0];
        ASSERT(weight->shape()[1] == K, "LinearIndexed: weight shape mismatch.");
        ASSERT(out->shape()[0] == B && out->shape()[1] == N, "LinearIndexed: output shape mismatch.");
        const auto *rows = reinterpret_cast<const int32_t *>(row_indices->data());

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32: {
            auto *out_ptr = reinterpret_cast<float *>(out->data());
            const auto *in_ptr = reinterpret_cast<const float *>(in->data());
            const auto *w_ptr = reinterpret_cast<const float *>(weight->data());
            const auto *b_ptr = bias ? reinterpret_cast<const float *>(bias->data()) : nullptr;
            #if defined(_OPENMP)
            #pragma omp parallel for collapse(2) schedule(static)
            #endif
            for (int64_t bi = 0; bi < static_cast<int64_t>(B); ++bi) {
                for (int64_t ni = 0; ni < static_cast<int64_t>(N); ++ni) {
                    const size_t b = static_cast<size_t>(bi);
                    const size_t n = static_cast<size_t>(ni);
                    const int32_t r = rows[b];
                    ASSERT(r >= 0 && static_cast<size_t>(r) < M, "LinearIndexed: row index out of range.");
                    const float *x_row = in_ptr + static_cast<size_t>(r) * K;
                    const float *w_row = w_ptr + n * K;
                    float acc = b_ptr ? b_ptr[n] : 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        acc += x_row[k] * w_row[k];
                    }
                    out_ptr[b * N + n] = acc;
                }
            }
            return;
        }
        case LLAISYS_DTYPE_BF16: {
            auto *out_ptr = reinterpret_cast<llaisys::bf16_t *>(out->data());
            const auto *in_ptr = reinterpret_cast<const llaisys::bf16_t *>(in->data());
            const auto *w_ptr = reinterpret_cast<const llaisys::bf16_t *>(weight->data());
            const auto *b_ptr = bias ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr;
            #if defined(_OPENMP)
            #pragma omp parallel for collapse(2) schedule(static)
            #endif
            for (int64_t bi = 0; bi < static_cast<int64_t>(B); ++bi) {
                for (int64_t ni = 0; ni < static_cast<int64_t>(N); ++ni) {
                    const size_t b = static_cast<size_t>(bi);
                    const size_t n = static_cast<size_t>(ni);
                    const int32_t r = rows[b];
                    ASSERT(r >= 0 && static_cast<size_t>(r) < M, "LinearIndexed: row index out of range.");
                    const auto *x_row = in_ptr + static_cast<size_t>(r) * K;
                    const auto *w_row = w_ptr + n * K;
                    float acc = b_ptr ? llaisys::utils::cast<float>(b_ptr[n]) : 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        acc += llaisys::utils::cast<float>(x_row[k]) * llaisys::utils::cast<float>(w_row[k]);
                    }
                    out_ptr[b * N + n] = llaisys::utils::cast<llaisys::bf16_t>(acc);
                }
            }
            return;
        }
        case LLAISYS_DTYPE_F16: {
            auto *out_ptr = reinterpret_cast<llaisys::fp16_t *>(out->data());
            const auto *in_ptr = reinterpret_cast<const llaisys::fp16_t *>(in->data());
            const auto *w_ptr = reinterpret_cast<const llaisys::fp16_t *>(weight->data());
            const auto *b_ptr = bias ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr;
            #if defined(_OPENMP)
            #pragma omp parallel for collapse(2) schedule(static)
            #endif
            for (int64_t bi = 0; bi < static_cast<int64_t>(B); ++bi) {
                for (int64_t ni = 0; ni < static_cast<int64_t>(N); ++ni) {
                    const size_t b = static_cast<size_t>(bi);
                    const size_t n = static_cast<size_t>(ni);
                    const int32_t r = rows[b];
                    ASSERT(r >= 0 && static_cast<size_t>(r) < M, "LinearIndexed: row index out of range.");
                    const auto *x_row = in_ptr + static_cast<size_t>(r) * K;
                    const auto *w_row = w_ptr + n * K;
                    float acc = b_ptr ? llaisys::utils::cast<float>(b_ptr[n]) : 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        acc += llaisys::utils::cast<float>(x_row[k]) * llaisys::utils::cast<float>(w_row[k]);
                    }
                    out_ptr[b * N + n] = llaisys::utils::cast<llaisys::fp16_t>(acc);
                }
            }
            return;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }
    
    } // namespace llaisys::ops::cpu
