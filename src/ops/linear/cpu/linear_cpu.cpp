#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void linear_(T *out,
             const T *in,
             const T *weight,
             const T *bias,
             size_t M,
             size_t K,
             size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {

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
    
    } // namespace llaisys::ops::cpu
