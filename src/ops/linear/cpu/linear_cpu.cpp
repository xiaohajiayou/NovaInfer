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
