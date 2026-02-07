#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <cmath>

template <typename T>
void rms_norm_(T *out,
               const T *in,
               const T *weight,
               float eps,
               size_t rows,
               size_t dim) {

    for (size_t i = 0; i < rows; ++i) {
        const T* x_row = in + i * dim;
        T* y_row = out + i * dim;

        // ---- Step 1: 计算 mean square ----
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float v = llaisys::utils::cast<float>(x_row[j]);
            sum_sq += v * v;
        }

        float mean_sq = sum_sq / static_cast<float>(dim);
        float scale = 1.0f / std::sqrt(mean_sq + eps);

        // ---- Step 2: 归一化并乘权重 ----
        for (size_t j = 0; j < dim; ++j) {
            float v = llaisys::utils::cast<float>(x_row[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            float out_val = v * scale * w;
            y_row[j] = llaisys::utils::cast<T>(out_val);
        }
    }
}

namespace llaisys::ops::cpu {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    const size_t rows = in->shape()[0];
    const size_t dim  = in->shape()[1];

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float*>(out->data()),
            reinterpret_cast<const float*>(in->data()),
            reinterpret_cast<const float*>(weight->data()),
            eps, rows, dim);

    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t*>(out->data()),
            reinterpret_cast<const llaisys::bf16_t*>(in->data()),
            reinterpret_cast<const llaisys::bf16_t*>(weight->data()),
            eps, rows, dim);

    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t*>(out->data()),
            reinterpret_cast<const llaisys::fp16_t*>(in->data()),
            reinterpret_cast<const llaisys::fp16_t*>(weight->data()),
            eps, rows, dim);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
    
} // namespace llaisys::ops::cpu
    