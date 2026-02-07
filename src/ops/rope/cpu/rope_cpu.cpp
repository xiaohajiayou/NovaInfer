#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <vector>
#include <cmath>

template <typename T>
void rope_(T* out,
           const T* in,
           const int64_t* pos_ids,
           float theta,
           size_t seqlen,
           size_t nhead,
           size_t dim) {
    
    const size_t half = dim / 2;

    // ====== 1️⃣ 预计算 inv_freq（double 精度，对齐 PyTorch）======
    std::vector<double> inv_freq(half);
    for (size_t j = 0; j < half; ++j) {
        double exponent = static_cast<double>(j) / static_cast<double>(half);
        inv_freq[j] = std::pow(static_cast<double>(theta), -exponent);
    }

    // ====== 2️⃣ 主循环 ======
    for (size_t i = 0; i < seqlen; ++i) {
        double pos = static_cast<double>(pos_ids[i]);

        for (size_t h = 0; h < nhead; ++h) {
            const T* in_row  = in  + (i * nhead + h) * dim;
            T*       out_row = out + (i * nhead + h) * dim;

            for (size_t j = 0; j < half; ++j) {
                float a = llaisys::utils::cast<float>(in_row[j]);
                float b = llaisys::utils::cast<float>(in_row[j + half]);

                // double 精度角度
                double angle = pos * inv_freq[j];
                double c = std::cos(angle);
                double s = std::sin(angle);

                float a_out = static_cast<float>(a * c - b * s);
                float b_out = static_cast<float>(b * c + a * s);

                out_row[j]        = llaisys::utils::cast<T>(a_out);
                out_row[j + half] = llaisys::utils::cast<T>(b_out);
            }
        }
    }
}


namespace llaisys::ops::cpu {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    const size_t seqlen = in->shape()[0];
    const size_t nhead  = in->shape()[1];
    const size_t dim    = in->shape()[2];

    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float*>(out->data()),
            reinterpret_cast<const float*>(in->data()),
            pos_ptr, theta, seqlen, nhead, dim);

    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t*>(out->data()),
            reinterpret_cast<const llaisys::fp16_t*>(in->data()),
            pos_ptr, theta, seqlen, nhead, dim);

    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t*>(out->data()),
            reinterpret_cast<const llaisys::bf16_t*>(in->data()),
            pos_ptr, theta, seqlen, nhead, dim);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace
    