#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void swiglu_(
    T* out,
    const T* gate,
    const T* up,
    size_t numel)
{
    for (size_t i = 0; i < numel; ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);

        // 数值稳定版 sigmoid(x)
        float sigmoid;
        if (g >= 0.f) {
            float z = std::exp(-g);
            sigmoid = 1.f / (1.f + z);
        } else {
            float z = std::exp(g);
            sigmoid = z / (1.f + z);
        }

        float silu = g * sigmoid;     // x * sigmoid(x)
        float result = u * silu;

        out[i] = llaisys::utils::cast<T>(result);
    }
}

} // namespace llaisys::ops::cpu


namespace llaisys::ops::cpu {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);

    ASSERT(out->ndim() == 2, "swiglu: out must be 2D");
    ASSERT(gate->ndim() == 2 && up->ndim() == 2, "swiglu: inputs must be 2D");
    ASSERT(out->shape() == gate->shape() &&
            gate->shape() == up->shape(),
            "swiglu: shape mismatch");
    ASSERT(out->dtype() == gate->dtype() &&
            gate->dtype() == up->dtype(),
            "swiglu: dtype mismatch");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
            "swiglu: tensors must be contiguous");

    size_t numel = out->numel();

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(
                        reinterpret_cast<float *>(out->data()),
                        reinterpret_cast<const float *>(gate->data()),
                        reinterpret_cast<const float *>(up->data()),
                        numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(
                        reinterpret_cast<llaisys::bf16_t *>(out->data()),
                        reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
                        reinterpret_cast<const llaisys::bf16_t *>(up->data()),
                        numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(
                        reinterpret_cast<llaisys::fp16_t *>(out->data()),
                        reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
                        reinterpret_cast<const llaisys::fp16_t *>(up->data()),
                        numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::cpu