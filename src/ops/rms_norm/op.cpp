#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/rms_norm_cuda.hpp"
#endif

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    ASSERT(out->ndim() == 2, "RMSNorm: out must be 2-D.");
    ASSERT(in->ndim() == 2, "RMSNorm: in must be 2-D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1-D.");

    ASSERT(out->shape()[0] == in->shape()[0], "RMSNorm: batch size mismatch.");
    ASSERT(out->shape()[1] == in->shape()[1], "RMSNorm: hidden size mismatch.");
    ASSERT(weight->shape()[0] == in->shape()[1], "RMSNorm: weight size mismatch.");

    ASSERT(out->dtype() == in->dtype(), "RMSNorm: out/in dtype mismatch.");
    ASSERT(weight->dtype() == in->dtype(), "RMSNorm: weight dtype mismatch.");

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out, in, weight, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::rms_norm(out, in, weight, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
