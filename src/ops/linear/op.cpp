#include "op.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    CHECK_SAME_DEVICE(out, in, weight, bias);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
    const size_t batch_size = in->shape()[0];
    const size_t input_dim = in->shape()[1];
    const size_t output_dim = weight->shape()[0];
    ASSERT(input_dim == weight->shape()[1], "Linear: input dimension must match weight dimension.");
    ASSERT(out->shape()[0] == batch_size && out->shape()[1] == output_dim, "Linear: output shape must match.");
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out, in, weight, bias);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
