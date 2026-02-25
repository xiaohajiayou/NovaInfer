#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/linear_cuda.hpp"
#endif

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
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::linear(out, in, weight, bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}

void linear_indexed(tensor_t out, tensor_t in, tensor_t row_indices, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    CHECK_SAME_DEVICE(out, in, row_indices, weight, bias);
    ASSERT(out->isContiguous() && in->isContiguous() && row_indices->isContiguous() &&
               weight->isContiguous() && bias->isContiguous(),
           "LinearIndexed: all tensors must be contiguous.");
    ASSERT(row_indices->dtype() == LLAISYS_DTYPE_I32, "LinearIndexed: row_indices must be int32.");
    ASSERT(in->ndim() == 2 && out->ndim() == 2 && weight->ndim() == 2 && row_indices->ndim() == 1,
           "LinearIndexed: shape ranks must be in[2], out[2], weight[2], row_indices[1].");
    const size_t batch_size = row_indices->shape()[0];
    const size_t input_dim = in->shape()[1];
    const size_t output_dim = weight->shape()[0];
    ASSERT(input_dim == weight->shape()[1], "LinearIndexed: input dimension must match weight dimension.");
    ASSERT(out->shape()[0] == batch_size && out->shape()[1] == output_dim,
           "LinearIndexed: output shape must match.");

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear_indexed(out, in, row_indices, weight, bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::linear_indexed(out, in, row_indices, weight, bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
