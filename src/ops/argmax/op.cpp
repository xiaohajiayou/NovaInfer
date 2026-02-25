#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/argmax_cuda.hpp"
#endif
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // Only support contiguous inputs.
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64.");
    ASSERT(max_val->dtype() == vals->dtype(), "Argmax: max_val dtype must match vals.");
    
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");
    ASSERT(max_idx->ndim() == 1 && max_val->ndim() == 1 && vals->ndim() == 1, "Argmax: all tensors must be 1D.");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void argmax_rows(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "ArgmaxRows: max_idx must be int64.");
    ASSERT(max_val->dtype() == vals->dtype(), "ArgmaxRows: max_val dtype must match vals.");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "ArgmaxRows: all tensors must be contiguous.");
    ASSERT(max_idx->ndim() == 1 && max_val->ndim() == 1 && vals->ndim() == 2,
           "ArgmaxRows: max_idx/max_val must be 1D and vals must be 2D.");
    const size_t nrow = vals->shape()[0];
    const size_t ncol = vals->shape()[1];
    ASSERT(max_idx->shape()[0] == nrow && max_val->shape()[0] == nrow,
           "ArgmaxRows: output shape mismatch.");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax_rows(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), nrow, ncol);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::argmax_rows(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), nrow, ncol);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
