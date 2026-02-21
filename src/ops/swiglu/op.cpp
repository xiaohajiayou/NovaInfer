#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/swiglu_cuda.hpp"
#endif
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->ndim() == 2, "Swiglu: out must be 2-D.");
    ASSERT(gate->ndim() == 2, "Swiglu: gate must be 2-D.");
    ASSERT(up->ndim() == 2, "Swiglu: up must be 2-D.");
    ASSERT(out->shape()[0] == gate->shape()[0], "Swiglu: gate length mismatch.");
    ASSERT(out->shape()[1] == up->shape()[1], "Swiglu: up length mismatch.");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "Swiglu: all tensors must be contiguous.");
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out, gate, up);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::swiglu(out, gate, up);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
