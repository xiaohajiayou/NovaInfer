#include "swiglu_cuda.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace llaisys::device::nvidia {
void cuda_check(cudaError_t rc, const char *what, const char *file, int line);
} // namespace llaisys::device::nvidia

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace llaisys::ops::cuda {

namespace {

template <typename T>
__global__ void swiglu_kernel(T *out,
                              const T *gate,
                              const T *up,
                              std::uint64_t numel) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                            + static_cast<std::uint64_t>(threadIdx.x);
    if (i >= numel) {
        return;
    }

    const float g = llaisys::device::nvidia::dtype::to_float<T>(gate[i]);
    const float u = llaisys::device::nvidia::dtype::to_float<T>(up[i]);
    const float sig = (g >= 0.0f) ? (1.0f / (1.0f + expf(-g))) : (expf(g) / (1.0f + expf(g)));
    out[i] = llaisys::device::nvidia::dtype::from_float<T>(u * (g * sig));
}

template <typename T>
void launch_swiglu(tensor_t out, tensor_t gate, tensor_t up, std::uint64_t numel) {
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((numel + static_cast<std::uint64_t>(kBlock) - 1)
                                       / static_cast<std::uint64_t>(kBlock));
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    swiglu_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(gate->data()),
        reinterpret_cast<const T *>(up->data()),
        numel);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    const std::uint64_t numel = static_cast<std::uint64_t>(out->numel());
    if (numel == 0) {
        return;
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_swiglu<float>(out, gate, up, numel);
        return;
    case LLAISYS_DTYPE_F16:
        launch_swiglu<llaisys::fp16_t>(out, gate, up, numel);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_swiglu<llaisys::bf16_t>(out, gate, up, numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops::cuda
