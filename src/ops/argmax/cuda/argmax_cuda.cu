#include "argmax_cuda.hpp"

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
__global__ void argmax_kernel(std::int64_t *out_idx,
                              T *out_val,
                              const T *vals,
                              std::int32_t numel) {
    __shared__ float s_val[256];
    __shared__ std::int32_t s_idx[256];

    const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
    float best_val = llaisys::device::nvidia::dtype::to_float<T>(vals[0]);
    std::int32_t best_idx = 0;
    for (std::int32_t i = tid; i < numel; i += blockDim.x) {
        const float v = llaisys::device::nvidia::dtype::to_float<T>(vals[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (std::int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float rhs_v = s_val[tid + stride];
            const std::int32_t rhs_i = s_idx[tid + stride];
            if (rhs_v > s_val[tid]) {
                s_val[tid] = rhs_v;
                s_idx[tid] = rhs_i;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[0] = static_cast<std::int64_t>(s_idx[0]);
        out_val[0] = llaisys::device::nvidia::dtype::from_float<T>(s_val[0]);
    }
}

template <typename T>
void launch_argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, std::int32_t numel) {
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    argmax_kernel<T><<<1, 256, 0, stream>>>(
        reinterpret_cast<std::int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    const std::int32_t numel = static_cast<std::int32_t>(size);
    if (numel <= 0) {
        return;
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_argmax<float>(max_idx, max_val, vals, numel);
        return;
    case LLAISYS_DTYPE_F16:
        launch_argmax<llaisys::fp16_t>(max_idx, max_val, vals, numel);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_argmax<llaisys::bf16_t>(max_idx, max_val, vals, numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cuda
