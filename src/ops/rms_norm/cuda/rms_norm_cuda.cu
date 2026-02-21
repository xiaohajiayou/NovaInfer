#include "rms_norm_cuda.hpp"

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
__global__ void rms_norm_kernel(T *out,
                                const T *in,
                                const T *weight,
                                float eps,
                                std::int32_t rows,
                                std::int32_t dim) {
    const std::int32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float smem[256];
    float local_sum = 0.0f;
    for (std::int32_t j = threadIdx.x; j < dim; j += blockDim.x) {
        const float v = llaisys::device::nvidia::dtype::to_float<T>(
            in[static_cast<std::size_t>(row) * static_cast<std::size_t>(dim) + static_cast<std::size_t>(j)]);
        local_sum += v * v;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (std::int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(smem[0] / static_cast<float>(dim) + eps);
    for (std::int32_t j = threadIdx.x; j < dim; j += blockDim.x) {
        const std::size_t idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim) + static_cast<std::size_t>(j);
        const float in_v = llaisys::device::nvidia::dtype::to_float<T>(in[idx]);
        const float w_v = llaisys::device::nvidia::dtype::to_float<T>(weight[j]);
        out[idx] = llaisys::device::nvidia::dtype::from_float<T>(in_v * inv_rms * w_v);
    }
}

template <typename T>
void launch_rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps, std::int32_t rows, std::int32_t dim) {
    constexpr int kBlock = 256;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    rms_norm_kernel<T><<<rows, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(in->data()),
        reinterpret_cast<const T *>(weight->data()),
        eps,
        rows,
        dim);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    const std::int32_t rows = static_cast<std::int32_t>(in->shape()[0]);
    const std::int32_t dim = static_cast<std::int32_t>(in->shape()[1]);
    if (rows <= 0 || dim <= 0) {
        return;
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_rms_norm<float>(out, in, weight, eps, rows, dim);
        return;
    case LLAISYS_DTYPE_F16:
        launch_rms_norm<llaisys::fp16_t>(out, in, weight, eps, rows, dim);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_rms_norm<llaisys::bf16_t>(out, in, weight, eps, rows, dim);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops::cuda
