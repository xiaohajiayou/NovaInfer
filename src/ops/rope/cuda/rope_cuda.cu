#include "rope_cuda.hpp"

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
__global__ void rope_kernel(T *out,
                            const T *in,
                            const std::int64_t *pos_ids,
                            float theta,
                            std::int32_t seqlen,
                            std::int32_t nhead,
                            std::int32_t dim) {
    const std::int32_t half = dim / 2;
    const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::int32_t total = seqlen * nhead * half;
    if (idx >= total) {
        return;
    }

    const std::int32_t j = idx % half;
    const std::int32_t tmp = idx / half;
    const std::int32_t h = tmp % nhead;
    const std::int32_t i = tmp / nhead;

    const float pos = static_cast<float>(pos_ids[i]);
    const float exponent = static_cast<float>(j) / static_cast<float>(half);
    const float inv_freq = powf(theta, -exponent);
    const float angle = pos * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);

    const std::size_t base = (static_cast<std::size_t>(i) * static_cast<std::size_t>(nhead)
                             + static_cast<std::size_t>(h)) * static_cast<std::size_t>(dim);
    const float a = llaisys::device::nvidia::dtype::to_float<T>(in[base + static_cast<std::size_t>(j)]);
    const float b = llaisys::device::nvidia::dtype::to_float<T>(in[base + static_cast<std::size_t>(j + half)]);
    out[base + static_cast<std::size_t>(j)] = llaisys::device::nvidia::dtype::from_float<T>(a * c - b * s);
    out[base + static_cast<std::size_t>(j + half)] = llaisys::device::nvidia::dtype::from_float<T>(b * c + a * s);
}

template <typename T>
void launch_rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, std::int32_t seqlen, std::int32_t nhead, std::int32_t dim) {
    const std::int32_t total = seqlen * nhead * (dim / 2);
    constexpr int kBlock = 256;
    const int grid = (total + kBlock - 1) / kBlock;
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    rope_kernel<T><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(in->data()),
        reinterpret_cast<const std::int64_t *>(pos_ids->data()),
        theta,
        seqlen,
        nhead,
        dim);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    const std::int32_t seqlen = static_cast<std::int32_t>(in->shape()[0]);
    const std::int32_t nhead = static_cast<std::int32_t>(in->shape()[1]);
    const std::int32_t dim = static_cast<std::int32_t>(in->shape()[2]);
    if (seqlen <= 0 || nhead <= 0 || dim <= 0) {
        return;
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_rope<float>(out, in, pos_ids, theta, seqlen, nhead, dim);
        return;
    case LLAISYS_DTYPE_F16:
        launch_rope<llaisys::fp16_t>(out, in, pos_ids, theta, seqlen, nhead, dim);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_rope<llaisys::bf16_t>(out, in, pos_ids, theta, seqlen, nhead, dim);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops::cuda
