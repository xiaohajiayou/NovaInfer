#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>
#include "llaisys.h"

namespace llaisys::device::nvidia {

void cuda_check(cudaError_t rc, const char *what, const char *file, int line);

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace {

__global__ void contiguous_strided_copy_kernel(std::uint8_t *dst,
                                               const std::uint8_t *src,
                                               const std::uint64_t *shape,
                                               const long long *strides,
                                               int ndim,
                                               std::uint64_t numel,
                                               std::uint64_t elem_size) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= numel) {
        return;
    }

    std::uint64_t rem = linear;
    long long src_elem_index = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        const std::uint64_t dim = shape[d];
        const std::uint64_t idx = rem % dim;
        rem /= dim;
        src_elem_index += static_cast<long long>(idx) * strides[d];
    }

    const std::uint8_t *src_ptr = src + static_cast<std::uint64_t>(src_elem_index) * elem_size;
    std::uint8_t *dst_ptr = dst + linear * elem_size;
    for (std::uint64_t b = 0; b < elem_size; ++b) {
        dst_ptr[b] = src_ptr[b];
    }
}

} // namespace

void contiguous_strided_copy(void *dst,
                             const void *src,
                             const size_t *shape,
                             const ptrdiff_t *strides,
                             size_t ndim,
                             size_t numel,
                             size_t elem_size,
                             llaisysStream_t stream) {
    if (numel == 0 || elem_size == 0) {
        return;
    }

    std::vector<std::uint64_t> shape_h(ndim, 1);
    std::vector<long long> strides_h(ndim, 0);
    for (size_t i = 0; i < ndim; ++i) {
        shape_h[i] = static_cast<std::uint64_t>(shape[i]);
        strides_h[i] = static_cast<long long>(strides[i]);
    }

    std::uint64_t *shape_d = nullptr;
    long long *strides_d = nullptr;
    const size_t shape_bytes = ndim * sizeof(std::uint64_t);
    const size_t strides_bytes = ndim * sizeof(long long);

    LLAISYS_CUDA_CHECK(cudaMalloc(&shape_d, shape_bytes));
    LLAISYS_CUDA_CHECK(cudaMalloc(&strides_d, strides_bytes));

    auto cu_stream = reinterpret_cast<cudaStream_t>(stream);
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(shape_d, shape_h.data(), shape_bytes, cudaMemcpyHostToDevice, cu_stream));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(strides_d, strides_h.data(), strides_bytes, cudaMemcpyHostToDevice, cu_stream));

    constexpr int kBlock = 256;
    const int grid = static_cast<int>((numel + static_cast<size_t>(kBlock) - 1) / static_cast<size_t>(kBlock));
    contiguous_strided_copy_kernel<<<grid, kBlock, 0, cu_stream>>>(reinterpret_cast<std::uint8_t *>(dst),
                                                                    reinterpret_cast<const std::uint8_t *>(src),
                                                                    shape_d,
                                                                    strides_d,
                                                                    static_cast<int>(ndim),
                                                                    static_cast<std::uint64_t>(numel),
                                                                    static_cast<std::uint64_t>(elem_size));
    LLAISYS_CUDA_CHECK(cudaGetLastError());

    LLAISYS_CUDA_CHECK(cudaFree(shape_d));
    LLAISYS_CUDA_CHECK(cudaFree(strides_d));
}

} // namespace llaisys::device::nvidia
