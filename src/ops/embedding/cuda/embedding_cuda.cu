#include "embedding_cuda.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace llaisys::device::nvidia {
void cuda_check(cudaError_t rc, const char *what, const char *file, int line);
} // namespace llaisys::device::nvidia

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace llaisys::ops::cuda {

namespace {

__global__ void embedding_u8_kernel(std::uint8_t *out,
                                    const std::int64_t *index,
                                    const std::uint8_t *weight,
                                    std::uint64_t num_indices,
                                    std::uint64_t embedding_dim,
                                    std::uint64_t vocab_size,
                                    std::uint64_t elem_size) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                                 + static_cast<std::uint64_t>(threadIdx.x);
    const std::uint64_t total = num_indices * embedding_dim;
    if (linear >= total) {
        return;
    }

    const std::uint64_t row = linear / embedding_dim;
    const std::uint64_t col = linear % embedding_dim;
    const std::int64_t idx = index[row];
    if (idx < 0 || static_cast<std::uint64_t>(idx) >= vocab_size) {
        return;
    }

    const std::uint64_t src_elem = static_cast<std::uint64_t>(idx) * embedding_dim + col;
    const std::uint64_t dst_elem = row * embedding_dim + col;
    const std::uint8_t *src_ptr = weight + src_elem * elem_size;
    std::uint8_t *dst_ptr = out + dst_elem * elem_size;

    for (std::uint64_t b = 0; b < elem_size; ++b) {
        dst_ptr[b] = src_ptr[b];
    }
}

} // namespace

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    const std::uint64_t num_indices = static_cast<std::uint64_t>(index->shape()[0]);
    const std::uint64_t embedding_dim = static_cast<std::uint64_t>(weight->shape()[1]);
    const std::uint64_t vocab_size = static_cast<std::uint64_t>(weight->shape()[0]);
    const std::uint64_t elem_size = static_cast<std::uint64_t>(out->elementSize());
    const std::uint64_t total = num_indices * embedding_dim;
    if (total == 0) {
        return;
    }

    constexpr int kBlock = 256;
    const int grid = static_cast<int>((total + static_cast<std::uint64_t>(kBlock) - 1)
                                      / static_cast<std::uint64_t>(kBlock));
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    embedding_u8_kernel<<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<std::uint8_t *>(out->data()),
        reinterpret_cast<const std::int64_t *>(index->data()),
        reinterpret_cast<const std::uint8_t *>(weight->data()),
        num_indices,
        embedding_dim,
        vocab_size,
        elem_size);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::cuda
