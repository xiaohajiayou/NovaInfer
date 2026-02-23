#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/embedding_cuda.hpp"
#endif
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->ndim() == 2, "Embedding: out must be 2-D.");
    ASSERT(index->ndim() == 1, "Embedding: index must be 1-D.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2-D.");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: index length mismatch.");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: embedding dim mismatch.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");
    ASSERT(out->dtype() == weight->dtype(), "Embedding: out/weight dtype mismatch.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out, index, weight);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::embedding(out, index, weight);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
