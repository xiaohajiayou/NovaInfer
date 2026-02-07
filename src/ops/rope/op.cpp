#include "op.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    ASSERT(in->ndim() == 3, "RoPE: input must be 3-D [seqlen, nhead, dim]");
    ASSERT(out->shape() == in->shape(), "RoPE: out shape mismatch");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1-D");
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "RoPE: seqlen mismatch");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64");
    ASSERT(out->dtype() == in->dtype(), "RoPE: dtype mismatch");
    ASSERT(in->shape()[2] % 2 == 0, "RoPE: dim must be even");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: tensors must be contiguous");

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out, in, pos_ids, theta);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
