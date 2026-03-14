#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/sampler_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/sampler_cuda.hpp"
#endif

namespace llaisys::ops {

void sample_rows(tensor_t sampled_ids,
                 tensor_t logits,
                 tensor_t temperatures,
                 tensor_t top_ps,
                 tensor_t top_ks,
                 tensor_t seeds,
                 tensor_t has_seeds) {
    CHECK_SAME_DEVICE(sampled_ids, logits, temperatures, top_ps, top_ks, seeds, has_seeds);
    ASSERT(sampled_ids->dtype() == LLAISYS_DTYPE_I64, "Sampler: sampled_ids must be int64.");
    ASSERT(logits->ndim() == 2 && logits->isContiguous(), "Sampler: logits must be contiguous 2D.");
    ASSERT(sampled_ids->ndim() == 1 && sampled_ids->isContiguous(), "Sampler: sampled_ids must be contiguous 1D.");
    ASSERT(temperatures->dtype() == LLAISYS_DTYPE_F32 && temperatures->ndim() == 1 && temperatures->isContiguous(),
           "Sampler: temperatures must be contiguous float32 1D.");
    ASSERT(top_ps->dtype() == LLAISYS_DTYPE_F32 && top_ps->ndim() == 1 && top_ps->isContiguous(),
           "Sampler: top_ps must be contiguous float32 1D.");
    ASSERT(top_ks->dtype() == LLAISYS_DTYPE_I32 && top_ks->ndim() == 1 && top_ks->isContiguous(),
           "Sampler: top_ks must be contiguous int32 1D.");
    ASSERT(seeds->dtype() == LLAISYS_DTYPE_I64 && seeds->ndim() == 1 && seeds->isContiguous(),
           "Sampler: seeds must be contiguous int64 1D.");
    ASSERT(has_seeds->dtype() == LLAISYS_DTYPE_I32 && has_seeds->ndim() == 1 && has_seeds->isContiguous(),
           "Sampler: has_seeds must be contiguous int32 1D.");

    const size_t nrow = logits->shape()[0];
    const size_t ncol = logits->shape()[1];
    ASSERT(sampled_ids->shape()[0] == nrow, "Sampler: sampled_ids shape mismatch.");
    ASSERT(temperatures->shape()[0] == nrow && top_ps->shape()[0] == nrow && top_ks->shape()[0] == nrow &&
               seeds->shape()[0] == nrow && has_seeds->shape()[0] == nrow,
           "Sampler: control tensor shape mismatch.");

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sample_rows(sampled_ids->data(),
                                logits->data(),
                                logits->dtype(),
                                temperatures->data(),
                                top_ps->data(),
                                top_ks->data(),
                                seeds->data(),
                                has_seeds->data(),
                                nrow,
                                ncol,
                                kSamplerCandidateCap);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::sample_rows(sampled_ids->data(),
                                 logits->data(),
                                 logits->dtype(),
                                 temperatures->data(),
                                 top_ps->data(),
                                 top_ks->data(),
                                 seeds->data(),
                                 has_seeds->data(),
                                 nrow,
                                 ncol,
                                 kSamplerCandidateCap);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
