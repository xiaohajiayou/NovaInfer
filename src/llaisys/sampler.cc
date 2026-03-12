#include "llaisys/models/model.h"

#include "llaisys_tensor.hpp"
#include "../ops/argmax/op.hpp"
#include "../tensor/tensor.hpp"
#include "../utils/nvtx.hpp"

#include <algorithm>
#include <cstddef>

namespace {

struct SamplerScratchState {
    llaisys::tensor_t max_val_buf{};
    size_t max_val_capacity{0};
    llaisysDataType_t max_val_dtype{LLAISYS_DTYPE_INVALID};
    llaisysDeviceType_t max_val_device_type{LLAISYS_DEVICE_CPU};
    int max_val_device_id{0};

    llaisys::tensor_t ensure_max_val(size_t n,
                                     llaisysDataType_t dtype,
                                     llaisysDeviceType_t device_type,
                                     int device_id) {
        if (n == 0) {
            return nullptr;
        }
        const bool recreate = (max_val_buf == nullptr) || (max_val_capacity < n) || (max_val_dtype != dtype) ||
                              (max_val_device_type != device_type) || (max_val_device_id != device_id);
        if (recreate) {
            size_t cap = std::max<size_t>(1, max_val_capacity);
            while (cap < n) {
                cap <<= 1;
            }
            max_val_buf = llaisys::Tensor::create({cap}, dtype, device_type, device_id);
            max_val_capacity = cap;
            max_val_dtype = dtype;
            max_val_device_type = device_type;
            max_val_device_id = device_id;
        }
        if (max_val_capacity == n) {
            return max_val_buf;
        }
        return max_val_buf->slice(0, 0, n);
    }
};

SamplerScratchState &sampler_scratch_state() {
    static thread_local SamplerScratchState state{};
    return state;
}

} // namespace

__C int32_t llaisysSamplerSample(const struct SamplerInput *input,
                                 struct SamplerOutput *output) {
    LLAISYS_NVTX_SCOPE("api/sampler_sample");
    if (input == nullptr || output == nullptr || input->logits == nullptr || output->sampled_ids == nullptr) {
        return -1;
    }
    try {
        const llaisys::tensor_t logits = input->logits->tensor;
        llaisys::tensor_t sampled_ids = output->sampled_ids->tensor;
        if (logits == nullptr || sampled_ids == nullptr) {
            return -1;
        }
        if (logits->ndim() != 2 || !logits->isContiguous()) {
            return -1;
        }
        if (sampled_ids->ndim() != 1 || sampled_ids->dtype() != LLAISYS_DTYPE_I64 || !sampled_ids->isContiguous()) {
            return -1;
        }
        if (sampled_ids->deviceType() != logits->deviceType() || sampled_ids->deviceId() != logits->deviceId()) {
            return -1;
        }

        const size_t n_outputs = logits->shape()[0];
        if (n_outputs == 0) {
            return 0;
        }
        if (sampled_ids->shape()[0] < n_outputs) {
            return -1;
        }
        if (sampled_ids->shape()[0] > n_outputs) {
            sampled_ids = sampled_ids->slice(0, 0, n_outputs);
        }

        llaisys::tensor_t max_idx = sampled_ids;
        llaisys::tensor_t max_val =
            sampler_scratch_state().ensure_max_val(n_outputs, logits->dtype(), logits->deviceType(), logits->deviceId());
        if (max_val == nullptr) {
            return -1;
        }
        {
            LLAISYS_NVTX_SCOPE("sample/argmax_rows");
            llaisys::ops::argmax_rows(max_idx, max_val, logits);
        }
        return 0;
    } catch (...) {
        return -2;
    }
}
