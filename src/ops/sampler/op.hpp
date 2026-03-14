#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

constexpr int kSamplerCandidateCap = 256;

void sample_rows(tensor_t sampled_ids,
                 tensor_t logits,
                 tensor_t temperatures,
                 tensor_t top_ps,
                 tensor_t top_ks,
                 tensor_t seeds,
                 tensor_t has_seeds);

} // namespace llaisys::ops
