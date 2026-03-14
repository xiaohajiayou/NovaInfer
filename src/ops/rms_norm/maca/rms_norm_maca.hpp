#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);

} // namespace llaisys::ops::cuda

