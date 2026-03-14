#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

} // namespace llaisys::ops::cuda
