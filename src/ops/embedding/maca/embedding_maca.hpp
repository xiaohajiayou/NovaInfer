#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {

void embedding(tensor_t out, tensor_t index, tensor_t weight);

} // namespace llaisys::ops::cuda

