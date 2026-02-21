#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {

void add(tensor_t c, tensor_t a, tensor_t b);

} // namespace llaisys::ops::cuda
