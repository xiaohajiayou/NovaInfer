#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cuda {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);

} // namespace llaisys::ops::cuda

