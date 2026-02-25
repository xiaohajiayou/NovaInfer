#pragma once
#include "llaisys.h"
#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
void linear_indexed(tensor_t out, tensor_t in, tensor_t row_indices, tensor_t weight, tensor_t bias);
}
