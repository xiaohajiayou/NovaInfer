#pragma once
#include "llaisys.h"
#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
}
