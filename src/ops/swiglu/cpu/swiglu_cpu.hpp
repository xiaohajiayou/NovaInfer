#pragma once
#include "llaisys.h"
#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}