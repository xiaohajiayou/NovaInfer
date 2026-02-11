#pragma once
#include "llaisys.h"
#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
    void self_attention(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        float scale);
    void self_attention_masked(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        tensor_t mask,
        float scale);
}
