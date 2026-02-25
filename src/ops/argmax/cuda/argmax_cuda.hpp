#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size);
void argmax_rows(std::byte *max_idx,
                 std::byte *max_val,
                 const std::byte *vals,
                 llaisysDataType_t type,
                 size_t nrow,
                 size_t ncol);

} // namespace llaisys::ops::cuda
