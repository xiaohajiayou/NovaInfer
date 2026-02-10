#pragma once

#include "llaisys/tensor.h"

#include <vector>

namespace llaisys::runtime::weights {

// Replace one weight slot safely:
// - same handle: no-op
// - different old handle: destroy old, then set new
void replace_slot(llaisysTensor_t *slot, llaisysTensor_t new_handle);

// Destroy all handles referenced by slots with pointer deduplication.
void destroy_unique(const std::vector<llaisysTensor_t *> &slots);

} // namespace llaisys::runtime::weights
