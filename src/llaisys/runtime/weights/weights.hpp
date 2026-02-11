#pragma once

#include "llaisys/tensor.h"

#include <vector>

namespace llaisys::runtime::weights {

// Replace one weight slot safely:
// - same handle: no-op
// - different old handle: destroy old, then set new
// slot: pointer to destination slot inside model weight table.
// new_handle: new tensor handle to store into slot (ownership transferred).
void replace_slot(llaisysTensor_t *slot, llaisysTensor_t new_handle);

// Destroy all handles referenced by slots with pointer deduplication.
// slots: list of slot pointers; each slot is nulled after processing.
void destroy_unique(const std::vector<llaisysTensor_t *> &slots);

} // namespace llaisys::runtime::weights
