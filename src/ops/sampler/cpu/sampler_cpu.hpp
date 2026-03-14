#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

void sample_rows(std::byte *sampled_ids,
                 const std::byte *logits,
                 llaisysDataType_t logits_type,
                 const std::byte *temperatures,
                 const std::byte *top_ps,
                 const std::byte *top_ks,
                 const std::byte *seeds,
                 const std::byte *has_seeds,
                 size_t nrow,
                 size_t ncol,
                 int candidate_cap);

} // namespace llaisys::ops::cpu
