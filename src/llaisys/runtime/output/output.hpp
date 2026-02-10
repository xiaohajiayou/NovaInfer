#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::runtime::output {

class OutputBuffer {
public:
    explicit OutputBuffer(size_t voc) : voc_(voc) {}

    void clear();
    void reserve_rows(size_t n_rows);
    void append_row(const std::byte *row, llaisysDataType_t dtype, int32_t output_id);

    float *logits() noexcept;
    float *logits_ith(int32_t i) noexcept;
    int32_t n_outputs() const noexcept;
    const int32_t *output_ids() const noexcept;

private:
    size_t voc_;
    std::vector<float> logits_f32_;
    std::vector<int32_t> output_ids_;
};

} // namespace llaisys::runtime::output
