#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::runtime::output {

// Stores per-step logits rows in contiguous f32 memory and tracks row->token mapping.
class OutputBuffer {
public:
    // voc: model vocabulary size used as row width.
    explicit OutputBuffer(size_t voc) : voc_(voc) {}

    // Reset buffered logits rows and output-id mapping for a new decode step.
    void clear();
    // Pre-reserve capacity for n_rows logits rows to avoid repeated reallocations.
    void reserve_rows(size_t n_rows);
    // Append one logits row and remember which token index produced this row.
    // row: pointer to source logits row.
    // dtype: source row dtype (will be converted to f32).
    // output_id: token index inside current batch.
    void append_row(const std::byte *row, llaisysDataType_t dtype, int64_t output_id);
    // Append one output token index without materializing logits.
    void append_output_id(int64_t output_id);

    // Return pointer to the first logits row, or nullptr when empty.
    float *logits() noexcept;
    // Return pointer to i-th logits row, or nullptr when out of range.
    float *logits_ith(int32_t i) noexcept;
    // Return number of buffered logits rows.
    int32_t n_outputs() const noexcept;
    // Return row->token-index mapping buffer, or nullptr when empty.
    const int64_t *output_ids() const noexcept;
    // Append one sampled token id for the latest decode row.
    void append_sampled_id(int64_t sampled_id);
    // Return sampled token ids aligned with output_ids(), or nullptr when empty.
    const int64_t *sampled_ids() const noexcept;

private:
    // Vocabulary size (number of logits per row).
    size_t voc_;
    // Contiguous logits buffer in row-major layout [n_outputs, voc_].
    std::vector<float> logits_f32_;
    // output_ids_[j] is the batch token index corresponding to row j.
    std::vector<int64_t> output_ids_;
    // sampled_ids_[j] is sampled token id for output row j.
    std::vector<int64_t> sampled_ids_;
};

} // namespace llaisys::runtime::output
