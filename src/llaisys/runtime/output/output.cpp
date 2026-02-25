#include "output.hpp"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

#include <cstring>

namespace llaisys::runtime::output {

namespace {

// Convert one logits row from its runtime dtype into f32 output storage.
void copy_row_to_f32(const std::byte *src, llaisysDataType_t dtype, size_t n, float *dst) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float *in = reinterpret_cast<const float *>(src);
        std::memcpy(dst, in, n * sizeof(float));
        return;
    }
    case LLAISYS_DTYPE_F16: {
        const llaisys::fp16_t *in = reinterpret_cast<const llaisys::fp16_t *>(src);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = llaisys::utils::cast<float>(in[i]);
        }
        return;
    }
    case LLAISYS_DTYPE_BF16: {
        const llaisys::bf16_t *in = reinterpret_cast<const llaisys::bf16_t *>(src);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = llaisys::utils::cast<float>(in[i]);
        }
        return;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

void OutputBuffer::clear() {
    // Keep allocated memory for next step; reset logical contents only.
    logits_f32_.clear();
    output_ids_.clear();
    sampled_ids_.clear();
}

void OutputBuffer::reserve_rows(size_t n_rows) {
    // Reserve payload and mapping buffers together to keep append_row amortized O(1).
    logits_f32_.reserve(n_rows * voc_);
    output_ids_.reserve(n_rows);
    sampled_ids_.reserve(n_rows);
}

void OutputBuffer::append_row(const std::byte *row, llaisysDataType_t dtype, int32_t output_id) {
    ASSERT(row != nullptr, "output: row must not be null");
    const size_t old = logits_f32_.size();
    logits_f32_.resize(old + voc_);
    copy_row_to_f32(row, dtype, voc_, logits_f32_.data() + old);
    output_ids_.push_back(output_id);
}

void OutputBuffer::append_output_id(int32_t output_id) {
    output_ids_.push_back(output_id);
}

float *OutputBuffer::logits() noexcept {
    return logits_f32_.empty() ? nullptr : logits_f32_.data();
}

float *OutputBuffer::logits_ith(int32_t i) noexcept {
    if (i < 0 || static_cast<size_t>(i) >= output_ids_.size()) {
        return nullptr;
    }
    // Rows are tightly packed in row-major order.
    return logits_f32_.data() + static_cast<size_t>(i) * voc_;
}

int32_t OutputBuffer::n_outputs() const noexcept {
    return static_cast<int32_t>(output_ids_.size());
}

const int32_t *OutputBuffer::output_ids() const noexcept {
    return output_ids_.empty() ? nullptr : output_ids_.data();
}

void OutputBuffer::append_sampled_id(int32_t sampled_id) {
    sampled_ids_.push_back(sampled_id);
}

const int32_t *OutputBuffer::sampled_ids() const noexcept {
    return sampled_ids_.empty() ? nullptr : sampled_ids_.data();
}

} // namespace llaisys::runtime::output
