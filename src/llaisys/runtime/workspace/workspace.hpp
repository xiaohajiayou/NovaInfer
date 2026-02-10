#pragma once

#include "../../../tensor/tensor.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace llaisys::runtime::workspace {

struct Qwen2WorkspaceView {
    tensor_t hidden;
    tensor_t normed;
    tensor_t q_proj;
    tensor_t k_proj;
    tensor_t v_proj;
    tensor_t rope_q;
    tensor_t rope_k;
    tensor_t attn_out;
    tensor_t attn_proj;
    tensor_t mlp_normed;
    tensor_t gate;
    tensor_t up;
    tensor_t swiglu;
    tensor_t down;
    tensor_t logits;
    tensor_t pos_ids;
    tensor_t k_ctx;
    tensor_t v_ctx;
};

// Grow-only workspace arena for Qwen2 compute buffers.
// Main buffers are allocated from one contiguous tensor and then sliced/viewed.
class Qwen2Workspace {
public:
    Qwen2Workspace(size_t hs,
                   size_t nh,
                   size_t nkvh,
                   size_t dh,
                   size_t di,
                   size_t voc,
                   size_t maxseq,
                   llaisysDataType_t dtype,
                   llaisysDeviceType_t device_type,
                   int device_id);

    void reserve(size_t ntoken);
    size_t token_capacity() const noexcept { return token_cap_; }
    const Qwen2WorkspaceView &view() const noexcept { return view_; }

private:
    struct Layout {
        size_t hidden;
        size_t normed;
        size_t q_proj;
        size_t k_proj;
        size_t v_proj;
        size_t rope_q;
        size_t rope_k;
        size_t attn_out;
        size_t attn_proj;
        size_t mlp_normed;
        size_t gate;
        size_t up;
        size_t swiglu;
        size_t down;
        size_t logits;
        size_t k_ctx;
        size_t v_ctx;
        size_t total_main;
        size_t total_i64;
    };

    Layout build_layout_(size_t ntoken) const;
    tensor_t slice_main_(const Layout &layout, size_t start, size_t n, const std::vector<size_t> &shape) const;
    tensor_t slice_i64_(const Layout &layout, size_t start, size_t n, const std::vector<size_t> &shape) const;

    size_t hs_;
    size_t nh_;
    size_t nkvh_;
    size_t dh_;
    size_t di_;
    size_t voc_;
    size_t maxseq_;
    llaisysDataType_t dtype_;
    llaisysDeviceType_t device_type_;
    int device_id_;

    size_t token_cap_{0};
    tensor_t main_arena_;
    tensor_t i64_arena_;
    Qwen2WorkspaceView view_{};
};

using qwen2_workspace_t = std::unique_ptr<Qwen2Workspace>;

} // namespace llaisys::runtime::workspace
