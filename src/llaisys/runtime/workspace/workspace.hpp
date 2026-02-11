#pragma once

#include "../../../tensor/tensor.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace llaisys::runtime::workspace {

// Materialized tensor views sliced from workspace arenas for one decode step.
struct Qwen2WorkspaceView {
    // [ntoken, hs]
    tensor_t hidden;
    // [ntoken, hs]
    tensor_t normed;
    // [ntoken, nh * dh]
    tensor_t q_proj;
    // [ntoken, nkvh * dh]
    tensor_t k_proj;
    // [ntoken, nkvh * dh]
    tensor_t v_proj;
    // [ntoken, nh, dh]
    tensor_t rope_q;
    // [ntoken, nkvh, dh]
    tensor_t rope_k;
    // [ntoken, nh, dh]
    tensor_t attn_out;
    // [ntoken, hs]
    tensor_t attn_proj;
    // [ntoken, hs]
    tensor_t mlp_normed;
    // [ntoken, di]
    tensor_t gate;
    // [ntoken, di]
    tensor_t up;
    // [ntoken, di]
    tensor_t swiglu;
    // [ntoken, hs]
    tensor_t down;
    // [ntoken, voc]
    tensor_t logits;
    // [ntoken] i64 logical positions.
    tensor_t pos_ids;
    // [maxseq, nkvh, dh] transient key context gather buffer.
    tensor_t k_ctx;
    // [maxseq, nkvh, dh] transient value context gather buffer.
    tensor_t v_ctx;
};

// Grow-only workspace arena for Qwen2 compute buffers.
// Main buffers are allocated from one contiguous tensor and then sliced/viewed.
class Qwen2Workspace {
public:
    // hs: hidden size.
    // nh: attention head count.
    // nkvh: kv head count.
    // dh: per-head dimension.
    // di: intermediate feed-forward size.
    // voc: vocabulary size.
    // maxseq: KV cache slot count used by gather buffers.
    // dtype: compute dtype for main arena.
    // device_type/device_id: target device where arenas are allocated.
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

    // Ensure workspace capacity for at least ntoken tokens (grow-only).
    void reserve(size_t ntoken);
    // Current token capacity of all token-shaped views.
    size_t token_capacity() const noexcept { return token_cap_; }
    // Access sliced views for current capacity.
    const Qwen2WorkspaceView &view() const noexcept { return view_; }

private:
    // Linear layout (element offsets) of each sub-buffer in main/i64 arenas.
    struct Layout {
        // Offset of hidden buffer in main arena.
        size_t hidden;
        // Offset of normed buffer in main arena.
        size_t normed;
        // Offset of q_proj buffer in main arena.
        size_t q_proj;
        // Offset of k_proj buffer in main arena.
        size_t k_proj;
        // Offset of v_proj buffer in main arena.
        size_t v_proj;
        // Offset of rope_q buffer in main arena.
        size_t rope_q;
        // Offset of rope_k buffer in main arena.
        size_t rope_k;
        // Offset of attn_out buffer in main arena.
        size_t attn_out;
        // Offset of attn_proj buffer in main arena.
        size_t attn_proj;
        // Offset of mlp_normed buffer in main arena.
        size_t mlp_normed;
        // Offset of gate buffer in main arena.
        size_t gate;
        // Offset of up buffer in main arena.
        size_t up;
        // Offset of swiglu buffer in main arena.
        size_t swiglu;
        // Offset of down buffer in main arena.
        size_t down;
        // Offset of logits buffer in main arena.
        size_t logits;
        // Offset of k_ctx buffer in main arena.
        size_t k_ctx;
        // Offset of v_ctx buffer in main arena.
        size_t v_ctx;
        // Total element count in main arena.
        size_t total_main;
        // Total element count in i64 arena.
        size_t total_i64;
    };

    // Build linear layout for a given token count.
    Layout build_layout_(size_t ntoken) const;
    // Slice typed view from main arena by [start, start+n) and reshape.
    tensor_t slice_main_(const Layout &layout, size_t start, size_t n, const std::vector<size_t> &shape) const;
    // Slice typed view from i64 arena by [start, start+n) and reshape.
    tensor_t slice_i64_(const Layout &layout, size_t start, size_t n, const std::vector<size_t> &shape) const;

    // Model config: hidden size.
    size_t hs_;
    // Model config: attention head count.
    size_t nh_;
    // Model config: kv head count.
    size_t nkvh_;
    // Model config: per-head dimension.
    size_t dh_;
    // Model config: MLP intermediate dimension.
    size_t di_;
    // Model config: vocabulary size.
    size_t voc_;
    // Runtime config: maximum kv sequence slots.
    size_t maxseq_;
    // Compute dtype for main arena.
    llaisysDataType_t dtype_;
    // Target device type for all arena allocations.
    llaisysDeviceType_t device_type_;
    // Target device index for all arena allocations.
    int device_id_;

    // Current reserved token capacity.
    size_t token_cap_{0};
    // Contiguous arena for float-like intermediate buffers.
    tensor_t main_arena_;
    // Contiguous arena for i64 position buffers.
    tensor_t i64_arena_;
    // Materialized per-buffer views over arenas.
    Qwen2WorkspaceView view_{};
};

using qwen2_workspace_t = std::unique_ptr<Qwen2Workspace>;

} // namespace llaisys::runtime::workspace
