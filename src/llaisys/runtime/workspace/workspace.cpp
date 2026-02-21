#include "workspace.hpp"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

namespace llaisys::runtime::workspace {

namespace {

// Small helper to keep allocation call-site concise.
tensor_t make_tensor(const std::vector<size_t> &shape,
                     llaisysDataType_t dtype,
                     llaisysDeviceType_t device_type,
                     int device_id) {
    return Tensor::create(shape, dtype, device_type, device_id);
}

} // namespace

Qwen2Workspace::Qwen2Workspace(size_t hs,
                               size_t nh,
                               size_t nkvh,
                               size_t dh,
                               size_t di,
                               size_t voc,
                               size_t maxseq,
                               llaisysDataType_t dtype,
                               llaisysDeviceType_t device_type,
                               int device_id)
    : hs_(hs),
      nh_(nh),
      nkvh_(nkvh),
      dh_(dh),
      di_(di),
      voc_(voc),
      maxseq_(maxseq),
      dtype_(dtype),
      device_type_(device_type),
      device_id_(device_id) {}

Qwen2Workspace::Layout Qwen2Workspace::build_layout_(size_t ntoken) const {
    // All offsets are counted in elements (not bytes).
    Layout layout{};
    size_t off = 0;
    layout.hidden = off;
    off += ntoken * hs_;
    layout.normed = off;
    off += ntoken * hs_;
    layout.q_proj = off;
    off += ntoken * (nh_ * dh_);
    layout.k_proj = off;
    off += ntoken * (nkvh_ * dh_);
    layout.v_proj = off;
    off += ntoken * (nkvh_ * dh_);
    layout.rope_q = off;
    off += ntoken * (nh_ * dh_);
    layout.rope_k = off;
    off += ntoken * (nkvh_ * dh_);
    layout.attn_out = off;
    off += ntoken * (nh_ * dh_);
    layout.attn_proj = off;
    off += ntoken * hs_;
    layout.mlp_normed = off;
    off += ntoken * hs_;
    layout.gate = off;
    off += ntoken * di_;
    layout.up = off;
    off += ntoken * di_;
    layout.swiglu = off;
    off += ntoken * di_;
    layout.down = off;
    off += ntoken * hs_;
    layout.logits = off;
    off += ntoken * voc_;
    layout.k_ctx = off;
    off += maxseq_ * nkvh_ * dh_;
    layout.v_ctx = off;
    off += maxseq_ * nkvh_ * dh_;
    layout.total_main = off;
    // i64 arena packs [input_ids, pos_ids].
    layout.total_i64 = ntoken * 2;
    // u8 arena is used by SLOT dense-mask materialization.
    layout.total_u8 = ntoken * maxseq_;
    return layout;
}

tensor_t Qwen2Workspace::slice_main_(const Layout &layout,
                                     size_t start,
                                     size_t n,
                                     const std::vector<size_t> &shape) const {
    ASSERT(main_arena_ != nullptr, "workspace: main arena is null");
    ASSERT(start + n <= layout.total_main, "workspace: main slice out of range");
    tensor_t t = main_arena_->slice(0, start, start + n);
    return t->view(shape);
}

tensor_t Qwen2Workspace::slice_i64_(const Layout &layout,
                                    size_t start,
                                    size_t n,
                                    const std::vector<size_t> &shape) const {
    ASSERT(i64_arena_ != nullptr, "workspace: i64 arena is null");
    ASSERT(start + n <= layout.total_i64, "workspace: i64 slice out of range");
    tensor_t t = i64_arena_->slice(0, start, start + n);
    return t->view(shape);
}

tensor_t Qwen2Workspace::slice_u8_(const Layout &layout,
                                   size_t start,
                                   size_t n,
                                   const std::vector<size_t> &shape) const {
    ASSERT(u8_arena_ != nullptr, "workspace: u8 arena is null");
    ASSERT(start + n <= layout.total_u8, "workspace: u8 slice out of range");
    tensor_t t = u8_arena_->slice(0, start, start + n);
    return t->view(shape);
}

void Qwen2Workspace::reserve(size_t ntoken) {
    CHECK_ARGUMENT(ntoken > 0, "workspace: ntoken must be > 0");
    // Grow-only: reuse current arenas when capacity already satisfies request.
    if (token_cap_ >= ntoken && main_arena_ != nullptr && i64_arena_ != nullptr && u8_arena_ != nullptr) {
        return;
    }

    token_cap_ = ntoken;
    const Layout layout = build_layout_(token_cap_);

    main_arena_ = make_tensor({layout.total_main}, dtype_, device_type_, device_id_);
    i64_arena_ = make_tensor({layout.total_i64}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    u8_arena_ = make_tensor({layout.total_u8}, LLAISYS_DTYPE_U8, device_type_, device_id_);

    // Materialize all tensor views once after (re)allocation.
    view_.hidden = slice_main_(layout, layout.hidden, token_cap_ * hs_, {token_cap_, hs_});
    view_.normed = slice_main_(layout, layout.normed, token_cap_ * hs_, {token_cap_, hs_});
    view_.q_proj = slice_main_(layout, layout.q_proj, token_cap_ * (nh_ * dh_), {token_cap_, nh_ * dh_});
    view_.k_proj = slice_main_(layout, layout.k_proj, token_cap_ * (nkvh_ * dh_), {token_cap_, nkvh_ * dh_});
    view_.v_proj = slice_main_(layout, layout.v_proj, token_cap_ * (nkvh_ * dh_), {token_cap_, nkvh_ * dh_});
    view_.rope_q = slice_main_(layout, layout.rope_q, token_cap_ * (nh_ * dh_), {token_cap_, nh_, dh_});
    view_.rope_k = slice_main_(layout, layout.rope_k, token_cap_ * (nkvh_ * dh_), {token_cap_, nkvh_, dh_});
    view_.attn_out = slice_main_(layout, layout.attn_out, token_cap_ * (nh_ * dh_), {token_cap_, nh_, dh_});
    view_.attn_proj = slice_main_(layout, layout.attn_proj, token_cap_ * hs_, {token_cap_, hs_});
    view_.mlp_normed = slice_main_(layout, layout.mlp_normed, token_cap_ * hs_, {token_cap_, hs_});
    view_.gate = slice_main_(layout, layout.gate, token_cap_ * di_, {token_cap_, di_});
    view_.up = slice_main_(layout, layout.up, token_cap_ * di_, {token_cap_, di_});
    view_.swiglu = slice_main_(layout, layout.swiglu, token_cap_ * di_, {token_cap_, di_});
    view_.down = slice_main_(layout, layout.down, token_cap_ * hs_, {token_cap_, hs_});
    view_.logits = slice_main_(layout, layout.logits, token_cap_ * voc_, {token_cap_, voc_});
    view_.k_ctx = slice_main_(layout, layout.k_ctx, maxseq_ * nkvh_ * dh_, {maxseq_, nkvh_, dh_});
    view_.v_ctx = slice_main_(layout, layout.v_ctx, maxseq_ * nkvh_ * dh_, {maxseq_, nkvh_, dh_});
    view_.input_ids = slice_i64_(layout, 0, token_cap_, {token_cap_});
    view_.pos_ids = slice_i64_(layout, token_cap_, token_cap_, {token_cap_});
    view_.attn_mask_flat = slice_u8_(layout, 0, token_cap_ * maxseq_, {token_cap_ * maxseq_});
}

} // namespace llaisys::runtime::workspace
