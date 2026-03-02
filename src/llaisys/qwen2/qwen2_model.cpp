#include "qwen2_model.hpp"
#include "llaisys/models/model.h"
#include "../runtime/kv_cache/paged_kv.hpp"
#include "../runtime/kv_cache/unified_kv.hpp"
#include "../../core/llaisys_core.hpp"
#include <cctype>
#include <cstdlib>

namespace llaisys::models::qwen2 {

namespace {

using KvStatus = llaisys::runtime::kv_cache::KvStatus;
using KvCacheLayout = llaisys::runtime::kv_cache::KvCacheLayout;
using KvCacheBase = llaisys::runtime::kv_cache::KvCacheBase;

bool attention_mode_supported(llaisysDeviceType_t device_type, int32_t mode) {
    if (mode == ATTENTION_MODE_BLOCK) {
        return true;
    }
    if (mode == ATTENTION_MODE_SLOT) {
#ifdef ENABLE_NVIDIA_API
        if (device_type == LLAISYS_DEVICE_NVIDIA) {
            // SLOT path is currently CPU-only in this implementation.
            return false;
        }
#endif
        return true;
    }
    return false;
}

tensor_t kv_layer_k_from_cache(KvCacheBase *cache, KvCacheLayout layout, size_t layer) {
    if (layout == KvCacheLayout::SLOT) {
        auto *impl = dynamic_cast<llaisys::runtime::kv_cache::UnifiedKvImpl *>(cache);
        CHECK_ARGUMENT(impl != nullptr, "Qwen2: KvCacheBase is not UnifiedKvImpl");
        return impl->layer_k(layer);
    }
    auto *impl = dynamic_cast<llaisys::runtime::kv_cache::PagedKvImpl *>(cache);
    CHECK_ARGUMENT(impl != nullptr, "Qwen2: KvCacheBase is not PagedKvImpl");
    return impl->layer_k(layer);
}

tensor_t kv_layer_v_from_cache(KvCacheBase *cache, KvCacheLayout layout, size_t layer) {
    if (layout == KvCacheLayout::SLOT) {
        auto *impl = dynamic_cast<llaisys::runtime::kv_cache::UnifiedKvImpl *>(cache);
        CHECK_ARGUMENT(impl != nullptr, "Qwen2: KvCacheBase is not UnifiedKvImpl");
        return impl->layer_v(layer);
    }
    auto *impl = dynamic_cast<llaisys::runtime::kv_cache::PagedKvImpl *>(cache);
    CHECK_ARGUMENT(impl != nullptr, "Qwen2: KvCacheBase is not PagedKvImpl");
    return impl->layer_v(layer);
}

llaisysMemcpyKind_t choose_memcpy_kind(llaisysDeviceType_t dst, llaisysDeviceType_t src) {
    if (dst == LLAISYS_DEVICE_CPU && src == LLAISYS_DEVICE_CPU) {
        return LLAISYS_MEMCPY_H2H;
    }
    if (dst != LLAISYS_DEVICE_CPU && src == LLAISYS_DEVICE_CPU) {
        return LLAISYS_MEMCPY_H2D;
    }
    if (dst == LLAISYS_DEVICE_CPU && src != LLAISYS_DEVICE_CPU) {
        return LLAISYS_MEMCPY_D2H;
    }
    return LLAISYS_MEMCPY_D2D;
}

void runtime_copy_bytes(llaisysDeviceType_t dst_dev, std::byte *dst, llaisysDeviceType_t src_dev, const std::byte *src, size_t nbytes) {
    if (nbytes == 0) {
        return;
    }
    const llaisysMemcpyKind_t kind = choose_memcpy_kind(dst_dev, src_dev);
    utils::NvtxScope nvtx_scope(utils::nvtx_memcpy_tag(kind, false));
    const auto *api = core::context().runtime().api();
    api->memcpy_sync(dst, src, nbytes, kind);
}

#ifdef ENABLE_NVIDIA_API
ops::cuda::PagedAttentionBackend parse_paged_attn_backend_env() {
    const char *raw = std::getenv("LLAISYS_CUDA_PAGED_ATTN_BACKEND");
    if (raw == nullptr) {
        return ops::cuda::PagedAttentionBackend::NATIVE;
    }
    std::string v(raw);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "flashinfer") {
        return ops::cuda::PagedAttentionBackend::FLASHINFER;
    }
    if (v == "cudnn") {
        return ops::cuda::PagedAttentionBackend::CUDNN;
    }
    return ops::cuda::PagedAttentionBackend::NATIVE;
}
#endif

} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       int *device_ids,
                       int ndevice)
    : meta_(meta),
      device_type_(device) {

    if (device_ids != nullptr && ndevice > 0) {
        device_id_ = device_ids[0];
    }

#ifdef ENABLE_NVIDIA_API
    if (device_type_ == LLAISYS_DEVICE_NVIDIA) {
        paged_attn_backend_ = parse_paged_attn_backend_env();
    }
#endif

    check_meta_invariants_();
    init_weight_slots_();
}

Qwen2Model::~Qwen2Model() {
    destroy_weights_();

    delete[] weights_.attn_norm_w;
    delete[] weights_.attn_q_w;
    delete[] weights_.attn_q_b;
    delete[] weights_.attn_k_w;
    delete[] weights_.attn_k_b;
    delete[] weights_.attn_v_w;
    delete[] weights_.attn_v_b;
    delete[] weights_.attn_o_w;
    delete[] weights_.mlp_norm_w;
    delete[] weights_.mlp_gate_w;
    delete[] weights_.mlp_up_w;
    delete[] weights_.mlp_down_w;
}

void Qwen2Model::init_weight_slots_() {
    const size_t nlayer = meta_.nlayer;

    weights_.in_embed = nullptr;
    weights_.out_embed = nullptr;
    weights_.out_norm_w = nullptr;

    weights_.attn_norm_w = new llaisysTensor_t[nlayer]();
    weights_.attn_q_w = new llaisysTensor_t[nlayer]();
    weights_.attn_q_b = new llaisysTensor_t[nlayer]();
    weights_.attn_k_w = new llaisysTensor_t[nlayer]();
    weights_.attn_k_b = new llaisysTensor_t[nlayer]();
    weights_.attn_v_w = new llaisysTensor_t[nlayer]();
    weights_.attn_v_b = new llaisysTensor_t[nlayer]();
    weights_.attn_o_w = new llaisysTensor_t[nlayer]();
    weights_.mlp_norm_w = new llaisysTensor_t[nlayer]();
    weights_.mlp_gate_w = new llaisysTensor_t[nlayer]();
    weights_.mlp_up_w = new llaisysTensor_t[nlayer]();
    weights_.mlp_down_w = new llaisysTensor_t[nlayer]();
}

void Qwen2Model::init_runtime_state_() {
    const size_t max_model_len = runtime_.max_model_len > 0 ? runtime_.max_model_len : static_cast<size_t>(meta_.maxseq);
    const size_t kv_capacity = runtime_.kv_cache_capacity_tokens > 0 ? runtime_.kv_cache_capacity_tokens : max_model_len;
    if (runtime_.kv_layout == runtime::kv_cache::KvCacheLayout::SLOT) {
        runtime_.kv_cache = std::make_unique<runtime::kv_cache::UnifiedKvImpl>(kv_capacity, 1);
    } else {
        runtime_.kv_cache = std::make_unique<runtime::kv_cache::PagedKvImpl>(kv_capacity, 1, runtime_.kv_block_size);
    }
    runtime_.kv_cache->init_storage(meta_.nlayer, meta_.nkvh, meta_.dh, meta_.dtype, device_type_, device_id_);
    runtime_.output = std::make_unique<runtime::output::OutputBuffer>(meta_.voc);
    runtime_.kv_peak_used_tokens = 0;
}

int Qwen2Model::configure_runtime(runtime::kv_cache::KvCacheLayout kv_layout,
                                  size_t kv_block_size,
                                  size_t kv_cache_capacity_tokens,
                                  int64_t max_model_len) {
    if (kv_block_size == 0) {
        return -1;
    }
    const size_t max_len = max_model_len > 0 ? static_cast<size_t>(max_model_len) : static_cast<size_t>(meta_.maxseq);
    if (max_len == 0 || max_len > static_cast<size_t>(meta_.maxseq)) {
        return -1;
    }
    const size_t kv_capacity = kv_cache_capacity_tokens > 0 ? kv_cache_capacity_tokens : max_len;
    if (kv_capacity == 0) {
        return -1;
    }

    runtime_.kv_layout = kv_layout;
    runtime_.kv_block_size = kv_block_size;
    runtime_.max_model_len = max_len;
    runtime_.kv_cache_capacity_tokens = kv_capacity;
    runtime_.kv_cache.reset();
    runtime_.output.reset();
    workspace_.reset();
    init_runtime_state_();
    return 0;
}

void Qwen2Model::check_meta_invariants_() const {
    CHECK_ARGUMENT(meta_.nlayer > 0, "Qwen2: nlayer must be > 0");
    CHECK_ARGUMENT(meta_.hs > 0, "Qwen2: hs must be > 0");
    CHECK_ARGUMENT(meta_.nh > 0, "Qwen2: nh must be > 0");
    CHECK_ARGUMENT(meta_.nkvh > 0, "Qwen2: nkvh must be > 0");
    CHECK_ARGUMENT(meta_.dh > 0, "Qwen2: dh must be > 0");
    CHECK_ARGUMENT(meta_.di > 0, "Qwen2: di must be > 0");
    CHECK_ARGUMENT(meta_.maxseq > 0, "Qwen2: maxseq must be > 0");
    CHECK_ARGUMENT(meta_.voc > 0, "Qwen2: voc must be > 0");

    CHECK_ARGUMENT(meta_.hs == meta_.nh * meta_.dh, "Qwen2: hs must equal nh * dh");
    CHECK_ARGUMENT(meta_.nkvh <= meta_.nh, "Qwen2: nkvh must be <= nh");

    CHECK_ARGUMENT(meta_.dtype == LLAISYS_DTYPE_F32 ||
                       meta_.dtype == LLAISYS_DTYPE_F16 ||
                       meta_.dtype == LLAISYS_DTYPE_BF16,
                   "Qwen2: dtype must be one of F32/F16/BF16");
}

tensor_t Qwen2Model::create_zero_tensor_(const std::vector<size_t> &shape, llaisysDataType_t dtype) const {
    tensor_t t = Tensor::create(shape, dtype, device_type_, device_id_);
    size_t numel = 1;
    for (size_t d : shape) {
        numel *= d;
    }
    const size_t nbytes = numel * utils::dsize(dtype);
    std::vector<std::byte> zeros(nbytes, std::byte{0});
    t->load(zeros.data());
    return t;
}

void Qwen2Model::check_tensor_(const llaisysTensor_t handle,
                               const std::vector<size_t> &shape,
                               const char *name,
                               bool required) const {
    if (!handle) {
        CHECK_ARGUMENT(!required, std::string("Qwen2: missing required weight: ") + name);
        return;
    }

    const tensor_t &t = handle->tensor;
    CHECK_ARGUMENT(t->dtype() == meta_.dtype, std::string("Qwen2: dtype mismatch for ") + name);
    CHECK_ARGUMENT(t->deviceType() == device_type_ && t->deviceId() == device_id_,
                   std::string("Qwen2: device mismatch for ") + name);
    CHECK_ARGUMENT(t->shape() == shape, std::string("Qwen2: shape mismatch for ") + name);
    CHECK_ARGUMENT(t->isContiguous(), std::string("Qwen2: tensor must be contiguous for ") + name);
}

tensor_t Qwen2Model::bias_or_zero_(llaisysTensor_t handle, const tensor_t &zero_bias) const {
    if (handle) {
        return handle->tensor;
    }
    return zero_bias;
}

void Qwen2Model::validate_or_die_() {
    if (validated_) {
        return;
    }

    check_meta_invariants_();

    const size_t hs = meta_.hs;
    const size_t nh = meta_.nh;
    const size_t nkvh = meta_.nkvh;
    const size_t dh = meta_.dh;
    const size_t di = meta_.di;
    const size_t voc = meta_.voc;

    // Zero biases used where the model does not expose bias slots.
    zero_bias_attn_o_ = create_zero_tensor_({hs}, meta_.dtype);
    zero_bias_attn_q_ = create_zero_tensor_({nh * dh}, meta_.dtype);
    zero_bias_attn_k_ = create_zero_tensor_({nkvh * dh}, meta_.dtype);
    zero_bias_attn_v_ = create_zero_tensor_({nkvh * dh}, meta_.dtype);
    zero_bias_mlp_gate_ = create_zero_tensor_({di}, meta_.dtype);
    zero_bias_mlp_up_ = create_zero_tensor_({di}, meta_.dtype);
    zero_bias_mlp_down_ = create_zero_tensor_({hs}, meta_.dtype);
    zero_bias_logits_ = create_zero_tensor_({voc}, meta_.dtype);

    // Global weights.
    check_tensor_(weights_.in_embed, {voc, hs}, "in_embed", true);
    check_tensor_(weights_.out_embed, {voc, hs}, "out_embed", true);
    check_tensor_(weights_.out_norm_w, {hs}, "out_norm_w", true);

    // Per-layer weights.
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        check_tensor_(weights_.attn_norm_w[i], {hs}, "attn_norm_w", true);
        check_tensor_(weights_.attn_q_w[i], {nh * dh, hs}, "attn_q_w", true);
        check_tensor_(weights_.attn_q_b[i], {nh * dh}, "attn_q_b", false);
        check_tensor_(weights_.attn_k_w[i], {nkvh * dh, hs}, "attn_k_w", true);
        check_tensor_(weights_.attn_k_b[i], {nkvh * dh}, "attn_k_b", false);
        check_tensor_(weights_.attn_v_w[i], {nkvh * dh, hs}, "attn_v_w", true);
        check_tensor_(weights_.attn_v_b[i], {nkvh * dh}, "attn_v_b", false);
        check_tensor_(weights_.attn_o_w[i], {hs, nh * dh}, "attn_o_w", true);

        check_tensor_(weights_.mlp_norm_w[i], {hs}, "mlp_norm_w", true);
        check_tensor_(weights_.mlp_gate_w[i], {di, hs}, "mlp_gate_w", true);
        check_tensor_(weights_.mlp_up_w[i], {di, hs}, "mlp_up_w", true);
        check_tensor_(weights_.mlp_down_w[i], {hs, di}, "mlp_down_w", true);
    }

    validated_ = true;
}

tensor_t Qwen2Model::slice_tokens_(const tensor_t &t, size_t len) const {
    if (t->shape()[0] == len) {
        return t;
    }
    return t->slice(0, 0, len);
}

tensor_t Qwen2Model::view_2d_to_3d_(const tensor_t &t, size_t len, size_t nhead, size_t dim) const {
    tensor_t sliced = slice_tokens_(t, len);
    return sliced->view({len, nhead, dim});
}

void Qwen2Model::ensure_workspace_(size_t ntoken) {
    if (!workspace_) {
        const size_t kv_capacity =
            runtime_.kv_cache_capacity_tokens > 0 ? runtime_.kv_cache_capacity_tokens : meta_.maxseq;
        workspace_ = std::make_unique<runtime::workspace::Qwen2Workspace>(
            meta_.hs,
            meta_.nh,
            meta_.nkvh,
            meta_.dh,
            meta_.di,
            meta_.voc,
            kv_capacity,
            meta_.dtype,
            device_type_,
            device_id_);
    }
    workspace_->reserve(ntoken);
}

void Qwen2Model::fill_pos_ids_from_values_(const tensor_t &pos_ids, const std::vector<int64_t> &pos_values) {
    LLAISYS_NVTX_SCOPE("decode/fill_pos_ids");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Qwen2: pos_ids dtype must be int64");
    ASSERT(pos_ids->shape()[0] == pos_values.size(), "Qwen2: pos_ids length mismatch");
    const size_t nbytes = pos_values.size() * sizeof(int64_t);
    runtime_copy_bytes(pos_ids->deviceType(),
                       pos_ids->data(),
                       LLAISYS_DEVICE_CPU,
                       reinterpret_cast<const std::byte *>(pos_values.data()),
                       nbytes);
}

void Qwen2Model::build_hidden_and_pos_(const std::vector<int64_t> &tokens,
                                       const std::vector<int64_t> &pos_values,
                                       tensor_t *hidden,
                                       tensor_t *pos_ids) {
    CHECK_ARGUMENT(hidden != nullptr, "Qwen2: hidden output pointer is null");
    CHECK_ARGUMENT(pos_ids != nullptr, "Qwen2: pos_ids output pointer is null");
    ASSERT(workspace_ != nullptr, "Qwen2: workspace is null");
    CHECK_ARGUMENT(tokens.size() == pos_values.size(), "Qwen2: tokens/pos_values size mismatch");

    const size_t ntoken = tokens.size();
    const auto &ws = workspace_->view();

    tensor_t input_ids = slice_tokens_(ws.input_ids, ntoken);
    {
        LLAISYS_NVTX_SCOPE("decode/input_ids_load");
        input_ids->load(tokens.data());
    }

    *hidden = slice_tokens_(ws.hidden, ntoken);
    {
        LLAISYS_NVTX_SCOPE("decode/embedding");
        ops::embedding(*hidden, input_ids, weights_.in_embed->tensor);
    }

    *pos_ids = slice_tokens_(ws.pos_ids, ntoken);
    fill_pos_ids_from_values_(*pos_ids, pos_values);
}

void Qwen2Model::copy_token_into_cache_(tensor_t &cache, int32_t slot, const tensor_t &src, size_t token_idx) {
    LLAISYS_NVTX_SCOPE("decode/copy_token_into_cache");
    ASSERT(cache->deviceType() == src->deviceType(), "Qwen2: cache/src device mismatch");
    ASSERT(cache->dtype() == src->dtype(), "Qwen2: cache/src dtype mismatch");
    ASSERT(cache->shape()[1] == src->shape()[1] && cache->shape()[2] == src->shape()[2],
           "Qwen2: cache/src head shape mismatch");
    const size_t stride_elems = cache->shape()[1] * cache->shape()[2];
    const size_t elem_size = utils::dsize(cache->dtype());
    const size_t stride_bytes = stride_elems * elem_size;
    ASSERT(slot >= 0 && static_cast<size_t>(slot) < cache->shape()[0], "Qwen2: KV slot out of range");
    ASSERT(token_idx < src->shape()[0], "Qwen2: src token index out of range");

    std::byte *dst = cache->data() + static_cast<ptrdiff_t>(slot) * static_cast<ptrdiff_t>(stride_bytes);
    const std::byte *src_ptr = src->data() + static_cast<ptrdiff_t>(token_idx) * static_cast<ptrdiff_t>(stride_bytes);
    runtime_copy_bytes(cache->deviceType(), dst, src->deviceType(), src_ptr, stride_bytes);
}

tensor_t Qwen2Model::gather_cache_by_slots_(const tensor_t &cache, const std::vector<int32_t> &slots, size_t len, const tensor_t &buffer) {
    LLAISYS_NVTX_SCOPE("decode/gather_cache_by_slots");
    ASSERT(cache->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2: cache must be on CPU");
    ASSERT(buffer->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2: buffer must be on CPU");
    ASSERT(cache->dtype() == buffer->dtype(), "Qwen2: cache/buffer dtype mismatch");
    ASSERT(buffer->shape()[1] == cache->shape()[1] && buffer->shape()[2] == cache->shape()[2],
           "Qwen2: cache/buffer shape mismatch");
    ASSERT(len <= slots.size(), "Qwen2: gather length exceeds slot list");

    const size_t stride_elems = cache->shape()[1] * cache->shape()[2];
    const size_t elem_size = utils::dsize(cache->dtype());
    const size_t stride_bytes = stride_elems * elem_size;
    const std::byte *src_base = cache->data();
    std::byte *dst_base = buffer->data();
    for (size_t i = 0; i < len; ++i) {
        const int32_t slot = slots[i];
        ASSERT(slot >= 0 && static_cast<size_t>(slot) < cache->shape()[0], "Qwen2: gather slot out of range");
        const std::byte *src = src_base + static_cast<ptrdiff_t>(slot) * static_cast<ptrdiff_t>(stride_bytes);
        std::byte *dst = dst_base + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(stride_bytes);
        std::memcpy(dst, src, stride_bytes);
    }
    return buffer->slice(0, 0, len);
}

tensor_t Qwen2Model::run_attention_layer_(size_t layer,
                                          size_t ntoken,
                                          const tensor_t &attn_normed,
                                          const tensor_t &pos_ids,
                                          const AttentionExecState &attn_state) {
    const auto &ws = workspace_->view();
    const size_t nh = meta_.nh;
    const size_t nkvh = meta_.nkvh;
    const size_t dh = meta_.dh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    tensor_t q_proj = slice_tokens_(ws.q_proj, ntoken);
    tensor_t k_proj = slice_tokens_(ws.k_proj, ntoken);
    tensor_t v_proj = slice_tokens_(ws.v_proj, ntoken);
    ops::linear(q_proj, attn_normed, weights_.attn_q_w[layer]->tensor, bias_or_zero_(weights_.attn_q_b[layer], zero_bias_attn_q_));
    ops::linear(k_proj, attn_normed, weights_.attn_k_w[layer]->tensor, bias_or_zero_(weights_.attn_k_b[layer], zero_bias_attn_k_));
    ops::linear(v_proj, attn_normed, weights_.attn_v_w[layer]->tensor, bias_or_zero_(weights_.attn_v_b[layer], zero_bias_attn_v_));

    tensor_t q_3d = view_2d_to_3d_(q_proj, ntoken, nh, dh);
    tensor_t k_new_3d = view_2d_to_3d_(k_proj, ntoken, nkvh, dh);
    tensor_t v_new_3d = view_2d_to_3d_(v_proj, ntoken, nkvh, dh);

    tensor_t rope_q = slice_tokens_(ws.rope_q, ntoken);
    tensor_t rope_k = slice_tokens_(ws.rope_k, ntoken);
    ops::rope(rope_q, q_3d, pos_ids, meta_.theta);
    ops::rope(rope_k, k_new_3d, pos_ids, meta_.theta);

    tensor_t layer_k_cache = kv_layer_k_from_cache(runtime_.kv_cache.get(), runtime_.kv_layout, layer);
    tensor_t layer_v_cache = kv_layer_v_from_cache(runtime_.kv_cache.get(), runtime_.kv_layout, layer);
#ifdef ENABLE_NVIDIA_API
    if (device_type_ == LLAISYS_DEVICE_NVIDIA) {
        CHECK_ARGUMENT(attn_state.slot_mapping != nullptr, "Qwen2: missing slot_mapping");
        ops::cuda::reshape_and_cache(
            layer_k_cache, layer_v_cache, rope_k, v_new_3d, attn_state.slot_mapping);
    } else
#endif
    {
        const auto *slot_map =
            attn_state.paged_attention ? reinterpret_cast<const int32_t *>(attn_state.slot_mapping->data()) : nullptr;
        for (size_t i = 0; i < ntoken; ++i) {
            const int32_t slot = attn_state.paged_attention ? slot_map[i] : attn_state.slot_idxs[i];
            copy_token_into_cache_(layer_k_cache, slot, rope_k, i);
            copy_token_into_cache_(layer_v_cache, slot, v_new_3d, i);
        }
    }

    tensor_t attn_out = slice_tokens_(ws.attn_out, ntoken);
    if (attn_state.paged_attention) {
#ifdef ENABLE_NVIDIA_API
        if (device_type_ == LLAISYS_DEVICE_NVIDIA) {
            const ops::cuda::CommonAttentionMetadata prepared{
                reinterpret_cast<const int32_t *>(attn_state.q_seq_rows->data()),
                reinterpret_cast<const int32_t *>(attn_state.q_pos->data()),
                reinterpret_cast<const int32_t *>(attn_state.block_tables->data()),
                reinterpret_cast<const int32_t *>(attn_state.seq_lens->data()),
                static_cast<int32_t>(attn_state.seq_lens->shape()[0]),
            };
            ops::cuda::dispatch_attention_with_backend(
                attn_out,
                rope_q,
                layer_k_cache,
                layer_v_cache,
                prepared,
                paged_attn_backend_,
                attn_state.block_table_width,
                static_cast<int32_t>(runtime_.kv_block_size),
                scale);
        } else
#endif
        {
            ops::self_attention_paged(
                attn_out,
                rope_q,
                layer_k_cache,
                layer_v_cache,
                attn_state.q_seq_rows,
                attn_state.q_pos,
                attn_state.block_tables,
                attn_state.seq_lens,
                attn_state.block_table_width,
                static_cast<int32_t>(runtime_.kv_block_size),
                scale);
        }
    } else {
        const size_t kvlen = attn_state.used_slots.size();
        tensor_t k_full = gather_cache_by_slots_(layer_k_cache, attn_state.used_slots, kvlen, ws.k_ctx);
        tensor_t v_full = gather_cache_by_slots_(layer_v_cache, attn_state.used_slots, kvlen, ws.v_ctx);
        ops::self_attention_masked(attn_out, rope_q, k_full, v_full, attn_state.attn_mask, scale);
    }
    return attn_out->view({ntoken, nh * dh});
}

int32_t Qwen2Model::prepare_slot_attention_state_(size_t ntoken,
                                                  const tensor_t &seq_ids_t,
                                                  const tensor_t &pos_ids_host_t,
                                                  AttentionExecState *state) {
    ASSERT(state != nullptr, "Qwen2: attention state is null");
    const auto &ws = workspace_->view();
    const auto *seq_host = reinterpret_cast<const int64_t *>(seq_ids_t->data());
    const auto *pos_host = reinterpret_cast<const int64_t *>(pos_ids_host_t->data());

    std::vector<std::vector<int64_t>> seq_sets(ntoken);
    std::vector<int64_t> pos_values(ntoken, 0);
    for (size_t i = 0; i < ntoken; ++i) {
        seq_sets[i].push_back(seq_host[i]);
        pos_values[i] = pos_host[i];
    }

    runtime::kv_cache::KvUBatch kv_ubatch;
    kv_ubatch.seq_sets = seq_sets;
    kv_ubatch.pos_values = pos_values;
    runtime::kv_cache::KvSlotInfo slot_info;
    bool kv_applied = false;
    try {
        auto sinfos = runtime_.kv_cache->prepare({kv_ubatch});
        if (sinfos.empty()) {
            return 1;
        }
        slot_info = sinfos[0];
        const KvStatus apply_rc = runtime_.kv_cache->apply_ubatch(slot_info, kv_ubatch);
        if (apply_rc == KvStatus::OOM_SLOT) {
            return 1;
        }
        if (apply_rc != KvStatus::OK) {
            return -1;
        }
        kv_applied = true;
        CHECK_ARGUMENT(slot_info.n_stream() == 1, "Qwen2: unsupported slot_info stream shape");
        CHECK_ARGUMENT(slot_info.size() == ntoken, "Qwen2: slot_info size mismatch");
        state->slot_idxs = slot_info.idxs[0];

        runtime_.kv_cache->used_slots(&state->used_slots);
        const size_t kvlen = state->used_slots.size();
        CHECK_ARGUMENT(kvlen > 0, "Qwen2: no used KV slots");

        std::vector<uint8_t> host_mask(ntoken * kvlen, static_cast<uint8_t>(0));
        for (size_t i = 0; i < ntoken; ++i) {
            bool has_visible = false;
            for (size_t k = 0; k < kvlen; ++k) {
                const bool visible = runtime_.kv_cache->slot_visible_for(
                    state->used_slots[k], seq_sets[i].data(), static_cast<int32_t>(seq_sets[i].size()), pos_values[i]);
                host_mask[i * kvlen + k] = visible ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
                has_visible = has_visible || visible;
            }
            CHECK_ARGUMENT(has_visible, "Qwen2: empty attention context for token");
        }

        tensor_t attn_mask_flat = ws.attn_mask_flat->slice(0, 0, ntoken * kvlen);
        attn_mask_flat->load(host_mask.data());
        state->attn_mask = attn_mask_flat->view({ntoken, kvlen});
        state->paged_attention = false;
        state->block_table_width = 0;
        return 0;
    } catch (const std::invalid_argument &) {
        if (kv_applied) {
            runtime_.kv_cache->rollback_ubatch(slot_info, kv_ubatch);
        }
        return 1;
    } catch (...) {
        if (kv_applied) {
            runtime_.kv_cache->rollback_ubatch(slot_info, kv_ubatch);
        }
        return -2;
    }
}

int32_t Qwen2Model::validate_and_bind_block_attention_state_(const ::AttentionMetadata &attn,
                                                             size_t ntoken,
                                                             AttentionExecState *state) {
    ASSERT(state != nullptr, "Qwen2: attention state is null");
    auto validate_block_meta_tensor_1d = [this, ntoken](llaisysTensor_t handle, llaisysDataType_t dtype, const char *name) -> tensor_t {
        CHECK_ARGUMENT(handle != nullptr && handle->tensor != nullptr, "Qwen2: missing BLOCK attention metadata tensor");
        tensor_t t = handle->tensor;
        CHECK_ARGUMENT(t->ndim() == 1, "Qwen2: BLOCK metadata ndim mismatch");
        CHECK_ARGUMENT(t->dtype() == dtype, "Qwen2: BLOCK metadata dtype mismatch");
        CHECK_ARGUMENT(t->isContiguous(), "Qwen2: BLOCK metadata must be contiguous");
        CHECK_ARGUMENT(t->shape()[0] == ntoken, "Qwen2: BLOCK metadata token length mismatch");
        CHECK_ARGUMENT(t->deviceType() == device_type_ && t->deviceId() == device_id_, "Qwen2: BLOCK metadata device mismatch");
        (void)name;
        return t;
    };

    state->q_seq_rows = validate_block_meta_tensor_1d(attn.q_seq_rows, LLAISYS_DTYPE_I32, "q_seq_rows");
    state->q_pos = validate_block_meta_tensor_1d(attn.q_pos, LLAISYS_DTYPE_I32, "q_pos");
    state->slot_mapping = validate_block_meta_tensor_1d(attn.slot_mapping, LLAISYS_DTYPE_I32, "slot_mapping");

    CHECK_ARGUMENT(attn.context_lens != nullptr && attn.context_lens->tensor != nullptr, "Qwen2: missing context_lens");
    state->seq_lens = attn.context_lens->tensor;
    CHECK_ARGUMENT(state->seq_lens->ndim() == 1 && state->seq_lens->dtype() == LLAISYS_DTYPE_I32 && state->seq_lens->isContiguous(),
                   "Qwen2: context_lens must be contiguous i32 1D");
    CHECK_ARGUMENT(state->seq_lens->deviceType() == device_type_ && state->seq_lens->deviceId() == device_id_,
                   "Qwen2: context_lens device mismatch");

    state->block_table_width = attn.block_table_width;
    CHECK_ARGUMENT(state->block_table_width > 0, "Qwen2: invalid block_table_width");
    const int32_t n_batch_seq = static_cast<int32_t>(state->seq_lens->shape()[0]);
    CHECK_ARGUMENT(n_batch_seq > 0, "Qwen2: empty BLOCK batch");

    CHECK_ARGUMENT(attn.block_tables != nullptr && attn.block_tables->tensor != nullptr, "Qwen2: missing block_tables");
    state->block_tables = attn.block_tables->tensor;
    const size_t block_table_len = static_cast<size_t>(n_batch_seq) * static_cast<size_t>(state->block_table_width);
    CHECK_ARGUMENT(state->block_tables->ndim() == 1 && state->block_tables->dtype() == LLAISYS_DTYPE_I32 &&
                       state->block_tables->isContiguous() && state->block_tables->shape()[0] == block_table_len,
                   "Qwen2: invalid block_tables");
    CHECK_ARGUMENT(state->block_tables->deviceType() == device_type_ && state->block_tables->deviceId() == device_id_,
                   "Qwen2: block_tables device mismatch");

    state->paged_attention = true;
    return 0;
}

int32_t Qwen2Model::forward(const ::ModelForwardInput &input, ::ModelForwardOutput *output) {
    LLAISYS_NVTX_SCOPE("forward/main");
    auto fail = [](const char *msg) -> int32_t {
        std::cerr << "[Qwen2Model::forward] " << msg << std::endl;
        return -1;
    };
    if (input.input_ids == nullptr || input.pos_ids == nullptr) {
        return fail("missing input_ids/pos_ids");
    }

    try {
        const tensor_t input_ids = input.input_ids->tensor;
        const tensor_t pos_ids = input.pos_ids->tensor;
        if (input_ids == nullptr || pos_ids == nullptr) {
            return fail("null input_ids/pos_ids tensor");
        }
        if (input_ids->ndim() != 1 || input_ids->dtype() != LLAISYS_DTYPE_I64 || !input_ids->isContiguous()) {
            return fail("invalid input_ids");
        }
        if (pos_ids->ndim() != 1 || pos_ids->dtype() != LLAISYS_DTYPE_I64 || !pos_ids->isContiguous()) {
            return fail("invalid pos_ids");
        }
        if (input_ids->deviceType() != device_type_ || input_ids->deviceId() != device_id_) {
            return fail("input_ids device mismatch");
        }
        if (pos_ids->deviceType() != device_type_ || pos_ids->deviceId() != device_id_) {
            return fail("pos_ids device mismatch");
        }
        const size_t ntoken = input_ids->shape()[0];
        if (ntoken == 0 || pos_ids->shape()[0] != ntoken) {
            return fail("ntoken mismatch");
        }

        if (input.attention.mode != ATTENTION_MODE_SLOT && input.attention.mode != ATTENTION_MODE_BLOCK) {
            return fail("invalid attention mode");
        }
        if (!attention_mode_supported(device_type_, input.attention.mode)) {
            return fail("attention mode unsupported");
        }
        if ((runtime_.kv_layout == KvCacheLayout::BLOCK && input.attention.mode != ATTENTION_MODE_BLOCK) ||
            (runtime_.kv_layout == KvCacheLayout::SLOT && input.attention.mode != ATTENTION_MODE_SLOT)) {
            return fail("attention mode != kv layout");
        }

        tensor_t pos_ids_host_t = nullptr;
        tensor_t seq_ids_t = nullptr;
        if (input.attention.mode == ATTENTION_MODE_SLOT) {
            if (input.attention.pos_ids_host == nullptr || input.attention.pos_ids_host->tensor == nullptr) {
                return fail("missing pos_ids_host");
            }
            pos_ids_host_t = input.attention.pos_ids_host->tensor;
            if (pos_ids_host_t == nullptr) {
                return fail("null pos_ids_host tensor");
            }
            if (pos_ids_host_t->ndim() != 1 || pos_ids_host_t->dtype() != LLAISYS_DTYPE_I64 || !pos_ids_host_t->isContiguous() ||
                pos_ids_host_t->shape()[0] != ntoken || pos_ids_host_t->deviceType() != LLAISYS_DEVICE_CPU) {
                return fail("invalid pos_ids_host");
            }

            if (input.attention.seq_ids == nullptr || input.attention.seq_ids->tensor == nullptr) {
                return fail("missing seq_ids");
            }
            seq_ids_t = input.attention.seq_ids->tensor;
            if (seq_ids_t == nullptr) {
                return fail("null seq_ids tensor");
            }
            if (seq_ids_t->ndim() != 1 || seq_ids_t->dtype() != LLAISYS_DTYPE_I64 || !seq_ids_t->isContiguous() ||
                seq_ids_t->shape()[0] != ntoken || seq_ids_t->deviceType() != LLAISYS_DEVICE_CPU) {
                return fail("invalid seq_ids tensor");
            }
        }

        tensor_t logits_mask_t = nullptr;
        const int8_t *logits_host = nullptr;
        if (input.logits_mask == nullptr || input.logits_mask->tensor == nullptr) {
            return fail("missing logits_mask");
        }
        logits_mask_t = input.logits_mask->tensor;
        if (logits_mask_t->deviceType() != LLAISYS_DEVICE_CPU) {
            return fail("logits_mask device");
        }
        if (logits_mask_t->ndim() != 1 || logits_mask_t->dtype() != LLAISYS_DTYPE_I8 || !logits_mask_t->isContiguous() ||
            logits_mask_t->shape()[0] != ntoken) {
            return fail("invalid logits_mask");
        }
        logits_host = reinterpret_cast<const int8_t *>(logits_mask_t->data());

        ASSERT(runtime_.output != nullptr, "Qwen2: output buffer is null");
        runtime_.output->clear();
        runtime_.output->reserve_rows(ntoken);
        step_logits_ = nullptr;

        validate_or_die_();
        ASSERT(runtime_.kv_cache != nullptr, "Qwen2: kv_cache is null");
        ensure_workspace_(ntoken);
        const auto &ws = workspace_->view();

        tensor_t hidden = slice_tokens_(ws.hidden, ntoken);
        {
            LLAISYS_NVTX_SCOPE("forward/embedding");
            ops::embedding(hidden, input_ids, weights_.in_embed->tensor);
        }

        std::vector<int32_t> collected_rows{};
        collected_rows.reserve(ntoken);
        for (size_t i = 0; i < ntoken; ++i) {
            if (logits_host[i] != 0) {
                collected_rows.push_back(static_cast<int32_t>(i));
            }
        }

        AttentionExecState attn_state{};
        attn_state = AttentionExecState{};
        int32_t attn_rc = 0;
        if (input.attention.mode == ATTENTION_MODE_SLOT) {
            attn_rc = prepare_slot_attention_state_(ntoken, seq_ids_t, pos_ids_host_t, &attn_state);
        } else {
            attn_rc = validate_and_bind_block_attention_state_(input.attention, ntoken, &attn_state);
        }
        if (attn_rc != 0) {
            std::cerr << "[Qwen2Model::forward] prepare attention state rc=" << attn_rc << std::endl;
            return attn_rc;
        }

        for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
            tensor_t attn_normed = slice_tokens_(ws.normed, ntoken);
            ops::rms_norm(attn_normed, hidden, weights_.attn_norm_w[layer]->tensor, meta_.epsilon);

            tensor_t attn_out_2d = run_attention_layer_(
                layer,
                ntoken,
                attn_normed,
                pos_ids,
                attn_state);
            tensor_t attn_proj = slice_tokens_(ws.attn_proj, ntoken);
            ops::linear(attn_proj, attn_out_2d, weights_.attn_o_w[layer]->tensor, zero_bias_attn_o_);
            ops::add(hidden, hidden, attn_proj);

            tensor_t mlp_normed = slice_tokens_(ws.mlp_normed, ntoken);
            ops::rms_norm(mlp_normed, hidden, weights_.mlp_norm_w[layer]->tensor, meta_.epsilon);

            tensor_t gate = slice_tokens_(ws.gate, ntoken);
            tensor_t up = slice_tokens_(ws.up, ntoken);
            ops::linear(gate, mlp_normed, weights_.mlp_gate_w[layer]->tensor, zero_bias_mlp_gate_);
            ops::linear(up, mlp_normed, weights_.mlp_up_w[layer]->tensor, zero_bias_mlp_up_);

            tensor_t swiglu = slice_tokens_(ws.swiglu, ntoken);
            ops::swiglu(swiglu, gate, up);

            tensor_t down = slice_tokens_(ws.down, ntoken);
            ops::linear(down, swiglu, weights_.mlp_down_w[layer]->tensor, zero_bias_mlp_down_);
            ops::add(hidden, hidden, down);
        }

        if (!collected_rows.empty()) {
            tensor_t final_normed = slice_tokens_(ws.normed, ntoken);
            ops::rms_norm(final_normed, hidden, weights_.out_norm_w->tensor, meta_.epsilon);

            const size_t n_outputs = collected_rows.size();
            tensor_t logits = slice_tokens_(ws.logits, n_outputs);
            if (n_outputs == ntoken) {
                ops::linear(logits, final_normed, weights_.out_embed->tensor, zero_bias_logits_);
            } else {
                std::vector<int64_t> output_rows(n_outputs, 0);
                for (size_t i = 0; i < n_outputs; ++i) {
                    output_rows[i] = static_cast<int64_t>(collected_rows[i]);
                }
                tensor_t output_rows_t = slice_tokens_(ws.argmax_idx, n_outputs);
                output_rows_t->load(output_rows.data());
                tensor_t selected_hidden = slice_tokens_(ws.hidden, n_outputs);
                ops::embedding(selected_hidden, output_rows_t, final_normed);
                ops::linear(logits, selected_hidden, weights_.out_embed->tensor, zero_bias_logits_);
            }
            step_logits_ = logits;

            tensor_t max_idx = slice_tokens_(ws.argmax_idx, n_outputs);
            tensor_t max_val = slice_tokens_(ws.argmax_val, n_outputs);
            ops::argmax_rows(max_idx, max_val, logits);

            std::vector<int64_t> sampled_host(n_outputs, 0);
            runtime_copy_bytes(
                LLAISYS_DEVICE_CPU,
                reinterpret_cast<std::byte *>(sampled_host.data()),
                device_type_,
                max_idx->data(),
                n_outputs * sizeof(int64_t));

            for (size_t i = 0; i < n_outputs; ++i) {
                runtime_.output->append_output_id(static_cast<int64_t>(collected_rows[i]));
                runtime_.output->append_sampled_id(sampled_host[i]);
            }
        }

        if (output == nullptr) {
            return 0;
        }

        const int32_t n_outputs = runtime_.output->n_outputs();
        output->n_outputs = n_outputs;
        if (output->output_ids != nullptr) {
            const int64_t *output_ids = runtime_.output->output_ids();
            if (n_outputs > 0 && output_ids == nullptr) {
                return -2;
            }
            tensor_t out_ids = output->output_ids->tensor;
            if (out_ids == nullptr || out_ids->dtype() != LLAISYS_DTYPE_I64 || out_ids->ndim() != 1 || !out_ids->isContiguous() ||
                out_ids->shape()[0] < static_cast<size_t>(n_outputs)) {
                std::cerr << "[Qwen2Model::forward] invalid output_ids tensor" << std::endl;
                return -1;
            }
            if (n_outputs > 0) {
                if (out_ids->shape()[0] > static_cast<size_t>(n_outputs)) {
                    out_ids = out_ids->slice(0, 0, static_cast<size_t>(n_outputs));
                }
                out_ids->load(output_ids);
            }
        }
        if (output->logits == nullptr) {
            std::cerr << "[Qwen2Model::forward] missing output logits handle" << std::endl;
            return -1;
        }
        output->logits->tensor = step_logits_;
        return 0;
    } catch (const std::invalid_argument &e) {
        std::cerr << "[Qwen2Model::forward] invalid_argument: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -2;
    }
}

tensor_t Qwen2Model::kv_layer_k(size_t layer) const {
    ASSERT(runtime_.kv_cache != nullptr, "Qwen2: kv_cache is null");
    return kv_layer_k_from_cache(runtime_.kv_cache.get(), runtime_.kv_layout, layer);
}

tensor_t Qwen2Model::kv_layer_v(size_t layer) const {
    ASSERT(runtime_.kv_cache != nullptr, "Qwen2: kv_cache is null");
    return kv_layer_v_from_cache(runtime_.kv_cache.get(), runtime_.kv_layout, layer);
}

void Qwen2Model::destroy_weights_() {
    std::vector<llaisysTensor_t *> slots{};
    slots.reserve(3 + meta_.nlayer * 12);
    slots.push_back(&weights_.in_embed);
    slots.push_back(&weights_.out_embed);
    slots.push_back(&weights_.out_norm_w);
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        slots.push_back(&weights_.attn_norm_w[i]);
        slots.push_back(&weights_.attn_q_w[i]);
        slots.push_back(&weights_.attn_q_b[i]);
        slots.push_back(&weights_.attn_k_w[i]);
        slots.push_back(&weights_.attn_k_b[i]);
        slots.push_back(&weights_.attn_v_w[i]);
        slots.push_back(&weights_.attn_v_b[i]);
        slots.push_back(&weights_.attn_o_w[i]);
        slots.push_back(&weights_.mlp_norm_w[i]);
        slots.push_back(&weights_.mlp_gate_w[i]);
        slots.push_back(&weights_.mlp_up_w[i]);
        slots.push_back(&weights_.mlp_down_w[i]);
    }
    runtime::weights::destroy_unique(slots);
}

} // namespace llaisys::models::qwen2
