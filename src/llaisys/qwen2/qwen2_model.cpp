#include "qwen2_model.hpp"
#include "llaisys/models/model.h"
#include "../kv_cache/paged_kv.hpp"
#include "../../core/llaisys_core.hpp"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <chrono>
#ifdef ENABLE_NCCL_API
#include <cuda_runtime_api.h>
#include <nccl.h>
#endif

namespace llaisys::models::qwen2 {

namespace {

using KvStatus = llaisys::kv_cache::KvStatus;

bool attention_phase_valid(int32_t phase) {
    return phase == ATTENTION_PHASE_PREFILL || phase == ATTENTION_PHASE_DECODE;
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
        if (paged_attn_backend_ == ops::cuda::PagedAttentionBackend::CUDNN) {
            cudnn_runtime_state_ = ops::cuda::create_cudnn_runtime_state();
            CHECK_ARGUMENT(cudnn_runtime_state_ != nullptr, "Qwen2: failed to create CUDNN runtime state");
        }
    }
#endif

    check_meta_invariants_();
    init_weight_slots_();
}

int Qwen2Model::bind_parallel_context(int32_t tp_size,
                                      int32_t tp_rank,
                                      int32_t local_rank,
                                      const int *device_ids,
                                      int32_t ndevice,
                                      void *nccl_comm) {
    const int32_t resolved_tp_size = std::max<int32_t>(1, tp_size);
    const int32_t resolved_tp_rank = std::max<int32_t>(0, tp_rank);
    const int32_t resolved_local_rank = std::max<int32_t>(0, local_rank);
    if (resolved_tp_rank >= resolved_tp_size) {
        std::fprintf(stderr,
                     "[ERROR] Qwen2 bind_parallel_context invalid rank: tp_size=%d tp_rank=%d\n",
                     resolved_tp_size,
                     resolved_tp_rank);
        return -1;
    }
    if (resolved_tp_size > 1 && (ndevice != resolved_tp_size || device_ids == nullptr)) {
        std::fprintf(stderr,
                     "[ERROR] Qwen2 bind_parallel_context invalid devices: tp_size=%d ndevice=%d\n",
                     resolved_tp_size,
                     ndevice);
        return -1;
    }

    tp_size_ = resolved_tp_size;
    tp_rank_ = resolved_tp_rank;
    local_rank_ = resolved_local_rank;
    tp_device_ids_.clear();
    if (device_ids != nullptr && ndevice > 0) {
        tp_device_ids_.reserve(static_cast<size_t>(ndevice));
        for (int32_t i = 0; i < ndevice; ++i) {
            tp_device_ids_.push_back(device_ids[i]);
        }
    }
    if (tp_size_ > 1) {
        device_id_ = tp_device_ids_[static_cast<size_t>(tp_rank_)];
    }
    tp_nh_local_ = (tp_size_ > 1) ? (meta_.nh / static_cast<size_t>(tp_size_)) : meta_.nh;
    tp_nkvh_local_ = (tp_size_ > 1) ? (meta_.nkvh / static_cast<size_t>(tp_size_)) : meta_.nkvh;
    tp_di_local_ = (tp_size_ > 1) ? (meta_.di / static_cast<size_t>(tp_size_)) : meta_.di;
#ifdef ENABLE_NCCL_API
    tp_nccl_comm_ = nccl_comm;
#else
    (void)nccl_comm;
#endif
    parallel_bound_ = true;

    check_meta_invariants_();
#ifndef ENABLE_NCCL_API
    if (tp_size_ > 1) {
        std::fprintf(stderr, "[ERROR] Qwen2 TP requires ENABLE_NCCL_API build\n");
        return -1;
    }
#endif

    std::fprintf(stderr,
                 "[qwen2.parallel] bound tp_size=%d tp_rank=%d local_rank=%d ndevice=%d nh_local=%zu nkvh_local=%zu di_local=%zu\n",
                 tp_size_,
                 tp_rank_,
                 local_rank_,
                 static_cast<int>(tp_device_ids_.size()),
                 tp_nh_local_,
                 tp_nkvh_local_,
                 tp_di_local_);
    return 0;
}

Qwen2Model::~Qwen2Model() {
#ifdef ENABLE_NVIDIA_API
    ops::cuda::destroy_cudnn_runtime_state(cudnn_runtime_state_);
    cudnn_runtime_state_ = nullptr;
#endif
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

bool Qwen2Model::bind_kv_state_handle(const void *kv_state_handle) noexcept {
    if (kv_state_handle == nullptr) {
        return false;
    }
    if (bound_kv_state_handle_ == nullptr) {
        bound_kv_state_handle_ = kv_state_handle;
        return true;
    }
    return bound_kv_state_handle_ == kv_state_handle;
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
    const size_t nkvh_local = tp_nkvh_local_ > 0 ? tp_nkvh_local_ : meta_.nkvh;
    runtime_.kv_cache = std::make_unique<kv_cache::PagedKvImpl>(kv_capacity, 1, runtime_.kv_block_size);
    runtime_.kv_cache->init_storage(meta_.nlayer, nkvh_local, meta_.dh, meta_.dtype, device_type_, device_id_);
    runtime_.kv_peak_used_tokens = 0;
}

int Qwen2Model::configure_runtime(size_t kv_block_size,
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

    runtime_.kv_block_size = kv_block_size;
    runtime_.max_model_len = max_len;
    runtime_.kv_cache_capacity_tokens = kv_capacity;
    runtime_.kv_cache.reset();
    workspace_.reset();
    init_runtime_state_();
    return 0;
}

#ifdef ENABLE_NCCL_API
int Qwen2Model::tp_allreduce_sum_(const tensor_t &tensor) const {
    if (tp_size_ <= 1) {
        return 0;
    }
    if (tensor == nullptr || tp_nccl_comm_ == nullptr) {
        return -1;
    }
    if (tensor->deviceType() != LLAISYS_DEVICE_NVIDIA || tensor->deviceId() != device_id_ || !tensor->isContiguous()) {
        return -1;
    }
    ncclDataType_t nccl_dtype{};
    switch (tensor->dtype()) {
    case LLAISYS_DTYPE_F16:
        nccl_dtype = ncclHalf;
        break;
    case LLAISYS_DTYPE_BF16:
        nccl_dtype = ncclBfloat16;
        break;
    case LLAISYS_DTYPE_F32:
        nccl_dtype = ncclFloat32;
        break;
    default:
        std::fprintf(stderr, "[ERROR] TP allreduce unsupported dtype=%d\n", static_cast<int>(tensor->dtype()));
        return -1;
    }
    llaisys::core::context().setDevice(tensor->deviceType(), tensor->deviceId());
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    const size_t count = tensor->numel();
    const ncclResult_t rc =
        ncclAllReduce(tensor->data(), tensor->data(), count, nccl_dtype, ncclSum, reinterpret_cast<ncclComm_t>(tp_nccl_comm_), stream);
    if (rc != ncclSuccess) {
        std::fprintf(stderr, "[ERROR] ncclAllReduce failed: %s\n", ncclGetErrorString(rc));
        return -1;
    }
    return 0;
}
#else
int Qwen2Model::tp_allreduce_sum_(const tensor_t &) const {
    return -1;
}
#endif

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
    CHECK_ARGUMENT(tp_size_ >= 1, "Qwen2: tp_size must be >= 1");
    if (tp_size_ > 1) {
        CHECK_ARGUMENT((meta_.nh % static_cast<size_t>(tp_size_)) == 0, "Qwen2: nh must be divisible by tp_size");
        CHECK_ARGUMENT((meta_.nkvh % static_cast<size_t>(tp_size_)) == 0, "Qwen2: nkvh must be divisible by tp_size");
        CHECK_ARGUMENT((meta_.di % static_cast<size_t>(tp_size_)) == 0, "Qwen2: di must be divisible by tp_size");
        CHECK_ARGUMENT(tp_nkvh_local_ > 0, "Qwen2: nkvh_local must be > 0");
        CHECK_ARGUMENT((tp_nh_local_ % tp_nkvh_local_) == 0, "Qwen2: nh_local must be divisible by nkvh_local");
    }

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
    (void)zero_bias;
    if (handle) {
        return handle->tensor;
    }
    return nullptr;
}

void Qwen2Model::validate_or_die_() {
    if (validated_) {
        return;
    }

    check_meta_invariants_();

    const size_t hs = meta_.hs;
    const size_t nh = tp_nh_local_ > 0 ? tp_nh_local_ : meta_.nh;
    const size_t nkvh = tp_nkvh_local_ > 0 ? tp_nkvh_local_ : meta_.nkvh;
    const size_t dh = meta_.dh;
    const size_t di = tp_di_local_ > 0 ? tp_di_local_ : meta_.di;
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
        const size_t nh_local = tp_nh_local_ > 0 ? tp_nh_local_ : meta_.nh;
        const size_t di_local = tp_di_local_ > 0 ? tp_di_local_ : meta_.di;
        workspace_ = std::make_unique<workspace::Qwen2Workspace>(
            meta_.hs,
            nh_local,
            tp_nkvh_local_ > 0 ? tp_nkvh_local_ : meta_.nkvh,
            meta_.dh,
            di_local,
            meta_.voc,
            kv_capacity,
            meta_.dtype,
            device_type_,
            device_id_);
    }
    workspace_->reserve(ntoken);
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
    if (stride_bytes == 0) {
        return;
    }
    const llaisysMemcpyKind_t kind = llaisys::utils::infer_memcpy_kind(cache->deviceType(), src->deviceType());
    utils::NvtxScope nvtx_scope(utils::nvtx_memcpy_tag(kind, false));
    const auto *api = core::context().runtime().api();
    api->memcpy_sync(dst, src_ptr, stride_bytes, kind);
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

tensor_t Qwen2Model::run_block_attention_layer_(size_t layer,
                                                size_t ntoken,
                                                const tensor_t &attn_normed,
                                                const tensor_t &pos_ids,
                                                const AttentionExecState &attn_state) {
    LLAISYS_NVTX_SCOPE("forward/attention_layer_block");
    const auto &ws = workspace_->view();
    const size_t nh = tp_nh_local_ > 0 ? tp_nh_local_ : meta_.nh;
    const size_t nkvh = tp_nkvh_local_ > 0 ? tp_nkvh_local_ : meta_.nkvh;
    const size_t dh = meta_.dh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    tensor_t q_proj = slice_tokens_(ws.q_proj, ntoken);
    tensor_t k_proj = slice_tokens_(ws.k_proj, ntoken);
    tensor_t v_proj = slice_tokens_(ws.v_proj, ntoken);
    {
        LLAISYS_NVTX_SCOPE("forward/attn/qkv_proj");
        ops::linear(q_proj, attn_normed, weights_.attn_q_w[layer]->tensor, bias_or_zero_(weights_.attn_q_b[layer], zero_bias_attn_q_));
        ops::linear(k_proj, attn_normed, weights_.attn_k_w[layer]->tensor, bias_or_zero_(weights_.attn_k_b[layer], zero_bias_attn_k_));
        ops::linear(v_proj, attn_normed, weights_.attn_v_w[layer]->tensor, bias_or_zero_(weights_.attn_v_b[layer], zero_bias_attn_v_));
    }

    tensor_t q_3d = view_2d_to_3d_(q_proj, ntoken, nh, dh);
    tensor_t k_new_3d = view_2d_to_3d_(k_proj, ntoken, nkvh, dh);
    tensor_t v_new_3d = view_2d_to_3d_(v_proj, ntoken, nkvh, dh);

    tensor_t rope_q = slice_tokens_(ws.rope_q, ntoken);
    tensor_t rope_k = slice_tokens_(ws.rope_k, ntoken);
    {
        LLAISYS_NVTX_SCOPE("forward/attn/rope");
        ops::rope(rope_q, q_3d, pos_ids, meta_.theta);
        ops::rope(rope_k, k_new_3d, pos_ids, meta_.theta);
    }

    CHECK_ARGUMENT(runtime_.kv_cache != nullptr, "Qwen2: kv_cache is not initialized");
    tensor_t layer_k_cache = runtime_.kv_cache->layer_k(layer);
    tensor_t layer_v_cache = runtime_.kv_cache->layer_v(layer);
    CHECK_ARGUMENT(attn_state.slot_mapping != nullptr, "Qwen2: missing slot_mapping");
    {
        LLAISYS_NVTX_SCOPE("forward/attn/cache_update");
#ifdef ENABLE_NVIDIA_API
        if (device_type_ == LLAISYS_DEVICE_NVIDIA) {
            ops::cuda::reshape_and_cache(
                layer_k_cache, layer_v_cache, rope_k, v_new_3d, attn_state.slot_mapping);
        } else
#endif
        {
            const auto *slot_map = reinterpret_cast<const int32_t *>(attn_state.slot_mapping->data());
            for (size_t i = 0; i < ntoken; ++i) {
                const int32_t slot = slot_map[i];
                CHECK_ARGUMENT(slot >= 0, "Qwen2: invalid slot_mapping value");
                copy_token_into_cache_(layer_k_cache, slot, rope_k, i);
                copy_token_into_cache_(layer_v_cache, slot, v_new_3d, i);
            }
        }
    }

    tensor_t attn_out = slice_tokens_(ws.attn_out, ntoken);
    {
        LLAISYS_NVTX_SCOPE("forward/attn/core");
#ifdef ENABLE_NVIDIA_API
        if (device_type_ == LLAISYS_DEVICE_NVIDIA) {
            ops::cuda::CudnnRuntimeState *cudnn_state = nullptr;
            if (attn_state.use_cudnn_backend) {
                cudnn_state = cudnn_runtime_state_;
                CHECK_ARGUMENT(cudnn_state != nullptr, "Qwen2: missing CUDNN runtime state");
            }
            const ops::cuda::CommonAttentionMetadata prepared{
                reinterpret_cast<const int32_t *>(attn_state.cu_seqlens_q != nullptr ? attn_state.cu_seqlens_q->data() : nullptr),
                reinterpret_cast<const int32_t *>(attn_state.cu_seqlens_k != nullptr ? attn_state.cu_seqlens_k->data() : nullptr),
                reinterpret_cast<const int32_t *>(attn_state.block_tables != nullptr ? attn_state.block_tables->data() : nullptr),
                reinterpret_cast<const int32_t *>(attn_state.slot_mapping != nullptr ? attn_state.slot_mapping->data() : nullptr),
                cudnn_state,
                reinterpret_cast<const int32_t *>(attn_state.cudnn_seq_lens_q != nullptr ? attn_state.cudnn_seq_lens_q->data() : nullptr),
                reinterpret_cast<const int32_t *>(attn_state.cudnn_seq_lens_kv != nullptr ? attn_state.cudnn_seq_lens_kv->data() : nullptr),
                reinterpret_cast<const int32_t *>(attn_state.cudnn_page_table != nullptr ? attn_state.cudnn_page_table->data() : nullptr),
                reinterpret_cast<const int32_t *>(
                    attn_state.cudnn_qo_ragged_offset != nullptr ? attn_state.cudnn_qo_ragged_offset->data() : nullptr),
                int(attn_state.cudnn_b_exec),
                int(attn_state.cudnn_warmup_b),
                int(attn_state.n_batch_seq),
                int(attn_state.max_seqlen_q),
                int(attn_state.max_seqlen_k),
                static_cast<ops::cuda::AttentionPhase>(attn_state.attention_phase),
            };
            const auto backend =
                attn_state.use_cudnn_backend ? paged_attn_backend_ : ops::cuda::PagedAttentionBackend::NATIVE;
            ops::cuda::self_attention_paged_with_backend(
                attn_out,
                rope_q,
                layer_k_cache,
                layer_v_cache,
                prepared,
                backend,
                attn_state.block_table_width,
                static_cast<int32_t>(runtime_.kv_block_size),
                scale);
        } else
#endif
        {
            CHECK_ARGUMENT(attn_state.cu_seqlens_q != nullptr && attn_state.cu_seqlens_k != nullptr && attn_state.block_tables != nullptr,
                           "Qwen2: missing native BLOCK metadata");
            ops::self_attention_paged(
                attn_out,
                rope_q,
                layer_k_cache,
                layer_v_cache,
                attn_state.cu_seqlens_q,
                attn_state.cu_seqlens_k,
                attn_state.block_tables,
                attn_state.slot_mapping,
                attn_state.max_seqlen_q,
                attn_state.max_seqlen_k,
                attn_state.block_table_width,
                static_cast<int32_t>(runtime_.kv_block_size),
                scale);
        }
    }
    return attn_out->view({ntoken, nh * dh});
}

int32_t Qwen2Model::validate_and_bind_block_attention_state_(const ::AttentionMetadata &attn,
                                                             size_t ntoken,
                                                             AttentionExecState *state) {
    LLAISYS_NVTX_SCOPE("attn_meta/block/validate");
    ASSERT(state != nullptr, "Qwen2: attention state is null");
    auto validate_block_meta_tensor_1d = [this](llaisysTensor_t handle, llaisysDataType_t dtype, const char *name) -> tensor_t {
        CHECK_ARGUMENT(handle != nullptr && handle->tensor != nullptr, "Qwen2: missing BLOCK attention metadata tensor");
        tensor_t t = handle->tensor;
        CHECK_ARGUMENT(t->ndim() == 1, "Qwen2: BLOCK metadata ndim mismatch");
        CHECK_ARGUMENT(t->dtype() == dtype, "Qwen2: BLOCK metadata dtype mismatch");
        CHECK_ARGUMENT(t->isContiguous(), "Qwen2: BLOCK metadata must be contiguous");
        CHECK_ARGUMENT(t->deviceType() == device_type_ && t->deviceId() == device_id_, "Qwen2: BLOCK metadata device mismatch");
        (void)name;
        return t;
    };

    bool want_cudnn_backend = false;
#ifdef ENABLE_NVIDIA_API
    want_cudnn_backend = (device_type_ == LLAISYS_DEVICE_NVIDIA) &&
                         (paged_attn_backend_ == ops::cuda::PagedAttentionBackend::CUDNN);
#endif
    const auto has_tensor = [](llaisysTensor_t handle) -> bool {
        return handle != nullptr && handle->tensor != nullptr;
    };
    const bool has_cudnn_meta =
        has_tensor(attn.cudnn_seq_lens_q) &&
        has_tensor(attn.cudnn_seq_lens_kv) &&
        has_tensor(attn.cudnn_page_table) &&
        (attn.phase != ATTENTION_PHASE_PREFILL || has_tensor(attn.cudnn_qo_ragged_offset));
    if (want_cudnn_backend) {
        CHECK_ARGUMENT(has_cudnn_meta, "Qwen2: CUDNN backend requested but missing CUDNN BLOCK metadata");
#ifdef ENABLE_NVIDIA_API
        CHECK_ARGUMENT(cudnn_runtime_state_ != nullptr, "Qwen2: CUDNN runtime state is not initialized");
#endif
    }
    const bool use_cudnn_backend = want_cudnn_backend;
    state->use_cudnn_backend = use_cudnn_backend;

    state->max_seqlen_q = int(attn.max_seqlen_q);
    state->max_seqlen_k = int(attn.max_seqlen_k);
    CHECK_ARGUMENT(state->max_seqlen_q > 0, "Qwen2: invalid max_seqlen_q");
    CHECK_ARGUMENT(state->max_seqlen_k > 0, "Qwen2: invalid max_seqlen_k");
    CHECK_ARGUMENT(attention_phase_valid(attn.phase), "Qwen2: invalid attention phase");
    state->attention_phase = int(attn.phase);
    if (state->attention_phase == ATTENTION_PHASE_DECODE) {
        CHECK_ARGUMENT(state->max_seqlen_q == 1, "Qwen2: decode BLOCK phase requires max_seqlen_q == 1");
    }

    state->slot_mapping = validate_block_meta_tensor_1d(attn.slot_mapping, LLAISYS_DTYPE_I32, "slot_mapping");
    CHECK_ARGUMENT(state->slot_mapping->shape()[0] == ntoken, "Qwen2: slot_mapping token length mismatch");

    state->block_table_width = attn.block_table_width;
    CHECK_ARGUMENT(state->block_table_width > 0, "Qwen2: invalid block_table_width");

    CHECK_ARGUMENT(attn.block_tables != nullptr && attn.block_tables->tensor != nullptr, "Qwen2: missing block_tables");
    state->block_tables = attn.block_tables->tensor;
    CHECK_ARGUMENT(state->block_tables->ndim() == 1 && state->block_tables->dtype() == LLAISYS_DTYPE_I32 &&
                       state->block_tables->isContiguous(),
                   "Qwen2: invalid block_tables");
    CHECK_ARGUMENT((state->block_tables->shape()[0] % static_cast<size_t>(state->block_table_width)) == 0,
                   "Qwen2: block_tables length not divisible by block_table_width");
    state->n_batch_seq = static_cast<int32_t>(state->block_tables->shape()[0] / static_cast<size_t>(state->block_table_width));
    CHECK_ARGUMENT(state->n_batch_seq > 0, "Qwen2: empty BLOCK batch");
    const size_t block_table_len = static_cast<size_t>(state->n_batch_seq) * static_cast<size_t>(state->block_table_width);
    CHECK_ARGUMENT(state->block_tables->ndim() == 1 && state->block_tables->dtype() == LLAISYS_DTYPE_I32 &&
                       state->block_tables->isContiguous() && state->block_tables->shape()[0] == block_table_len,
                   "Qwen2: invalid block_tables");
    CHECK_ARGUMENT(state->block_tables->deviceType() == device_type_ && state->block_tables->deviceId() == device_id_,
                   "Qwen2: block_tables device mismatch");

    if (use_cudnn_backend) {
        state->cu_seqlens_q = nullptr;
        state->cu_seqlens_k = nullptr;
        state->cudnn_seq_lens_q = validate_block_meta_tensor_1d(attn.cudnn_seq_lens_q, LLAISYS_DTYPE_I32, "cudnn_seq_lens_q");
        state->cudnn_seq_lens_kv = validate_block_meta_tensor_1d(attn.cudnn_seq_lens_kv, LLAISYS_DTYPE_I32, "cudnn_seq_lens_kv");
        state->cudnn_page_table = validate_block_meta_tensor_1d(attn.cudnn_page_table, LLAISYS_DTYPE_I32, "cudnn_page_table");
        state->cudnn_qo_ragged_offset = nullptr;
        state->cudnn_b_exec = int(attn.cudnn_b_exec);
        state->cudnn_warmup_b = std::max<int32_t>(state->cudnn_b_exec, int(attn.cudnn_warmup_b));
        CHECK_ARGUMENT(state->cudnn_b_exec > 0, "Qwen2: invalid cudnn_b_exec");
        CHECK_ARGUMENT(state->cudnn_seq_lens_q->shape()[0] == static_cast<size_t>(state->cudnn_b_exec),
                       "Qwen2: cudnn_seq_lens_q size mismatch");
        CHECK_ARGUMENT(state->cudnn_seq_lens_kv->shape()[0] == static_cast<size_t>(state->cudnn_b_exec),
                       "Qwen2: cudnn_seq_lens_kv size mismatch");
        CHECK_ARGUMENT(
            state->cudnn_page_table->shape()[0] ==
                static_cast<size_t>(state->cudnn_b_exec) * static_cast<size_t>(state->block_table_width),
            "Qwen2: cudnn_page_table size mismatch");
        if (state->attention_phase == ATTENTION_PHASE_PREFILL) {
            state->cudnn_qo_ragged_offset =
                validate_block_meta_tensor_1d(attn.cudnn_qo_ragged_offset, LLAISYS_DTYPE_I32, "cudnn_qo_ragged_offset");
            CHECK_ARGUMENT(
                state->cudnn_qo_ragged_offset->shape()[0] == static_cast<size_t>(state->cudnn_b_exec + 1),
                "Qwen2: cudnn_qo_ragged_offset size mismatch");
        }
    } else {
        state->cudnn_seq_lens_q = nullptr;
        state->cudnn_seq_lens_kv = nullptr;
        state->cudnn_page_table = nullptr;
        state->cudnn_qo_ragged_offset = nullptr;
        state->cudnn_b_exec = 0;
        state->cudnn_warmup_b = 0;

        state->cu_seqlens_q = validate_block_meta_tensor_1d(attn.cu_seqlens_q, LLAISYS_DTYPE_I32, "cu_seqlens_q");
        CHECK_ARGUMENT(state->cu_seqlens_q->shape()[0] >= 2, "Qwen2: cu_seqlens_q must have at least 2 elements");
        const int32_t n_batch_seq_from_cu = static_cast<int32_t>(state->cu_seqlens_q->shape()[0]) - 1;
        CHECK_ARGUMENT(n_batch_seq_from_cu == state->n_batch_seq, "Qwen2: cu_seqlens_q batch size mismatch with block_tables");

        state->cu_seqlens_k = validate_block_meta_tensor_1d(attn.cu_seqlens_k, LLAISYS_DTYPE_I32, "cu_seqlens_k");
        CHECK_ARGUMENT(state->cu_seqlens_k->shape()[0] == static_cast<size_t>(n_batch_seq_from_cu + 1),
                       "Qwen2: cu_seqlens_k size mismatch");
    }

    state->paged_attention = true;
    return 0;
}

int32_t Qwen2Model::forward(const ::ModelForwardInput &input, ::ModelForwardOutput *output) {
    LLAISYS_NVTX_SCOPE("forward/main");
    auto fail = [](const char *msg) -> int32_t {
        std::cerr << "[Qwen2Model::forward] " << msg << std::endl;
        return -1;
    };
    try {
        if (!parallel_bound_) {
            return fail("parallel context is not bound");
        }
        llaisys::core::context().setDevice(device_type_, device_id_);
        if (input.input_ids == nullptr || input.pos_ids == nullptr) {
            return fail("missing input_ids/pos_ids");
        }
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

        if (!attention_phase_valid(input.attention.phase)) {
            return fail("invalid attention phase");
        }

        if (input.logits_indices == nullptr || input.logits_indices->tensor == nullptr) {
            return fail("missing logits_indices");
        }
        tensor_t logits_indices = input.logits_indices->tensor;
        if (logits_indices->ndim() != 1 || logits_indices->dtype() != LLAISYS_DTYPE_I64 || !logits_indices->isContiguous()) {
            return fail("invalid logits_indices");
        }
        if (logits_indices->deviceType() != device_type_ || logits_indices->deviceId() != device_id_) {
            return fail("logits_indices device mismatch");
        }
        const size_t n_outputs = logits_indices->shape()[0];
        if (n_outputs > ntoken) {
            return fail("logits_indices length > ntoken");
        }
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

        AttentionExecState attn_state{};
        {
            LLAISYS_NVTX_SCOPE("forward/prepare_attention_state");
            const int32_t rc = validate_and_bind_block_attention_state_(input.attention, ntoken, &attn_state);
            if (rc != 0) {
                std::cerr << "[Qwen2Model::forward] bind block attention state rc=" << rc << std::endl;
                return rc;
            }
        }

        for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
            LLAISYS_NVTX_SCOPE("forward/layer");
            tensor_t attn_normed = slice_tokens_(ws.normed, ntoken);
            ops::rms_norm(attn_normed, hidden, weights_.attn_norm_w[layer]->tensor, meta_.epsilon);

            tensor_t attn_out_2d = run_block_attention_layer_(
                layer,
                ntoken,
                attn_normed,
                pos_ids,
                attn_state);
            tensor_t attn_proj = slice_tokens_(ws.attn_proj, ntoken);
            {
                LLAISYS_NVTX_SCOPE("forward/layer/attn_out_proj");
                ops::linear(attn_proj, attn_out_2d, weights_.attn_o_w[layer]->tensor, nullptr);
                if (tp_size_ > 1) {
                    const int rc_tp = tp_allreduce_sum_(attn_proj);
                    if (rc_tp != 0) {
                        return fail("tp allreduce failed at attn_out_proj");
                    }
                }
                ops::add(hidden, hidden, attn_proj);
            }

            tensor_t mlp_normed = slice_tokens_(ws.mlp_normed, ntoken);
            ops::rms_norm(mlp_normed, hidden, weights_.mlp_norm_w[layer]->tensor, meta_.epsilon);

            tensor_t gate = slice_tokens_(ws.gate, ntoken);
            tensor_t up = slice_tokens_(ws.up, ntoken);
            {
                LLAISYS_NVTX_SCOPE("forward/layer/mlp");
                ops::linear(gate, mlp_normed, weights_.mlp_gate_w[layer]->tensor, nullptr);
                ops::linear(up, mlp_normed, weights_.mlp_up_w[layer]->tensor, nullptr);

                tensor_t swiglu = slice_tokens_(ws.swiglu, ntoken);
                ops::swiglu(swiglu, gate, up);

                tensor_t down = slice_tokens_(ws.down, ntoken);
                ops::linear(down, swiglu, weights_.mlp_down_w[layer]->tensor, nullptr);
                if (tp_size_ > 1) {
                    const int rc_tp = tp_allreduce_sum_(down);
                    if (rc_tp != 0) {
                        return fail("tp allreduce failed at mlp_down");
                    }
                }
                ops::add(hidden, hidden, down);
            }
        }

        tensor_t final_normed = slice_tokens_(ws.normed, ntoken);
        {
            LLAISYS_NVTX_SCOPE("forward/final_norm");
            ops::rms_norm(final_normed, hidden, weights_.out_norm_w->tensor, meta_.epsilon);
        }

        tensor_t logits = slice_tokens_(ws.logits, n_outputs);
        if (n_outputs == 0) {
            step_logits_ = logits;
        } else {
            tensor_t selected_hidden = slice_tokens_(ws.hidden, n_outputs);
            {
                LLAISYS_NVTX_SCOPE("forward/logits_select");
                ops::embedding(selected_hidden, logits_indices, final_normed);
            }
            {
                LLAISYS_NVTX_SCOPE("forward/logits_proj");
                ops::linear(logits, selected_hidden, weights_.out_embed->tensor, nullptr);
            }
            step_logits_ = logits;
        }

        if (output == nullptr) {
            return 0;
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
    return runtime_.kv_cache->layer_k(layer);
}

tensor_t Qwen2Model::kv_layer_v(size_t layer) const {
    ASSERT(runtime_.kv_cache != nullptr, "Qwen2: kv_cache is null");
    return runtime_.kv_cache->layer_v(layer);
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
    weights::destroy_unique(slots);
}

} // namespace llaisys::models::qwen2
