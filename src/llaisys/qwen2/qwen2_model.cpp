#include "qwen2_model.hpp"

namespace llaisys::models::qwen2 {

namespace {

using KvStatus = llaisys::runtime::kv_cache::KvStatus;

size_t numel_from_shape(const std::vector<size_t> &shape) {
    size_t n = 1;
    for (size_t d : shape) {
        n *= d;
    }
    return n;
}

tensor_t make_tensor(const std::vector<size_t> &shape,
                     llaisysDataType_t dtype,
                     llaisysDeviceType_t device_type,
                     int device_id) {
    return Tensor::create(shape, dtype, device_type, device_id);
}

} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       int *device_ids,
                       int ndevice)
    : meta_(meta), device_type_(device) {
    CHECK_ARGUMENT(device_type_ == LLAISYS_DEVICE_CPU, "Qwen2: only CPU is supported in this stage");

    if (device_ids != nullptr && ndevice > 0) {
        device_id_ = device_ids[0];
    }

    check_meta_invariants_();
    init_weight_slots_();
    init_kv_cache_();
    output_ = std::make_unique<runtime::output::OutputBuffer>(meta_.voc);
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

void Qwen2Model::init_kv_cache_() {
    caches_.clear();
    caches_.reserve(meta_.nlayer);

    const std::vector<size_t> cache_shape{meta_.maxseq, meta_.nkvh, meta_.dh};
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        LayerCache cache{};
        cache.k_cache = make_tensor(cache_shape, meta_.dtype, device_type_, device_id_);
        cache.v_cache = make_tensor(cache_shape, meta_.dtype, device_type_, device_id_);
        caches_.push_back(std::move(cache));
    }
    kv_cache_ = std::make_unique<runtime::kv_cache::KvCache>(meta_.maxseq);
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
    tensor_t t = make_tensor(shape, dtype, device_type_, device_id_);
    const size_t nbytes = numel_from_shape(shape) * utils::dsize(dtype);
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
        workspace_ = std::make_unique<runtime::workspace::Qwen2Workspace>(
            meta_.hs,
            meta_.nh,
            meta_.nkvh,
            meta_.dh,
            meta_.di,
            meta_.voc,
            meta_.maxseq,
            meta_.dtype,
            device_type_,
            device_id_);
    }
    workspace_->reserve(ntoken);
}

void Qwen2Model::fill_pos_ids_from_values_(const tensor_t &pos_ids, const std::vector<int64_t> &pos_values) {
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Qwen2: pos_ids dtype must be int64");
    ASSERT(pos_ids->shape()[0] == pos_values.size(), "Qwen2: pos_ids length mismatch");
    int64_t *ptr = reinterpret_cast<int64_t *>(pos_ids->data());
    std::memcpy(ptr, pos_values.data(), pos_values.size() * sizeof(int64_t));
}

void Qwen2Model::copy_token_into_cache_(tensor_t &cache, int32_t slot, const tensor_t &src, size_t token_idx) {
    ASSERT(cache->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2: cache must be on CPU");
    ASSERT(src->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2: src must be on CPU");
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
    std::memcpy(dst, src_ptr, stride_bytes);
}

tensor_t Qwen2Model::gather_cache_by_slots_(const tensor_t &cache, const std::vector<int32_t> &slots, size_t len, const tensor_t &buffer) {
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

bool Qwen2Model::validate_decode_batch_(const LlaisysBatch &batch) const {
    if (batch.n_tokens <= 0) {
        return false;
    }
    if (batch.embd != nullptr) {
        return false;
    }
    if (batch.token == nullptr) {
        return false;
    }

    const size_t ntoken = static_cast<size_t>(batch.n_tokens);
    for (size_t i = 0; i < ntoken; ++i) {
        const int32_t nseq = batch.n_seq_id ? batch.n_seq_id[i] : 1;
        if (nseq <= 0) {
            return false;
        }
        if (batch.seq_id != nullptr && batch.seq_id[i] == nullptr) {
            return false;
        }
    }

    return true;
}

void Qwen2Model::append_output_logits_row_(size_t row_idx, int32_t output_id) {
    ASSERT(workspace_ != nullptr, "Qwen2: workspace is null when collecting logits");
    ASSERT(output_ != nullptr, "Qwen2: output buffer is null");
    const auto &ws = workspace_->view();
    const size_t voc = meta_.voc;
    const size_t row_bytes = voc * utils::dsize(meta_.dtype);
    const std::byte *row = ws.logits->data() + row_idx * row_bytes;
    output_->append_row(row, meta_.dtype, output_id);
}

int32_t Qwen2Model::decode(const LlaisysBatch &batch) {
    if (!validate_decode_batch_(batch)) {
        return -1;
    }

    const size_t ntoken = static_cast<size_t>(batch.n_tokens);
    ASSERT(output_ != nullptr, "Qwen2: output buffer is null");
    output_->clear();
    output_->reserve_rows(ntoken);

    std::vector<std::vector<int64_t>> seq_sets(ntoken);
    std::vector<int64_t> pos_values(ntoken);
    llaisys::runtime::kv_cache::KvCache::ubatch kv_ubatch;
    llaisys::runtime::kv_cache::KvCache::slot_info slot_info;
    bool kv_applied = false;

    try {
        validate_or_die_();
        ASSERT(kv_cache_ != nullptr, "Qwen2: kv_cache is null");

        ensure_workspace_(ntoken);
        const auto &ws = workspace_->view();

        std::vector<int64_t> tokens(ntoken);
        std::vector<int32_t> nseq_values(ntoken, 0);

        std::unordered_map<int64_t, int64_t> next_pos_by_seq;
        for (size_t i = 0; i < ntoken; ++i) {
            tokens[i] = batch.token[i];

            const int32_t req_nseq = batch.n_seq_id ? batch.n_seq_id[i] : 1;
            const int64_t *src_seq_ptr = (batch.seq_id && batch.seq_id[i]) ? batch.seq_id[i] : nullptr;

            std::vector<int64_t> uniq_seq;
            uniq_seq.reserve(static_cast<size_t>(req_nseq));
            if (src_seq_ptr == nullptr) {
                if (batch.seq_id != nullptr) {
                    return -1;
                }
                uniq_seq.push_back(0);
            } else {
                for (int32_t j = 0; j < req_nseq; ++j) {
                    const int64_t sid = src_seq_ptr[j];
                    if (std::find(uniq_seq.begin(), uniq_seq.end(), sid) == uniq_seq.end()) {
                        uniq_seq.push_back(sid);
                    }
                }
            }
            if (uniq_seq.empty()) {
                return -1;
            }
            seq_sets[i] = std::move(uniq_seq);
            nseq_values[i] = static_cast<int32_t>(seq_sets[i].size());

            int64_t expected_pos = -1;
            for (int64_t sid : seq_sets[i]) {
                auto it = next_pos_by_seq.find(sid);
                if (it == next_pos_by_seq.end()) {
                    it = next_pos_by_seq.emplace(sid, kv_cache_->seq_pos_max(sid) + 1).first;
                }
                if (expected_pos < 0) {
                    expected_pos = it->second;
                } else if (expected_pos != it->second) {
                    return -1;
                }
            }
            const int64_t pos = batch.pos ? batch.pos[i] : expected_pos;
            if (pos != expected_pos) {
                return -1;
            }
            pos_values[i] = pos;
            for (int64_t sid : seq_sets[i]) {
                next_pos_by_seq[sid] = pos + 1;
            }
        }

        kv_ubatch.seq_sets = seq_sets;
        kv_ubatch.pos_values = pos_values;
        auto sinfos = kv_cache_->prepare({kv_ubatch});
        if (sinfos.empty()) {
            return 1;
        }
        slot_info = sinfos[0];
        const KvStatus apply_rc = kv_cache_->apply_ubatch(slot_info, kv_ubatch);
        if (apply_rc == KvStatus::OOM_SLOT) {
            return 1;
        }
        if (apply_rc != KvStatus::OK) {
            return -1;
        }
        CHECK_ARGUMENT(slot_info.n_stream() == 1, "Qwen2: unsupported slot_info stream shape");
        CHECK_ARGUMENT(slot_info.size() == ntoken, "Qwen2: slot_info size mismatch");
        const auto &slot_idxs = slot_info.idxs[0];
        kv_applied = true;

        tensor_t input_ids = make_tensor({ntoken}, LLAISYS_DTYPE_I64, device_type_, device_id_);
        input_ids->load(tokens.data());

        tensor_t hidden = slice_tokens_(ws.hidden, ntoken);
        ops::embedding(hidden, input_ids, weights_.in_embed->tensor);

        tensor_t pos_ids = slice_tokens_(ws.pos_ids, ntoken);
        fill_pos_ids_from_values_(pos_ids, pos_values);

        std::vector<int32_t> used_slots;
        kv_cache_->used_slots(&used_slots);
        const size_t kvlen = used_slots.size();
        CHECK_ARGUMENT(kvlen > 0, "Qwen2: no used KV slots");

        tensor_t attn_mask = make_tensor({ntoken, kvlen}, LLAISYS_DTYPE_U8, device_type_, device_id_);
        uint8_t *mask_ptr = reinterpret_cast<uint8_t *>(attn_mask->data());
        for (size_t i = 0; i < ntoken; ++i) {
            bool has_visible = false;
            for (size_t k = 0; k < kvlen; ++k) {
                const bool visible =
                    kv_cache_->slot_visible_for(used_slots[k], seq_sets[i].data(), nseq_values[i], pos_values[i]);
                mask_ptr[i * kvlen + k] = visible ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
                has_visible = has_visible || visible;
            }
            CHECK_ARGUMENT(has_visible, "Qwen2: empty attention context for token");
        }

        const size_t nh = meta_.nh;
        const size_t nkvh = meta_.nkvh;
        const size_t dh = meta_.dh;
        const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

        for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
            tensor_t attn_normed = slice_tokens_(ws.normed, ntoken);
            ops::rms_norm(attn_normed, hidden, weights_.attn_norm_w[layer]->tensor, meta_.epsilon);

            tensor_t q_proj = slice_tokens_(ws.q_proj, ntoken);
            tensor_t k_proj = slice_tokens_(ws.k_proj, ntoken);
            tensor_t v_proj = slice_tokens_(ws.v_proj, ntoken);

            ops::linear(q_proj, attn_normed, weights_.attn_q_w[layer]->tensor,
                        bias_or_zero_(weights_.attn_q_b[layer], zero_bias_attn_q_));
            ops::linear(k_proj, attn_normed, weights_.attn_k_w[layer]->tensor,
                        bias_or_zero_(weights_.attn_k_b[layer], zero_bias_attn_k_));
            ops::linear(v_proj, attn_normed, weights_.attn_v_w[layer]->tensor,
                        bias_or_zero_(weights_.attn_v_b[layer], zero_bias_attn_v_));

            tensor_t q_3d = view_2d_to_3d_(q_proj, ntoken, nh, dh);
            tensor_t k_new_3d = view_2d_to_3d_(k_proj, ntoken, nkvh, dh);
            tensor_t v_new_3d = view_2d_to_3d_(v_proj, ntoken, nkvh, dh);

            tensor_t rope_q = slice_tokens_(ws.rope_q, ntoken);
            tensor_t rope_k = slice_tokens_(ws.rope_k, ntoken);
            ops::rope(rope_q, q_3d, pos_ids, meta_.theta);
            ops::rope(rope_k, k_new_3d, pos_ids, meta_.theta);

            for (size_t i = 0; i < ntoken; ++i) {
                copy_token_into_cache_(caches_[layer].k_cache, slot_idxs[i], rope_k, i);
                copy_token_into_cache_(caches_[layer].v_cache, slot_idxs[i], v_new_3d, i);
            }

            tensor_t k_full = gather_cache_by_slots_(caches_[layer].k_cache, used_slots, kvlen, ws.k_ctx);
            tensor_t v_full = gather_cache_by_slots_(caches_[layer].v_cache, used_slots, kvlen, ws.v_ctx);

            tensor_t attn_out = slice_tokens_(ws.attn_out, ntoken);
            ops::self_attention_masked(attn_out, rope_q, k_full, v_full, attn_mask, scale);

            tensor_t attn_out_2d = attn_out->view({ntoken, nh * dh});

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

        tensor_t final_normed = slice_tokens_(ws.normed, ntoken);
        ops::rms_norm(final_normed, hidden, weights_.out_norm_w->tensor, meta_.epsilon);
        tensor_t logits = slice_tokens_(ws.logits, ntoken);
        ops::linear(logits, final_normed, weights_.out_embed->tensor, zero_bias_logits_);

        for (size_t i = 0; i < ntoken; ++i) {
            const bool collect = (batch.logits == nullptr) ? (i + 1 == ntoken) : (batch.logits[i] != 0);
            if (collect) {
                append_output_logits_row_(i, static_cast<int32_t>(i));
            }
        }

        return 0;
    } catch (const std::invalid_argument &) {
        if (kv_applied) {
            kv_cache_->rollback_ubatch(slot_info, kv_ubatch);
        }
        return 1;
    } catch (...) {
        if (kv_applied) {
            kv_cache_->rollback_ubatch(slot_info, kv_ubatch);
        }
        return -2;
    }
}

float *Qwen2Model::logits() noexcept {
    return output_ ? output_->logits() : nullptr;
}

float *Qwen2Model::logits_ith(int32_t i) noexcept {
    return output_ ? output_->logits_ith(i) : nullptr;
}

int32_t Qwen2Model::n_outputs() const noexcept {
    return output_ ? output_->n_outputs() : 0;
}

const int32_t *Qwen2Model::output_ids() const noexcept {
    return output_ ? output_->output_ids() : nullptr;
}

KvStatus Qwen2Model::kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1) {
    ASSERT(kv_cache_ != nullptr, "Qwen2: kv_cache is null");
    std::vector<int32_t> src_slots;
    std::vector<int32_t> dst_slots;
    const KvStatus rc = kv_cache_->seq_cp(dst_seq, src_seq, p0, p1, &src_slots, &dst_slots);
    if (rc != KvStatus::OK) {
        return rc;
    }

    const size_t copy_len = src_slots.size();
    const size_t stride_elems = meta_.nkvh * meta_.dh;
    const size_t stride_bytes = stride_elems * utils::dsize(meta_.dtype);
    for (size_t i = 0; i < copy_len; ++i) {
        const int32_t src_slot = src_slots[i];
        const int32_t dst_slot = dst_slots[i];
        if (src_slot == dst_slot) {
            continue;
        }
        for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
            std::byte *k_dst = caches_[layer].k_cache->data() + static_cast<ptrdiff_t>(dst_slot) * static_cast<ptrdiff_t>(stride_bytes);
            std::byte *v_dst = caches_[layer].v_cache->data() + static_cast<ptrdiff_t>(dst_slot) * static_cast<ptrdiff_t>(stride_bytes);
            const std::byte *k_src = caches_[layer].k_cache->data() + static_cast<ptrdiff_t>(src_slot) * static_cast<ptrdiff_t>(stride_bytes);
            const std::byte *v_src = caches_[layer].v_cache->data() + static_cast<ptrdiff_t>(src_slot) * static_cast<ptrdiff_t>(stride_bytes);
            std::memcpy(k_dst, k_src, stride_bytes);
            std::memcpy(v_dst, v_src, stride_bytes);
        }
    }
    return KvStatus::OK;
}

KvStatus Qwen2Model::kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
    ASSERT(kv_cache_ != nullptr, "Qwen2: kv_cache is null");
    return kv_cache_->seq_rm(seq_id, p0, p1);
}

KvStatus Qwen2Model::kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    ASSERT(kv_cache_ != nullptr, "Qwen2: kv_cache is null");
    return kv_cache_->seq_add(seq_id, p0, p1, delta);
}

KvStatus Qwen2Model::kv_seq_keep(int64_t seq_id) {
    ASSERT(kv_cache_ != nullptr, "Qwen2: kv_cache is null");
    return kv_cache_->seq_keep(seq_id);
}

int64_t Qwen2Model::kv_seq_pos_max(int64_t seq_id) const noexcept {
    if (!kv_cache_) {
        return -1;
    }
    return kv_cache_->seq_pos_max(seq_id);
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
