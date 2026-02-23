#pragma once

#include "../../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace llaisys::runtime::kv_cache {

enum class KvStatus : int32_t {
    OK = 0,
    OOM_SLOT = 1,
    INVALID_SEQ = 2,
    INVALID_POS = 3,
    EMPTY_RANGE = 4,
    INTERNAL_ERROR = 5,
};

enum class KvCacheLayout : uint8_t {
    SLOT = 0,
    BLOCK = 1,
};

struct KvSlotInfo {
    using idx_vec_t = std::vector<int32_t>;
    int32_t s0{0};
    int32_t s1{0};
    std::vector<int64_t> strm;
    std::vector<idx_vec_t> idxs;

    int32_t head() const noexcept {
        if (idxs.empty() || idxs[0].empty()) {
            return -1;
        }
        return idxs[0][0];
    }

    void resize(size_t n) {
        strm.resize(n);
        idxs.resize(n);
    }

    size_t size() const noexcept {
        if (idxs.empty()) {
            return 0;
        }
        return idxs[0].size();
    }

    size_t n_stream() const noexcept { return idxs.size(); }
    bool empty() const noexcept { return idxs.empty(); }
};

using KvSlotInfoVec = std::vector<KvSlotInfo>;

struct KvUBatch {
    std::vector<std::vector<int64_t>> seq_sets;
    std::vector<int64_t> pos_values;
};

class KVStorage {
public:
    struct LayerCache {
        tensor_t k_cache;
        tensor_t v_cache;
        tensor_t k_linear;
        tensor_t v_linear;
    };

    void init(size_t nlayer,
              size_t nblocks,
              size_t block_size,
              size_t nkvh,
              size_t dh,
              llaisysDataType_t dtype,
              llaisysDeviceType_t device_type,
              int device_id);

    size_t n_layer() const noexcept { return layers_.size(); }
    size_t n_blocks() const noexcept { return nblocks_; }
    size_t block_size() const noexcept { return block_size_; }
    size_t token_capacity() const noexcept { return nblocks_ * block_size_; }

    tensor_t layer_k(size_t layer) const;
    tensor_t layer_v(size_t layer) const;
    tensor_t layer_k_block(size_t layer) const;
    tensor_t layer_v_block(size_t layer) const;

private:
    size_t nblocks_{0};
    size_t block_size_{0};
    tensor_t k_arena_;
    tensor_t v_arena_;
    std::vector<LayerCache> layers_;
};

class KvCacheBase {
public:
    virtual ~KvCacheBase() = default;

    virtual void init_storage(size_t nlayer,
                              size_t nkvh,
                              size_t dh,
                              llaisysDataType_t dtype,
                              llaisysDeviceType_t device_type,
                              int device_id) = 0;

    virtual KvSlotInfoVec prepare(const std::vector<KvUBatch> &ubatches) = 0;
    virtual KvStatus apply_ubatch(const KvSlotInfo &sinfo, const KvUBatch &ubatch) = 0;
    virtual void rollback_ubatch(const KvSlotInfo &sinfo, const KvUBatch &ubatch) = 0;
    virtual KvStatus request_free(int64_t seq_id) {
        (void) seq_id;
        std::cerr << "[ERROR] KvCacheBase::request_free is not implemented for this KV layout." << std::endl;
        return KvStatus::INTERNAL_ERROR;
    }
    virtual KvStatus reset_prefix_cache() {
        return KvStatus::OK;
    }
    virtual KvStatus seq_cp(int64_t dst_seq,
                            int64_t src_seq,
                            int64_t p0,
                            int64_t p1,
                            std::vector<int32_t> *src_slots,
                            std::vector<int32_t> *dst_slots) {
        (void) dst_seq;
        (void) src_seq;
        (void) p0;
        (void) p1;
        (void) src_slots;
        (void) dst_slots;
        std::cerr << "[ERROR] KvCacheBase::seq_cp is not implemented for this KV layout." << std::endl;
        return KvStatus::INTERNAL_ERROR;
    }
    virtual KvStatus seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
        (void) seq_id;
        (void) p0;
        (void) p1;
        std::cerr << "[ERROR] KvCacheBase::seq_rm is not implemented for this KV layout." << std::endl;
        return KvStatus::INTERNAL_ERROR;
    }
    virtual KvStatus seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
        (void) seq_id;
        (void) p0;
        (void) p1;
        (void) delta;
        std::cerr << "[ERROR] KvCacheBase::seq_add is not implemented for this KV layout." << std::endl;
        return KvStatus::INTERNAL_ERROR;
    }
    virtual KvStatus seq_keep(int64_t seq_id) {
        (void) seq_id;
        std::cerr << "[ERROR] KvCacheBase::seq_keep is not implemented for this KV layout." << std::endl;
        return KvStatus::INTERNAL_ERROR;
    }
    virtual int64_t seq_pos_max(int64_t seq_id) const noexcept = 0;
    virtual void used_slots(std::vector<int32_t> *out) const = 0;
    virtual bool slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const = 0;
};

} // namespace llaisys::runtime::kv_cache
