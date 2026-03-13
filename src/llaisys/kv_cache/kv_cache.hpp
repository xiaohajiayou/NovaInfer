#pragma once

#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace llaisys::kv_cache {

enum class KvStatus : int32_t {
    OK = 0,
    OOM_KV = 1,
    INVALID_SEQ = 2,
    INVALID_POS = 3,
    EMPTY_RANGE = 4,
    INTERNAL_ERROR = 5,
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

} // namespace llaisys::kv_cache
