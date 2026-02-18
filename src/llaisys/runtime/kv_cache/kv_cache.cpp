#include "kv_cache.hpp"

#include "../../../utils/check.hpp"

namespace llaisys::runtime::kv_cache {

void KVStorage::init(size_t nlayer,
                     size_t nblocks,
                     size_t block_size,
                     size_t nkvh,
                     size_t dh,
                     llaisysDataType_t dtype,
                     llaisysDeviceType_t device_type,
                     int device_id) {
    CHECK_ARGUMENT(nlayer > 0, "kv_storage: nlayer must be > 0");
    CHECK_ARGUMENT(nblocks > 0, "kv_storage: nblocks must be > 0");
    CHECK_ARGUMENT(block_size > 0, "kv_storage: block_size must be > 0");
    CHECK_ARGUMENT(nkvh > 0, "kv_storage: nkvh must be > 0");
    CHECK_ARGUMENT(dh > 0, "kv_storage: dh must be > 0");

    nblocks_ = nblocks;
    block_size_ = block_size;

    const size_t tokens_per_layer = nblocks_ * block_size_;
    k_arena_ = Tensor::create({nlayer * nblocks_, block_size_, nkvh, dh}, dtype, device_type, device_id);
    v_arena_ = Tensor::create({nlayer * nblocks_, block_size_, nkvh, dh}, dtype, device_type, device_id);

    layers_.clear();
    layers_.reserve(nlayer);
    for (size_t il = 0; il < nlayer; ++il) {
        const size_t b0 = il * nblocks_;
        const size_t b1 = b0 + nblocks_;
        tensor_t k_block = k_arena_->slice(0, b0, b1);
        tensor_t v_block = v_arena_->slice(0, b0, b1);
        tensor_t k_linear = k_block->view({tokens_per_layer, nkvh, dh});
        tensor_t v_linear = v_block->view({tokens_per_layer, nkvh, dh});
        layers_.push_back({k_block, v_block, k_linear, v_linear});
    }
}

tensor_t KVStorage::layer_k(size_t layer) const {
    CHECK_ARGUMENT(layer < layers_.size(), "kv_storage: layer_k index out of range");
    return layers_[layer].k_linear;
}

tensor_t KVStorage::layer_v(size_t layer) const {
    CHECK_ARGUMENT(layer < layers_.size(), "kv_storage: layer_v index out of range");
    return layers_[layer].v_linear;
}

tensor_t KVStorage::layer_k_block(size_t layer) const {
    CHECK_ARGUMENT(layer < layers_.size(), "kv_storage: layer_k_block index out of range");
    return layers_[layer].k_cache;
}

tensor_t KVStorage::layer_v_block(size_t layer) const {
    CHECK_ARGUMENT(layer < layers_.size(), "kv_storage: layer_v_block index out of range");
    return layers_[layer].v_cache;
}

} // namespace llaisys::runtime::kv_cache
