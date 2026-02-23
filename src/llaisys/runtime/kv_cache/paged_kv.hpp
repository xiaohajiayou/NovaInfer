#pragma once

#include "kv_cache.hpp"
#include "../../../utils/check.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llaisys::runtime::kv_cache {

class BlockPool {
public:
    explicit BlockPool(size_t num_blocks);

    size_t num_blocks() const noexcept { return num_blocks_; }
    size_t num_free_blocks() const noexcept { return free_ids_.size(); }
    float usage() const noexcept;

    bool alloc_n(size_t n, std::vector<int32_t> *out_ids);
    void free_one(int32_t block_id);
    void free_many(const std::vector<int32_t> &block_ids);

    int32_t ref_count(int32_t block_id) const;
    void incref(int32_t block_id);
    void decref(int32_t block_id);

private:
    size_t num_blocks_{0};
    std::vector<int32_t> ref_counts_;
    std::vector<int32_t> free_ids_;
};

class BlockTable {
public:
    explicit BlockTable(size_t block_size) : block_size_(block_size) {
        CHECK_ARGUMENT(block_size_ > 0, "block_table: block_size must be > 0");
    }

    size_t block_size() const noexcept { return block_size_; }
    size_t num_blocks() const noexcept { return block_ids_.size(); }
    const std::vector<int32_t> &block_ids() const noexcept { return block_ids_; }

    int32_t block_for_pos(int64_t pos) const;
    int64_t max_pos() const noexcept { return max_pos_; }
    int32_t back_block() const;

    void append_block(int32_t block_id);
    void pop_back_block();
    void clear();
    void set_max_pos(int64_t pos) noexcept { max_pos_ = pos; }

private:
    size_t block_size_{1};
    std::vector<int32_t> block_ids_{};
    int64_t max_pos_{-1};
};

class PagedKvImpl final : public KvCacheBase {
public:
    PagedKvImpl(size_t maxseq, uint32_t n_stream, size_t block_size);

    void init_storage(size_t nlayer,
                      size_t nkvh,
                      size_t dh,
                      llaisysDataType_t dtype,
                      llaisysDeviceType_t device_type,
                      int device_id) override;
    tensor_t layer_k(size_t layer) const;
    tensor_t layer_v(size_t layer) const;

    KvSlotInfoVec prepare(const std::vector<KvUBatch> &ubatches) override;
    KvStatus apply_ubatch(const KvSlotInfo &sinfo, const KvUBatch &ubatch) override;
    void rollback_ubatch(const KvSlotInfo &sinfo, const KvUBatch &ubatch) override;
    KvStatus request_free(int64_t seq_id) override;
    KvStatus reset_prefix_cache() override;
    int64_t seq_pos_max(int64_t seq_id) const noexcept override;
    void used_slots(std::vector<int32_t> *out) const override;
    bool slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const override;
    bool build_attention_plan(const std::vector<std::vector<int64_t>> &seq_sets,
                              const std::vector<int64_t> &qpos,
                              std::vector<int32_t> *used_slots,
                              std::vector<uint8_t> *mask) const;
    bool build_attention_plan_csr(const std::vector<std::vector<int64_t>> &seq_sets,
                                  const std::vector<int64_t> &qpos,
                                  std::vector<int32_t> *used_slots,
                                  std::vector<int32_t> *row_ptr,
                                  std::vector<int32_t> *col_idx) const;

private:
    int32_t compute_slot_(int32_t block_id, int32_t block_offset) const;
    uint32_t stream_for_seq_(int64_t seq_id, std::unordered_map<int64_t, uint32_t> *seq_to_stream) const;
    KvStatus validate_ubatch_(const KvUBatch &ub, std::vector<uint32_t> *token_streams) const;

    struct ShadowState {
        BlockPool pool;
        std::unordered_map<int64_t, std::vector<int32_t>> req_to_blocks;
        std::unordered_map<int64_t, uint32_t> seq_to_stream;
        std::unordered_map<int64_t, std::unique_ptr<BlockTable>> block_tables;
        explicit ShadowState(size_t nblocks) : pool(nblocks) {
        }
    };

    ShadowState clone_state_() const;
    KvStatus allocate_slots_(const KvUBatch &ub,
                             const std::vector<uint32_t> &token_streams,
                             ShadowState *state,
                             std::vector<int32_t> *out_slots) const;

    size_t maxseq_{0};
    uint32_t n_stream_{1};
    size_t block_size_{1};
    size_t nblocks_{0};
    std::unique_ptr<KVStorage> storage_;
    std::unique_ptr<BlockPool> block_pool_;
    std::unordered_map<int64_t, std::vector<int32_t>> req_to_blocks_;
    std::unordered_map<int64_t, uint32_t> seq_to_stream_;
    std::unordered_map<int64_t, std::unique_ptr<BlockTable>> block_tables_;
};

} // namespace llaisys::runtime::kv_cache
