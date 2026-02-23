#pragma once

#include "kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaisys::runtime::kv_cache {

class KvCells {
public:
    void reset();
    void reset_shift();

    uint32_t size() const noexcept;
    void resize(uint32_t n);

    bool is_empty(uint32_t i) const;

    uint32_t get_used() const;
    uint32_t used_min() const;
    uint32_t used_max_p1() const;
    bool get_has_shift() const;

    KvCells cp(const std::vector<int32_t> &idxs) const;
    void set(const std::vector<int32_t> &idxs, const KvCells &other);

    void rm(uint32_t i);
    bool seq_rm(uint32_t i, int64_t seq_id);
    bool seq_keep(uint32_t i, int64_t seq_id);

    int seq_count(uint32_t i) const;
    bool seq_has(uint32_t i, int64_t seq_id) const;
    void seq_add(uint32_t i, int64_t seq_id);
    int64_t seq_get(uint32_t i) const;

    int64_t seq_pos_min(int64_t seq_id) const;
    int64_t seq_pos_max(int64_t seq_id) const;

    int64_t pos_get(uint32_t i) const;
    bool pos_in(uint32_t i, int64_t p0, int64_t p1) const;
    void pos_set(uint32_t i, int64_t p);
    bool pos_add(uint32_t i, int64_t d);

    const std::unordered_set<int64_t> &seq_set(uint32_t i) const;

private:
    struct Cell {
        int64_t pos = -1;
        int64_t shift = 0;
        std::unordered_set<int64_t> seq;
    };

    bool has_shift_ = false;
    std::set<uint32_t> used_;
    std::vector<Cell> cells_;
    std::unordered_map<int64_t, std::map<int64_t, int>> seq_pos_;

    void seq_pos_inc_(int64_t s, int64_t p);
    void seq_pos_dec_(int64_t s, int64_t p);
    void seq_pos_rm_(uint32_t i);
    void seq_pos_add_(uint32_t i);
};

class UnifiedKvImpl final : public KvCacheBase {
public:
    UnifiedKvImpl(size_t maxseq, uint32_t n_stream);

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

    KvStatus seq_cp(int64_t dst_seq,
                    int64_t src_seq,
                    int64_t p0,
                    int64_t p1,
                    std::vector<int32_t> *src_slots,
                    std::vector<int32_t> *dst_slots) override;
    KvStatus seq_rm(int64_t seq_id, int64_t p0, int64_t p1) override;
    KvStatus seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) override;
    KvStatus seq_keep(int64_t seq_id) override;
    int64_t seq_pos_max(int64_t seq_id) const noexcept override;
    void used_slots(std::vector<int32_t> *out) const override;
    bool slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const override;

private:
    KvSlotInfo find_slot(const KvUBatch &ubatch, bool cont) const;
    uint32_t stream_for_seq_(int64_t seq_id);
    uint32_t stream_for_seq_const_(int64_t seq_id) const;
    int64_t seq_pos_max_(int64_t seq_id) const noexcept;
    bool normalize_range_(int64_t seq_id, int64_t p0, int64_t p1, int64_t *out_p0, int64_t *out_p1) const;
    size_t free_slot_count_stream_(uint32_t stream) const noexcept;
    KvStatus validate_ubatch_(const KvUBatch &ub, std::vector<uint32_t> *token_streams) const;
    KvSlotInfo find_slot_(const KvUBatch &ub, bool cont) const;

    size_t maxseq_{0};
    uint32_t n_stream_{1};
    std::unique_ptr<KVStorage> storage_;

    std::vector<KvCells> v_cells_;
    std::vector<uint32_t> v_heads_;

    mutable std::unordered_map<int64_t, uint32_t> seq_to_stream_;
    mutable std::unordered_map<int64_t, std::vector<int32_t>> seq_slots_cache_;
    mutable std::unordered_set<int64_t> seq_slots_dirty_;
};

} // namespace llaisys::runtime::kv_cache
