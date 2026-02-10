#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace llaisys::runtime::kv_cache {

class KvCache {
public:
    explicit KvCache(size_t maxseq);

    size_t maxseq() const noexcept { return maxseq_; }
    size_t free_slot_count() const noexcept { return free_slots_.size(); }

    int alloc_tokens(int64_t seq_id, size_t ntoken, int64_t pos_start, std::vector<int32_t> *out_slots);
    const std::vector<int32_t> *seq_slots(int64_t seq_id) const noexcept;

    int seq_cp_prepare(int64_t dst_seq,
                       int64_t src_seq,
                       int64_t p0,
                       int64_t p1,
                       std::vector<int32_t> *src_slots,
                       std::vector<int32_t> *dst_slots);
    int seq_rm(int64_t seq_id, int64_t p0, int64_t p1);
    int seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);
    int seq_keep(int64_t seq_id);
    int64_t seq_pos_max(int64_t seq_id) const noexcept;

private:
    struct SeqState {
        std::vector<int32_t> pos_to_slot;
    };

    SeqState &ensure_seq_(int64_t seq_id);
    int32_t alloc_slot_(int64_t seq_id, int64_t pos);
    void free_slot_(int32_t slot);
    void clear_seq_(int64_t seq_id);

    size_t maxseq_;
    std::unordered_map<int64_t, SeqState> seq_states_;
    std::vector<int32_t> free_slots_;
    std::vector<int64_t> slot_seq_;
    std::vector<int64_t> slot_pos_;
};

} // namespace llaisys::runtime::kv_cache
