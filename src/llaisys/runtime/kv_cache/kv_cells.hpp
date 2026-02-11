#pragma once

#include <cstdint>
#include <map>
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

} // namespace llaisys::runtime::kv_cache
