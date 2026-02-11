#include "kv_cells.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace llaisys::runtime::kv_cache {

void KvCells::reset() {
    for (auto &cell : cells_) {
        cell.pos = -1;
        cell.shift = 0;
        cell.seq.clear();
    }
    has_shift_ = false;
    used_.clear();
    seq_pos_.clear();
}

void KvCells::reset_shift() {
    has_shift_ = false;
    for (auto &cell : cells_) {
        cell.shift = 0;
    }
}

uint32_t KvCells::size() const noexcept {
    return static_cast<uint32_t>(cells_.size());
}

void KvCells::resize(uint32_t n) {
    cells_.resize(n);
    reset();
}

bool KvCells::is_empty(uint32_t i) const {
    assert(i < cells_.size());
    return cells_[i].pos == -1;
}

uint32_t KvCells::get_used() const {
    return static_cast<uint32_t>(used_.size());
}

uint32_t KvCells::used_min() const {
    return used_.empty() ? 0u : *used_.begin();
}

uint32_t KvCells::used_max_p1() const {
    return used_.empty() ? 0u : (*used_.rbegin() + 1u);
}

bool KvCells::get_has_shift() const {
    return has_shift_;
}

KvCells KvCells::cp(const std::vector<int32_t> &idxs) const {
    KvCells res;
    res.resize(static_cast<uint32_t>(idxs.size()));
    for (size_t j = 0; j < idxs.size(); ++j) {
        const auto idx = static_cast<uint32_t>(idxs[j]);
        assert(idx < cells_.size());
        const auto &src = cells_[idx];
        if (src.pos >= 0) {
            res.cells_[j] = src;
            res.used_.insert(static_cast<uint32_t>(j));
            for (int64_t s : src.seq) {
                res.seq_pos_inc_(s, src.pos);
            }
        }
    }
    return res;
}

void KvCells::set(const std::vector<int32_t> &idxs, const KvCells &other) {
    assert(idxs.size() == other.cells_.size());
    for (size_t j = 0; j < idxs.size(); ++j) {
        const auto idx = static_cast<uint32_t>(idxs[j]);
        assert(idx < cells_.size());

        if (cells_[idx].pos >= 0) {
            seq_pos_rm_(idx);
            used_.erase(idx);
        }

        cells_[idx] = other.cells_[j];
        if (cells_[idx].pos >= 0) {
            used_.insert(idx);
            seq_pos_add_(idx);
        }
    }
}

void KvCells::rm(uint32_t i) {
    assert(i < cells_.size());
    assert(cells_[i].pos >= 0);
    seq_pos_rm_(i);
    cells_[i].seq.clear();
    cells_[i].pos = -1;
    cells_[i].shift = 0;
    used_.erase(i);
}

bool KvCells::seq_rm(uint32_t i, int64_t seq_id) {
    assert(i < cells_.size());
    auto &cell = cells_[i];
    assert(cell.pos >= 0);
    const auto it = cell.seq.find(seq_id);
    assert(it != cell.seq.end());
    cell.seq.erase(it);
    seq_pos_dec_(seq_id, cell.pos);
    if (cell.seq.empty()) {
        cell.pos = -1;
        cell.shift = 0;
        used_.erase(i);
        return true;
    }
    return false;
}

bool KvCells::seq_keep(uint32_t i, int64_t seq_id) {
    assert(i < cells_.size());
    auto &cell = cells_[i];
    if (cell.pos < 0) {
        return false;
    }
    if (cell.seq.count(seq_id) != 0) {
        seq_pos_rm_(i);
        cell.seq.clear();
        cell.seq.insert(seq_id);
        seq_pos_inc_(seq_id, cell.pos);
        return false;
    }
    seq_pos_rm_(i);
    cell.seq.clear();
    cell.pos = -1;
    cell.shift = 0;
    used_.erase(i);
    return true;
}

int KvCells::seq_count(uint32_t i) const {
    assert(i < cells_.size());
    return static_cast<int>(cells_[i].seq.size());
}

bool KvCells::seq_has(uint32_t i, int64_t seq_id) const {
    assert(i < cells_.size());
    return cells_[i].seq.count(seq_id) != 0;
}

void KvCells::seq_add(uint32_t i, int64_t seq_id) {
    assert(i < cells_.size());
    auto &cell = cells_[i];
    assert(cell.pos >= 0);
    if (cell.seq.insert(seq_id).second) {
        seq_pos_inc_(seq_id, cell.pos);
    }
}

int64_t KvCells::seq_get(uint32_t i) const {
    assert(i < cells_.size());
    assert(cells_[i].seq.size() == 1);
    return *cells_[i].seq.begin();
}

int64_t KvCells::seq_pos_min(int64_t seq_id) const {
    const auto it = seq_pos_.find(seq_id);
    if (it == seq_pos_.end() || it->second.empty()) {
        return -1;
    }
    return it->second.begin()->first;
}

int64_t KvCells::seq_pos_max(int64_t seq_id) const {
    const auto it = seq_pos_.find(seq_id);
    if (it == seq_pos_.end() || it->second.empty()) {
        return -1;
    }
    return it->second.rbegin()->first;
}

int64_t KvCells::pos_get(uint32_t i) const {
    assert(i < cells_.size());
    assert(cells_[i].pos >= 0);
    return cells_[i].pos;
}

bool KvCells::pos_in(uint32_t i, int64_t p0, int64_t p1) const {
    assert(i < cells_.size());
    return cells_[i].pos >= p0 && cells_[i].pos < p1;
}

void KvCells::pos_set(uint32_t i, int64_t p) {
    assert(i < cells_.size());
    assert(cells_[i].pos == -1);
    cells_[i].pos = p;
    used_.insert(i);
}

bool KvCells::pos_add(uint32_t i, int64_t d) {
    assert(i < cells_.size());
    auto &cell = cells_[i];
    assert(cell.pos >= 0);
    seq_pos_rm_(i);
    cell.pos += d;
    cell.shift += d;
    has_shift_ = true;
    if (cell.pos < 0) {
        cell.seq.clear();
        cell.pos = -1;
        cell.shift = 0;
        used_.erase(i);
        return true;
    }
    seq_pos_add_(i);
    return false;
}

const std::unordered_set<int64_t> &KvCells::seq_set(uint32_t i) const {
    assert(i < cells_.size());
    return cells_[i].seq;
}

void KvCells::seq_pos_inc_(int64_t s, int64_t p) {
    seq_pos_[s][p]++;
}

void KvCells::seq_pos_dec_(int64_t s, int64_t p) {
    auto it = seq_pos_.find(s);
    assert(it != seq_pos_.end());
    auto pit = it->second.find(p);
    assert(pit != it->second.end());
    pit->second -= 1;
    if (pit->second == 0) {
        it->second.erase(pit);
    }
    if (it->second.empty()) {
        seq_pos_.erase(it);
    }
}

void KvCells::seq_pos_rm_(uint32_t i) {
    const auto &cell = cells_[i];
    for (int64_t s : cell.seq) {
        seq_pos_dec_(s, cell.pos);
    }
}

void KvCells::seq_pos_add_(uint32_t i) {
    const auto &cell = cells_[i];
    for (int64_t s : cell.seq) {
        seq_pos_inc_(s, cell.pos);
    }
}

} // namespace llaisys::runtime::kv_cache
