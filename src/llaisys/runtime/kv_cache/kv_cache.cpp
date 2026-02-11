#include "kv_cache.hpp"

#include <algorithm>

namespace llaisys::runtime::kv_cache {

KvCache::KvCache(size_t maxseq) : maxseq_(maxseq) {
    // Free-list starts with all slots available.
    free_slots_.reserve(maxseq_);
    slot_seq_sets_.assign(maxseq_, {});
    slot_pos_.assign(maxseq_, -1);
    for (int32_t s = static_cast<int32_t>(maxseq_) - 1; s >= 0; --s) {
        free_slots_.push_back(s);
    }
}

KvCache::SeqState &KvCache::ensure_seq_(int64_t seq_id) {
    // Default-construct sequence state on first access.
    return seq_states_[seq_id];
}

int32_t KvCache::alloc_slot_(int64_t pos) {
    // No eviction policy in stage-0/1: allocation fails when free-list is empty.
    if (free_slots_.empty()) {
        return -1;
    }
    const int32_t slot = free_slots_.back();
    free_slots_.pop_back();
    slot_seq_sets_[slot].clear();
    slot_pos_[slot] = pos;
    return slot;
}

void KvCache::free_slot_(int32_t slot) {
    // Ignore invalid indices to keep cleanup paths idempotent.
    if (slot < 0 || static_cast<size_t>(slot) >= slot_seq_sets_.size()) {
        return;
    }
    slot_seq_sets_[slot].clear();
    slot_pos_[slot] = -1;
    free_slots_.push_back(slot);
}

void KvCache::clear_seq_(int64_t seq_id) {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return;
    }
    for (int32_t slot : it->second.pos_to_slot) {
        if (slot < 0 || static_cast<size_t>(slot) >= slot_seq_sets_.size()) {
            continue;
        }
        auto &set = slot_seq_sets_[slot];
        set.erase(std::remove(set.begin(), set.end(), seq_id), set.end());
        if (set.empty()) {
            free_slot_(slot);
        }
    }
    seq_states_.erase(it);
}

KvStatus KvCache::alloc_tokens(int64_t seq_id, size_t ntoken, int64_t pos_start, std::vector<int32_t> *out_slots) {
    if (ntoken == 0) {
        return KvStatus::EMPTY_RANGE;
    }
    if (free_slots_.size() < ntoken) {
        return KvStatus::OOM_SLOT;
    }
    SeqState &seq = ensure_seq_(seq_id);
    if (pos_start < 0 || static_cast<size_t>(pos_start) != seq.pos_to_slot.size()) {
        return KvStatus::INVALID_POS;
    }
    if (seq.pos_to_slot.size() + ntoken > maxseq_) {
        return KvStatus::INVALID_POS;
    }

    if (out_slots) {
        out_slots->clear();
        out_slots->reserve(ntoken);
    }

    for (size_t i = 0; i < ntoken; ++i) {
        const int32_t slot = alloc_slot_(pos_start + static_cast<int64_t>(i));
        if (slot < 0) {
            return KvStatus::OOM_SLOT;
        }
        slot_seq_sets_[slot].push_back(seq_id);
        // pos_to_slot order is logical position order.
        seq.pos_to_slot.push_back(slot);
        if (out_slots) {
            out_slots->push_back(slot);
        }
    }
    return KvStatus::OK;
}

KvStatus KvCache::alloc_token(const int64_t *seq_ids, int32_t n_seq_id, int64_t pos, int32_t *out_slot) {
    if (seq_ids == nullptr || n_seq_id <= 0) {
        return KvStatus::INVALID_SEQ;
    }
    if (pos < 0) {
        return KvStatus::INVALID_POS;
    }
    if (free_slots_.empty()) {
        return KvStatus::OOM_SLOT;
    }

    // Validate and de-duplicate incoming seq ids.
    std::vector<int64_t> uniq;
    uniq.reserve(static_cast<size_t>(n_seq_id));
    for (int32_t i = 0; i < n_seq_id; ++i) {
        const int64_t sid = seq_ids[i];
        if (std::find(uniq.begin(), uniq.end(), sid) != uniq.end()) {
            continue;
        }
        SeqState &seq = ensure_seq_(sid);
        if (seq.pos_to_slot.size() != static_cast<size_t>(pos)) {
            return KvStatus::INVALID_POS;
        }
        uniq.push_back(sid);
    }

    const int32_t slot = alloc_slot_(pos);
    if (slot < 0) {
        return KvStatus::OOM_SLOT;
    }
    auto &owners = slot_seq_sets_[slot];
    owners.reserve(uniq.size());
    for (int64_t sid : uniq) {
        owners.push_back(sid);
        seq_states_[sid].pos_to_slot.push_back(slot);
    }
    if (out_slot) {
        *out_slot = slot;
    }
    return KvStatus::OK;
}

const std::vector<int32_t> *KvCache::seq_slots(int64_t seq_id) const noexcept {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return nullptr;
    }
    return &it->second.pos_to_slot;
}

KvStatus KvCache::seq_cp_prepare(int64_t dst_seq,
                                 int64_t src_seq,
                                 int64_t p0,
                                 int64_t p1,
                                 std::vector<int32_t> *src_slots,
                                 std::vector<int32_t> *dst_slots) {
    auto src_it = seq_states_.find(src_seq);
    if (src_it == seq_states_.end()) {
        return KvStatus::INVALID_SEQ;
    }

    const size_t src_len = src_it->second.pos_to_slot.size();
    if (p0 < 0 || p1 < 0 || p0 > p1 || static_cast<size_t>(p1) > src_len) {
        return KvStatus::INVALID_POS;
    }
    if (p0 == p1) {
        return KvStatus::EMPTY_RANGE;
    }

    const size_t copy_len = static_cast<size_t>(p1 - p0);
    SeqState &dst = ensure_seq_(dst_seq);
    if (dst.pos_to_slot.size() + copy_len > maxseq_) {
        return KvStatus::INVALID_POS;
    }

    if (src_slots) {
        src_slots->clear();
        src_slots->reserve(copy_len);
    }
    if (dst_slots) {
        dst_slots->clear();
        dst_slots->reserve(copy_len);
    }

    for (size_t i = 0; i < copy_len; ++i) {
        const int32_t src_slot = src_it->second.pos_to_slot[static_cast<size_t>(p0) + i];
        dst.pos_to_slot.push_back(src_slot);
        auto &owners = slot_seq_sets_[src_slot];
        if (std::find(owners.begin(), owners.end(), dst_seq) == owners.end()) {
            owners.push_back(dst_seq);
        }
        if (src_slots) {
            src_slots->push_back(src_slot);
        }
        if (dst_slots) {
            dst_slots->push_back(src_slot);
        }
    }
    return KvStatus::OK;
}

KvStatus KvCache::seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return KvStatus::INVALID_SEQ;
    }

    auto &slots = it->second.pos_to_slot;
    if (p0 < 0 || p1 < 0 || p0 > p1 || static_cast<size_t>(p1) > slots.size()) {
        return KvStatus::INVALID_POS;
    }
    if (p0 == p1) {
        return KvStatus::EMPTY_RANGE;
    }

    const size_t begin = static_cast<size_t>(p0);
    const size_t end = static_cast<size_t>(p1);
    for (size_t i = begin; i < end; ++i) {
        const int32_t slot = slots[i];
        auto &owners = slot_seq_sets_[slot];
        owners.erase(std::remove(owners.begin(), owners.end(), seq_id), owners.end());
        if (owners.empty()) {
            free_slot_(slot);
        }
    }
    // Keep pos_to_slot compact so logical position equals vector index.
    slots.erase(slots.begin() + static_cast<ptrdiff_t>(begin), slots.begin() + static_cast<ptrdiff_t>(end));
    for (size_t i = begin; i < slots.size(); ++i) {
        slot_pos_[slots[i]] = static_cast<int64_t>(i);
    }
    if (slots.empty()) {
        seq_states_.erase(it);
    }
    return KvStatus::OK;
}

KvStatus KvCache::seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return KvStatus::INVALID_SEQ;
    }
    auto &slots = it->second.pos_to_slot;
    if (p0 < 0 || p1 < 0 || p0 > p1 || static_cast<size_t>(p1) > slots.size()) {
        return KvStatus::INVALID_POS;
    }
    if (p0 == p1) {
        return KvStatus::EMPTY_RANGE;
    }
    if (delta == 0) {
        return KvStatus::OK;
    }
    // Position-shift semantics are reserved but not implemented in current stage.
    return KvStatus::INVALID_POS;
}

KvStatus KvCache::seq_keep(int64_t seq_id) {
    if (seq_states_.find(seq_id) == seq_states_.end()) {
        return KvStatus::INVALID_SEQ;
    }

    std::vector<int64_t> to_remove;
    to_remove.reserve(seq_states_.size());
    for (const auto &kv : seq_states_) {
        if (kv.first != seq_id) {
            to_remove.push_back(kv.first);
        }
    }
    for (int64_t other : to_remove) {
        clear_seq_(other);
    }
    return KvStatus::OK;
}

int64_t KvCache::seq_pos_max(int64_t seq_id) const noexcept {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end() || it->second.pos_to_slot.empty()) {
        return -1;
    }
    return static_cast<int64_t>(it->second.pos_to_slot.size() - 1);
}

void KvCache::used_slots(std::vector<int32_t> *out) const {
    if (out == nullptr) {
        return;
    }
    out->clear();
    out->reserve(maxseq_);
    for (size_t i = 0; i < maxseq_; ++i) {
        if (slot_pos_[i] >= 0 && !slot_seq_sets_[i].empty()) {
            out->push_back(static_cast<int32_t>(i));
        }
    }
    std::sort(out->begin(), out->end(), [&](int32_t a, int32_t b) {
        const int64_t pa = slot_pos_[static_cast<size_t>(a)];
        const int64_t pb = slot_pos_[static_cast<size_t>(b)];
        if (pa != pb) {
            return pa < pb;
        }
        return a < b;
    });
}

bool KvCache::slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const {
    if (slot < 0 || static_cast<size_t>(slot) >= maxseq_) {
        return false;
    }
    if (seq_ids == nullptr || n_seq_id <= 0) {
        return false;
    }
    if (slot_pos_[static_cast<size_t>(slot)] < 0) {
        return false;
    }
    if (slot_pos_[static_cast<size_t>(slot)] > qpos) {
        return false;
    }

    const auto &owners = slot_seq_sets_[static_cast<size_t>(slot)];
    for (int32_t i = 0; i < n_seq_id; ++i) {
        if (std::find(owners.begin(), owners.end(), seq_ids[i]) != owners.end()) {
            return true;
        }
    }
    return false;
}

} // namespace llaisys::runtime::kv_cache
