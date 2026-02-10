#include "kv_cache.hpp"

namespace llaisys::runtime::kv_cache {

KvCache::KvCache(size_t maxseq) : maxseq_(maxseq) {
    free_slots_.reserve(maxseq_);
    slot_seq_.assign(maxseq_, -1);
    slot_pos_.assign(maxseq_, -1);
    for (int32_t s = static_cast<int32_t>(maxseq_) - 1; s >= 0; --s) {
        free_slots_.push_back(s);
    }
}

KvCache::SeqState &KvCache::ensure_seq_(int64_t seq_id) {
    return seq_states_[seq_id];
}

int32_t KvCache::alloc_slot_(int64_t seq_id, int64_t pos) {
    if (free_slots_.empty()) {
        return -1;
    }
    const int32_t slot = free_slots_.back();
    free_slots_.pop_back();
    slot_seq_[slot] = seq_id;
    slot_pos_[slot] = pos;
    return slot;
}

void KvCache::free_slot_(int32_t slot) {
    if (slot < 0 || static_cast<size_t>(slot) >= slot_seq_.size()) {
        return;
    }
    slot_seq_[slot] = -1;
    slot_pos_[slot] = -1;
    free_slots_.push_back(slot);
}

void KvCache::clear_seq_(int64_t seq_id) {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return;
    }
    for (int32_t slot : it->second.pos_to_slot) {
        free_slot_(slot);
    }
    seq_states_.erase(it);
}

int KvCache::alloc_tokens(int64_t seq_id, size_t ntoken, int64_t pos_start, std::vector<int32_t> *out_slots) {
    if (ntoken == 0) {
        return 4;
    }
    if (free_slots_.size() < ntoken) {
        return 1;
    }
    SeqState &seq = ensure_seq_(seq_id);
    if (seq.pos_to_slot.size() + ntoken > maxseq_) {
        return 3;
    }

    if (out_slots) {
        out_slots->clear();
        out_slots->reserve(ntoken);
    }

    for (size_t i = 0; i < ntoken; ++i) {
        const int32_t slot = alloc_slot_(seq_id, pos_start + static_cast<int64_t>(i));
        if (slot < 0) {
            return 1;
        }
        seq.pos_to_slot.push_back(slot);
        if (out_slots) {
            out_slots->push_back(slot);
        }
    }
    return 0;
}

const std::vector<int32_t> *KvCache::seq_slots(int64_t seq_id) const noexcept {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return nullptr;
    }
    return &it->second.pos_to_slot;
}

int KvCache::seq_cp_prepare(int64_t dst_seq,
                            int64_t src_seq,
                            int64_t p0,
                            int64_t p1,
                            std::vector<int32_t> *src_slots,
                            std::vector<int32_t> *dst_slots) {
    auto src_it = seq_states_.find(src_seq);
    if (src_it == seq_states_.end()) {
        return 2;
    }

    const size_t src_len = src_it->second.pos_to_slot.size();
    if (p0 < 0 || p1 < 0 || p0 > p1 || static_cast<size_t>(p1) > src_len) {
        return 3;
    }
    if (p0 == p1) {
        return 4;
    }

    const size_t copy_len = static_cast<size_t>(p1 - p0);
    if (free_slots_.size() < copy_len) {
        return 1;
    }

    SeqState &dst = ensure_seq_(dst_seq);
    const size_t dst_pos_start = dst.pos_to_slot.size();

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
        const int32_t dst_slot = alloc_slot_(dst_seq, static_cast<int64_t>(dst_pos_start + i));
        if (dst_slot < 0) {
            return 1;
        }
        dst.pos_to_slot.push_back(dst_slot);
        if (src_slots) {
            src_slots->push_back(src_slot);
        }
        if (dst_slots) {
            dst_slots->push_back(dst_slot);
        }
    }
    return 0;
}

int KvCache::seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return 2;
    }

    auto &slots = it->second.pos_to_slot;
    if (p0 < 0 || p1 < 0 || p0 > p1 || static_cast<size_t>(p1) > slots.size()) {
        return 3;
    }
    if (p0 == p1) {
        return 4;
    }

    const size_t begin = static_cast<size_t>(p0);
    const size_t end = static_cast<size_t>(p1);
    for (size_t i = begin; i < end; ++i) {
        free_slot_(slots[i]);
    }
    slots.erase(slots.begin() + static_cast<ptrdiff_t>(begin), slots.begin() + static_cast<ptrdiff_t>(end));
    for (size_t i = begin; i < slots.size(); ++i) {
        slot_pos_[slots[i]] = static_cast<int64_t>(i);
    }
    if (slots.empty()) {
        seq_states_.erase(it);
    }
    return 0;
}

int KvCache::seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        return 2;
    }
    auto &slots = it->second.pos_to_slot;
    if (p0 < 0 || p1 < 0 || p0 > p1 || static_cast<size_t>(p1) > slots.size()) {
        return 3;
    }
    if (p0 == p1) {
        return 4;
    }
    if (delta == 0) {
        return 0;
    }
    return 3;
}

int KvCache::seq_keep(int64_t seq_id) {
    if (seq_states_.find(seq_id) == seq_states_.end()) {
        return 2;
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
    return 0;
}

int64_t KvCache::seq_pos_max(int64_t seq_id) const noexcept {
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end() || it->second.pos_to_slot.empty()) {
        return -1;
    }
    return static_cast<int64_t>(it->second.pos_to_slot.size() - 1);
}

} // namespace llaisys::runtime::kv_cache
