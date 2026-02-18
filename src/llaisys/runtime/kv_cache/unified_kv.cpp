#include "unified_kv.hpp"

#include "../../../utils/check.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <unordered_map>
#include <unordered_set>

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

UnifiedKvImpl::UnifiedKvImpl(size_t maxseq, uint32_t n_stream)
    : maxseq_(maxseq), n_stream_(std::max(1u, n_stream)) {
    v_cells_.resize(n_stream_);
    v_heads_.assign(n_stream_, 0);
    for (auto &cells : v_cells_) {
        cells.resize(static_cast<uint32_t>(maxseq_));
    }
}

void UnifiedKvImpl::init_storage(size_t nlayer,
                                 size_t nkvh,
                                 size_t dh,
                                 llaisysDataType_t dtype,
                                 llaisysDeviceType_t device_type,
                                 int device_id) {
    CHECK_ARGUMENT(maxseq_ > 0, "unified_kv: maxseq must be > 0");
    CHECK_ARGUMENT(nlayer > 0, "unified_kv: nlayer must be > 0");
    CHECK_ARGUMENT(nkvh > 0, "unified_kv: nkvh must be > 0");
    CHECK_ARGUMENT(dh > 0, "unified_kv: dh must be > 0");
    storage_ = std::make_unique<KVStorage>();
    storage_->init(nlayer, maxseq_, 1, nkvh, dh, dtype, device_type, device_id);
}

tensor_t UnifiedKvImpl::layer_k(size_t layer) const {
    CHECK_ARGUMENT(storage_ != nullptr, "unified_kv: storage is not initialized");
    return storage_->layer_k(layer);
}

tensor_t UnifiedKvImpl::layer_v(size_t layer) const {
    CHECK_ARGUMENT(storage_ != nullptr, "unified_kv: storage is not initialized");
    return storage_->layer_v(layer);
}

size_t UnifiedKvImpl::free_slot_count_stream_(uint32_t stream) const noexcept {
    if (stream >= n_stream_) {
        return 0;
    }
    size_t n = 0;
    const auto &cells = v_cells_[stream];
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i)) {
            ++n;
        }
    }
    return n;
}

uint32_t UnifiedKvImpl::stream_for_seq_(int64_t seq_id) {
    if (n_stream_ == 1) {
        return 0;
    }
    auto it = seq_to_stream_.find(seq_id);
    if (it != seq_to_stream_.end()) {
        return it->second;
    }
    const uint32_t stream = (seq_id >= 0 && static_cast<uint64_t>(seq_id) < n_stream_) ? static_cast<uint32_t>(seq_id) : 0u;
    seq_to_stream_[seq_id] = stream;
    return stream;
}

uint32_t UnifiedKvImpl::stream_for_seq_const_(int64_t seq_id) const {
    if (n_stream_ == 1) {
        return 0;
    }
    auto it = seq_to_stream_.find(seq_id);
    if (it != seq_to_stream_.end()) {
        return it->second;
    }
    return (seq_id >= 0 && static_cast<uint64_t>(seq_id) < n_stream_) ? static_cast<uint32_t>(seq_id) : 0u;
}

int64_t UnifiedKvImpl::seq_pos_max_(int64_t seq_id) const noexcept {
    const uint32_t stream = stream_for_seq_const_(seq_id);
    if (stream >= n_stream_) {
        return -1;
    }
    return v_cells_[stream].seq_pos_max(seq_id);
}

bool UnifiedKvImpl::normalize_range_(int64_t seq_id, int64_t p0, int64_t p1, int64_t *out_p0, int64_t *out_p1) const {
    if (out_p0 == nullptr || out_p1 == nullptr) {
        return false;
    }
    int64_t begin = p0 < 0 ? 0 : p0;
    int64_t end = p1;
    if (end < 0) {
        const int64_t pmax = seq_pos_max_(seq_id);
        end = (pmax < 0) ? 0 : (pmax + 1);
    }
    if (begin > end) {
        return false;
    }
    *out_p0 = begin;
    *out_p1 = end;
    return true;
}

KvStatus UnifiedKvImpl::validate_ubatch_(const KvUBatch &ub, std::vector<uint32_t> *token_streams) const {
    if (ub.seq_sets.size() != ub.pos_values.size()) {
        return KvStatus::INVALID_POS;
    }
    if (ub.seq_sets.empty()) {
        return KvStatus::EMPTY_RANGE;
    }

    std::unordered_map<int64_t, int64_t> next_pos_by_seq;
    next_pos_by_seq.reserve(ub.seq_sets.size() * 2);

    std::vector<uint32_t> streams;
    streams.resize(ub.seq_sets.size(), 0);

    for (size_t i = 0; i < ub.seq_sets.size(); ++i) {
        const auto &seqs = ub.seq_sets[i];
        if (seqs.empty()) {
            return KvStatus::INVALID_SEQ;
        }

        std::unordered_set<int64_t> dedup;
        dedup.reserve(seqs.size());

        int64_t expected_pos = -1;
        int32_t stream = -1;
        for (int64_t sid : seqs) {
            if (!dedup.insert(sid).second) {
                continue;
            }
            const uint32_t s = stream_for_seq_const_(sid);
            if (stream < 0) {
                stream = static_cast<int32_t>(s);
            } else if (stream != static_cast<int32_t>(s)) {
                return KvStatus::INVALID_SEQ;
            }

            auto it = next_pos_by_seq.find(sid);
            if (it == next_pos_by_seq.end()) {
                it = next_pos_by_seq.emplace(sid, seq_pos_max_(sid) + 1).first;
            }
            if (expected_pos < 0) {
                expected_pos = it->second;
            } else if (expected_pos != it->second) {
                return KvStatus::INVALID_POS;
            }
        }

        if (stream < 0) {
            return KvStatus::INVALID_SEQ;
        }
        if (ub.pos_values[i] != expected_pos) {
            return KvStatus::INVALID_POS;
        }
        streams[i] = static_cast<uint32_t>(stream);
        for (int64_t sid : dedup) {
            next_pos_by_seq[sid] = ub.pos_values[i] + 1;
        }
    }

    if (token_streams) {
        *token_streams = std::move(streams);
    }
    return KvStatus::OK;
}

KvSlotInfo UnifiedKvImpl::find_slot_(const KvUBatch &ub, bool cont) const {
    std::vector<uint32_t> token_streams;
    if (validate_ubatch_(ub, &token_streams) != KvStatus::OK) {
        return {};
    }

    KvSlotInfo res{};
    res.s0 = std::numeric_limits<int32_t>::max();
    res.s1 = 0;

    std::vector<uint32_t> streams_order;
    streams_order.reserve(n_stream_);
    std::unordered_map<uint32_t, size_t> stream_to_group;
    for (uint32_t st : token_streams) {
        if (stream_to_group.find(st) == stream_to_group.end()) {
            stream_to_group[st] = streams_order.size();
            streams_order.push_back(st);
        }
    }

    res.resize(streams_order.size());
    for (size_t g = 0; g < streams_order.size(); ++g) {
        const uint32_t st = streams_order[g];
        res.strm[g] = static_cast<int64_t>(st);
        res.s0 = std::min(res.s0, static_cast<int32_t>(st));
        res.s1 = std::max(res.s1, static_cast<int32_t>(st));

        uint32_t n_tokens = 0;
        for (uint32_t ts : token_streams) {
            if (ts == st) {
                ++n_tokens;
            }
        }

        const auto &cells = v_cells_[st];
        uint32_t head_cur = v_heads_[st];

        if (head_cur > cells.get_used() + 2 * n_tokens) {
            head_cur = 0;
        }
        if (n_tokens > cells.size()) {
            return {};
        }

        uint32_t n_tested = 0;
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; ++i) {
                const uint32_t idx = head_cur;
                head_cur += 1;
                n_tested += 1;

                const bool can_use = cells.is_empty(idx);
                if (can_use) {
                    res.idxs[g].push_back(static_cast<int32_t>(idx));
                } else if (cont) {
                    break;
                }
            }

            if (res.idxs[g].size() == n_tokens) {
                break;
            }
            if (cont) {
                res.idxs[g].clear();
            }
            if (n_tested >= cells.size()) {
                return {};
            }
        }
    }

    if (res.s0 == std::numeric_limits<int32_t>::max()) {
        res.s0 = 0;
    }
    return res;
}

KvSlotInfo UnifiedKvImpl::find_slot(const KvUBatch &ubatch, bool cont) const {
    return find_slot_(ubatch, cont);
}

KvSlotInfoVec UnifiedKvImpl::prepare(const std::vector<KvUBatch> &ubatches) {
    KvSlotInfoVec res;

    struct state_t {
        KvSlotInfo sinfo;
        std::vector<uint32_t> heads_old;
        std::vector<KvCells> cells_old;
    };

    std::vector<state_t> states;
    states.reserve(ubatches.size());
    res.reserve(ubatches.size());

    bool success = true;
    for (const auto &ub : ubatches) {
        std::vector<uint32_t> token_streams;
        if (validate_ubatch_(ub, &token_streams) != KvStatus::OK) {
            success = false;
            break;
        }
        const auto sinfo_new = find_slot_(ub, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        state_t state{};
        state.sinfo = sinfo_new;
        state.heads_old = v_heads_;
        state.cells_old.reserve(sinfo_new.n_stream());
        for (size_t s = 0; s < sinfo_new.n_stream(); ++s) {
            const uint32_t stream = static_cast<uint32_t>(sinfo_new.strm[s]);
            state.cells_old.push_back(v_cells_[stream].cp(sinfo_new.idxs[s]));
        }
        states.push_back(std::move(state));

        if (apply_ubatch(sinfo_new, ub) != KvStatus::OK) {
            success = false;
            break;
        }
        res.push_back(sinfo_new);
    }

    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        for (size_t s = 0; s < it->sinfo.n_stream(); ++s) {
            const uint32_t stream = static_cast<uint32_t>(it->sinfo.strm[s]);
            v_cells_[stream].set(it->sinfo.idxs[s], it->cells_old[s]);
        }
        v_heads_ = it->heads_old;
    }

    if (!success) {
        return {};
    }
    return res;
}

KvStatus UnifiedKvImpl::apply_ubatch(const KvSlotInfo &sinfo, const KvUBatch &ub) {
    if (ub.seq_sets.size() != ub.pos_values.size()) {
        return KvStatus::INVALID_POS;
    }
    if (ub.seq_sets.empty()) {
        return KvStatus::EMPTY_RANGE;
    }

    std::vector<uint32_t> token_streams;
    const KvStatus vrc = validate_ubatch_(ub, &token_streams);
    if (vrc != KvStatus::OK) {
        return vrc;
    }

    std::unordered_map<uint32_t, size_t> stream_to_group;
    for (size_t g = 0; g < sinfo.n_stream(); ++g) {
        stream_to_group[static_cast<uint32_t>(sinfo.strm[g])] = g;
    }
    std::vector<size_t> stream_offsets(sinfo.n_stream(), 0);

    std::unordered_map<int64_t, int64_t> seq_pos_max_rm;

    for (size_t i = 0; i < ub.seq_sets.size(); ++i) {
        const uint32_t st = token_streams[i];
        const auto git = stream_to_group.find(st);
        if (git == stream_to_group.end()) {
            return KvStatus::INVALID_SEQ;
        }
        const size_t g = git->second;
        if (stream_offsets[g] >= sinfo.idxs[g].size()) {
            return KvStatus::INVALID_POS;
        }
        const int32_t idx = sinfo.idxs[g][stream_offsets[g]++];
        auto &cells = v_cells_[st];

        if (idx < 0 || static_cast<uint32_t>(idx) >= cells.size()) {
            return KvStatus::INVALID_POS;
        }

        if (!cells.is_empty(static_cast<uint32_t>(idx))) {
            const int64_t pos_old = cells.pos_get(static_cast<uint32_t>(idx));
            for (int64_t sid : cells.seq_set(static_cast<uint32_t>(idx))) {
                auto it = seq_pos_max_rm.find(sid);
                if (it == seq_pos_max_rm.end()) {
                    seq_pos_max_rm[sid] = pos_old;
                } else {
                    it->second = std::max(it->second, pos_old);
                }
            }
            cells.rm(static_cast<uint32_t>(idx));
        }

        cells.pos_set(static_cast<uint32_t>(idx), ub.pos_values[i]);
        for (int64_t sid : ub.seq_sets[i]) {
            stream_for_seq_(sid);
            cells.seq_add(static_cast<uint32_t>(idx), sid);
        }
    }

    for (const auto &[sid, pmax_rm] : seq_pos_max_rm) {
        const uint32_t st = stream_for_seq_(sid);
        auto &cells = v_cells_[st];
        const int64_t pmin = cells.seq_pos_min(sid);
        if (pmin >= 0 && pmin <= pmax_rm) {
            (void) seq_rm(sid, pmin, pmax_rm + 1);
        }
    }

    for (size_t g = 0; g < sinfo.n_stream(); ++g) {
        const uint32_t st = static_cast<uint32_t>(sinfo.strm[g]);
        if (!sinfo.idxs[g].empty()) {
            v_heads_[st] = static_cast<uint32_t>((static_cast<size_t>(sinfo.idxs[g].back()) + 1) % maxseq_);
        }
    }

    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();

    return KvStatus::OK;
}

void UnifiedKvImpl::rollback_ubatch(const KvSlotInfo &sinfo, const KvUBatch &ub) {
    if (ub.seq_sets.empty() || ub.seq_sets.size() != ub.pos_values.size()) {
        return;
    }
    std::vector<uint32_t> token_streams;
    if (validate_ubatch_(ub, &token_streams) != KvStatus::OK) {
        return;
    }
    std::unordered_map<uint32_t, size_t> stream_to_group;
    for (size_t g = 0; g < sinfo.n_stream(); ++g) {
        stream_to_group[static_cast<uint32_t>(sinfo.strm[g])] = g;
    }
    std::vector<size_t> stream_offsets(sinfo.n_stream(), 0);

    for (size_t i = 0; i < ub.seq_sets.size(); ++i) {
        const auto it = stream_to_group.find(token_streams[i]);
        if (it == stream_to_group.end()) {
            continue;
        }
        const size_t g = it->second;
        if (stream_offsets[g] >= sinfo.idxs[g].size()) {
            continue;
        }
        const int32_t idx = sinfo.idxs[g][stream_offsets[g]++];
        auto &cells = v_cells_[token_streams[i]];
        if (idx < 0 || static_cast<uint32_t>(idx) >= cells.size()) {
            continue;
        }
        if (!cells.is_empty(static_cast<uint32_t>(idx))) {
            cells.rm(static_cast<uint32_t>(idx));
        }
    }

    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();
}

KvStatus UnifiedKvImpl::seq_cp(int64_t dst_seq,
                               int64_t src_seq,
                               int64_t p0,
                               int64_t p1,
                               std::vector<int32_t> *src_slots,
                               std::vector<int32_t> *dst_slots) {
    const uint32_t ssrc = stream_for_seq_(src_seq);
    const uint32_t sdst = stream_for_seq_(dst_seq);
    auto &cells_src = v_cells_[ssrc];
    auto &cells_dst = v_cells_[sdst];

    int64_t begin = -1;
    int64_t end = -1;
    if (!normalize_range_(src_seq, p0, p1, &begin, &end)) {
        return KvStatus::INVALID_POS;
    }
    if (begin == end) {
        return KvStatus::EMPTY_RANGE;
    }

    if (src_slots) {
        src_slots->clear();
    }
    if (dst_slots) {
        dst_slots->clear();
    }

    if (ssrc == sdst) {
        for (uint32_t i = 0; i < cells_src.size(); ++i) {
            if (!cells_src.is_empty(i) && cells_src.pos_in(i, begin, end) && cells_src.seq_has(i, src_seq)) {
                cells_src.seq_add(i, dst_seq);
                if (src_slots) {
                    src_slots->push_back(static_cast<int32_t>(i));
                }
                if (dst_slots) {
                    dst_slots->push_back(static_cast<int32_t>(i));
                }
            }
        }
        seq_slots_cache_.clear();
        seq_slots_dirty_.clear();
        return KvStatus::OK;
    }

    cells_dst.reset();
    for (uint32_t i = 0; i < cells_src.size(); ++i) {
        if (!cells_src.is_empty(i) && cells_src.pos_in(i, begin, end) && cells_src.seq_has(i, src_seq)) {
            cells_dst.pos_set(i, cells_src.pos_get(i));
            cells_dst.seq_add(i, dst_seq);
            if (src_slots) {
                src_slots->push_back(static_cast<int32_t>(i));
            }
            if (dst_slots) {
                dst_slots->push_back(static_cast<int32_t>(i));
            }
        }
    }
    v_heads_[sdst] = v_heads_[ssrc];
    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();
    return KvStatus::OK;
}

KvStatus UnifiedKvImpl::seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
    if (seq_pos_max_(seq_id) < 0) {
        return KvStatus::INVALID_SEQ;
    }
    const uint32_t stream = stream_for_seq_(seq_id);
    auto &cells = v_cells_[stream];
    auto &head = v_heads_[stream];

    int64_t begin = -1;
    int64_t end = -1;
    if (!normalize_range_(seq_id, p0, p1, &begin, &end)) {
        return KvStatus::INVALID_POS;
    }
    if (begin == end) {
        return KvStatus::EMPTY_RANGE;
    }

    uint32_t new_head = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, begin, end)) {
            continue;
        }
        if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();
    return KvStatus::OK;
}

KvStatus UnifiedKvImpl::seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    if (seq_pos_max_(seq_id) < 0) {
        return KvStatus::INVALID_SEQ;
    }
    const uint32_t stream = stream_for_seq_(seq_id);
    auto &cells = v_cells_[stream];
    auto &head = v_heads_[stream];

    int64_t begin = -1;
    int64_t end = -1;
    if (!normalize_range_(seq_id, p0, p1, &begin, &end)) {
        return KvStatus::INVALID_POS;
    }
    if (begin == end) {
        return KvStatus::EMPTY_RANGE;
    }
    if (delta == 0) {
        return KvStatus::OK;
    }

    uint32_t new_head = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, begin, end)) {
            continue;
        }
        if (cells.seq_has(i, seq_id) && cells.pos_add(i, delta)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    head = new_head != cells.size() ? new_head : 0;
    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();
    return KvStatus::OK;
}

KvStatus UnifiedKvImpl::seq_keep(int64_t seq_id) {
    if (seq_pos_max_(seq_id) < 0) {
        return KvStatus::INVALID_SEQ;
    }
    const uint32_t stream = stream_for_seq_(seq_id);
    auto &cells = v_cells_[stream];
    auto &head = v_heads_[stream];

    uint32_t new_head = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();
    return KvStatus::OK;
}

int64_t UnifiedKvImpl::seq_pos_max(int64_t seq_id) const noexcept {
    return seq_pos_max_(seq_id);
}

void UnifiedKvImpl::used_slots(std::vector<int32_t> *out) const {
    if (out == nullptr) {
        return;
    }
    out->clear();
    const auto &cells = v_cells_[0];
    out->reserve(cells.get_used());
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i)) {
            out->push_back(static_cast<int32_t>(i));
        }
    }
    std::sort(out->begin(), out->end(), [&](int32_t a, int32_t b) {
        const int64_t pa = cells.pos_get(static_cast<uint32_t>(a));
        const int64_t pb = cells.pos_get(static_cast<uint32_t>(b));
        if (pa != pb) {
            return pa < pb;
        }
        return a < b;
    });
}

bool UnifiedKvImpl::slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const {
    if (slot < 0 || seq_ids == nullptr || n_seq_id <= 0) {
        return false;
    }
    const auto &cells = v_cells_[0];
    if (static_cast<uint32_t>(slot) >= cells.size()) {
        return false;
    }
    if (cells.is_empty(static_cast<uint32_t>(slot))) {
        return false;
    }
    if (cells.pos_get(static_cast<uint32_t>(slot)) > qpos) {
        return false;
    }
    for (int32_t i = 0; i < n_seq_id; ++i) {
        if (cells.seq_has(static_cast<uint32_t>(slot), seq_ids[i])) {
            return true;
        }
    }
    return false;
}

} // namespace llaisys::runtime::kv_cache
