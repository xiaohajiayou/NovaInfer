#include "kv_cache.hpp"

#include "../../../utils/check.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace llaisys::runtime::kv_cache {

KvCache::KvCache(size_t maxseq, uint32_t n_stream) : maxseq_(maxseq), n_stream_(std::max(1u, n_stream)) {
    v_cells_.resize(n_stream_);
    v_heads_.assign(n_stream_, 0);
    for (auto &cells : v_cells_) {
        cells.resize(static_cast<uint32_t>(maxseq_));
    }
}

void KvCache::init_storage(size_t nlayer,
                           size_t nkvh,
                           size_t dh,
                           llaisysDataType_t dtype,
                           llaisysDeviceType_t device_type,
                           int device_id) {
    CHECK_ARGUMENT(maxseq_ > 0, "kv_cache: maxseq must be > 0");
    CHECK_ARGUMENT(nlayer > 0, "kv_cache: nlayer must be > 0");
    CHECK_ARGUMENT(nkvh > 0, "kv_cache: nkvh must be > 0");
    CHECK_ARGUMENT(dh > 0, "kv_cache: dh must be > 0");

    k_arena_ = Tensor::create({nlayer, maxseq_, nkvh, dh}, dtype, device_type, device_id);
    v_arena_ = Tensor::create({nlayer, maxseq_, nkvh, dh}, dtype, device_type, device_id);
    layers_.clear();
    layers_.reserve(nlayer);
    for (size_t il = 0; il < nlayer; ++il) {
        tensor_t k_cache = k_arena_->slice(0, il, il + 1)->view({maxseq_, nkvh, dh});
        tensor_t v_cache = v_arena_->slice(0, il, il + 1)->view({maxseq_, nkvh, dh});
        layers_.push_back({k_cache, v_cache});
    }
}

tensor_t KvCache::layer_k(size_t layer) const {
    CHECK_ARGUMENT(layer < layers_.size(), "kv_cache: layer_k index out of range");
    return layers_[layer].k_cache;
}

tensor_t KvCache::layer_v(size_t layer) const {
    CHECK_ARGUMENT(layer < layers_.size(), "kv_cache: layer_v index out of range");
    return layers_[layer].v_cache;
}

size_t KvCache::free_slot_count_stream_(uint32_t stream) const noexcept {
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

size_t KvCache::free_slot_count() const noexcept {
    size_t n = 0;
    for (uint32_t s = 0; s < n_stream_; ++s) {
        n += free_slot_count_stream_(s);
    }
    return n;
}

uint32_t KvCache::stream_for_seq_(int64_t seq_id) {
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

uint32_t KvCache::stream_for_seq_const_(int64_t seq_id) const {
    if (n_stream_ == 1) {
        return 0;
    }
    auto it = seq_to_stream_.find(seq_id);
    if (it != seq_to_stream_.end()) {
        return it->second;
    }
    return (seq_id >= 0 && static_cast<uint64_t>(seq_id) < n_stream_) ? static_cast<uint32_t>(seq_id) : 0u;
}

int64_t KvCache::seq_pos_max_(int64_t seq_id) const noexcept {
    const uint32_t stream = stream_for_seq_const_(seq_id);
    if (stream >= n_stream_) {
        return -1;
    }
    return v_cells_[stream].seq_pos_max(seq_id);
}

bool KvCache::normalize_range_(int64_t seq_id, int64_t p0, int64_t p1, int64_t *out_p0, int64_t *out_p1) const {
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

int32_t KvCache::alloc_slot_(uint32_t stream, int64_t pos) {
    if (stream >= n_stream_ || maxseq_ == 0) {
        return -1;
    }
    auto &cells = v_cells_[stream];
    auto &head = v_heads_[stream];

    size_t visited = 0;
    uint32_t cur = head;
    while (visited < maxseq_) {
        if (cells.is_empty(cur)) {
            cells.pos_set(cur, pos);
            head = static_cast<uint32_t>((cur + 1) % maxseq_);
            return static_cast<int32_t>(cur);
        }
        cur = static_cast<uint32_t>((cur + 1) % maxseq_);
        ++visited;
    }
    return -1;
}

KvStatus KvCache::validate_ubatch_(const ubatch &ub, std::vector<uint32_t> *token_streams) const {
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

KvCache::slot_info KvCache::find_slot(const ubatch &ub, bool cont) const {
    std::vector<uint32_t> token_streams;
    if (validate_ubatch_(ub, &token_streams) != KvStatus::OK) {
        return {};
    }

    slot_info res{};
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

KvStatus KvCache::apply_ubatch(const slot_info &sinfo, const ubatch &ub) {
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

void KvCache::rollback_ubatch(const slot_info &sinfo, const ubatch &ub) {
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

KvCache::slot_info_vec_t KvCache::prepare(const std::vector<ubatch> &ubatches) {
    slot_info_vec_t res;

    struct state_t {
        slot_info sinfo;
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
        const auto sinfo_new = find_slot(ub, false);
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

bool KvCache::update(bool do_shift, const stream_copy_info &sc_info) {
    bool updated = false;

    if (!sc_info.empty() && n_stream_ > 1) {
        const size_t n_copy = sc_info.ssrc.size();
        for (size_t i = 0; i < n_copy; ++i) {
            const uint32_t ssrc = sc_info.ssrc[i];
            const uint32_t sdst = sc_info.sdst[i];
            if (ssrc >= n_stream_ || sdst >= n_stream_ || ssrc == sdst) {
                continue;
            }
            v_cells_[sdst] = v_cells_[ssrc];
            v_heads_[sdst] = v_heads_[ssrc];
            updated = true;
        }
    }

    if (do_shift) {
        for (auto &cells : v_cells_) {
            if (cells.get_has_shift()) {
                cells.reset_shift();
                updated = true;
            }
        }
    }

    if (updated) {
        seq_slots_cache_.clear();
        seq_slots_dirty_.clear();
    }
    return updated;
}

KvStatus KvCache::alloc_tokens(int64_t seq_id, size_t ntoken, int64_t pos_start, std::vector<int32_t> *out_slots) {
    if (ntoken == 0) {
        return KvStatus::EMPTY_RANGE;
    }
    const uint32_t stream = stream_for_seq_(seq_id);
    if (free_slot_count_stream_(stream) < ntoken) {
        return KvStatus::OOM_SLOT;
    }
    const int64_t expected_pos = seq_pos_max_(seq_id) + 1;
    if (pos_start < 0 || pos_start != expected_pos) {
        return KvStatus::INVALID_POS;
    }

    std::vector<int32_t> local_slots;
    local_slots.reserve(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        const int32_t slot = alloc_slot_(stream, pos_start + static_cast<int64_t>(i));
        if (slot < 0) {
            for (int32_t s : local_slots) {
                v_cells_[stream].rm(static_cast<uint32_t>(s));
            }
            return KvStatus::OOM_SLOT;
        }
        v_cells_[stream].seq_add(static_cast<uint32_t>(slot), seq_id);
        local_slots.push_back(slot);
    }
    if (out_slots) {
        *out_slots = local_slots;
    }
    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();
    return KvStatus::OK;
}

KvStatus KvCache::alloc_token(const int64_t *seq_ids, int32_t n_seq_id, int64_t pos, int32_t *out_slot) {
    if (seq_ids == nullptr || n_seq_id <= 0) {
        return KvStatus::INVALID_SEQ;
    }
    if (pos < 0) {
        return KvStatus::INVALID_POS;
    }
    std::vector<int64_t> uniq;
    uniq.reserve(static_cast<size_t>(n_seq_id));
    uint32_t stream = std::numeric_limits<uint32_t>::max();
    for (int32_t i = 0; i < n_seq_id; ++i) {
        const int64_t sid = seq_ids[i];
        if (std::find(uniq.begin(), uniq.end(), sid) != uniq.end()) {
            continue;
        }
        const uint32_t s = stream_for_seq_(sid);
        if (stream == std::numeric_limits<uint32_t>::max()) {
            stream = s;
        } else if (stream != s) {
            return KvStatus::INVALID_SEQ;
        }
        if (seq_pos_max_(sid) + 1 != pos) {
            return KvStatus::INVALID_POS;
        }
        uniq.push_back(sid);
    }

    if (stream == std::numeric_limits<uint32_t>::max()) {
        return KvStatus::INVALID_SEQ;
    }
    if (free_slot_count_stream_(stream) == 0) {
        return KvStatus::OOM_SLOT;
    }
    const int32_t slot = alloc_slot_(stream, pos);
    if (slot < 0) {
        return KvStatus::OOM_SLOT;
    }
    for (int64_t sid : uniq) {
        v_cells_[stream].seq_add(static_cast<uint32_t>(slot), sid);
    }
    if (out_slot) {
        *out_slot = slot;
    }
    seq_slots_cache_.clear();
    seq_slots_dirty_.clear();
    return KvStatus::OK;
}

KvStatus KvCache::find_slot(const std::vector<std::vector<int64_t>> &seq_sets,
                            const std::vector<int64_t> &pos_values,
                            slot_info *out_sinfo) const {
    ubatch ub{};
    ub.seq_sets = seq_sets;
    ub.pos_values = pos_values;
    std::vector<uint32_t> token_streams;
    const KvStatus rc = validate_ubatch_(ub, &token_streams);
    if (rc != KvStatus::OK) {
        return rc;
    }
    const auto sinfo = find_slot(ub, false);
    if (sinfo.empty()) {
        return KvStatus::OOM_SLOT;
    }
    if (out_sinfo) {
        *out_sinfo = sinfo;
    }
    return KvStatus::OK;
}

KvStatus KvCache::apply_ubatch(const std::vector<std::vector<int64_t>> &seq_sets,
                               const std::vector<int64_t> &pos_values,
                               const slot_info &sinfo) {
    ubatch ub{};
    ub.seq_sets = seq_sets;
    ub.pos_values = pos_values;
    return apply_ubatch(sinfo, ub);
}

void KvCache::rollback_ubatch(const std::vector<std::vector<int64_t>> &seq_sets,
                              const std::vector<int64_t> &pos_values,
                              const slot_info &sinfo) {
    ubatch ub{};
    ub.seq_sets = seq_sets;
    ub.pos_values = pos_values;
    rollback_ubatch(sinfo, ub);
}

const std::vector<int32_t> *KvCache::seq_slots(int64_t seq_id) const noexcept {
    if (seq_pos_max_(seq_id) < 0) {
        return nullptr;
    }
    const uint32_t stream = stream_for_seq_const_(seq_id);
    auto &slots = seq_slots_cache_[seq_id];
    slots.clear();
    slots.reserve(maxseq_);
    const auto &cells = v_cells_[stream];
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
            slots.push_back(static_cast<int32_t>(i));
        }
    }
    std::sort(slots.begin(), slots.end(), [&](int32_t a, int32_t b) {
        const int64_t pa = cells.pos_get(static_cast<uint32_t>(a));
        const int64_t pb = cells.pos_get(static_cast<uint32_t>(b));
        if (pa != pb) {
            return pa < pb;
        }
        return a < b;
    });
    return &slots;
}

KvStatus KvCache::seq_cp(int64_t dst_seq,
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

KvStatus KvCache::seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
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

KvStatus KvCache::seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
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

KvStatus KvCache::seq_keep(int64_t seq_id) {
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

int64_t KvCache::seq_pos_max(int64_t seq_id) const noexcept {
    return seq_pos_max_(seq_id);
}

void KvCache::used_slots(std::vector<int32_t> *out) const {
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

bool KvCache::slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const {
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
