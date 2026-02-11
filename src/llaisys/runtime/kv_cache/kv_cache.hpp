#pragma once

#include "kv_cells.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaisys::runtime::kv_cache {

enum class KvStatus : int32_t {
    OK = 0,
    OOM_SLOT = 1,
    INVALID_SEQ = 2,
    INVALID_POS = 3,
    EMPTY_RANGE = 4,
    INTERNAL_ERROR = 5,
};

class KvCache {
public:
    struct stream_copy_info {
        bool empty() const noexcept { return ssrc.empty(); }
        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    struct slot_info {
        using idx_vec_t = std::vector<int32_t>;

        int32_t s0{0};
        int32_t s1{0};

        std::vector<int64_t> strm;
        std::vector<idx_vec_t> idxs;

        int32_t head() const noexcept {
            if (idxs.empty() || idxs[0].empty()) {
                return -1;
            }
            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const noexcept {
            if (idxs.empty()) {
                return 0;
            }
            return idxs[0].size();
        }

        size_t n_stream() const noexcept { return idxs.size(); }
        bool empty() const noexcept { return idxs.empty(); }

        void clear() {
            strm.clear();
            idxs.clear();
            s0 = 0;
            s1 = 0;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    struct ubatch {
        std::vector<std::vector<int64_t>> seq_sets;
        std::vector<int64_t> pos_values;
    };

    explicit KvCache(size_t maxseq, uint32_t n_stream = 1);

    size_t maxseq() const noexcept { return maxseq_; }
    uint32_t n_stream() const noexcept { return n_stream_; }
    size_t free_slot_count() const noexcept;

    slot_info_vec_t prepare(const std::vector<ubatch> &ubatches);
    bool update(bool do_shift, const stream_copy_info &sc_info);

    slot_info find_slot(const ubatch &ubatch, bool cont) const;
    KvStatus apply_ubatch(const slot_info &sinfo, const ubatch &ubatch);
    void rollback_ubatch(const slot_info &sinfo, const ubatch &ubatch);

    // Compatibility wrappers reserved for future runner/scheduler integrations.
    KvStatus alloc_tokens(int64_t seq_id, size_t ntoken, int64_t pos_start, std::vector<int32_t> *out_slots);
    KvStatus alloc_token(const int64_t *seq_ids, int32_t n_seq_id, int64_t pos, int32_t *out_slot);
    KvStatus find_slot(const std::vector<std::vector<int64_t>> &seq_sets,
                       const std::vector<int64_t> &pos_values,
                       slot_info *out_sinfo) const;
    KvStatus apply_ubatch(const std::vector<std::vector<int64_t>> &seq_sets,
                          const std::vector<int64_t> &pos_values,
                          const slot_info &sinfo);
    void rollback_ubatch(const std::vector<std::vector<int64_t>> &seq_sets,
                         const std::vector<int64_t> &pos_values,
                         const slot_info &sinfo);

    const std::vector<int32_t> *seq_slots(int64_t seq_id) const noexcept;

    KvStatus seq_cp(int64_t dst_seq,
                    int64_t src_seq,
                    int64_t p0,
                    int64_t p1,
                    std::vector<int32_t> *src_slots,
                    std::vector<int32_t> *dst_slots);
    KvStatus seq_rm(int64_t seq_id, int64_t p0, int64_t p1);
    KvStatus seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);
    KvStatus seq_keep(int64_t seq_id);
    int64_t seq_pos_max(int64_t seq_id) const noexcept;

    void used_slots(std::vector<int32_t> *out) const;
    bool slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const;

private:
    uint32_t stream_for_seq_(int64_t seq_id);
    uint32_t stream_for_seq_const_(int64_t seq_id) const;

    int64_t seq_pos_max_(int64_t seq_id) const noexcept;
    bool normalize_range_(int64_t seq_id, int64_t p0, int64_t p1, int64_t *out_p0, int64_t *out_p1) const;
    size_t free_slot_count_stream_(uint32_t stream) const noexcept;
    int32_t alloc_slot_(uint32_t stream, int64_t pos);
    KvStatus validate_ubatch_(const ubatch &ub, std::vector<uint32_t> *token_streams) const;

    size_t maxseq_;
    uint32_t n_stream_;

    std::vector<KvCells> v_cells_;
    std::vector<uint32_t> v_heads_;

    mutable std::unordered_map<int64_t, uint32_t> seq_to_stream_;
    mutable std::unordered_map<int64_t, std::vector<int32_t>> seq_slots_cache_;
    mutable std::unordered_set<int64_t> seq_slots_dirty_;
};

} // namespace llaisys::runtime::kv_cache
