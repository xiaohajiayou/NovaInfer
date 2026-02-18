#include "paged_kv.hpp"

#include "../../../utils/check.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace llaisys::runtime::kv_cache {

BlockPool::BlockPool(size_t num_blocks) : num_blocks_(num_blocks) {
    CHECK_ARGUMENT(num_blocks_ > 0, "block_pool: num_blocks must be > 0");
    ref_counts_.assign(num_blocks_, 0);
    free_ids_.reserve(num_blocks_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        free_ids_.push_back(static_cast<int32_t>(num_blocks_ - 1 - i));
    }
}

float BlockPool::usage() const noexcept {
    if (num_blocks_ == 0) {
        return 0.0f;
    }
    return static_cast<float>(num_blocks_ - free_ids_.size()) / static_cast<float>(num_blocks_);
}

bool BlockPool::alloc_n(size_t n, std::vector<int32_t> *out_ids) {
    CHECK_ARGUMENT(out_ids != nullptr, "block_pool: out_ids must not be null");
    if (free_ids_.size() < n) {
        return false;
    }
    out_ids->clear();
    out_ids->reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const int32_t id = free_ids_.back();
        free_ids_.pop_back();
        CHECK_ARGUMENT(id >= 0 && static_cast<size_t>(id) < num_blocks_, "block_pool: invalid free block id");
        CHECK_ARGUMENT(ref_counts_[static_cast<size_t>(id)] == 0, "block_pool: free block has non-zero ref count");
        ref_counts_[static_cast<size_t>(id)] = 1;
        out_ids->push_back(id);
    }
    return true;
}

void BlockPool::free_one(int32_t block_id) {
    CHECK_ARGUMENT(block_id >= 0 && static_cast<size_t>(block_id) < num_blocks_, "block_pool: block_id out of range");
    const size_t idx = static_cast<size_t>(block_id);
    CHECK_ARGUMENT(ref_counts_[idx] >= 0, "block_pool: negative ref count");
    if (ref_counts_[idx] == 0) {
        return;
    }
    ref_counts_[idx] = 0;
    free_ids_.push_back(block_id);
}

void BlockPool::free_many(const std::vector<int32_t> &block_ids) {
    for (const int32_t id : block_ids) {
        free_one(id);
    }
}

int32_t BlockPool::ref_count(int32_t block_id) const {
    CHECK_ARGUMENT(block_id >= 0 && static_cast<size_t>(block_id) < num_blocks_, "block_pool: block_id out of range");
    return ref_counts_[static_cast<size_t>(block_id)];
}

void BlockPool::incref(int32_t block_id) {
    CHECK_ARGUMENT(block_id >= 0 && static_cast<size_t>(block_id) < num_blocks_, "block_pool: block_id out of range");
    auto &rc = ref_counts_[static_cast<size_t>(block_id)];
    CHECK_ARGUMENT(rc > 0, "block_pool: incref on free block");
    ++rc;
}

void BlockPool::decref(int32_t block_id) {
    CHECK_ARGUMENT(block_id >= 0 && static_cast<size_t>(block_id) < num_blocks_, "block_pool: block_id out of range");
    auto &rc = ref_counts_[static_cast<size_t>(block_id)];
    CHECK_ARGUMENT(rc > 0, "block_pool: decref on free block");
    --rc;
    if (rc == 0) {
        free_ids_.push_back(block_id);
    }
}

int32_t BlockTable::block_for_pos(int64_t pos) const {
    CHECK_ARGUMENT(pos >= 0, "block_table: pos must be non-negative");
    const size_t idx = static_cast<size_t>(pos) / block_size_;
    CHECK_ARGUMENT(idx < block_ids_.size(), "block_table: pos out of allocated range");
    return block_ids_[idx];
}

int32_t BlockTable::back_block() const {
    CHECK_ARGUMENT(!block_ids_.empty(), "block_table: no blocks");
    return block_ids_.back();
}

void BlockTable::append_block(int32_t block_id) {
    CHECK_ARGUMENT(block_id >= 0, "block_table: block_id must be non-negative");
    block_ids_.push_back(block_id);
}

void BlockTable::pop_back_block() {
    CHECK_ARGUMENT(!block_ids_.empty(), "block_table: no blocks");
    block_ids_.pop_back();
}

void BlockTable::clear() {
    block_ids_.clear();
    max_pos_ = -1;
}

PagedKvImpl::PagedKvImpl(size_t maxseq, uint32_t n_stream, size_t block_size)
    : maxseq_(maxseq),
      n_stream_(std::max(1u, n_stream)),
      block_size_(std::max<size_t>(1, block_size)),
      nblocks_((maxseq_ + block_size_ - 1) / block_size_) {
    CHECK_ARGUMENT(maxseq_ > 0, "paged_kv: maxseq must be > 0");
    CHECK_ARGUMENT(nblocks_ > 0, "paged_kv: nblocks must be > 0");
}

void PagedKvImpl::init_storage(size_t nlayer,
                               size_t nkvh,
                               size_t dh,
                               llaisysDataType_t dtype,
                               llaisysDeviceType_t device_type,
                               int device_id) {
    storage_ = std::make_unique<KVStorage>();
    storage_->init(nlayer, nblocks_, block_size_, nkvh, dh, dtype, device_type, device_id);
    block_pool_ = std::make_unique<BlockPool>(nblocks_);
    req_to_blocks_.clear();
    seq_to_stream_.clear();
    block_tables_.clear();
}

tensor_t PagedKvImpl::layer_k(size_t layer) const {
    CHECK_ARGUMENT(storage_ != nullptr, "paged_kv: storage is not initialized");
    return storage_->layer_k(layer);
}

tensor_t PagedKvImpl::layer_v(size_t layer) const {
    CHECK_ARGUMENT(storage_ != nullptr, "paged_kv: storage is not initialized");
    return storage_->layer_v(layer);
}

int32_t PagedKvImpl::compute_slot_(int32_t block_id, int32_t block_offset) const {
    const size_t slot = static_cast<size_t>(block_id) * block_size_ + static_cast<size_t>(block_offset);
    CHECK_ARGUMENT(slot < maxseq_, "paged_kv: slot out of range");
    return static_cast<int32_t>(slot);
}

uint32_t PagedKvImpl::stream_for_seq_(int64_t seq_id, std::unordered_map<int64_t, uint32_t> *seq_to_stream) const {
    auto &m = *seq_to_stream;
    if (n_stream_ == 1) {
        m[seq_id] = 0;
        return 0;
    }
    auto it = m.find(seq_id);
    if (it != m.end()) {
        return it->second;
    }
    const uint32_t stream = (seq_id >= 0 && static_cast<uint64_t>(seq_id) < n_stream_) ? static_cast<uint32_t>(seq_id) : 0u;
    m[seq_id] = stream;
    return stream;
}

KvStatus PagedKvImpl::validate_ubatch_(const KvUBatch &ub, std::vector<uint32_t> *token_streams) const {
    if (ub.seq_sets.size() != ub.pos_values.size()) {
        return KvStatus::INVALID_POS;
    }
    if (ub.seq_sets.empty()) {
        return KvStatus::EMPTY_RANGE;
    }

    std::vector<uint32_t> streams(ub.seq_sets.size(), 0);
    for (size_t i = 0; i < ub.seq_sets.size(); ++i) {
        if (ub.seq_sets[i].empty()) {
            return KvStatus::INVALID_SEQ;
        }
        int32_t stream = -1;
        for (int64_t sid : ub.seq_sets[i]) {
            const auto it = seq_to_stream_.find(sid);
            const uint32_t s =
                (it != seq_to_stream_.end())
                    ? it->second
                    : ((sid >= 0 && static_cast<uint64_t>(sid) < n_stream_) ? static_cast<uint32_t>(sid) : 0u);
            if (stream < 0) {
                stream = static_cast<int32_t>(s);
            } else if (stream != static_cast<int32_t>(s)) {
                return KvStatus::INVALID_SEQ;
            }
        }
        if (stream < 0) {
            return KvStatus::INVALID_SEQ;
        }
        streams[i] = static_cast<uint32_t>(stream);
    }
    if (token_streams) {
        *token_streams = std::move(streams);
    }
    return KvStatus::OK;
}

bool PagedKvImpl::normalize_range_(int64_t seq_id, int64_t p0, int64_t p1, int64_t *out_p0, int64_t *out_p1) const {
    if (out_p0 == nullptr || out_p1 == nullptr) {
        return false;
    }
    const int64_t pmax = seq_pos_max(seq_id);
    if (pmax < 0) {
        return false;
    }
    int64_t begin = p0 < 0 ? 0 : p0;
    int64_t end = p1 < 0 ? (pmax + 1) : p1;
    if (begin > end) {
        return false;
    }
    *out_p0 = begin;
    *out_p1 = end;
    return true;
}

PagedKvImpl::ShadowState PagedKvImpl::clone_state_() const {
    ShadowState s(nblocks_);
    CHECK_ARGUMENT(block_pool_ != nullptr, "paged_kv: block_pool is not initialized");
    s.pool = *block_pool_;
    s.req_to_blocks = req_to_blocks_;
    s.seq_to_stream = seq_to_stream_;
    for (const auto &[sid, table] : block_tables_) {
        if (table) {
            s.block_tables[sid] = std::make_unique<BlockTable>(*table);
        }
    }
    return s;
}

KvStatus PagedKvImpl::allocate_slots_(const KvUBatch &ub,
                                      const std::vector<uint32_t> &token_streams,
                                      ShadowState *state,
                                      std::vector<int32_t> *out_slots) const {
    CHECK_ARGUMENT(state != nullptr, "paged_kv: state must not be null");
    CHECK_ARGUMENT(out_slots != nullptr, "paged_kv: out_slots must not be null");
    out_slots->clear();
    out_slots->reserve(ub.seq_sets.size());

    for (size_t i = 0; i < ub.seq_sets.size(); ++i) {
        const int64_t pos = ub.pos_values[i];
        if (pos < 0) {
            return KvStatus::INVALID_POS;
        }
        const uint32_t stream = token_streams[i];

        std::unordered_set<int64_t> dedup;
        dedup.reserve(ub.seq_sets[i].size());
        for (int64_t sid : ub.seq_sets[i]) {
            dedup.insert(sid);
        }

        const size_t block_idx = static_cast<size_t>(pos) / block_size_;
        const int32_t block_off = static_cast<int32_t>(static_cast<size_t>(pos) % block_size_);
        int32_t block_id = -1;

        for (int64_t sid : dedup) {
            const uint32_t st = stream_for_seq_(sid, &state->seq_to_stream);
            if (st != stream) {
                return KvStatus::INVALID_SEQ;
            }

            auto &table_ptr = state->block_tables[sid];
            if (!table_ptr) {
                table_ptr = std::make_unique<BlockTable>(block_size_);
            }
            auto &table = *table_ptr;
            if (table.max_pos() + 1 != pos) {
                return KvStatus::INVALID_POS;
            }

            auto &blocks = state->req_to_blocks[sid];
            if (block_idx < blocks.size()) {
                const int32_t existing = blocks[block_idx];
                if (block_id < 0) {
                    block_id = existing;
                } else if (block_id != existing) {
                    return KvStatus::INVALID_SEQ;
                }
            } else if (block_idx == blocks.size()) {
                if (block_id < 0) {
                    std::vector<int32_t> alloc;
                    if (!state->pool.alloc_n(1, &alloc) || alloc.empty()) {
                        return KvStatus::OOM_SLOT;
                    }
                    block_id = alloc[0];
                } else {
                    state->pool.incref(block_id);
                }
                blocks.push_back(block_id);
                table.append_block(block_id);
            } else {
                return KvStatus::INVALID_POS;
            }
            table.set_max_pos(pos);
        }

        if (block_id < 0) {
            return KvStatus::INTERNAL_ERROR;
        }
        out_slots->push_back(compute_slot_(block_id, block_off));
    }
    return KvStatus::OK;
}

KvSlotInfoVec PagedKvImpl::prepare(const std::vector<KvUBatch> &ubatches) {
    KvSlotInfoVec out;
    auto shadow = clone_state_();
    out.reserve(ubatches.size());

    for (const auto &ub : ubatches) {
        std::vector<uint32_t> token_streams;
        if (validate_ubatch_(ub, &token_streams) != KvStatus::OK) {
            return {};
        }
        std::vector<int32_t> slots;
        if (allocate_slots_(ub, token_streams, &shadow, &slots) != KvStatus::OK) {
            return {};
        }

        KvSlotInfo s{};
        s.resize(1);
        s.s0 = static_cast<int32_t>(token_streams.empty() ? 0 : token_streams[0]);
        s.s1 = s.s0;
        s.strm[0] = static_cast<int64_t>(s.s0);
        s.idxs[0] = std::move(slots);
        out.push_back(std::move(s));
    }
    return out;
}

KvStatus PagedKvImpl::apply_ubatch(const KvSlotInfo &, const KvUBatch &ubatch) {
    std::vector<uint32_t> token_streams;
    const KvStatus rc = validate_ubatch_(ubatch, &token_streams);
    if (rc != KvStatus::OK) {
        return rc;
    }
    auto shadow = clone_state_();
    std::vector<int32_t> slots;
    const KvStatus arc = allocate_slots_(ubatch, token_streams, &shadow, &slots);
    if (arc != KvStatus::OK) {
        return arc;
    }

    *block_pool_ = shadow.pool;
    req_to_blocks_ = std::move(shadow.req_to_blocks);
    seq_to_stream_ = std::move(shadow.seq_to_stream);
    block_tables_.clear();
    for (auto &[sid, table] : shadow.block_tables) {
        block_tables_[sid] = std::move(table);
    }
    return KvStatus::OK;
}

void PagedKvImpl::rollback_ubatch(const KvSlotInfo &, const KvUBatch &ubatch) {
    if (!block_pool_) {
        return;
    }
    std::unordered_map<int64_t, int32_t> rollback_n;
    for (size_t i = 0; i < ubatch.seq_sets.size(); ++i) {
        std::unordered_set<int64_t> dedup;
        for (int64_t sid : ubatch.seq_sets[i]) {
            dedup.insert(sid);
        }
        for (int64_t sid : dedup) {
            rollback_n[sid] += 1;
        }
    }

    for (const auto &[sid, n] : rollback_n) {
        auto it_table = block_tables_.find(sid);
        auto it_blocks = req_to_blocks_.find(sid);
        if (it_table == block_tables_.end() || it_blocks == req_to_blocks_.end() || !it_table->second) {
            continue;
        }
        auto &table = *it_table->second;
        auto &blocks = it_blocks->second;
        for (int32_t step = 0; step < n; ++step) {
            const int64_t pmax = table.max_pos();
            if (pmax < 0) {
                break;
            }
            const size_t block_idx = static_cast<size_t>(pmax) / block_size_;
            const int32_t off = static_cast<int32_t>(static_cast<size_t>(pmax) % block_size_);
            if (block_idx >= blocks.size()) {
                break;
            }
            const int32_t block_id = blocks[block_idx];
            if (off == 0 && !blocks.empty()) {
                table.pop_back_block();
                blocks.pop_back();
                block_pool_->decref(block_id);
            }
            table.set_max_pos(pmax - 1);
        }
    }
}

KvStatus PagedKvImpl::seq_cp(int64_t dst_seq,
                             int64_t src_seq,
                             int64_t p0,
                             int64_t p1,
                             std::vector<int32_t> *src_slots,
                             std::vector<int32_t> *dst_slots) {
    int64_t begin = -1;
    int64_t end = -1;
    if (!normalize_range_(src_seq, p0, p1, &begin, &end)) {
        return KvStatus::INVALID_POS;
    }
    if (begin == end) {
        return KvStatus::EMPTY_RANGE;
    }

    auto it_src = req_to_blocks_.find(src_seq);
    if (it_src == req_to_blocks_.end()) {
        return KvStatus::INVALID_SEQ;
    }
    auto &dst_blocks = req_to_blocks_[dst_seq];
    auto &dst_table_ptr = block_tables_[dst_seq];
    if (!dst_table_ptr) {
        dst_table_ptr = std::make_unique<BlockTable>(block_size_);
    }
    auto &dst_table = *dst_table_ptr;

    (void) stream_for_seq_(src_seq, &seq_to_stream_);
    (void) stream_for_seq_(dst_seq, &seq_to_stream_);

    if (src_slots) {
        src_slots->clear();
    }
    if (dst_slots) {
        dst_slots->clear();
    }

    for (int64_t pos = begin; pos < end; ++pos) {
        const size_t bidx = static_cast<size_t>(pos) / block_size_;
        const int32_t off = static_cast<int32_t>(static_cast<size_t>(pos) % block_size_);
        if (bidx >= it_src->second.size()) {
            return KvStatus::INVALID_POS;
        }
        const int32_t block_id = it_src->second[bidx];
        while (dst_blocks.size() <= bidx) {
            dst_blocks.push_back(block_id);
            dst_table.append_block(block_id);
            block_pool_->incref(block_id);
        }
        dst_table.set_max_pos(pos);
        const int32_t slot = compute_slot_(block_id, off);
        if (src_slots) {
            src_slots->push_back(slot);
        }
        if (dst_slots) {
            dst_slots->push_back(slot);
        }
    }
    return KvStatus::OK;
}

KvStatus PagedKvImpl::seq_rm(int64_t seq_id, int64_t p0, int64_t p1) {
    int64_t begin = -1;
    int64_t end = -1;
    if (!normalize_range_(seq_id, p0, p1, &begin, &end)) {
        return KvStatus::INVALID_POS;
    }
    if (begin == end) {
        return KvStatus::EMPTY_RANGE;
    }

    auto it_table = block_tables_.find(seq_id);
    auto it_blocks = req_to_blocks_.find(seq_id);
    if (it_table == block_tables_.end() || it_blocks == req_to_blocks_.end() || !it_table->second) {
        return KvStatus::INVALID_SEQ;
    }

    const int64_t pmax = it_table->second->max_pos();
    if (end != pmax + 1) {
        // Current paged metadata model only supports tail-truncation removal.
        // Reject non-tail removal explicitly instead of silently doing nothing.
        return KvStatus::INVALID_POS;
    }

    const int64_t new_max = begin - 1;
    const size_t keep_blocks = (new_max < 0) ? 0 : (static_cast<size_t>(new_max) / block_size_ + 1);
    while (it_blocks->second.size() > keep_blocks) {
        const int32_t bid = it_blocks->second.back();
        it_blocks->second.pop_back();
        it_table->second->pop_back_block();
        block_pool_->decref(bid);
    }
    it_table->second->set_max_pos(new_max);
    return KvStatus::OK;
}

KvStatus PagedKvImpl::seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    int64_t begin = -1;
    int64_t end = -1;
    if (!normalize_range_(seq_id, p0, p1, &begin, &end)) {
        return KvStatus::INVALID_POS;
    }
    if (begin == end) {
        return KvStatus::EMPTY_RANGE;
    }

    auto it_table = block_tables_.find(seq_id);
    if (it_table == block_tables_.end() || !it_table->second) {
        return KvStatus::INVALID_SEQ;
    }
    if (end != it_table->second->max_pos() + 1) {
        return KvStatus::INVALID_POS;
    }
    if (delta != 0) {
        return KvStatus::INVALID_POS;
    }
    return KvStatus::OK;
}

KvStatus PagedKvImpl::seq_keep(int64_t seq_id) {
    if (seq_pos_max(seq_id) < 0) {
        return KvStatus::INVALID_SEQ;
    }
    std::vector<int64_t> remove;
    remove.reserve(req_to_blocks_.size());
    for (const auto &[sid, _] : req_to_blocks_) {
        if (sid != seq_id) {
            remove.push_back(sid);
        }
    }
    for (int64_t sid : remove) {
        (void) seq_rm(sid, 0, -1);
    }
    return KvStatus::OK;
}

int64_t PagedKvImpl::seq_pos_max(int64_t seq_id) const noexcept {
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end() || !it->second) {
        return -1;
    }
    return it->second->max_pos();
}

void PagedKvImpl::used_slots(std::vector<int32_t> *out) const {
    CHECK_ARGUMENT(out != nullptr, "paged_kv: out must not be null");
    out->clear();
    out->reserve(maxseq_);

    for (const auto &[sid, table_ptr] : block_tables_) {
        (void) sid;
        if (!table_ptr) {
            continue;
        }
        const auto &table = *table_ptr;
        const int64_t pmax = table.max_pos();
        if (pmax < 0) {
            continue;
        }
        const auto &blocks = table.block_ids();
        const size_t max_bidx = static_cast<size_t>(pmax) / block_size_;
        for (size_t bidx = 0; bidx <= max_bidx && bidx < blocks.size(); ++bidx) {
            const int32_t bid = blocks[bidx];
            const size_t off_max = (bidx == max_bidx) ? (static_cast<size_t>(pmax) % block_size_) : (block_size_ - 1);
            for (size_t off = 0; off <= off_max; ++off) {
                out->push_back(compute_slot_(bid, static_cast<int32_t>(off)));
            }
        }
    }

    std::sort(out->begin(), out->end());
    out->erase(std::unique(out->begin(), out->end()), out->end());
}

bool PagedKvImpl::slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const {
    if (slot < 0 || seq_ids == nullptr || n_seq_id <= 0) {
        return false;
    }
    const int32_t block_id = static_cast<int32_t>(static_cast<size_t>(slot) / block_size_);
    const int32_t block_off = static_cast<int32_t>(static_cast<size_t>(slot) % block_size_);

    for (int32_t i = 0; i < n_seq_id; ++i) {
        const int64_t sid = seq_ids[i];
        auto it_table = block_tables_.find(sid);
        if (it_table == block_tables_.end() || !it_table->second) {
            continue;
        }
        const auto &table = *it_table->second;
        const int64_t pmax = table.max_pos();
        if (pmax < 0) {
            continue;
        }
        const int64_t vmax = std::min<int64_t>(pmax, qpos);
        if (vmax < 0) {
            continue;
        }
        const auto &blocks = table.block_ids();
        const size_t max_bidx = static_cast<size_t>(vmax) / block_size_;
        for (size_t bidx = 0; bidx <= max_bidx && bidx < blocks.size(); ++bidx) {
            if (blocks[bidx] != block_id) {
                continue;
            }
            const int64_t pos = static_cast<int64_t>(bidx * block_size_ + static_cast<size_t>(block_off));
            if (pos <= vmax) {
                return true;
            }
        }
    }
    return false;
}

bool PagedKvImpl::build_attention_plan(const std::vector<std::vector<int64_t>> &seq_sets,
                                       const std::vector<int64_t> &qpos,
                                       std::vector<int32_t> *used_slots,
                                       std::vector<uint8_t> *mask) const {
    CHECK_ARGUMENT(used_slots != nullptr, "paged_kv: used_slots must not be null");
    CHECK_ARGUMENT(mask != nullptr, "paged_kv: mask must not be null");

    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    const bool ok = build_attention_plan_csr(seq_sets, qpos, used_slots, &row_ptr, &col_idx);
    if (!ok) {
        return false;
    }

    const size_t ntoken = seq_sets.size();
    const size_t kvlen = used_slots->size();
    mask->assign(ntoken * kvlen, static_cast<uint8_t>(0));
    for (size_t i = 0; i < ntoken; ++i) {
        const int32_t rb = row_ptr[i];
        const int32_t re = row_ptr[i + 1];
        for (int32_t p = rb; p < re; ++p) {
            (*mask)[i * kvlen + static_cast<size_t>(col_idx[static_cast<size_t>(p)])] = static_cast<uint8_t>(1);
        }
    }
    return true;
}

bool PagedKvImpl::build_attention_plan_csr(const std::vector<std::vector<int64_t>> &seq_sets,
                                           const std::vector<int64_t> &qpos,
                                           std::vector<int32_t> *used_slots,
                                           std::vector<int32_t> *row_ptr,
                                           std::vector<int32_t> *col_idx) const {
    CHECK_ARGUMENT(used_slots != nullptr, "paged_kv: used_slots must not be null");
    CHECK_ARGUMENT(row_ptr != nullptr, "paged_kv: row_ptr must not be null");
    CHECK_ARGUMENT(col_idx != nullptr, "paged_kv: col_idx must not be null");
    if (seq_sets.size() != qpos.size()) {
        return false;
    }
    const size_t ntoken = seq_sets.size();
    if (ntoken == 0) {
        return false;
    }

    std::vector<std::unordered_set<int32_t>> token_slots(ntoken);
    std::unordered_set<int32_t> global_slots;

    for (size_t i = 0; i < ntoken; ++i) {
        std::unordered_set<int64_t> dedup_seq;
        dedup_seq.reserve(seq_sets[i].size());
        for (int64_t sid : seq_sets[i]) {
            dedup_seq.insert(sid);
        }
        if (dedup_seq.empty()) {
            return false;
        }

        auto &tok = token_slots[i];
        for (int64_t sid : dedup_seq) {
            auto it_table = block_tables_.find(sid);
            if (it_table == block_tables_.end() || !it_table->second) {
                continue;
            }
            const auto &table = *it_table->second;
            const int64_t pmax = table.max_pos();
            if (pmax < 0) {
                continue;
            }
            const int64_t vmax = std::min<int64_t>(pmax, qpos[i]);
            if (vmax < 0) {
                continue;
            }

            const auto &blocks = table.block_ids();
            const size_t max_bidx = static_cast<size_t>(vmax) / block_size_;
            for (size_t bidx = 0; bidx <= max_bidx && bidx < blocks.size(); ++bidx) {
                const int32_t bid = blocks[bidx];
                const size_t off_max = (bidx == max_bidx) ? (static_cast<size_t>(vmax) % block_size_) : (block_size_ - 1);
                for (size_t off = 0; off <= off_max; ++off) {
                    const int32_t slot = compute_slot_(bid, static_cast<int32_t>(off));
                    tok.insert(slot);
                    global_slots.insert(slot);
                }
            }
        }
        if (tok.empty()) {
            return false;
        }
    }

    used_slots->assign(global_slots.begin(), global_slots.end());
    std::sort(used_slots->begin(), used_slots->end());
    if (used_slots->empty()) {
        return false;
    }

    std::unordered_map<int32_t, size_t> slot_to_col;
    slot_to_col.reserve(used_slots->size() * 2);
    for (size_t c = 0; c < used_slots->size(); ++c) {
        slot_to_col[(*used_slots)[c]] = c;
    }

    row_ptr->assign(ntoken + 1, 0);
    col_idx->clear();
    col_idx->reserve(global_slots.size());
    for (size_t i = 0; i < ntoken; ++i) {
        std::vector<int32_t> cols;
        cols.reserve(token_slots[i].size());
        for (int32_t slot : token_slots[i]) {
            auto it = slot_to_col.find(slot);
            if (it == slot_to_col.end()) {
                continue;
            }
            cols.push_back(static_cast<int32_t>(it->second));
        }
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        if (cols.empty()) {
            return false;
        }
        col_idx->insert(col_idx->end(), cols.begin(), cols.end());
        (*row_ptr)[i + 1] = static_cast<int32_t>(col_idx->size());
    }
    if (row_ptr->back() != static_cast<int32_t>(col_idx->size())) {
        return false;
    }
    for (size_t i = 1; i < row_ptr->size(); ++i) {
        if ((*row_ptr)[i] < (*row_ptr)[i - 1]) {
            return false;
        }
    }

    return true;
}

} // namespace llaisys::runtime::kv_cache
