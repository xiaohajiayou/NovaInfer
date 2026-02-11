#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace llaisys::runtime::kv_cache {

// Internal KV operation status used in C++ layers.
enum class KvStatus : int32_t {
    OK = 0,
    OOM_SLOT = 1,
    INVALID_SEQ = 2,
    INVALID_POS = 3,
    EMPTY_RANGE = 4,
    INTERNAL_ERROR = 5,
};

/**
 * @brief Per-model KV slot allocator/metadata manager.
 *
 * This class manages a global slot pool (size = maxseq) and a per-sequence
 * logical view (pos -> slot). Tensor copy/read is handled elsewhere; this
 * class only tracks slot ownership and logical positions.
 */
class KvCache {
public:
    /**
     * @brief Construct KV cache metadata with a fixed slot capacity.
     * @param maxseq Total number of physical KV slots available in this cache.
     */
    explicit KvCache(size_t maxseq);

    /** @brief Get total physical slot capacity. */
    size_t maxseq() const noexcept { return maxseq_; }
    /** @brief Get current number of free physical slots. */
    size_t free_slot_count() const noexcept { return free_slots_.size(); }

    /**
     * @brief Allocate slots for a contiguous token range of one sequence.
     * @param seq_id Logical sequence identifier.
     * @param ntoken Number of new tokens to allocate.
     * @param pos_start Logical start position for the first token.
     * @param out_slots Optional output list of allocated slot ids.
     * @return KvStatus::OK on success.
     */
    KvStatus alloc_tokens(int64_t seq_id, size_t ntoken, int64_t pos_start, std::vector<int32_t> *out_slots);
    /**
     * @brief Allocate one slot for a token that belongs to one/multiple sequences.
     * @param seq_ids Sequence-id array.
     * @param n_seq_id Number of sequence ids in seq_ids.
     * @param pos Logical token position shared by the seq set.
     * @param out_slot Optional output slot id.
     * @return KvStatus::OK on success.
     */
    KvStatus alloc_token(const int64_t *seq_ids, int32_t n_seq_id, int64_t pos, int32_t *out_slot);

    /**
     * @brief Query all slot ids of a sequence in logical position order.
     * @param seq_id Logical sequence identifier.
     * @return Pointer to internal slot vector, or nullptr if seq does not exist.
     */
    const std::vector<int32_t> *seq_slots(int64_t seq_id) const noexcept;

    /**
     * @brief Prepare copy plan from one sequence range to another sequence tail.
     *
     * This allocates new destination slots and returns source/destination slot
     * mapping for upper-layer tensor copy.
     *
     * @param dst_seq Destination sequence id.
     * @param src_seq Source sequence id.
     * @param p0 Inclusive logical start pos in source sequence.
     * @param p1 Exclusive logical end pos in source sequence.
     * @param src_slots Optional output source slot list.
     * @param dst_slots Optional output destination slot list.
     * @return KvStatus::OK on success.
     */
    KvStatus seq_cp_prepare(int64_t dst_seq,
                            int64_t src_seq,
                            int64_t p0,
                            int64_t p1,
                            std::vector<int32_t> *src_slots,
                            std::vector<int32_t> *dst_slots);

    /**
     * @brief Remove [p0, p1) logical range from a sequence and free those slots.
     * @param seq_id Logical sequence identifier.
     * @param p0 Inclusive logical start position.
     * @param p1 Exclusive logical end position.
     * @return KvStatus::OK on success.
     */
    KvStatus seq_rm(int64_t seq_id, int64_t p0, int64_t p1);

    /**
     * @brief Shift logical positions in [p0, p1) by delta.
     * @param seq_id Logical sequence identifier.
     * @param p0 Inclusive logical start position.
     * @param p1 Exclusive logical end position.
     * @param delta Position shift value.
     * @return KvStatus::OK on success.
     */
    KvStatus seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);

    /**
     * @brief Keep only one sequence and clear all other sequences.
     * @param seq_id Sequence id to keep.
     * @return KvStatus::OK on success.
     */
    KvStatus seq_keep(int64_t seq_id);

    /**
     * @brief Get max logical position of a sequence.
     * @param seq_id Logical sequence identifier.
     * @return Max logical position, or -1 if sequence does not exist/empty.
     */
    int64_t seq_pos_max(int64_t seq_id) const noexcept;

    /**
     * @brief Collect all currently used slots sorted by logical position.
     * @param out Ordered slot-id list.
     */
    void used_slots(std::vector<int32_t> *out) const;

    /**
     * @brief Test whether one slot is visible to a query token.
     * @param slot Physical slot id.
     * @param seq_ids Query sequence-id set.
     * @param n_seq_id Number of query sequence ids.
     * @param qpos Query logical position.
     * @return True when seq-set intersects and slot_pos <= qpos.
     */
    bool slot_visible_for(int32_t slot, const int64_t *seq_ids, int32_t n_seq_id, int64_t qpos) const;

private:
    /**
     * @brief Per-sequence metadata.
     */
    struct SeqState {
        /** @brief Logical mapping: pos_to_slot[pos] = physical slot id. */
        std::vector<int32_t> pos_to_slot;
    };

    /**
     * @brief Ensure sequence state exists and return mutable reference.
     * @param seq_id Logical sequence identifier.
     * @return Mutable sequence metadata.
     */
    SeqState &ensure_seq_(int64_t seq_id);

    /**
     * @brief Allocate one physical slot and bind it to (seq_id, pos).
     * @param seq_id Logical sequence identifier.
     * @param pos Logical position in the sequence.
     * @return Slot id on success, -1 if slot pool is empty.
     */
    int32_t alloc_slot_(int64_t pos);

    /**
     * @brief Release one physical slot back to free list.
     * @param slot Physical slot id to free.
     */
    void free_slot_(int32_t slot);

    /**
     * @brief Remove one sequence and free all of its slots.
     * @param seq_id Logical sequence identifier.
     */
    void clear_seq_(int64_t seq_id);

    /** @brief Total physical slot capacity. */
    size_t maxseq_;
    /** @brief Sequence table: seq_id -> sequence metadata. */
    std::unordered_map<int64_t, SeqState> seq_states_;
    /** @brief Free physical slot stack (LIFO). */
    std::vector<int32_t> free_slots_;
    /** @brief Slot owner map: slot_seq_sets_[slot] = associated seq-id set. */
    std::vector<std::vector<int64_t>> slot_seq_sets_;
    /** @brief Slot logical pos map: slot_pos_[slot] = logical position, -1 means free. */
    std::vector<int64_t> slot_pos_;
};

} // namespace llaisys::runtime::kv_cache
