from __future__ import annotations

from llaisys.engine.block_manager import BlockManager
from llaisys.engine.sequence import Sequence
from llaisys.engine.types import SamplingParams


def _seq(req_id: str, sid: int, toks: list[int], block_size: int = 4) -> Sequence:
    return Sequence(
        request_id=req_id,
        seq_id=sid,
        token_ids=toks,
        sampling_params=SamplingParams(max_new_tokens=8),
        block_size=block_size,
    )


def test_prefix_cache_match_and_reset():
    bm = BlockManager(block_size=4, num_blocks=8)

    s1 = _seq("req-1", 1, [1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
    bm.prepare_sequence(s1)
    assert s1.num_cached_tokens == 0
    bm.allocate(s1)

    s2 = _seq("req-2", 2, [1, 2, 3, 4, 5, 6, 7, 8, 9], block_size=4)
    bm.prepare_sequence(s2)
    bm.allocate(s2)
    assert s2.num_cached_tokens == 8

    st = bm.stats()
    assert st.prefix_hits == 1
    assert st.prefix_misses == 1
    assert st.prefix_saved_tokens == 8

    bm.reset_prefix_cache()
    st2 = bm.stats()
    assert st2.prefix_hits == 0
    assert st2.prefix_misses == 0
    assert st2.prefix_saved_tokens == 0

    s3 = _seq("req-3", 3, [1, 2, 3, 4], block_size=4)
    bm.prepare_sequence(s3)
    assert s3.num_cached_tokens == 0


def test_prefix_cache_hit_survives_donor_free():
    bm = BlockManager(block_size=4, num_blocks=8)
    donor = _seq("req-donor", 10, [1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
    bm.allocate(donor)

    hit = _seq("req-hit", 11, [1, 2, 3, 4], block_size=4)
    bm.prepare_sequence(hit)
    bm.allocate(hit)
    assert hit.num_cached_tokens == 4

    bm.deallocate(donor)
    miss = _seq("req-miss", 12, [1, 2, 3, 4], block_size=4)
    bm.prepare_sequence(miss)
    bm.allocate(miss)
    assert miss.num_cached_tokens == 4


def test_prefix_cache_allocation_reuses_donor_blocks():
    bm = BlockManager(block_size=4, num_blocks=8)

    donor = _seq("req-donor", 1, [1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
    bm.prepare_sequence(donor)
    bm.allocate(donor)
    donor_blocks = list(donor.block_table)
    assert len(donor_blocks) == 2

    hit = _seq("req-hit", 2, [1, 2, 3, 4, 5, 6, 7, 8, 9], block_size=4)
    bm.prepare_sequence(hit)
    bm.allocate(hit)
    assert hit.num_cached_tokens == 8

    # First two blocks are shared from donor prefix, the tail block is newly allocated.
    assert hit.block_table[:2] == donor_blocks
    assert len(hit.block_table) == 3
    assert hit.block_table[2] not in donor_blocks

    # Shared blocks keep non-zero refcount until both seqs are released.
    for bid in donor_blocks:
        assert bm.blocks[bid].ref_count == 2

    bm.deallocate(hit)
    for bid in donor_blocks:
        assert bm.blocks[bid].ref_count == 1
    bm.deallocate(donor)


def test_block_manager_release_reclaims_all_blocks():
    bm = BlockManager(block_size=4, num_blocks=8)
    donor = _seq("req-donor", 1, [1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
    hit = _seq("req-hit", 2, [1, 2, 3, 4, 5, 6, 7, 8, 9], block_size=4)

    bm.allocate(donor)
    bm.allocate(hit)
    st_mid = bm.stats()
    assert st_mid.used_blocks == 3
    assert st_mid.free_blocks == 5

    bm.deallocate(hit)
    bm.deallocate(donor)
    st_end = bm.stats()
    assert st_end.used_blocks == 0
    assert st_end.free_blocks == 8
    for block in bm.blocks:
        assert block.ref_count == 0


def test_prefix_cache_counters_and_peak_are_stable_across_rounds():
    bm = BlockManager(block_size=4, num_blocks=8)

    for round_idx in range(2):
        donor = _seq(f"req-donor-{round_idx}", 10 + round_idx, [1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        hit = _seq(f"req-hit-{round_idx}", 20 + round_idx, [1, 2, 3, 4, 5, 6, 7, 8, 9], block_size=4)

        bm.allocate(donor)
        bm.allocate(hit)
        assert hit.num_cached_tokens == 8

        bm.deallocate(hit)
        bm.deallocate(donor)
        st = bm.stats()
        assert st.used_blocks == 0
        assert st.free_blocks == 8

    st_final = bm.stats()
    # With persistent hash index, round-2 donor and round-2 hit both match prefix blocks.
    assert st_final.prefix_hits == 3
    assert st_final.prefix_misses == 1
    assert st_final.prefix_saved_tokens == 24
    assert st_final.peak_used_blocks == 3


def test_prefix_cache_disabled_keeps_prefix_counters_zero():
    bm = BlockManager(block_size=4, num_blocks=8, enable_prefix_caching=False)
    s1 = _seq("req-1", 1, [1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
    s2 = _seq("req-2", 2, [1, 2, 3, 4, 5, 6, 7, 8], block_size=4)

    bm.prepare_sequence(s1)
    bm.allocate(s1)
    bm.prepare_sequence(s2)
    bm.allocate(s2)

    st = bm.stats()
    assert st.prefix_hits == 0
    assert st.prefix_misses == 0
    assert st.prefix_saved_tokens == 0
    assert s2.num_cached_tokens == 0
