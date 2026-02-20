from __future__ import annotations

from llaisys.engine.scheduler import RequestScheduler
from llaisys.engine.sequence import Sequence
from llaisys.engine.types import SamplingParams


def _seq(req_id: str, sid: int, toks: list[int]) -> Sequence:
    return Sequence(
        request_id=req_id,
        seq_id=sid,
        token_ids=toks,
        sampling_params=SamplingParams(max_new_tokens=8),
        block_size=16,
    )


def test_scheduler_prefers_prefill_when_waiting_exists():
    sch = RequestScheduler(max_num_seqs=1, max_num_batched_tokens=4096, block_size=16, num_kvcache_blocks=0)

    s1 = _seq("req-1", 1, [11, 12])
    sch.add(s1)

    out1 = sch.schedule()
    assert out1 is not None
    assert out1.is_prefill is True
    assert [s.request_id for s in out1.scheduled_seqs] == ["req-1"]

    # New waiting request arrives while req-1 is already running.
    s2 = _seq("req-2", 2, [21, 22, 23])
    sch.add(s2)

    out2 = sch.schedule()
    assert out2 is not None
    assert out2.is_prefill is True
    assert [s.request_id for s in out2.scheduled_seqs] == ["req-2"]


def test_scheduler_block_stats_reflect_allocate_and_finish():
    sch = RequestScheduler(max_num_seqs=1, max_num_batched_tokens=4096, block_size=16, num_kvcache_blocks=4)
    s1 = _seq("req-1", 1, [i for i in range(20)])  # needs 2 blocks
    sch.add(s1)

    st0 = sch.block_stats()
    assert st0.used_blocks == 0
    assert st0.free_blocks == 4

    out = sch.schedule()
    assert out is not None and out.is_prefill is True

    st1 = sch.block_stats()
    assert st1.used_blocks == 2
    assert st1.free_blocks == 2

    sch.finish("req-1")
    st2 = sch.block_stats()
    assert st2.used_blocks == 0
    assert st2.free_blocks == 4


def test_scheduler_block_peak_watermark_persists_across_rounds():
    sch = RequestScheduler(max_num_seqs=1, max_num_batched_tokens=4096, block_size=16, num_kvcache_blocks=4)

    s1 = _seq("req-1", 1, [i for i in range(20)])  # 2 blocks
    sch.add(s1)
    out1 = sch.schedule()
    assert out1 is not None and out1.is_prefill is True
    sch.finish("req-1")

    s2 = _seq("req-2", 2, [i for i in range(20)])  # 2 blocks
    sch.add(s2)
    out2 = sch.schedule()
    assert out2 is not None and out2.is_prefill is True
    sch.finish("req-2")

    st = sch.block_stats()
    assert st.used_blocks == 0
    assert st.free_blocks == 4
    assert st.peak_used_blocks == 2
