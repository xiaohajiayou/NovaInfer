from __future__ import annotations

import numpy as np

from llaisys.engine.executor import Executor
from llaisys.engine.scheduler import SchedulerOutputs
from llaisys.engine.sequence import Sequence
from llaisys.engine.types import SamplingParams


class DummyWorker:
    def execute(self, plan):
        _ = plan
        # Return out order intentionally swapped to validate req-id mapping.
        output_ids = [1, 0]
        logits_rows = [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ]
        return output_ids, logits_rows


class CaptureWorker:
    def __init__(self):
        self.last_plan = None

    def execute(self, plan):
        self.last_plan = plan
        output_ids = [1]
        logits_rows = [np.array([0.0, 1.0], dtype=np.float32)]
        return output_ids, logits_rows


class FakeSampler:
    def sample(self, logits_rows, params):
        _ = (logits_rows, params)
        raise AssertionError("sample() should not be called in this test")

    def sample_per_row(self, logits_rows, params_rows):
        _ = logits_rows
        # Encode params identity into outputs for assertion.
        return [int(p.top_k) for p in params_rows]


def test_executor_uses_per_request_sampling_params_in_output_order():
    worker = DummyWorker()
    sampler = FakeSampler()
    ex = Executor(worker=worker, sampler=sampler)

    seq1 = Sequence(
        request_id="req-1",
        seq_id=1,
        token_ids=[11],
        sampling_params=SamplingParams(max_new_tokens=4, top_k=1),
        block_size=16,
    )
    seq2 = Sequence(
        request_id="req-2",
        seq_id=2,
        token_ids=[22],
        sampling_params=SamplingParams(max_new_tokens=4, top_k=1),
        block_size=16,
    )
    outputs = SchedulerOutputs(scheduled_seqs=[seq1, seq2], is_prefill=False)

    sampled, req_ids = ex.execute_scheduler_step(
        outputs,
        sampling_params_by_req={
            "req-1": SamplingParams(top_k=11),
            "req-2": SamplingParams(top_k=22),
        },
    )

    assert req_ids == ["req-2", "req-1"]
    assert sampled == [22, 11]


def test_executor_prefill_flattens_only_uncached_suffix():
    worker = CaptureWorker()
    ex = Executor(worker=worker, sampler=FakeSampler())
    seq = Sequence(
        request_id="req-1",
        seq_id=1,
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_new_tokens=4, top_k=7),
        block_size=4,
    )
    seq.num_cached_tokens = 2
    outputs = SchedulerOutputs(scheduled_seqs=[seq], is_prefill=True)

    sampled, req_ids = ex.execute_scheduler_step(
        outputs,
        sampling_params_by_req={"req-1": SamplingParams(top_k=7)},
    )
    assert req_ids == ["req-1"]
    assert sampled == [7]
    assert worker.last_plan is not None
    assert worker.last_plan.token_ids == [12, 13]
    assert worker.last_plan.pos_ids == [2, 3]
