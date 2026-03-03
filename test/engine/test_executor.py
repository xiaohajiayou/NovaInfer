from __future__ import annotations

import llaisys

from llaisys.engine.executor import Executor
from llaisys.engine.scheduler import SchedulerOutputs
from llaisys.engine.sequence import Sequence
from llaisys.engine.types import SamplingParams


class DummyWorker:
    def __init__(self):
        self._result = None

    def execute_model(self, outputs, sampling_params=None, sampling_params_by_req=None):
        _ = outputs
        _ = sampling_params
        _ = sampling_params_by_req
        # Return request order intentionally swapped to validate downstream mapping.
        sampled_ids = llaisys.Tensor((2,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
        sampled_ids.copy_from_sequence([22, 11])
        self._result = (sampled_ids, ["req-2", "req-1"])
        return None

    def sample_tokens(self, grammar_output=None):
        _ = grammar_output
        return self._result


class CaptureWorker:
    def __init__(self):
        self.last_outputs = None
        self.last_sampling_params = None
        self.last_sampling_params_by_req = None
        self._result = None

    def execute_model(self, outputs, sampling_params=None, sampling_params_by_req=None):
        self.last_outputs = outputs
        self.last_sampling_params = sampling_params
        self.last_sampling_params_by_req = sampling_params_by_req
        sampled_ids = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
        sampled_ids.copy_from_sequence([7])
        self._result = (sampled_ids, ["req-1"])
        return None

    def sample_tokens(self, grammar_output=None):
        _ = grammar_output
        return self._result


def test_executor_uses_per_request_sampling_params_in_output_order():
    worker = DummyWorker()
    ex = Executor(worker=worker)

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

    sampled_t, sampled_req_ids = ex.execute_scheduler_step(
        outputs,
        sampling_params_by_req={
            "req-1": SamplingParams(top_k=11),
            "req-2": SamplingParams(top_k=22),
        },
    )

    assert sampled_t.tolist() == [22, 11]
    assert sampled_req_ids == ["req-2", "req-1"]


def test_executor_prefill_flattens_only_uncached_suffix():
    worker = CaptureWorker()
    ex = Executor(worker=worker)
    seq = Sequence(
        request_id="req-1",
        seq_id=1,
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_new_tokens=4, top_k=7),
        block_size=4,
    )
    seq.num_cached_tokens = 2
    outputs = SchedulerOutputs(scheduled_seqs=[seq], is_prefill=True)

    sampled_t, sampled_req_ids = ex.execute_scheduler_step(
        outputs,
        sampling_params_by_req={"req-1": SamplingParams(top_k=7)},
    )
    assert sampled_t.tolist() == [7]
    assert sampled_req_ids == ["req-1"]
    assert worker.last_outputs is outputs
    assert worker.last_sampling_params is None
    assert worker.last_sampling_params_by_req is not None
    assert int(worker.last_sampling_params_by_req["req-1"].top_k) == 7
