from __future__ import annotations

from llaisys.engine.executor import Executor
from llaisys.engine.scheduler import SchedulerOutputs
from llaisys.engine.sequence import Sequence
from llaisys.engine.types import SamplingParams


class DummyWorker:
    def __init__(self):
        self._result = None

    def execute_model(self, outputs):
        _ = outputs
        # Return request order intentionally swapped to validate downstream mapping.
        self._result = [22, 11]
        return None

    def sample_tokens(self, grammar_output=None):
        _ = grammar_output
        return self._result


class CaptureWorker:
    def __init__(self):
        self.last_outputs = None
        self._result = None

    def execute_model(self, outputs):
        self.last_outputs = outputs
        self._result = [7]
        return None

    def sample_tokens(self, grammar_output=None):
        _ = grammar_output
        return self._result


def test_executor_returns_python_token_ids():
    worker = DummyWorker()
    ex = Executor(worker=worker)

    seq1 = Sequence(
        seq_id=1,
        token_ids=[11],
        sampling_params=SamplingParams(max_new_tokens=4, top_k=1),
        block_size=16,
    )
    seq2 = Sequence(
        seq_id=2,
        token_ids=[22],
        sampling_params=SamplingParams(max_new_tokens=4, top_k=1),
        block_size=16,
    )
    outputs = SchedulerOutputs(scheduled_seqs=[seq1, seq2], is_prefill=False)

    sampled = ex.execute_scheduler_step(outputs)
    assert sampled == [22, 11]


def test_executor_prefill_flattens_only_uncached_suffix():
    worker = CaptureWorker()
    ex = Executor(worker=worker)
    seq = Sequence(
        seq_id=1,
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_new_tokens=4, top_k=7),
        block_size=4,
    )
    seq.num_cached_tokens = 2
    outputs = SchedulerOutputs(scheduled_seqs=[seq], is_prefill=True)

    sampled = ex.execute_scheduler_step(outputs)
    assert sampled == [7]
    assert worker.last_outputs is outputs
