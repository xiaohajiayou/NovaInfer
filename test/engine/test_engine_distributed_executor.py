from unittest.mock import patch

import pytest

from llaisys.engine.llm_engine import LLMEngine


class _DummyExecutor:
    def __init__(self):
        self.worker = object()
        self.closed = False

    def close(self):
        self.closed = True


def test_engine_selects_mp_executor_when_requested():
    dummy = _DummyExecutor()
    with patch("llaisys.engine.llm_engine.MPExecutor", return_value=dummy) as mp_ctor:
        engine = LLMEngine(
            model_type="dummy",
            model_path="/tmp/unused",
            max_model_len=16,
            end_token_id=5,
            num_kvcache_blocks=4,
            device="nvidia",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
        )
    assert mp_ctor.call_count == 1
    assert engine._worker is dummy.worker  # noqa: SLF001
    engine.close()
    assert dummy.closed is True


def test_engine_rejects_unknown_distributed_executor_backend():
    with pytest.raises(NotImplementedError, match="unsupported distributed_executor_backend"):
        LLMEngine(
            model_type="dummy",
            model_path="/tmp/unused",
            max_model_len=16,
            end_token_id=5,
            num_kvcache_blocks=4,
            distributed_executor_backend="bogus",
        )
