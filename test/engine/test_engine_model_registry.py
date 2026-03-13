from unittest.mock import patch

from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.model_registry import ModelRegistry
from llaisys.engine.types import SamplingParams
from test.utils.dummy_model_runner import DummyModelRunner
from test.utils.engine_testkit import InjectedWorker


def test_engine_passes_model_registry_to_worker():
    registry = ModelRegistry(_factories={})
    runner = DummyModelRunner(max_seq_len=16, end_token_id=5)
    fake_worker = InjectedWorker(runner)

    with patch("llaisys.engine.llm_engine.Worker", side_effect=lambda *args, **kwargs: fake_worker) as worker_ctor:
        engine = LLMEngine(
            model_type="dummy",
            model_path="/tmp/unused",
            model_registry=registry,
            max_model_len=16,
            end_token_id=5,
            num_kvcache_blocks=4,
        )

    assert worker_ctor.call_count == 1
    assert worker_ctor.call_args.kwargs["model_registry"] is registry

    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=3, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.token_ids == [1, 2, 3, 4, 5]
    assert out.finish_reason == "eos_token"


if __name__ == "__main__":
    test_engine_passes_model_registry_to_worker()
    print("\033[92mtest_engine_model_registry passed!\033[0m")
