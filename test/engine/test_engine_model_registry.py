from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.model_registry import ModelRegistry
from llaisys.engine.types import SamplingParams
from llaisys.libllaisys.model import KvCacheLayout
from test.utils.dummy_model_runner import DummyModelRunner


def _build_dummy_runner(model_path, device):
    _ = (model_path, device)
    return DummyModelRunner(max_seq_len=16, end_token_id=5, kv_cache_layout=KvCacheLayout.BLOCK)


def test_engine_uses_model_registry_for_worker_creation():
    registry = ModelRegistry(_factories={})
    registry.register("dummy", _build_dummy_runner)

    engine = LLMEngine(
        model_type="dummy",
        model_path="/tmp/unused",
        model_registry=registry,
    )
    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=3, top_k=1, top_p=1.0, temperature=1.0),
    )

    assert out.token_ids == [1, 2, 3, 4, 5]
    assert out.finish_reason == "eos_token"


if __name__ == "__main__":
    test_engine_uses_model_registry_for_worker_creation()
    print("\033[92mtest_engine_model_registry passed!\033[0m")
