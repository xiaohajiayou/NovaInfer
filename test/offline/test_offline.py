from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.types import SamplingParams
from llaisys.libllaisys.model import KvCacheLayout
from test.utils.dummy_model_runner import DummyModelRunner



class DummyRunner(DummyModelRunner):
    pass


def test_offline_engine_argmax_loop():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=4, kv_cache_layout=KvCacheLayout.BLOCK))
    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.token_ids == [1, 2, 3, 4]
    assert out.finish_reason == "eos_token"
    assert out.text is None
    assert out.usage == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


def test_offline_engine_stream_loop():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=4, kv_cache_layout=KvCacheLayout.BLOCK))
    chunks = list(
        engine.stream(
            inputs=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
        )
    )
    token_chunks = [c for c in chunks if not c.is_finished]
    assert [int(c.token_id) for c in token_chunks] == [3, 4]
    assert "".join((c.text_delta or "") for c in token_chunks) == ""
    assert chunks[-1].is_finished is True
    assert chunks[-1].finish_reason == "eos_token"


if __name__ == "__main__":
    test_offline_engine_argmax_loop()
    test_offline_engine_stream_loop()
    print("\033[92mtest_offline passed!\033[0m")
