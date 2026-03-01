import pytest

from llaisys.engine.model_registry import ModelRegistry
from llaisys.engine.types import SamplingParams
from llaisys.entrypoints.llm import LLM
from llaisys.libllaisys.model import KvCacheLayout
from test.utils.dummy_model_runner import DummyModelRunner



class DummyRunner(DummyModelRunner):
    pass


def _build_llm() -> LLM:
    registry = ModelRegistry(_factories={})
    registry.register(
        "dummy",
        lambda model_path, device: DummyRunner(
            max_seq_len=32,
            end_token_id=5,
            kv_cache_layout=KvCacheLayout.BLOCK,
        ),
    )
    llm = LLM(
        model="/tmp/unused",
        model_type="dummy",
        model_registry=registry,
    )
    llm._encode_prompt = lambda prompt: [1, 2] if prompt == "p0" else [2, 3]
    return llm


def test_llm_generate_token_batch_output_shape():
    llm = _build_llm()
    out = llm.generate([[1, 2]], sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0))
    assert isinstance(out, list)
    assert len(out) == 1
    row = out[0]
    assert row["finish_reason"] == "eos_token"
    assert row["status"] == "finished_stopped"
    assert row["token_ids"] == [3, 4, 5]


def test_llm_generate_rejects_legacy_token_list_input():
    llm = _build_llm()
    with pytest.raises(TypeError, match="inputs must be str, list\\[str\\], or list\\[list\\[int\\]\\] for batch mode"):
        _ = llm.generate([1, 2], sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0))


def test_llm_generate_single_prompt_output_shape():
    llm = _build_llm()
    out = llm.generate("p0", sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0))
    assert isinstance(out, list)
    assert len(out) == 1
    row = out[0]
    assert row["finish_reason"] == "eos_token"
    assert row["status"] == "finished_stopped"
    assert row["token_ids"] == [3, 4, 5]
    assert row["text"] is None
    assert row["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


def test_llm_generate_prompt_batch_and_params_list():
    llm = _build_llm()
    params = [
        SamplingParams(max_new_tokens=2, top_k=1, top_p=1.0, temperature=1.0),
        SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
    ]
    out = llm.generate(["p0", "p1"], sampling_params=params)
    assert len(out) == 2
    assert out[0]["finish_reason"] == "length"
    assert out[0]["token_ids"] == [3, 4]
    assert out[1]["finish_reason"] == "eos_token"
    assert out[1]["token_ids"] == [4, 5]


def test_llm_stream_single_prompt():
    llm = _build_llm()
    chunks = list(llm.stream("p0", sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0)))
    token_chunks = [c for c in chunks if not c.is_finished]
    assert [int(c.token_id) for c in token_chunks] == [3, 4, 5]
    assert chunks[-1].is_finished is True
    assert chunks[-1].finish_reason == "eos_token"


if __name__ == "__main__":
    test_llm_generate_token_batch_output_shape()
    test_llm_generate_rejects_legacy_token_list_input()
    test_llm_generate_single_prompt_output_shape()
    test_llm_generate_prompt_batch_and_params_list()
    test_llm_stream_single_prompt()
    print("\033[92mtest_llm_entrypoint passed!\033[0m")
