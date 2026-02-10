
import numpy as np

from llaisys.engine.model_registry import ModelRegistry
from llaisys.engine.types import SamplingParams
from llaisys.entrypoints.llm import LLM



class DummyRunner:
    def __init__(self):
        self.max_seq_len = 32
        self.end_token_id = 5

    def decode_batch(self, token_ids, pos_ids=None, seq_ids=None, logits_mask=None):
        _ = (pos_ids, seq_ids)
        if logits_mask is None:
            logits_mask = [0] * len(token_ids)
            logits_mask[-1] = 1

        out_ids = []
        rows = []
        for i, tok in enumerate(token_ids):
            if int(logits_mask[i]) == 0:
                continue
            out_ids.append(i)
            row = np.zeros((16,), dtype=np.float32)
            row[(int(tok) + 1) % 16] = 1.0
            rows.append(row)
        return out_ids, rows

    def decode_tokens(self, token_ids):
        return "".join(chr(ord("a") + int(t)) for t in token_ids)


def _build_llm() -> LLM:
    registry = ModelRegistry(_factories={})
    registry.register("dummy", lambda model_path, device: DummyRunner())
    llm = LLM(
        model="/tmp/unused",
        model_type="dummy",
        model_registry=registry,
    )
    llm._encode_prompt = lambda prompt: [1, 2] if prompt == "p0" else [2, 3]
    return llm


def test_llm_generate_token_compat():
    llm = _build_llm()
    out = llm.generate([1, 2], sampling_params=SamplingParams(max_new_tokens=8))
    assert out == [1, 2, 3, 4, 5]


def test_llm_generate_single_prompt_output_shape():
    llm = _build_llm()
    out = llm.generate("p0", sampling_params=SamplingParams(max_new_tokens=8))
    assert isinstance(out, list)
    assert len(out) == 1
    row = out[0]
    assert row["finish_reason"] == "eos_token"
    assert row["status"] == "finished_stopped"
    assert row["token_ids"] == [3, 4, 5]
    assert row["text"] == "def"
    assert row["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


def test_llm_generate_prompt_batch_and_params_list():
    llm = _build_llm()
    params = [
        SamplingParams(max_new_tokens=2),
        SamplingParams(max_new_tokens=8),
    ]
    out = llm.generate(["p0", "p1"], sampling_params=params)
    assert len(out) == 2
    assert out[0]["finish_reason"] == "length"
    assert out[0]["token_ids"] == [3, 4]
    assert out[1]["finish_reason"] == "eos_token"
    assert out[1]["token_ids"] == [4, 5]


def test_llm_stream_single_prompt():
    llm = _build_llm()
    chunks = list(llm.stream("p0", sampling_params=SamplingParams(max_new_tokens=8)))
    token_chunks = [c for c in chunks if not c.is_finished]
    assert [int(c.token_id) for c in token_chunks] == [3, 4, 5]
    assert chunks[-1].is_finished is True
    assert chunks[-1].finish_reason == "eos_token"


if __name__ == "__main__":
    test_llm_generate_token_compat()
    test_llm_generate_single_prompt_output_shape()
    test_llm_generate_prompt_batch_and_params_list()
    test_llm_stream_single_prompt()
    print("\033[92mtest_llm_entrypoint passed!\033[0m")
