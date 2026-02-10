import io
import sys

import numpy as np

from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.types import SamplingParams

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class DummyRunner:
    def __init__(self):
        self.max_seq_len = 32
        self.end_token_id = 4

    def decode_batch(self, token_ids, pos_ids=None, seq_ids=None, logits_mask=None):
        if logits_mask is None:
            logits_mask = [0] * len(token_ids)
            logits_mask[-1] = 1

        out_ids = []
        rows = []
        for i, tok in enumerate(token_ids):
            if int(logits_mask[i]) == 0:
                continue
            out_ids.append(i)
            row = np.zeros((8,), dtype=np.float32)
            nxt = (int(tok) + 1) % 8
            row[nxt] = 1.0
            rows.append(row)
        return out_ids, rows

    def decode_tokens(self, token_ids):
        return "".join(chr(ord("a") + int(t)) for t in token_ids)


def test_offline_engine_argmax_loop():
    engine = LLMEngine(model_runner=DummyRunner())
    out = engine.generate(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=8))
    assert out.token_ids == [1, 2, 3, 4]
    assert out.finish_reason == "eos_token"
    assert out.text == "de"
    assert out.usage == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


def test_offline_engine_stream_loop():
    engine = LLMEngine(model_runner=DummyRunner())
    chunks = list(engine.stream(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=8)))
    token_chunks = [c for c in chunks if not c.is_finished]
    assert [int(c.token_id) for c in token_chunks] == [3, 4]
    assert "".join((c.text_delta or "") for c in token_chunks) == "de"
    assert chunks[-1].is_finished is True
    assert chunks[-1].finish_reason == "eos_token"


if __name__ == "__main__":
    test_offline_engine_argmax_loop()
    test_offline_engine_stream_loop()
    print("\033[92mtest_offline passed!\033[0m")
