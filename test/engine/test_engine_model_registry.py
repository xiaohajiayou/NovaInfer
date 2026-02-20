
import numpy as np

from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.model_registry import ModelRegistry
from llaisys.engine.types import SamplingParams



class DummyRunner:
    def __init__(self):
        self.max_seq_len = 16
        self.end_token_id = 5

    def decode_batch(self, token_ids, pos_ids=None, seq_ids=None, logits_mask=None):
        if logits_mask is None:
            logits_mask = [0] * len(token_ids)
            logits_mask[-1] = 1
        out_ids = []
        rows = []
        for i, tok in enumerate(token_ids):
            if not logits_mask[i]:
                continue
            out_ids.append(i)
            row = np.zeros((8,), dtype=np.float32)
            row[(int(tok) + 1) % 8] = 1.0
            rows.append(row)
        return out_ids, rows


def _build_dummy_runner(model_path, device):
    _ = (model_path, device)
    return DummyRunner()


def test_engine_uses_model_registry_for_worker_creation():
    registry = ModelRegistry(_factories={})
    registry.register("dummy", _build_dummy_runner)

    engine = LLMEngine(
        model_type="dummy",
        model_path="/tmp/unused",
        model_registry=registry,
    )
    out = engine.generate(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=3))

    assert out.token_ids == [1, 2, 3, 4, 5]
    assert out.finish_reason == "eos_token"


if __name__ == "__main__":
    test_engine_uses_model_registry_for_worker_creation()
    print("\033[92mtest_engine_model_registry passed!\033[0m")
