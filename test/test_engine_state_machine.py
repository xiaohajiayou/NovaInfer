
import numpy as np

from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.types import RequestStatus, SamplingParams



class DummyRunner:
    def __init__(self, max_seq_len=32, end_token_id=4):
        self.max_seq_len = max_seq_len
        self.end_token_id = end_token_id

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
            nxt = (int(tok) + 1) % 16
            row[nxt] = 1.0
            rows.append(row)
        return out_ids, rows

    def decode_tokens(self, token_ids):
        return "".join(chr(ord("a") + int(t)) for t in token_ids)


def test_state_machine_stopped_path():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=4))
    out = engine.generate(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=8))

    assert out.status == RequestStatus.FINISHED_STOPPED
    assert out.finish_reason == "eos_token"
    assert out.token_ids == [1, 2, 3, 4]

    history = engine.get_request_history(out.request_id)
    assert history == [
        RequestStatus.WAITING,
        RequestStatus.RUNNING,
        RequestStatus.FINISHED_STOPPED,
    ]


def test_state_machine_length_capped_path():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=99))
    out = engine.generate(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=2))

    assert out.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert out.finish_reason == "length"
    assert out.token_ids == [1, 2, 3, 4]

    history = engine.get_request_history(out.request_id)
    assert history == [
        RequestStatus.WAITING,
        RequestStatus.RUNNING,
        RequestStatus.FINISHED_LENGTH_CAPPED,
    ]


def test_state_machine_aborted_on_invalid_prompt():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=1, end_token_id=4))
    out = engine.generate(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=2))
    assert out.status == RequestStatus.FINISHED_ABORTED
    assert out.finish_reason == "aborted"

    req_id = engine.last_request_id
    assert req_id is not None
    assert engine.get_request_status(req_id) == RequestStatus.FINISHED_ABORTED
    history = engine.get_request_history(req_id)
    assert history == [
        RequestStatus.WAITING,
        RequestStatus.RUNNING,
        RequestStatus.FINISHED_ABORTED,
    ]


def test_submit_step_collect_contract():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=4))
    req_id = engine.submit(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=8))

    # drive engine loop explicitly
    for _ in range(16):
        out = engine.collect(req_id)
        if out is not None:
            break
        _ = engine.step()

    out = engine.collect(req_id)
    assert out is not None
    assert out.request_id == req_id
    assert out.status == RequestStatus.FINISHED_STOPPED
    assert out.finish_reason == "eos_token"
    assert out.token_ids == [1, 2, 3, 4]
    assert out.usage is not None
    assert out.usage["prompt_tokens"] == 2
    assert out.usage["completion_tokens"] == 2
    assert out.usage["total_tokens"] == 4
    assert out.text == "de"


def test_cancel_contract():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=99))
    req_id = engine.submit(inputs=[1, 2], sampling_params=SamplingParams(max_new_tokens=8))
    assert engine.cancel(req_id) is True

    out = engine.collect(req_id)
    assert out is not None
    assert out.status == RequestStatus.FINISHED_ABORTED
    assert out.finish_reason == "aborted"


def test_stop_string_contract():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=99))
    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=8, stop=("de",)),
    )
    assert out.status == RequestStatus.FINISHED_STOPPED
    assert out.finish_reason == "stop_string"
    # completion would be \"de...\"; output text is truncated at stop string.
    assert out.text == ""


if __name__ == "__main__":
    test_state_machine_stopped_path()
    test_state_machine_length_capped_path()
    test_state_machine_aborted_on_invalid_prompt()
    test_submit_step_collect_contract()
    test_cancel_contract()
    test_stop_string_contract()
    print("\033[92mtest_engine_state_machine passed!\033[0m")
