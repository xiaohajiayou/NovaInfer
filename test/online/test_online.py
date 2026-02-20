from __future__ import annotations

import numpy as np

from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.types import SamplingParams
from llaisys.server.async_engine import AsyncLLMEngine
from llaisys.server.openai_server import OpenAIServer
from llaisys.server.schemas import ChatCompletionRequest, ChatMessage


class DummyRunner:
    def __init__(self):
        self.max_seq_len = 64
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


def _make_server() -> OpenAIServer:
    engine = LLMEngine(model_runner=DummyRunner())
    async_engine = AsyncLLMEngine(engine=engine)
    return OpenAIServer(async_engine)


def test_online_chat_completion_non_stream():
    server = _make_server()
    req = ChatCompletionRequest(
        model="qwen2",
        messages=[ChatMessage(role="user", content="hello")],
        stream=False,
        max_tokens=8,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    resp = server.handle_chat(req)
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(resp["choices"][0]["message"]["content"], str)
    assert resp["status"].startswith("finished_")


def test_online_chat_completion_stream():
    server = _make_server()
    req = ChatCompletionRequest(
        model="qwen2",
        messages=[ChatMessage(role="user", content="hello")],
        stream=True,
        max_tokens=8,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    chunks = list(server.handle_chat_stream(req))
    assert len(chunks) > 0
    assert chunks[-1]["is_finished"] is True
    assert chunks[-1]["choices"][0]["finish_reason"] is not None


def test_online_cancel_request():
    server = _make_server()
    params = SamplingParams(max_new_tokens=32, top_k=1, top_p=1.0, temperature=1.0)
    req_id = server._async_engine.submit(inputs=[1, 2, 3], sampling_params=params)
    assert server.cancel(req_id) is True
    out = server._async_engine.collect(req_id)
    assert out is not None
    assert out.status.value.startswith("finished_")


def test_online_concurrent_requests():
    server = _make_server()
    params = SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0)
    req_id_a = server._async_engine.submit(inputs=[1, 2], sampling_params=params)
    req_id_b = server._async_engine.submit(inputs=[2, 3], sampling_params=params)

    for _ in range(32):
        server._async_engine.step()
        if server._async_engine.is_finished(req_id_a) and server._async_engine.is_finished(req_id_b):
            break

    out_a = server._async_engine.collect(req_id_a)
    out_b = server._async_engine.collect(req_id_b)
    assert out_a is not None and out_b is not None
    assert out_a.request_id != out_b.request_id
    assert out_a.status.value.startswith("finished_")
    assert out_b.status.value.startswith("finished_")
