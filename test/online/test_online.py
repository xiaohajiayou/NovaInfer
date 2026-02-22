from __future__ import annotations

import queue
import time

import numpy as np

from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.types import SamplingParams
from llaisys.server.async_engine import AsyncLLMEngine
from llaisys.server.openai_server import (
    OpenAIServer,
    _StreamingReasoningState,
    _ThinkingReasoningParser,
)
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


def test_online_chat_completion_stream_token_ids_match_non_stream():
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
    non_stream = server.handle_chat(req)
    expected = [int(t) for t in non_stream["token_ids"]]

    stream_req = ChatCompletionRequest(
        model="qwen2",
        messages=[ChatMessage(role="user", content="hello")],
        stream=True,
        max_tokens=8,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    chunks = list(server.handle_chat_stream(stream_req))
    stream_ids = [
        int(c["token_id"])
        for c in chunks
        if not c.get("is_finished") and c.get("token_id") is not None
    ]
    assert stream_ids == expected


def test_async_stream_late_subscribe_replays_prefix_and_tail():
    server = _make_server()
    params = SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0)
    prompt = [1, 2]
    req_id = server._async_engine.submit(inputs=prompt, sampling_params=params)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        current = server._async_engine.inner_engine.get_completion_tokens(req_id) or []
        if len(current) >= 2:
            break
        time.sleep(0.005)

    stream_q: queue.Queue = queue.Queue()
    server._async_engine._cmd_q.put(("watch_stream", req_id, stream_q))

    seen_ids: list[int] = []
    finished = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            item = stream_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        if item.token_id is not None:
            seen_ids.append(int(item.token_id))
        if item.is_finished:
            finished = True
            break

    assert finished, "stream should emit terminal chunk after late subscribe"
    out = server._async_engine.collect(req_id)
    assert out is not None
    expected = [int(t) for t in out.token_ids[len(prompt) :]]
    assert seen_ids == expected


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


def test_reasoning_parser_non_stream_split():
    parser = _ThinkingReasoningParser()
    reasoning, content = parser.extract_reasoning(
        "<think>internal</think>answer",
        include_reasoning=False,
    )
    assert reasoning is None
    assert content == "answer"


def test_reasoning_parser_non_stream_implicit_prefix_reasoning():
    parser = _ThinkingReasoningParser()
    reasoning, content = parser.extract_reasoning(
        "hidden chain</think>final answer",
        include_reasoning=False,
    )
    assert reasoning is None
    assert content == "final answer"


def test_reasoning_parser_streaming_split():
    parser = _ThinkingReasoningParser()
    state = _StreamingReasoningState()

    r1, c1 = parser.extract_reasoning_streaming(
        state,
        "<thi",
        include_reasoning=True,
        finalize=False,
    )
    assert r1 == "<thi"
    assert c1 is None

    r2, c2 = parser.extract_reasoning_streaming(
        state,
        "nk>abc</think>ok",
        include_reasoning=True,
        finalize=True,
    )
    assert r2 == "nk>abc"
    assert c2 == "ok"


def test_reasoning_parser_streaming_implicit_prefix_reasoning():
    parser = _ThinkingReasoningParser()
    state = _StreamingReasoningState()

    r1, c1 = parser.extract_reasoning_streaming(
        state,
        "hidden chain",
        include_reasoning=False,
        finalize=False,
    )
    assert r1 is None
    assert c1 is None

    r2, c2 = parser.extract_reasoning_streaming(
        state,
        "</think>final answer",
        include_reasoning=False,
        finalize=True,
    )
    assert r2 is None
    assert c2 == "final answer"
