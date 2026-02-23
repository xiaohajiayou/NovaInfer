from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from llaisys.engine.llm_engine import LLMEngine
from llaisys.server.async_engine import AsyncLLMEngine
from llaisys.server.http_server import LlaisysHTTPServer
from llaisys.server.openai_server import OpenAIServer

# Disable environment proxy for localhost tests; CI/dev shells may inject HTTP_PROXY.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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


def _start_http_server() -> LlaisysHTTPServer:
    engine = LLMEngine(model_runner=DummyRunner())
    async_engine = AsyncLLMEngine(engine=engine)
    server = OpenAIServer(async_engine)
    http = LlaisysHTTPServer(server, host="127.0.0.1", port=0)
    try:
        http.start()
    except PermissionError:
        pytest.skip("socket bind is not permitted in this environment")
    return http


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _NO_PROXY_OPENER.open(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_http_chat_completion_non_stream():
    http = _start_http_server()
    try:
        url = f"http://{http.host}:{http.port}/v1/chat/completions"
        payload = {
            "model": "qwen2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "max_tokens": 8,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
        }
        data = _post_json(url, payload)
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["status"].startswith("finished_")
        assert isinstance(data["request_id"], str)
    finally:
        http.stop()


def test_http_chat_completion_stream_sse():
    http = _start_http_server()
    try:
        url = f"http://{http.host}:{http.port}/v1/chat/completions"
        payload = {
            "model": "qwen2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 8,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        seen_done = False
        with _NO_PROXY_OPENER.open(req, timeout=10) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    seen_done = True
                    break
        assert seen_done is True
    finally:
        http.stop()
