from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from llaisys.engine.llm_engine import LLMEngine
from llaisys.libllaisys.model import KvCacheLayout
from llaisys.server.async_engine import AsyncLLMEngine
from llaisys.server.openai_server import OpenAIServer
from llaisys.server.schemas import ChatCompletionRequest, ChatMessage
from test.utils.dummy_model_runner import DummyModelRunner


class DummyRunner(DummyModelRunner):
    pass


def _make_server() -> OpenAIServer:
    engine = LLMEngine(
        model_runner=DummyRunner(max_seq_len=64, end_token_id=7, kv_cache_layout=KvCacheLayout.BLOCK)
    )
    async_engine = AsyncLLMEngine(engine=engine)
    return OpenAIServer(async_engine)


def _run_stream(server: OpenAIServer, content: str) -> list[dict]:
    req = ChatCompletionRequest(
        model="qwen2",
        messages=[ChatMessage(role="user", content=content)],
        stream=True,
        max_tokens=8,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    return list(server.handle_chat_stream(req))


def test_online_stream_request_isolation():
    server = _make_server()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_run_stream, server, "hello-a")
            fut_b = pool.submit(_run_stream, server, "hello-b")
            chunks_a = fut_a.result(timeout=5.0)
            chunks_b = fut_b.result(timeout=5.0)

        assert chunks_a and chunks_b

        req_ids_a = {c["request_id"] for c in chunks_a if "request_id" in c}
        req_ids_b = {c["request_id"] for c in chunks_b if "request_id" in c}
        assert len(req_ids_a) == 1
        assert len(req_ids_b) == 1
        assert req_ids_a != req_ids_b

        assert chunks_a[-1]["is_finished"] is True
        assert chunks_b[-1]["is_finished"] is True
        assert chunks_a[-1]["choices"][0]["finish_reason"] is not None
        assert chunks_b[-1]["choices"][0]["finish_reason"] is not None
    finally:
        server._async_engine.close()
