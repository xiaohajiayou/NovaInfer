from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from llaisys.engine.llm_engine import LLMEngine
from llaisys.server.async_engine import AsyncLLMEngine
from llaisys.server.openai_server import OpenAIServer
from llaisys.server.schemas import ChatCompletionRequest, ChatMessage


def _collect_stream(server: OpenAIServer, prompt: str, max_tokens: int = 12) -> list[dict]:
    req = ChatCompletionRequest(
        model="qwen2",
        messages=[ChatMessage(role="user", content=prompt)],
        stream=True,
        max_tokens=max_tokens,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    return list(server.handle_chat_stream(req))


def _assert_single_request_stream(chunks: list[dict]) -> str:
    assert chunks, "stream must not be empty"
    req_ids = [c.get("request_id") for c in chunks if c.get("request_id")]
    assert req_ids, "stream must contain request_id"
    unique = set(req_ids)
    assert len(unique) == 1, f"mixed request ids in one stream: {unique}"

    final = chunks[-1]
    assert final.get("is_finished") is True
    assert final.get("choices", [{}])[0].get("finish_reason") is not None
    return next(iter(unique))


@pytest.mark.requires_model
@pytest.mark.online
def test_online_real_model_multisession_stream_isolation(require_model_path: str):
    engine = LLMEngine(
        model_type="qwen2",
        model_path=require_model_path,
        kv_cache_auto_capacity=True,
    )
    async_engine = AsyncLLMEngine(engine=engine)
    server = OpenAIServer(async_engine)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_collect_stream, server, "请用一句中文介绍你自己。", 12)
            fut_b = pool.submit(_collect_stream, server, "Reply in English: what are you doing now?", 12)
            chunks_a = fut_a.result(timeout=180)
            chunks_b = fut_b.result(timeout=180)

        req_id_a = _assert_single_request_stream(chunks_a)
        req_id_b = _assert_single_request_stream(chunks_b)
        assert req_id_a != req_id_b, "two sessions should map to different request ids"

        # Non-final chunks should carry token ids. This catches common stream corruption.
        non_final_a = [c for c in chunks_a if not c.get("is_finished")]
        non_final_b = [c for c in chunks_b if not c.get("is_finished")]
        assert non_final_a and non_final_b
        assert any(c.get("token_id") is not None for c in non_final_a)
        assert any(c.get("token_id") is not None for c in non_final_b)
    finally:
        async_engine.close()
