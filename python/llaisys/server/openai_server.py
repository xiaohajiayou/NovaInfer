from __future__ import annotations

import time
from typing import Iterable

from ..engine.types import SamplingParams, StreamChunk
from .async_engine import AsyncLLMEngine
from .schemas import ChatCompletionRequest, ChatMessage


class OpenAIServer:
    """Minimal OpenAI-compatible handler (in-process, framework-agnostic)."""

    def __init__(self, async_engine: AsyncLLMEngine):
        self._async_engine = async_engine

    def close(self) -> None:
        close_fn = getattr(self._async_engine, "close", None)
        if callable(close_fn):
            close_fn()

    def handle_chat(self, req: ChatCompletionRequest) -> dict:
        prompt = self._messages_to_prompt(req.messages)
        params = self._to_sampling_params(req)
        out = self._async_engine.generate(inputs=prompt, sampling_params=params)
        completion_tokens = out.token_ids[out.usage["prompt_tokens"] :] if out.usage else out.token_ids
        text = out.text if out.text is not None else ""
        return {
            "id": f"chatcmpl-{out.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": out.finish_reason,
                }
            ],
            "usage": out.usage,
            "token_ids": completion_tokens,
            "request_id": out.request_id,
            "status": out.status.value,
        }

    def handle_chat_stream(self, req: ChatCompletionRequest) -> Iterable[dict]:
        prompt = self._messages_to_prompt(req.messages)
        params = self._to_sampling_params(req)
        for chunk in self._async_engine.stream(inputs=prompt, sampling_params=params):
            yield self._stream_chunk_to_openai(req.model, chunk)

    def cancel(self, request_id: str) -> bool:
        return self._async_engine.cancel(request_id)

    def _messages_to_prompt(self, messages: list[ChatMessage] | tuple[ChatMessage, ...]) -> list[int]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return self._async_engine.encode_chat_messages(payload)

    @staticmethod
    def _to_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
        return SamplingParams(
            max_new_tokens=req.max_tokens,
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
            stop=req.stop,
            stop_token_ids=req.stop_token_ids,
        )

    @staticmethod
    def _stream_chunk_to_openai(model: str, chunk: StreamChunk) -> dict:
        return {
            "id": f"chatcmpl-{chunk.request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.text_delta or ""},
                    "finish_reason": chunk.finish_reason if chunk.is_finished else None,
                }
            ],
            "request_id": chunk.request_id,
            "status": chunk.status.value,
            "is_finished": chunk.is_finished,
            "token_id": chunk.token_id,
        }
