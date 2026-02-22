from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from ..engine.types import SamplingParams, StreamChunk
from .async_engine import AsyncLLMEngine
from .schemas import ChatCompletionRequest, ChatMessage


@dataclass
class _ReasoningSplit:
    reasoning: str
    content: str


@dataclass
class _StreamingReasoningState:
    previous_text: str = ""


class _ThinkingReasoningParser:
    """vLLM-aligned (<think>...</think>) reasoning parser semantics."""

    _START = "<think>"
    _END = "</think>"

    def extract_reasoning(self, text: str, include_reasoning: bool) -> tuple[str | None, str]:
        # vLLM BaseThinkingReasoningParser.extract_reasoning:
        # remove prefix before first start token; if no end token, treat all as reasoning.
        parts = text.partition(self._START)
        body = parts[2] if parts[1] else parts[0]
        if self._END not in body:
            reasoning = body
            content = None
        else:
            reasoning, _, content = body.partition(self._END)
        out_reasoning = reasoning if include_reasoning and reasoning else None
        return out_reasoning, content or ""

    def extract_reasoning_streaming(
        self,
        state: _StreamingReasoningState,
        delta_text: str,
        include_reasoning: bool,
        *,
        finalize: bool,
    ) -> tuple[str | None, str | None]:
        # Text-equivalent of:
        # - vLLM BaseThinkingReasoningParser.extract_reasoning_streaming
        # - vLLM DeepSeekR1ReasoningParser override
        _ = finalize
        if not delta_text:
            return None, None

        previous_text = state.previous_text
        current_text = previous_text + delta_text
        state.previous_text = current_text

        start_in_prev = self._START in previous_text
        end_in_prev = self._END in previous_text
        start_in_delta = self._START in delta_text
        end_in_delta = self._END in delta_text

        reasoning: str | None = None
        content: str | None = None

        # BaseThinkingReasoningParser branches
        if start_in_prev:
            if end_in_delta:
                end_idx = delta_text.find(self._END)
                reasoning = delta_text[:end_idx]
                tail = delta_text[end_idx + len(self._END) :]
                content = tail if tail else None
            elif end_in_prev:
                content = delta_text
            else:
                reasoning = delta_text
        elif start_in_delta:
            if end_in_delta:
                start_idx = delta_text.find(self._START)
                end_idx = delta_text.find(self._END)
                reasoning = delta_text[start_idx + len(self._START) : end_idx]
                tail = delta_text[end_idx + len(self._END) :]
                content = tail if tail else None
            else:
                reasoning = delta_text
        else:
            content = delta_text

        # DeepSeekR1 override: models may omit start token.
        if (not start_in_prev) and (not start_in_delta):
            if end_in_delta:
                end_idx = delta_text.find(self._END)
                reasoning = delta_text[:end_idx]
                tail = delta_text[end_idx + len(self._END) :]
                content = tail if tail else None
            elif end_in_prev:
                content = delta_text
                reasoning = None
            else:
                reasoning = delta_text
                content = None

        out_reasoning = reasoning if include_reasoning and reasoning else None
        out_content = content if content else None
        return out_reasoning, out_content


class OpenAIServer:
    """Minimal OpenAI-compatible handler (in-process, framework-agnostic)."""

    def __init__(self, async_engine: AsyncLLMEngine):
        self._async_engine = async_engine
        self._default_sampling = self._async_engine.get_default_sampling_params()
        self._max_model_len = int(self._async_engine.get_max_model_len())
        self._reasoning_parser = _ThinkingReasoningParser()

    def close(self) -> None:
        close_fn = getattr(self._async_engine, "close", None)
        if callable(close_fn):
            close_fn()

    def handle_chat(self, req: ChatCompletionRequest) -> dict:
        prompt = self._messages_to_prompt(req.messages)
        params = self._to_sampling_params(req, prompt_len=len(prompt))
        out = self._async_engine.generate(inputs=prompt, sampling_params=params)
        completion_tokens = out.token_ids[out.usage["prompt_tokens"] :] if out.usage else out.token_ids
        raw_text = out.text if out.text is not None else ""
        reasoning, content = self._reasoning_parser.extract_reasoning(raw_text, include_reasoning=req.include_reasoning)
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning"] = reasoning
        return {
            "id": f"chatcmpl-{out.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
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
        params = self._to_sampling_params(req, prompt_len=len(prompt))
        parser_state = _StreamingReasoningState()
        for chunk in self._async_engine.stream(inputs=prompt, sampling_params=params):
            reasoning_delta, content_delta = self._reasoning_parser.extract_reasoning_streaming(
                parser_state,
                chunk.text_delta or "",
                include_reasoning=req.include_reasoning,
                finalize=bool(chunk.is_finished),
            )
            yield self._stream_chunk_to_openai(
                req.model,
                chunk,
                reasoning_delta=reasoning_delta,
                content_delta=content_delta,
            )

    def cancel(self, request_id: str) -> bool:
        return self._async_engine.cancel(request_id)

    def _messages_to_prompt(self, messages: list[ChatMessage] | tuple[ChatMessage, ...]) -> list[int]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return self._async_engine.encode_chat_messages(payload)

    def _to_sampling_params(self, req: ChatCompletionRequest, prompt_len: int) -> SamplingParams:
        # Match vLLM precedence:
        # request explicit params > model generation_config defaults > neutral defaults.
        context_budget = max(0, int(self._max_model_len) - int(prompt_len))
        cap = self._default_sampling.get("max_tokens")
        if cap is not None:
            context_budget = min(context_budget, int(cap))
        max_tokens = int(req.max_tokens) if req.max_tokens is not None else int(context_budget)
        top_k = req.top_k if req.top_k is not None else int(self._default_sampling.get("top_k", 0))
        top_p = req.top_p if req.top_p is not None else float(self._default_sampling.get("top_p", 1.0))
        temperature = (
            req.temperature
            if req.temperature is not None
            else float(self._default_sampling.get("temperature", 1.0))
        )
        return SamplingParams(
            max_new_tokens=max(0, int(max_tokens)),
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            stop=req.stop,
            stop_token_ids=req.stop_token_ids,
        )

    @staticmethod
    def _stream_chunk_to_openai(
        model: str,
        chunk: StreamChunk,
        reasoning_delta: str | None = None,
        content_delta: str | None = None,
    ) -> dict:
        delta: dict[str, str] = {}
        if content_delta:
            delta["content"] = content_delta
        if reasoning_delta:
            delta["reasoning"] = reasoning_delta
        if not delta:
            delta["content"] = ""
        return {
            "id": f"chatcmpl-{chunk.request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": chunk.finish_reason if chunk.is_finished else None,
                }
            ],
            "request_id": chunk.request_id,
            "status": chunk.status.value,
            "is_finished": chunk.is_finished,
            "token_id": chunk.token_id,
        }
