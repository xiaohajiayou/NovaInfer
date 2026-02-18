from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ..engine.engine_client import EngineClient
from ..engine.llm_engine import LLMEngine
from ..engine.model_registry import ModelRegistry
from ..engine.types import SamplingParams
from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout


class LLM:
    """Offline entrypoint aligned with vLLM-style LLM.generate semantics."""

    def __init__(
        self,
        model: Path | str,
        model_type: str = "qwen2",
        device: DeviceType = DeviceType.CPU,
        kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
        kv_cache_block_size: int = 16,
        max_model_len: int | None = None,
        kv_cache_capacity_tokens: int | None = None,
        model_registry: ModelRegistry | None = None,
    ):
        self._model_path = Path(model)
        self._tokenizer = None
        engine = LLMEngine(
            model_type=model_type,
            model_path=model,
            device=device,
            kv_cache_layout=kv_cache_layout,
            kv_cache_block_size=kv_cache_block_size,
            max_model_len=max_model_len,
            kv_cache_capacity_tokens=kv_cache_capacity_tokens,
            model_registry=model_registry,
        )
        self._engine_client = EngineClient(engine)

    def submit(
        self,
        inputs: Sequence[int] | str,
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stop_token_ids: Optional[Sequence[int]] = None,
        stop: Optional[Sequence[str]] = None,
        sampling_params: SamplingParams | None = None,
    ) -> str:
        token_ids = self._normalize_single_input(inputs)
        params = self._build_sampling_params(
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )
        return self._engine_client.submit(inputs=token_ids, sampling_params=params)

    def step(self):
        return self._engine_client.step()

    def collect(self, request_id: str):
        return self._engine_client.collect(request_id)

    def stream(
        self,
        inputs: Sequence[int] | str,
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stop_token_ids: Optional[Sequence[int]] = None,
        stop: Optional[Sequence[str]] = None,
        sampling_params: SamplingParams | None = None,
    ):
        token_ids = self._normalize_single_input(inputs)
        params = self._build_sampling_params(
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )
        return self._engine_client.stream(inputs=token_ids, sampling_params=params)

    def cancel(self, request_id: str) -> bool:
        return self._engine_client.cancel(request_id)

    def close(self) -> None:
        close_fn = getattr(self._engine_client, "close", None)
        if callable(close_fn):
            close_fn()

    def generate(
        self,
        inputs,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stop_token_ids: Optional[Sequence[int]] = None,
        stop: Optional[Sequence[str]] = None,
        use_tqdm: bool = False,
    ):
        _ = use_tqdm

        # Backward-compatible fast path: token ids in, token ids out.
        if self._is_token_id_list(inputs):
            params = self._build_sampling_params(
                sampling_params=sampling_params if isinstance(sampling_params, SamplingParams) else None,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stop_token_ids=stop_token_ids,
                stop=stop,
            )
            output = self._engine_client.generate(inputs=inputs, sampling_params=params)
            return output.token_ids

        batch_inputs, prompt_token_lens = self._normalize_batch_inputs(inputs)
        params_list = self._normalize_sampling_params_list(
            sampling_params=sampling_params,
            batch_size=len(batch_inputs),
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )

        request_ids = []
        for token_ids, params in zip(batch_inputs, params_list):
            req_id = self._engine_client.submit(inputs=token_ids, sampling_params=params)
            request_ids.append(req_id)

        pending = set(request_ids)
        outputs = {}
        while pending:
            _ = self._engine_client.step()
            for req_id in list(pending):
                out = self._engine_client.collect(req_id)
                if out is None:
                    continue
                outputs[req_id] = out
                pending.remove(req_id)

        packed = []
        for req_id, prompt_len in zip(request_ids, prompt_token_lens):
            out = outputs[req_id]
            completion_tokens = out.token_ids[prompt_len:]
            packed.append(
                {
                    "request_id": out.request_id,
                    "status": out.status.value,
                    "finish_reason": out.finish_reason,
                    "token_ids": completion_tokens,
                    "text": out.text,
                    "usage": out.usage,
                }
            )
        return packed

    def _build_sampling_params(
        self,
        sampling_params: SamplingParams | None,
        max_new_tokens: Optional[int],
        top_k: int,
        top_p: float,
        temperature: float,
        stop_token_ids: Optional[Sequence[int]],
        stop: Optional[Sequence[str]],
    ) -> SamplingParams:
        if sampling_params is not None:
            return sampling_params
        return SamplingParams(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_token_ids=tuple(stop_token_ids or ()),
            stop=tuple(stop or ()),
        )

    def _normalize_sampling_params_list(
        self,
        sampling_params: SamplingParams | list[SamplingParams] | None,
        batch_size: int,
        max_new_tokens: Optional[int],
        top_k: int,
        top_p: float,
        temperature: float,
        stop_token_ids: Optional[Sequence[int]],
        stop: Optional[Sequence[str]],
    ) -> list[SamplingParams]:
        if isinstance(sampling_params, list):
            if len(sampling_params) != batch_size:
                raise ValueError("sampling_params list length mismatch")
            return sampling_params
        one = self._build_sampling_params(
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )
        return [one for _ in range(batch_size)]

    def _normalize_single_input(self, inputs: Sequence[int] | str) -> list[int]:
        if isinstance(inputs, str):
            return self._encode_prompt(inputs)
        if self._is_token_id_list(inputs):
            return [int(t) for t in inputs]
        raise TypeError("inputs must be str or Sequence[int]")

    def _normalize_batch_inputs(self, inputs) -> tuple[list[list[int]], list[int]]:
        if isinstance(inputs, str):
            token_ids = self._encode_prompt(inputs)
            return [token_ids], [len(token_ids)]

        if isinstance(inputs, Sequence) and inputs and isinstance(inputs[0], str):
            token_batches = [self._encode_prompt(str(p)) for p in inputs]
            lens = [len(x) for x in token_batches]
            return token_batches, lens

        if isinstance(inputs, Sequence) and inputs and isinstance(inputs[0], Sequence):
            token_batches = []
            lens = []
            for seq in inputs:
                if not self._is_token_id_list(seq):
                    raise TypeError("batch token inputs must be list[list[int]]")
                ids = [int(t) for t in seq]
                token_batches.append(ids)
                lens.append(len(ids))
            return token_batches, lens

        raise TypeError("inputs must be str, list[str], or list[list[int]] for batch mode")

    @staticmethod
    def _is_token_id_list(inputs) -> bool:
        if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
            return False
        return all(isinstance(t, int) for t in inputs)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        return self._tokenizer

    def _encode_prompt(self, prompt: str) -> list[int]:
        tokenizer = self._get_tokenizer()
        # Align offline string-input behavior with chat-style parity tests.
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            return tokenizer.encode(text)
        return tokenizer.encode(prompt)
