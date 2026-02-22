from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from ..engine.engine_client import EngineClient
from ..engine.config import EngineConfig
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
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        kv_cache_auto_capacity: bool = False,
        kv_cache_memory_utilization: float = 0.9,
        model_registry: ModelRegistry | None = None,
    ):
        self._model_path = Path(model)
        self._tokenizer = None
        self._configure_cudnn_prebuild_hint_(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            kv_cache_block_size=kv_cache_block_size,
        )
        cfg = EngineConfig(
            model_type=model_type,
            model_path=model,
            device=device,
            kv_cache_layout=kv_cache_layout,
            kv_cache_block_size=kv_cache_block_size,
            max_model_len=max_model_len,
            kv_cache_capacity_tokens=kv_cache_capacity_tokens,
            max_num_seqs=(max(1, int(max_num_seqs)) if max_num_seqs is not None else 8),
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_auto_capacity=kv_cache_auto_capacity,
            kv_cache_memory_utilization=kv_cache_memory_utilization,
        ).normalized()
        engine = LLMEngine(
            config=cfg,
            model_registry=model_registry,
        )
        self._engine_client = EngineClient(engine)

    def _configure_cudnn_prebuild_hint_(
        self,
        max_num_seqs: int | None,
        max_model_len: int | None,
        kv_cache_block_size: int,
    ) -> None:
        # Auto-derive cuDNN prebuild bucket from scheduler shape limits.
        backend = os.environ.get("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "").strip().lower()
        if backend != "cudnn":
            return

        # Respect explicit user override if already set.
        if not os.environ.get("LLAISYS_CUDNN_PREBUILD_MAX_B"):
            seq_cap = max(1, int(max_num_seqs)) if max_num_seqs is not None else 8
            b_hint = 1
            while b_hint < seq_cap:
                b_hint <<= 1
            b_hint = max(64, b_hint)
            os.environ["LLAISYS_CUDNN_PREBUILD_MAX_B"] = str(b_hint)

        # Stabilize page-table capacity across decode steps by using config upper
        # bound, so cuDNN does not repeatedly rebuild plans on width growth.
        if not os.environ.get("LLAISYS_CUDNN_BLOCK_TABLE_CAP"):
            if max_model_len is not None and int(max_model_len) > 0 and int(kv_cache_block_size) > 0:
                cap = max(1, int(max_model_len) // int(kv_cache_block_size))
                os.environ["LLAISYS_CUDNN_BLOCK_TABLE_CAP"] = str(cap)

    def submit(
        self,
        inputs: Sequence[int] | str,
        max_new_tokens: Optional[int] = 16,
        top_k: int = 0,
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
        max_new_tokens: Optional[int] = 16,
        top_k: int = 0,
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

    def kv_cache_stats(self) -> dict:
        fn = getattr(self._engine_client, "kv_cache_stats", None)
        if callable(fn):
            return fn()
        return {}

    def generate(
        self,
        inputs,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        max_new_tokens: Optional[int] = 16,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stop_token_ids: Optional[Sequence[int]] = None,
        stop: Optional[Sequence[str]] = None,
        use_tqdm: bool = False,
    ):
        _ = use_tqdm

        # Normalize all accepted input forms into a uniform token-batch representation:
        # str -> [token_ids], list[str] -> list[token_ids], list[list[int]] -> unchanged.
        batch_inputs, prompt_token_lens = self._normalize_batch_inputs(inputs)
        # Build per-request sampling params (single config is broadcast to all requests).
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

        # Submit all requests first; actual decoding happens in step()/collect() loop below.
        request_ids = []
        for token_ids, params in zip(batch_inputs, params_list):
            req_id = self._engine_client.submit(inputs=token_ids, sampling_params=params)
            request_ids.append(req_id)

        # Drive the engine until every request becomes collectable.
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

        # Convert engine outputs to entrypoint payload shape.
        # token_ids are completion-only (prompt prefix removed).
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
        # Single prompt string -> single request batch.
        if isinstance(inputs, str):
            token_ids = self._encode_prompt(inputs)
            return [token_ids], [len(token_ids)]

        # Multiple prompt strings -> N requests.
        if isinstance(inputs, Sequence) and inputs and isinstance(inputs[0], str):
            token_batches = [self._encode_prompt(str(p)) for p in inputs]
            lens = [len(x) for x in token_batches]
            return token_batches, lens

        # Tokenized batch path (vLLM-style): list[list[int]].
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
