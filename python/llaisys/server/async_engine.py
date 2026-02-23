from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Sequence
import sys

from ..engine.llm_engine import LLMEngine
from ..engine.config import EngineConfig
from ..engine.model_registry import ModelRegistry
from ..engine.types import GenerationOutput, SamplingParams, StreamChunk
from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout


@dataclass
class _StreamState:
    emitted: int = 0
    prev_text: str = ""
    finished: bool = False


class AsyncLLMEngine:
    """Stage-2 async facade with queue-driven engine loop.

    API threads enqueue requests; a single loop thread mutates LLMEngine state.
    """

    def __init__(
        self,
        model_type: str = "qwen2",
        model_path: Path | str | None = None,
        device: DeviceType = DeviceType.CPU,
        kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
        kv_cache_block_size: int = 16,
        max_model_len: int | None = None,
        kv_cache_capacity_tokens: int | None = None,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int | None = None,
        kv_cache_auto_capacity: bool = False,
        kv_cache_memory_utilization: float = 0.9,
        model_registry: ModelRegistry | None = None,
        engine: LLMEngine | None = None,
    ):
        if engine is not None:
            self._engine = engine
        else:
            cfg = EngineConfig(
                model_type=model_type,
                model_path=model_path,
                device=device,
                kv_cache_layout=kv_cache_layout,
                kv_cache_block_size=kv_cache_block_size,
                max_model_len=max_model_len,
                kv_cache_capacity_tokens=kv_cache_capacity_tokens,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                kv_cache_auto_capacity=kv_cache_auto_capacity,
                kv_cache_memory_utilization=kv_cache_memory_utilization,
            ).normalized()
            self._engine = LLMEngine(
                config=cfg,
                model_registry=model_registry,
            )
        self._stream_queues: dict[str, list[Queue]] = {}
        self._stream_states: dict[str, _StreamState] = {}
        self._final_outputs: dict[str, GenerationOutput] = {}

        self._cmd_q: Queue = Queue()
        self._stop_event = Event()
        self._engine_closed = False
        self._loop_thread = Thread(target=self._loop, name="llaisys-async-engine", daemon=True)
        self._loop_thread.start()

    def close(self):
        if not self._stop_event.is_set():
            self._stop_event.set()
            self._loop_thread.join(timeout=2.0)

        if not self._engine_closed:
            close_fn = getattr(self._engine, "close", None)
            if callable(close_fn):
                close_fn()
            self._engine_closed = True

    def __del__(self):
        if sys.is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass

    def submit(self, inputs: Sequence[int], sampling_params: SamplingParams) -> str:
        reply_q: Queue = Queue(maxsize=1)
        self._cmd_q.put(("submit", [int(t) for t in inputs], sampling_params, reply_q))
        return str(reply_q.get())

    def step(self):
        # Compatibility path for stage-1 tests; async loop drives steps internally.
        return []

    def collect(self, request_id: str) -> GenerationOutput | None:
        reply_q: Queue = Queue(maxsize=1)
        self._cmd_q.put(("collect", str(request_id), reply_q))
        return reply_q.get()

    def cancel(self, request_id: str) -> bool:
        reply_q: Queue = Queue(maxsize=1)
        self._cmd_q.put(("cancel", str(request_id), reply_q))
        return bool(reply_q.get())

    def generate(self, inputs: Sequence[int], sampling_params: SamplingParams) -> GenerationOutput:
        req_id = self.submit(inputs=inputs, sampling_params=sampling_params)
        while True:
            out = self.collect(req_id)
            if out is not None:
                return out

    def stream(self, inputs: Sequence[int], sampling_params: SamplingParams):
        req_id = self.submit(inputs=inputs, sampling_params=sampling_params)
        stream_q: Queue = Queue()
        self._cmd_q.put(("watch_stream", req_id, stream_q))
        terminal_emitted = False
        while True:
            try:
                item = stream_q.get(timeout=1.0)
            except Empty:
                done = self.collect(req_id)
                if done is not None:
                    if not terminal_emitted:
                        yield StreamChunk(
                            request_id=req_id,
                            token_id=None,
                            text_delta=None,
                            status=done.status,
                            is_finished=True,
                            finish_reason=done.finish_reason,
                        )
                    return
                continue
            if item is None:
                return
            if item.is_finished:
                terminal_emitted = True
            yield item

    def is_finished(self, request_id: str) -> bool:
        reply_q: Queue = Queue(maxsize=1)
        self._cmd_q.put(("is_finished", str(request_id), reply_q))
        return bool(reply_q.get())

    def get_request_status(self, request_id: str):
        reply_q: Queue = Queue(maxsize=1)
        self._cmd_q.put(("status", str(request_id), reply_q))
        return reply_q.get()

    def encode_chat_messages(self, messages: list[dict]) -> list[int]:
        worker = self._engine._worker  # noqa: SLF001
        return worker.encode_chat_messages(messages)

    def get_default_sampling_params(self) -> dict:
        worker = self._engine._worker  # noqa: SLF001
        fn = getattr(worker, "get_default_sampling_params", None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
        return {"temperature": 1.0, "top_p": 1.0, "top_k": 0}

    def get_max_model_len(self) -> int:
        worker = self._engine._worker  # noqa: SLF001
        try:
            return int(worker.max_seq_len)
        except Exception:
            return 4096

    @property
    def inner_engine(self) -> LLMEngine:
        return self._engine

    def _loop(self):
        while not self._stop_event.is_set():
            self._drain_commands()
            outputs = self._engine.step()
            if outputs:
                for out in outputs:
                    req_id = out.request_id
                    # Ensure all newly generated tokens are emitted before
                    # sending terminal chunk for this request.
                    self._emit_pending_chunks_for_request(req_id)
                    self._final_outputs[req_id] = out
                    self._emit_chunk(req_id, token_id=None, text_delta=None, status=out.status, is_finished=True, finish_reason=out.finish_reason)
                    self._close_stream(req_id)
            self._emit_inflight_chunks()
            self._stop_event.wait(0.001)

    def _drain_commands(self):
        while True:
            try:
                cmd = self._cmd_q.get_nowait()
            except Empty:
                return
            op = cmd[0]
            if op == "submit":
                _, inputs, sampling_params, reply_q = cmd
                req_id = self._engine.submit(inputs=inputs, sampling_params=sampling_params)
                self._stream_states[req_id] = _StreamState()
                reply_q.put(req_id)
            elif op == "collect":
                _, req_id, reply_q = cmd
                reply_q.put(self._final_outputs.get(req_id))
            elif op == "cancel":
                _, req_id, reply_q = cmd
                ok = self._engine.cancel(req_id)
                reply_q.put(ok)
            elif op == "is_finished":
                _, req_id, reply_q = cmd
                reply_q.put(self._engine.is_finished(req_id))
            elif op == "status":
                _, req_id, reply_q = cmd
                reply_q.put(self._engine.get_request_status(req_id))
            elif op == "watch_stream":
                _, req_id, stream_q = cmd
                self._stream_queues.setdefault(req_id, []).append(stream_q)
                # Stream consumer may subscribe after some tokens are already
                # produced by the loop. Replay current completion tokens once
                # to avoid losing prefix chunks due to subscribe race.
                self._replay_generated_tokens(req_id, stream_q)
                if req_id in self._final_outputs:
                    out = self._final_outputs[req_id]
                    stream_q.put(
                        StreamChunk(
                            request_id=req_id,
                            token_id=None,
                            text_delta=None,
                            status=out.status,
                            is_finished=True,
                            finish_reason=out.finish_reason,
                        )
                    )
                    stream_q.put(None)
            else:
                pass

    def _replay_generated_tokens(self, req_id: str, stream_q: Queue) -> None:
        completion_tokens = self._engine.get_completion_tokens(req_id)
        status = self._engine.get_request_status(req_id)
        if completion_tokens is None or status is None or not completion_tokens:
            return

        tokens = [int(t) for t in completion_tokens]
        full_text = self._engine.decode_tokens(tokens)
        for idx, token_id in enumerate(tokens):
            text_delta = None
            if full_text is not None and idx == len(tokens) - 1:
                # Emit full decoded text once for parser state alignment.
                text_delta = full_text
            stream_q.put(
                StreamChunk(
                    request_id=req_id,
                    token_id=token_id,
                    text_delta=text_delta,
                    status=status,
                    is_finished=False,
                    finish_reason=None,
                )
            )

    def _emit_inflight_chunks(self):
        for req_id, state in list(self._stream_states.items()):
            if state.finished:
                continue
            self._emit_pending_chunks_for_request(req_id)

    def _emit_pending_chunks_for_request(self, req_id: str) -> None:
        state = self._stream_states.get(req_id)
        if state is None or state.finished:
            return
        completion_tokens = self._engine.get_completion_tokens(req_id)
        status = self._engine.get_request_status(req_id)
        if completion_tokens is None or status is None:
            return
        if len(completion_tokens) <= state.emitted:
            return

        new_tokens = completion_tokens[state.emitted :]
        full_text = self._engine.decode_tokens(completion_tokens)
        text_delta = None
        if full_text is not None:
            text_delta = full_text[len(state.prev_text) :] if full_text.startswith(state.prev_text) else full_text
            state.prev_text = full_text

        for idx, token_id in enumerate(new_tokens):
            self._emit_chunk(
                req_id=req_id,
                token_id=int(token_id),
                text_delta=text_delta if idx == len(new_tokens) - 1 else None,
                status=status,
                is_finished=False,
                finish_reason=None,
            )
        state.emitted = len(completion_tokens)

    def _emit_chunk(
        self,
        req_id: str,
        token_id: int | None,
        text_delta: str | None,
        status,
        is_finished: bool,
        finish_reason: str | None,
    ):
        chunk = StreamChunk(
            request_id=req_id,
            token_id=token_id,
            text_delta=text_delta,
            status=status,
            is_finished=is_finished,
            finish_reason=finish_reason,
        )
        for q in self._stream_queues.get(req_id, []):
            q.put(chunk)
        if is_finished:
            st = self._stream_states.get(req_id)
            if st is not None:
                st.finished = True

    def _close_stream(self, req_id: str):
        for q in self._stream_queues.get(req_id, []):
            q.put(None)
