from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout
from .executor import Executor
from .model_registry import ModelRegistry
from .output_processor import OutputProcessor
from .sequence import Sequence as EngineSequence
from .scheduler import RequestScheduler, SchedulerOutputs
from .types import (
    GenerationOutput,
    RequestState,
    RequestStatus,
    SamplingParams,
    StreamChunk,
    TERMINAL_REQUEST_STATUSES,
)
from .worker import Worker


_ALLOWED_TRANSITIONS = {
    RequestStatus.WAITING: {
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.RUNNING,
        RequestStatus.FINISHED_IGNORED,
        RequestStatus.FINISHED_ABORTED,
    },
    RequestStatus.WAITING_FOR_REMOTE_KVS: {
        RequestStatus.RUNNING,
        RequestStatus.PREEMPTED,
        RequestStatus.FINISHED_ABORTED,
    },
    RequestStatus.RUNNING: {
        RequestStatus.PREEMPTED,
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
        RequestStatus.FINISHED_ABORTED,
    },
    RequestStatus.PREEMPTED: {
        RequestStatus.WAITING,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.RUNNING,
        RequestStatus.FINISHED_ABORTED,
    },
}


@dataclass
class _RequestRuntime:
    request: RequestState
    sampling_params: SamplingParams
    seq: EngineSequence
    max_new_tokens: int = -1
    num_generated: int = 0
    matched_stop: str | None = None
    finished_output: GenerationOutput | None = None


class LLMEngine:
    """Stage-1 offline engine with explicit submit/step/collect/cancel contract."""

    def __init__(
        self,
        model_type: str = "qwen2",
        model_path: Path | str | None = None,
        device: DeviceType = DeviceType.CPU,
        kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
        kv_cache_block_size: int = 16,
        max_model_len: int | None = None,
        kv_cache_capacity_tokens: int | None = None,
        worker: Worker | None = None,
        model_runner=None,
        model_registry: ModelRegistry | None = None,
        max_batch_size: int = 8,
    ):
        self._worker = worker if worker is not None else Worker(
            model_type=model_type,
            model_path=model_path,
            device=device,
            kv_cache_layout=kv_cache_layout,
            kv_cache_block_size=kv_cache_block_size,
            max_model_len=max_model_len,
            kv_cache_capacity_tokens=kv_cache_capacity_tokens,
            model_runner=model_runner,
            model_registry=model_registry,
        )
        eff_max_model_len = int(self._worker.max_seq_len)
        eff_kv_capacity_tokens = (
            int(kv_cache_capacity_tokens) if kv_cache_capacity_tokens is not None else eff_max_model_len
        )
        eff_kv_capacity_tokens = max(1, eff_kv_capacity_tokens)
        eff_block_size = max(1, int(kv_cache_block_size))
        num_kvcache_blocks = (
            (eff_kv_capacity_tokens + eff_block_size - 1) // eff_block_size
            if kv_cache_layout == KvCacheLayout.BLOCK
            else 0
        )
        self._scheduler = RequestScheduler(
            max_num_seqs=max_batch_size,
            max_num_batched_tokens=eff_kv_capacity_tokens,
            block_size=eff_block_size,
            num_kvcache_blocks=num_kvcache_blocks,
        )
        self._executor = Executor(self._worker)
        self._output = OutputProcessor()

        self._requests: Dict[str, RequestState] = {}
        self._runtimes: Dict[str, _RequestRuntime] = {}
        self._request_counter = 0
        self._seq_counter = 0
        self._last_request_id: str | None = None
        self._max_batch_size = max(1, int(max_batch_size))

    def close(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None:
            close_fn = getattr(worker, "close", None)
            if callable(close_fn):
                close_fn()

    def submit(self, inputs: Sequence[int], sampling_params: SamplingParams) -> str:
        req = self._create_request([int(t) for t in inputs])
        seq = EngineSequence(
            request_id=req.request_id,
            seq_id=self._alloc_seq_id(),
            token_ids=[int(t) for t in inputs],
            sampling_params=sampling_params,
            block_size=self._scheduler.block_manager.block_size,
        )
        runtime = _RequestRuntime(
            request=req,
            sampling_params=sampling_params,
            seq=seq,
        )
        self._runtimes[req.request_id] = runtime
        self._scheduler.add(seq)
        return req.request_id

    def step(self) -> list[GenerationOutput]:
        sched = self._scheduler.schedule(max_num_seqs=self._max_batch_size)
        if sched is None:
            return []

        completions: list[GenerationOutput] = []
        active: list[_RequestRuntime] = []
        active_seqs: list[EngineSequence] = []

        for seq in sched.scheduled_seqs:
            req_id = seq.request_id
            runtime = self._runtimes.get(req_id)
            if runtime is None:
                self._scheduler.finish(req_id)
                continue

            req = runtime.request
            if req.status in TERMINAL_REQUEST_STATUSES:
                self._scheduler.finish(req_id)
                if runtime.finished_output is not None:
                    completions.append(runtime.finished_output)
                continue

            try:
                if req.status == RequestStatus.WAITING:
                    self._transition(req, RequestStatus.RUNNING)

                if not req.prompt_tokens:
                    completions.append(self._complete_request(runtime, RequestStatus.FINISHED_IGNORED, "empty"))
                    continue

                if len(req.prompt_tokens) > self._worker.max_seq_len:
                    raise ValueError("Prompt length exceeds maxseq")
                if len(req.prompt_tokens) > self._scheduler.max_num_batched_tokens:
                    raise ValueError("Prompt length exceeds scheduler token budget")

                if runtime.max_new_tokens < 0:
                    runtime.max_new_tokens = self._compute_max_new(len(req.prompt_tokens), runtime.sampling_params)
                if runtime.max_new_tokens <= 0:
                    completions.append(self._complete_request(runtime, RequestStatus.FINISHED_LENGTH_CAPPED, "length"))
                    continue
                active.append(runtime)
                active_seqs.append(seq)
            except Exception as exc:
                req.error = str(exc)
                completions.append(self._complete_request(runtime, RequestStatus.FINISHED_ABORTED, "aborted"))

        if not active_seqs:
            return completions

        # Current engine keeps one sampling config for the full step.
        # Mixed per-request sampling can be added after request-aware plan migration.
        first_runtime = active[0]
        sampled, sampled_req_ids = self._executor.execute_scheduler_step(
            SchedulerOutputs(scheduled_seqs=active_seqs, is_prefill=sched.is_prefill),
            first_runtime.sampling_params,
        )

        for req_id, token in zip(sampled_req_ids, sampled):
            runtime = self._runtimes.get(req_id)
            if runtime is None:
                continue
            req = runtime.request
            req.output_tokens.append(int(token))
            runtime.seq.append_token(int(token))
            runtime.num_generated += 1

            finish_reason = self._maybe_finish_reason(runtime, int(token))
            if finish_reason is not None:
                completions.append(self._complete_request(runtime, RequestStatus.FINISHED_STOPPED, finish_reason))
                continue

            if runtime.num_generated >= runtime.max_new_tokens:
                completions.append(self._complete_request(runtime, RequestStatus.FINISHED_LENGTH_CAPPED, "length"))

        return completions

    def collect(self, request_id: str) -> GenerationOutput | None:
        runtime = self._runtimes.get(request_id)
        if runtime is None:
            return None
        return runtime.finished_output

    def stream(self, inputs: Sequence[int], sampling_params: SamplingParams):
        request_id = self.submit(inputs=inputs, sampling_params=sampling_params)
        req = self._requests[request_id]
        prompt_len = len(req.prompt_tokens)
        emitted = 0
        prev_text = ""

        while True:
            done = self.collect(request_id)
            if done is not None:
                yield StreamChunk(
                    request_id=request_id,
                    token_id=None,
                    text_delta=None,
                    status=done.status,
                    is_finished=True,
                    finish_reason=done.finish_reason,
                )
                return

            _ = self.step()
            req = self._requests.get(request_id)
            if req is None:
                return

            completion_tokens = req.output_tokens[prompt_len:]
            if len(completion_tokens) <= emitted:
                continue

            new_tokens = [int(t) for t in completion_tokens[emitted:]]
            full_text = self._worker.decode_tokens(completion_tokens)
            text_delta = None
            if full_text is not None:
                text_delta = (
                    full_text[len(prev_text) :]
                    if full_text.startswith(prev_text)
                    else full_text
                )
                prev_text = full_text

            for idx, token_id in enumerate(new_tokens):
                yield StreamChunk(
                    request_id=request_id,
                    token_id=token_id,
                    text_delta=text_delta if idx == len(new_tokens) - 1 else None,
                    status=req.status,
                    is_finished=False,
                )
            emitted = len(completion_tokens)

    def generate(self, inputs: Sequence[int], sampling_params: SamplingParams) -> GenerationOutput:
        request_id = self.submit(inputs=inputs, sampling_params=sampling_params)
        while True:
            done = self.collect(request_id)
            if done is not None:
                return done

            _ = self.step()

            done = self.collect(request_id)
            if done is not None:
                return done

            if not self._scheduler.has_work():
                raise RuntimeError("Engine finished without output for request")

    def abort_request(self, request_id: str) -> bool:
        runtime = self._runtimes.get(request_id)
        if runtime is None:
            return False

        req = runtime.request
        if req.status in TERMINAL_REQUEST_STATUSES:
            return True

        req.error = "aborted by user"
        self._complete_request(runtime, RequestStatus.FINISHED_ABORTED, "aborted")
        return True

    def cancel(self, request_id: str) -> bool:
        return self.abort_request(request_id)

    def is_finished(self, request_id: str) -> bool:
        req = self._requests.get(request_id)
        if req is None:
            return True
        return req.status in TERMINAL_REQUEST_STATUSES

    def get_request_status(self, request_id: str):
        req = self._requests.get(request_id)
        return None if req is None else req.status

    def get_completion_tokens(self, request_id: str) -> list[int] | None:
        req = self._requests.get(request_id)
        if req is None:
            return None
        prompt_len = len(req.prompt_tokens)
        return [int(t) for t in req.output_tokens[prompt_len:]]

    def decode_tokens(self, token_ids: Sequence[int]) -> str | None:
        return self._worker.decode_tokens([int(t) for t in token_ids])

    def get_request_history(self, request_id: str):
        req = self._requests.get(request_id)
        if req is None:
            return None
        return list(req.history)

    @property
    def last_request_id(self) -> str | None:
        return self._last_request_id

    def _create_request(self, prompt_tokens: Sequence[int]) -> RequestState:
        self._request_counter += 1
        req_id = f"req-{self._request_counter}"
        req = RequestState(
            request_id=req_id,
            status=RequestStatus.WAITING,
            prompt_tokens=[int(t) for t in prompt_tokens],
            output_tokens=[int(t) for t in prompt_tokens],
            history=[RequestStatus.WAITING],
        )
        self._requests[req_id] = req
        self._last_request_id = req_id
        return req

    def _alloc_seq_id(self) -> int:
        self._seq_counter += 1
        return int(self._seq_counter)

    def _transition(self, req: RequestState, next_status: RequestStatus) -> None:
        current = req.status
        if current == next_status:
            return
        if current in TERMINAL_REQUEST_STATUSES:
            raise RuntimeError(f"invalid transition from terminal state: {current} -> {next_status}")

        allowed = _ALLOWED_TRANSITIONS.get(current, set())
        if next_status not in allowed:
            raise RuntimeError(f"invalid transition: {current} -> {next_status}")

        req.status = next_status
        req.history.append(next_status)

    def _complete_request(
        self,
        runtime: _RequestRuntime,
        terminal_status: RequestStatus,
        finish_reason: str,
    ) -> GenerationOutput:
        req = runtime.request
        if req.status not in TERMINAL_REQUEST_STATUSES:
            self._transition(req, terminal_status)

        text = self._decode_completion_text(req)
        if finish_reason == "stop_string" and runtime.matched_stop and text is not None:
            idx = text.find(runtime.matched_stop)
            if idx >= 0:
                text = text[:idx]

        output = self._output.finalize(
            request_id=req.request_id,
            prompt_len=len(req.prompt_tokens),
            token_ids=req.output_tokens,
            finish_reason=finish_reason,
            status=req.status,
            text=text,
        )
        runtime.finished_output = output
        self._scheduler.finish(req.request_id)
        self._worker.free_request(runtime.seq.seq_id)
        return output

    def _compute_max_new(self, prompt_len: int, sampling_params: SamplingParams) -> int:
        remaining = self._worker.max_seq_len - int(prompt_len)
        if remaining <= 0:
            return 0
        max_new = remaining
        if sampling_params.max_new_tokens is not None:
            max_new = min(max_new, int(sampling_params.max_new_tokens))
        return int(max_new)

    def _decode_completion_text(self, req: RequestState) -> str | None:
        completion_tokens = req.output_tokens[len(req.prompt_tokens) :]
        if not completion_tokens:
            return ""
        return self._worker.decode_tokens([int(t) for t in completion_tokens])

    def _maybe_finish_reason(self, runtime: _RequestRuntime, token_id: int) -> str | None:
        sampling_params = runtime.sampling_params
        if token_id == self._worker.end_token_id:
            return "eos_token"
        if sampling_params.stop_token_ids and token_id in set(sampling_params.stop_token_ids):
            return "stop_token"
        if sampling_params.stop:
            completion_text = self._decode_completion_text(runtime.request)
            if completion_text is not None:
                for stop_str in sampling_params.stop:
                    if stop_str and stop_str in completion_text:
                        runtime.matched_stop = stop_str
                        return "stop_string"
        return None
