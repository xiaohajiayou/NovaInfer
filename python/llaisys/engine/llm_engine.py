from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from ..libllaisys import DeviceType
from .executor import Executor
from .model_registry import ModelRegistry
from .output_processor import OutputProcessor
from .scheduler import RequestScheduler
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
    seq_id: int
    max_new_tokens: int = -1
    num_generated: int = 0
    prefill_done: bool = False
    matched_stop: str | None = None
    finished_output: GenerationOutput | None = None


class LLMEngine:
    """Stage-1 offline engine with explicit submit/step/collect/cancel contract."""

    def __init__(
        self,
        model_type: str = "qwen2",
        model_path: Path | str | None = None,
        device: DeviceType = DeviceType.CPU,
        worker: Worker | None = None,
        model_runner=None,
        model_registry: ModelRegistry | None = None,
    ):
        self._worker = worker if worker is not None else Worker(
            model_type=model_type,
            model_path=model_path,
            device=device,
            model_runner=model_runner,
            model_registry=model_registry,
        )
        self._scheduler = RequestScheduler()
        self._executor = Executor(self._worker)
        self._output = OutputProcessor()

        self._requests: Dict[str, RequestState] = {}
        self._runtimes: Dict[str, _RequestRuntime] = {}
        self._request_counter = 0
        self._seq_counter = 0
        self._last_request_id: str | None = None

    def submit(self, inputs: Sequence[int], sampling_params: SamplingParams) -> str:
        req = self._create_request([int(t) for t in inputs])
        runtime = _RequestRuntime(
            request=req,
            sampling_params=sampling_params,
            seq_id=self._alloc_seq_id(),
        )
        self._runtimes[req.request_id] = runtime
        self._scheduler.submit(req.request_id)
        return req.request_id

    def step(self) -> list[GenerationOutput]:
        req_id = self._scheduler.pick_next()
        if req_id is None:
            return []

        runtime = self._runtimes.get(req_id)
        if runtime is None:
            self._scheduler.finish(req_id)
            return []

        req = runtime.request
        if req.status in TERMINAL_REQUEST_STATUSES:
            self._scheduler.finish(req_id)
            return [runtime.finished_output] if runtime.finished_output is not None else []

        if req.status == RequestStatus.WAITING:
            self._transition(req, RequestStatus.RUNNING)

        try:
            if not req.prompt_tokens:
                return [self._complete_request(runtime, RequestStatus.FINISHED_IGNORED, "empty")]

            if len(req.prompt_tokens) > self._worker.max_seq_len:
                raise ValueError("Prompt length exceeds maxseq")

            if runtime.max_new_tokens < 0:
                runtime.max_new_tokens = self._compute_max_new(len(req.prompt_tokens), runtime.sampling_params)
            if runtime.max_new_tokens <= 0:
                return [self._complete_request(runtime, RequestStatus.FINISHED_LENGTH_CAPPED, "length")]

            if not runtime.prefill_done:
                plan = self._scheduler.build_prefill_plan(
                    req.prompt_tokens,
                    seq_id=runtime.seq_id,
                )
            else:
                plan = self._scheduler.build_decode_plan(
                    req.output_tokens[-1],
                    seq_id=runtime.seq_id,
                )

            step = self._executor.execute_step(plan, runtime.sampling_params)
            if not step.sampled_token_ids:
                raise RuntimeError("executor returned no sampled token")

            token = int(step.sampled_token_ids[-1])
            req.output_tokens.append(token)
            runtime.num_generated += 1
            runtime.prefill_done = True

            finish_reason = self._maybe_finish_reason(runtime, token)
            if finish_reason is not None:
                return [self._complete_request(runtime, RequestStatus.FINISHED_STOPPED, finish_reason)]

            if runtime.num_generated >= runtime.max_new_tokens:
                return [self._complete_request(runtime, RequestStatus.FINISHED_LENGTH_CAPPED, "length")]

            return []
        except Exception as exc:
            req.error = str(exc)
            if req.status not in TERMINAL_REQUEST_STATUSES:
                self._transition(req, RequestStatus.FINISHED_ABORTED)
            return [self._complete_request(runtime, RequestStatus.FINISHED_ABORTED, "aborted")]

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
