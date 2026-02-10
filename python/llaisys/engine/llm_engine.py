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
    BatchPlan,
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
        max_batch_size: int = 8,
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
        self._max_batch_size = max(1, int(max_batch_size))

    def close(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None:
            close_fn = getattr(worker, "close", None)
            if callable(close_fn):
                close_fn()

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
        req_ids = self._scheduler.pick_next_batch(self._max_batch_size)
        if not req_ids:
            return []

        completions: list[GenerationOutput] = []
        active: list[_RequestRuntime] = []
        combined_tokens: list[int] = []
        combined_logits_mask: list[int] = []
        combined_seq_ids: list[int] = []
        output_runtime_order: list[_RequestRuntime] = []

        for req_id in req_ids:
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

                if runtime.max_new_tokens < 0:
                    runtime.max_new_tokens = self._compute_max_new(len(req.prompt_tokens), runtime.sampling_params)
                if runtime.max_new_tokens <= 0:
                    completions.append(self._complete_request(runtime, RequestStatus.FINISHED_LENGTH_CAPPED, "length"))
                    continue

                if not runtime.prefill_done:
                    tokens = [int(t) for t in req.prompt_tokens]
                    logits_mask = [0] * len(tokens)
                    logits_mask[-1] = 1
                else:
                    tokens = [int(req.output_tokens[-1])]
                    logits_mask = [1]

                seq_ids = [int(runtime.seq_id)] * len(tokens)

                combined_tokens.extend(tokens)
                combined_logits_mask.extend(logits_mask)
                combined_seq_ids.extend(seq_ids)
                output_runtime_order.append(runtime)
                active.append(runtime)
            except Exception as exc:
                req.error = str(exc)
                completions.append(self._complete_request(runtime, RequestStatus.FINISHED_ABORTED, "aborted"))

        if not combined_tokens:
            return completions

        # Current engine keeps one sampling config for the full ubatch.
        # Multi-request mixed sampling params can be added in a later stage.
        first_runtime = active[0]
        step = self._executor.execute_step(
            BatchPlan(
                token_ids=combined_tokens,
                logits_mask=combined_logits_mask,
                seq_ids=combined_seq_ids,
            ),
            first_runtime.sampling_params,
        )

        sampled = [int(t) for t in step.sampled_token_ids]
        output_ids = [int(i) for i in step.output_ids] if step.output_ids else list(range(len(sampled)))
        if len(sampled) != len(output_ids):
            raise RuntimeError("executor returned inconsistent sampled/output sizes")
        if len(sampled) != len(output_runtime_order):
            raise RuntimeError("executor output size does not match planned request outputs")

        runtime_by_out_idx = {out_idx: rt for out_idx, rt in zip(output_ids, output_runtime_order)}

        for out_idx, token in zip(output_ids, sampled):
            runtime = runtime_by_out_idx.get(out_idx)
            if runtime is None:
                continue
            req = runtime.request
            req.output_tokens.append(int(token))
            runtime.num_generated += 1
            runtime.prefill_done = True

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
