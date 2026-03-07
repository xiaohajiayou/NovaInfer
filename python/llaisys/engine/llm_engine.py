from __future__ import annotations

from dataclasses import fields
from typing import Dict, Sequence

from ..libllaisys.model import KvCacheLayout
from .config import EngineConfig
from .executor import Executor
from .model_registry import ModelRegistry
from .sequence import Sequence as EngineSequence
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


class LLMEngine:
    """Stage-1 offline engine with explicit submit/step/collect/cancel contract."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        model_registry: ModelRegistry | None = None,
        **kwargs,
    ):
        if "max_num_seqs" not in kwargs and "max_batch_size" in kwargs:
            kwargs["max_num_seqs"] = kwargs["max_batch_size"]
        if "model_path" not in kwargs and "model" in kwargs:
            kwargs["model_path"] = kwargs["model"]
        config_fields = {field.name for field in fields(EngineConfig)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        unexpected = sorted(
            k for k in kwargs.keys() if k not in config_fields and k not in ("max_batch_size", "model")
        )
        if unexpected:
            names = ", ".join(unexpected)
            raise TypeError(f"LLMEngine got unexpected keyword argument(s): {names}")
        cfg = config or EngineConfig(**config_kwargs)
        self._config = cfg

        self._worker = Worker(
            config=cfg,
            model_registry=model_registry,
        )
        self._prefix_cache_enabled = (
            cfg.kv_cache_layout == KvCacheLayout.BLOCK and bool(cfg.enable_prefix_caching)
        )
        # Runtime-derived capacity/length are synced into cfg during model runner init.
        # Prefix caching is forced off when not using BLOCK layout.
        cfg.enable_prefix_caching = bool(self._prefix_cache_enabled)
        if cfg.end_token_id is None:
            raise ValueError("end_token_id is required")
        self._end_token_id = int(cfg.end_token_id)

        self._scheduler = RequestScheduler(cfg)
        self._executor = Executor(self._worker)

        self._requests: Dict[str, RequestState] = {}
        self._seq_by_id: Dict[int, EngineSequence] = {}
        self._request_id_by_seq_id: Dict[int, str] = {}
        self._seq_id_by_request: Dict[str, int] = {}
        self._finished_outputs: Dict[str, GenerationOutput] = {}
        self._request_counter = 0
        self._last_request_id: str | None = None
        self._max_batch_size = int(cfg.max_num_seqs)
        self._runtime_peak_used_tokens_observed = 0

    def close(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None:
            close_fn = getattr(worker, "close", None)
            if callable(close_fn):
                close_fn()

    def kv_cache_stats(self) -> dict:
        scheduler_stats = self._scheduler.block_stats()
        out = {
            "config": {
                "enable_prefix_caching": bool(self._config.enable_prefix_caching),
            },
            "allocator": {
                "block_size": int(scheduler_stats.block_size),
                "num_blocks": int(scheduler_stats.num_blocks),
                "used_blocks": int(scheduler_stats.used_blocks),
                "peak_used_blocks": int(getattr(scheduler_stats, "peak_used_blocks", 0)),
                "free_blocks": int(scheduler_stats.free_blocks),
                "usage": float(scheduler_stats.usage),
                "prefix_hits": int(scheduler_stats.prefix_hits),
                "prefix_misses": int(scheduler_stats.prefix_misses),
                "prefix_saved_tokens": int(scheduler_stats.prefix_saved_tokens),
            }
        }
        runner = self._worker.model_runner
        kv_stats_fn = getattr(runner, "kv_stats", None)
        if callable(kv_stats_fn):
            try:
                runtime = kv_stats_fn()
                if isinstance(runtime, dict):
                    peak = int(runtime.get("peak_used_tokens", 0) or 0)
                    observed = int(self._runtime_peak_used_tokens_observed)
                    if observed > peak:
                        runtime["peak_used_tokens"] = observed
                out["runtime"] = runtime
            except Exception:
                out["runtime"] = None
        return out

    def _observe_runtime_kv_peak(self) -> None:
        runner = self._worker.model_runner
        kv_stats_fn = getattr(runner, "kv_stats", None)
        if not callable(kv_stats_fn):
            return
        try:
            runtime = kv_stats_fn()
            if not isinstance(runtime, dict):
                return
            used = int(runtime.get("used_tokens", 0) or 0)
            peak = int(runtime.get("peak_used_tokens", 0) or 0)
            self._runtime_peak_used_tokens_observed = max(
                self._runtime_peak_used_tokens_observed, used, peak
            )
        except Exception:
            return

    def reset_prefix_cache(self) -> int:
        self._scheduler.block_manager.reset_prefix_cache()
        runner = self._worker.model_runner
        reset_fn = getattr(runner, "kv_reset_prefix_cache", None)
        if callable(reset_fn):
            try:
                return int(reset_fn())
            except Exception:
                return 5
        return 0

    def submit(self, inputs: Sequence[int], sampling_params: SamplingParams) -> str:
        req = self._create_request([int(t) for t in inputs])
        seq = EngineSequence(
            token_ids=[int(t) for t in inputs],
            sampling_params=sampling_params,
        )
        seq.max_tokens = self._compute_max_new(len(req.prompt_tokens), sampling_params)
        req.output_tokens = seq.token_ids
        seq_id = int(seq.seq_id)
        self._seq_by_id[seq_id] = seq
        self._request_id_by_seq_id[seq_id] = req.request_id
        self._seq_id_by_request[req.request_id] = seq_id
        self._scheduler.add(seq)
        return req.request_id

    def step(self) -> list[GenerationOutput]:
        sched = self._scheduler.schedule(max_num_seqs=self._max_batch_size)
        if sched is None:
            return []

        completions: list[GenerationOutput] = []
        for seq_id in list(sched.finished_seq_ids):
            sid = int(seq_id)
            req_key = self._request_id_by_seq_id.get(sid)
            if req_key is None:
                continue
            req = self._requests.get(req_key)
            seq = self._seq_by_id.get(sid)
            if req is None or seq is None:
                continue
            if req.status in TERMINAL_REQUEST_STATUSES:
                continue
            finish_reason = str(seq.finish_reason or "aborted")
            if finish_reason == "aborted" and req.error is None:
                req.error = "prompt exceeds model/scheduler budget"
            completions.append(
                self._complete_request(seq, finish_reason)
            )

        if not sched.scheduled_seqs:
            if sched.finished_seq_ids:
                self._executor.execute_scheduler_step(
                    sched,
                )
            return completions

        for seq in sched.scheduled_seqs:
            req_id = self._request_id_by_seq_id.get(int(seq.seq_id))
            if req_id is None:
                raise RuntimeError(f"missing request mapping for seq_id={seq.seq_id}")
            req = self._requests.get(req_id)
            if req is None:
                raise RuntimeError(f"missing request state for seq_id={seq.seq_id}")
            if req.status == RequestStatus.WAITING:
                self._transition(req, RequestStatus.RUNNING)

        sampled_token_ids = self._executor.execute_scheduler_step(sched)
        self._observe_runtime_kv_peak()
        if sampled_token_ids is None:
            return completions

        self._scheduler.postprocess(
            sched.scheduled_seqs,
            sampled_token_ids,
            self._end_token_id,
        )

        for seq in sched.scheduled_seqs:
            finish_reason = seq.finish_reason
            if finish_reason is not None:
                completions.append(
                    self._complete_request(seq, finish_reason)
                )

        return completions

    def collect(self, request_id: str) -> GenerationOutput | None:
        return self._finished_outputs.get(request_id)

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
        req = self._requests.get(request_id)
        seq_id = self._seq_id_by_request.get(request_id)
        seq = self._seq_by_id.get(int(seq_id)) if seq_id is not None else None
        if req is None or seq is None or seq_id is None:
            return False

        if req.status in TERMINAL_REQUEST_STATUSES:
            return True

        req.error = "aborted by user"
        self._complete_request(seq, "aborted")
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
        prompt_ids = [int(t) for t in prompt_tokens]
        req = RequestState(
            request_id=req_id,
            status=RequestStatus.WAITING,
            prompt_tokens=prompt_ids,
            output_tokens=list(prompt_ids),
            history=[RequestStatus.WAITING],
        )
        self._requests[req_id] = req
        self._last_request_id = req_id
        return req

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
        seq: EngineSequence,
        finish_reason: str,
    ) -> GenerationOutput:
        seq_id = int(seq.seq_id)
        request_id = self._request_id_by_seq_id.get(seq_id)
        if request_id is None:
            raise RuntimeError(f"missing request mapping for completion: seq_id={seq_id}")
        req_state = self._requests.get(request_id)
        if req_state is None:
            raise RuntimeError(f"missing request state for completion: request_id={request_id}")
        terminal_status = {
            "empty": RequestStatus.FINISHED_IGNORED,
            "length": RequestStatus.FINISHED_LENGTH_CAPPED,
            "aborted": RequestStatus.FINISHED_ABORTED,
        }.get(str(finish_reason), RequestStatus.FINISHED_STOPPED)
        if req_state.status not in TERMINAL_REQUEST_STATUSES:
            self._transition(req_state, terminal_status)

        total = len(req_state.output_tokens)
        prompt_len = len(req_state.prompt_tokens)
        completion = max(0, total - int(prompt_len))
        completion_tokens = req_state.output_tokens[prompt_len:]
        text = "" if not completion_tokens else self._worker.decode_tokens(completion_tokens)
        output = GenerationOutput(
            request_id=req_state.request_id,
            token_ids=list(req_state.output_tokens),
            finish_reason=str(finish_reason),
            status=req_state.status,
            text=text,
            usage={
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(completion),
                "total_tokens": int(total),
            },
        )
        self._finished_outputs[request_id] = output
        self._scheduler.finish(seq_id)
        self._worker.free_request(seq.seq_id)
        self._seq_by_id.pop(seq_id, None)
        self._request_id_by_seq_id.pop(seq_id, None)
        self._seq_id_by_request.pop(request_id, None)
        return output

    def _compute_max_new(self, prompt_len: int, sampling_params: SamplingParams) -> int:
        remaining = int(self._config.max_model_len) - int(prompt_len)
        if remaining <= 0:
            return 0
        max_new = remaining
        if sampling_params.max_new_tokens is not None:
            max_new = min(max_new, int(sampling_params.max_new_tokens))
        return int(max_new)
