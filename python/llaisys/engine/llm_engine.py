from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout
from .config import EngineConfig
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
class _RequestContext:
    # User-visible request state (status/history/tokens).
    request: RequestState
    # Effective sampling config bound to this request.
    sampling_params: SamplingParams
    # Scheduler-facing sequence state (block table / running queue metadata).
    seq: EngineSequence
    # Computed per-request generation cap; -1 means "not computed yet".
    max_new_tokens: int = -1
    # Number of completion tokens generated so far.
    num_generated: int = 0
    # Matched stop string for final text trimming.
    matched_stop: str | None = None
    # Cached terminal output returned by collect(request_id).
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
        max_num_batched_tokens: int | None = None,
        kv_cache_auto_capacity: bool = False,
        kv_cache_memory_utilization: float = 0.9,
        config: EngineConfig | None = None,
    ):
        cfg = (config or EngineConfig(
            model_type=model_type,
            model_path=model_path,
            device=device,
            kv_cache_layout=kv_cache_layout,
            kv_cache_block_size=kv_cache_block_size,
            max_model_len=max_model_len,
            kv_cache_capacity_tokens=kv_cache_capacity_tokens,
            max_num_seqs=max_batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_auto_capacity=kv_cache_auto_capacity,
            kv_cache_memory_utilization=kv_cache_memory_utilization,
        )).normalized()
        self._config = cfg

        self._worker = worker if worker is not None else Worker(
            config=cfg,
            model_runner=model_runner,
            model_registry=model_registry,
        )
        self._prefix_cache_enabled = (
            cfg.kv_cache_layout == KvCacheLayout.BLOCK and bool(cfg.enable_prefix_caching)
        )

        # Reconcile effective runtime capacities from the real runner and make
        # EngineConfig the single source of truth for downstream components.
        cfg.effective_max_model_len = int(self._worker.max_seq_len)
        cfg.effective_kv_cache_capacity_tokens = (
            int(self._worker.kv_cache_capacity_tokens)
            if getattr(self._worker, "kv_cache_capacity_tokens", None) is not None
            else int(cfg.effective_max_model_len)
        )
        cfg.effective_kv_cache_capacity_tokens = max(1, int(cfg.effective_kv_cache_capacity_tokens))
        eff_block_size = max(1, int(cfg.kv_cache_block_size))
        cfg.num_kvcache_blocks = (
            (int(cfg.effective_kv_cache_capacity_tokens) + eff_block_size - 1) // eff_block_size
            if cfg.kv_cache_layout == KvCacheLayout.BLOCK
            else 0
        )
        cfg.effective_enable_prefix_caching = bool(self._prefix_cache_enabled)
        cfg.normalized()

        self._scheduler = RequestScheduler(
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=(
                max(1, int(cfg.max_num_batched_tokens))
                if cfg.max_num_batched_tokens is not None
                else int(cfg.effective_kv_cache_capacity_tokens)
            ),
            block_size=eff_block_size,
            num_kvcache_blocks=int(cfg.num_kvcache_blocks),
            enable_prefix_cache=self._prefix_cache_enabled,
        )
        self._executor = Executor(self._worker)
        self._output = OutputProcessor()

        self._requests: Dict[str, RequestState] = {}
        self._request_contexts: Dict[str, _RequestContext] = {}
        self._request_counter = 0
        self._seq_counter = 0
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
                "effective_enable_prefix_caching": bool(self._prefix_cache_enabled),
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
            request_id=req.request_id,
            seq_id=self._alloc_seq_id(),
            token_ids=[int(t) for t in inputs],
            sampling_params=sampling_params,
            block_size=self._scheduler.block_manager.block_size,
        )
        request_context = _RequestContext(
            request=req,
            sampling_params=sampling_params,
            seq=seq,
        )
        self._request_contexts[req.request_id] = request_context
        self._scheduler.add(seq)
        return req.request_id

    def step(self) -> list[GenerationOutput]:
        # Step flow (reading guide):
        # 1) Ask scheduler for this round's work.
        # 2) Validate/filter scheduled requests into executable active set.
        # 3) Execute one model step for active set.
        # 4) Append sampled tokens, finalize finished requests, return completions.
        sched = self._scheduler.schedule(max_num_seqs=self._max_batch_size)
        if sched is None:
            # No schedulable work.
            # Guardrail: abort waiting requests that can never be admitted
            # (prompt already exceeds model/scheduler budget), so callers do
            # not spin forever waiting for an impossible request.
            completions: list[GenerationOutput] = []
            for seq in list(self._scheduler.waiting):
                request_context = self._request_contexts.get(seq.request_id)
                if request_context is None:
                    self._scheduler.finish(seq.request_id)
                    continue
                req = request_context.request
                if req.status in TERMINAL_REQUEST_STATUSES:
                    self._scheduler.finish(seq.request_id)
                    continue
                prompt_len = len(req.prompt_tokens)
                if prompt_len > self._worker.max_seq_len or prompt_len > self._scheduler.max_num_batched_tokens:
                    req.error = "prompt exceeds model/scheduler budget"
                    completions.append(self._complete_request(request_context, RequestStatus.FINISHED_ABORTED, "aborted"))
            return completions

        completions: list[GenerationOutput] = []
        active_contexts: list[_RequestContext] = []
        active_seqs: list[EngineSequence] = []

        # Build executable batch from scheduler output.
        # This stage handles state transition, budget checks, and per-request
        # generation-cap initialization.
        for seq in sched.scheduled_seqs:
            req_id = seq.request_id
            # 1) Scheduler may hold stale seqs; drop them if context is already gone.
            request_context = self._request_contexts.get(req_id)
            if request_context is None:
                self._scheduler.finish(req_id)
                continue

            req = request_context.request
            # 2) Terminal requests are removed from scheduler queues and their
            # cached output is surfaced once in this step result.
            if req.status in TERMINAL_REQUEST_STATUSES:
                self._scheduler.finish(req_id)
                if request_context.finished_output is not None:
                    completions.append(request_context.finished_output)
                continue

            try:
                # 3) Move WAITING -> RUNNING when the request is admitted.
                if req.status == RequestStatus.WAITING:
                    self._transition(req, RequestStatus.RUNNING)

                # 4) Empty prompt is a valid no-op request and is completed
                # immediately as ignored.
                if not req.prompt_tokens:
                    completions.append(self._complete_request(request_context, RequestStatus.FINISHED_IGNORED, "empty"))
                    continue

                # 5) Enforce hard budgets before execution.
                if len(req.prompt_tokens) > self._worker.max_seq_len:
                    raise ValueError("Prompt length exceeds maxseq")
                if len(req.prompt_tokens) > self._scheduler.max_num_batched_tokens:
                    raise ValueError("Prompt length exceeds scheduler token budget")

                # 6) Compute request-local generation cap once.
                if request_context.max_new_tokens < 0:
                    request_context.max_new_tokens = self._compute_max_new(
                        len(req.prompt_tokens), request_context.sampling_params
                    )
                # Cap can become zero if prompt already reaches model length.
                if request_context.max_new_tokens <= 0:
                    completions.append(
                        self._complete_request(request_context, RequestStatus.FINISHED_LENGTH_CAPPED, "length")
                    )
                    continue

                # 7) Request is executable in this round.
                active_contexts.append(request_context)
                active_seqs.append(seq)
            except Exception as exc:
                # Any per-request setup failure is converted into aborted output
                # so upper layers do not hang waiting for this request.
                req.error = str(exc)
                completions.append(self._complete_request(request_context, RequestStatus.FINISHED_ABORTED, "aborted"))

        if not active_seqs:
            return completions

        # Execute one decode/prefill step for active requests.
        sampling_params_by_req = {
            request_context.request.request_id: request_context.sampling_params
            for request_context in active_contexts
        }
        output_ids_t, sampled_t, token_idx_to_req_id = self._executor.execute_scheduler_step(
            SchedulerOutputs(scheduled_seqs=active_seqs, is_prefill=sched.is_prefill),
            sampling_params_by_req=sampling_params_by_req,
        )
        self._observe_runtime_kv_peak()
        if output_ids_t is None or sampled_t is None:
            return completions

        output_ids_cpu = output_ids_t.to(DeviceType.CPU, 0) if output_ids_t.device_type() != DeviceType.CPU else output_ids_t
        sampled_cpu = sampled_t.to(DeviceType.CPU, 0) if sampled_t.device_type() != DeviceType.CPU else sampled_t
        output_ids = output_ids_cpu.tolist()
        sampled = sampled_cpu.tolist()
        if len(output_ids) != len(sampled):
            raise RuntimeError("executor sampled/output mapping size mismatch")

        sampled_req_ids: list[str] = []
        for out_idx in output_ids:
            rid = token_idx_to_req_id.get(int(out_idx))
            if rid is None:
                raise RuntimeError("executor output id cannot be mapped to request")
            sampled_req_ids.append(rid)

        # Apply sampled tokens and finalize requests that hit stop/length.
        for req_id, token in zip(sampled_req_ids, sampled):
            request_context = self._request_contexts.get(req_id)
            if request_context is None:
                continue
            req = request_context.request
            req.output_tokens.append(int(token))
            request_context.seq.append_token(int(token))
            request_context.num_generated += 1

            finish_reason = self._maybe_finish_reason(request_context, int(token))
            if finish_reason is not None:
                completions.append(
                    self._complete_request(request_context, RequestStatus.FINISHED_STOPPED, finish_reason)
                )
                continue

            if request_context.num_generated >= request_context.max_new_tokens:
                completions.append(
                    self._complete_request(request_context, RequestStatus.FINISHED_LENGTH_CAPPED, "length")
                )

        return completions

    def collect(self, request_id: str) -> GenerationOutput | None:
        request_context = self._request_contexts.get(request_id)
        if request_context is None:
            return None
        return request_context.finished_output

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

            if not self._scheduler.has_work():
                raise RuntimeError("Engine finished without output for request")
            _ = self.step()

    def abort_request(self, request_id: str) -> bool:
        request_context = self._request_contexts.get(request_id)
        if request_context is None:
            return False

        req = request_context.request
        if req.status in TERMINAL_REQUEST_STATUSES:
            return True

        req.error = "aborted by user"
        self._complete_request(request_context, RequestStatus.FINISHED_ABORTED, "aborted")
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
        request_context: _RequestContext,
        terminal_status: RequestStatus,
        finish_reason: str,
    ) -> GenerationOutput:
        req = request_context.request
        if req.status not in TERMINAL_REQUEST_STATUSES:
            self._transition(req, terminal_status)

        text = self._decode_completion_text(req)
        if finish_reason == "stop_string" and request_context.matched_stop and text is not None:
            idx = text.find(request_context.matched_stop)
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
        request_context.finished_output = output
        self._scheduler.finish(req.request_id)
        self._worker.free_request(request_context.seq.seq_id)
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

    def _maybe_finish_reason(self, request_context: _RequestContext, token_id: int) -> str | None:
        sampling_params = request_context.sampling_params
        if (not bool(sampling_params.ignore_eos)) and token_id == self._worker.end_token_id:
            return "eos_token"
        if sampling_params.stop_token_ids and token_id in set(sampling_params.stop_token_ids):
            return "stop_token"
        if sampling_params.stop:
            completion_text = self._decode_completion_text(request_context.request)
            if completion_text is not None:
                for stop_str in sampling_params.stop:
                    if stop_str and stop_str in completion_text:
                        request_context.matched_stop = stop_str
                        return "stop_string"
        return None
