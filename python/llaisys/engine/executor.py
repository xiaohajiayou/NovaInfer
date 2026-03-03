from __future__ import annotations

from .scheduler import SchedulerOutputs
from .types import SamplingParams
from .worker import Worker


class Executor:
    """Coordinates one engine step (orchestration only, no tensor conversion)."""

    def __init__(self, worker: Worker):
        self._worker = worker

    def execute_scheduler_step(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ):
        self._worker.execute_model(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )
        sampled = self._worker.sample_tokens()
        if sampled is None:
            return None, []
        sampled_t, sampled_req_ids = sampled
        return sampled_t, sampled_req_ids
