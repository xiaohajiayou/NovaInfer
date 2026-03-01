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
        outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ):
        output_ids_t, sampled_t, token_idx_to_req_id = self._worker.execute(
            outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )
        return output_ids_t, sampled_t, token_idx_to_req_id
