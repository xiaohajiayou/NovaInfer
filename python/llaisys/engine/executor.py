from __future__ import annotations

from ..nvtx import nvtx_range
from .scheduler import SchedulerOutputs
from .worker import Worker


class Executor:
    """Coordinates one engine step and returns sampled token ids."""

    def __init__(self, worker: Worker):
        self._worker = worker

    def execute_scheduler_step(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> list[int] | None:
        with nvtx_range("py/executor/execute_scheduler_step"):
            with nvtx_range("py/executor/worker_execute_model"):
                self._worker.execute_model(scheduler_outputs)
            with nvtx_range("py/executor/worker_sample_tokens"):
                token_ids = self._worker.sample_tokens()
            return token_ids
