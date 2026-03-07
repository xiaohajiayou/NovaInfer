from __future__ import annotations

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
        self._worker.execute_model(scheduler_outputs)
        token_ids = self._worker.sample_tokens()
        return token_ids
