from __future__ import annotations

from ..utils.nvtx import nvtx_range
from .scheduler import SchedulerOutputs
from .worker import Worker


class Executor:
    """Coordinates one engine step and returns sampled token ids."""

    def __init__(self, worker: Worker):
        self._worker = worker

    @property
    def worker(self) -> Worker:
        return self._worker

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

    def free_request(self, seq_id: int) -> None:
        self._worker.free_request(int(seq_id))

    def reset_prefix_cache(self) -> int:
        runner = self._worker.model_runner
        reset_fn = getattr(runner, "kv_reset_prefix_cache", None)
        if callable(reset_fn):
            try:
                return int(reset_fn())
            except Exception:
                return 5
        return 0

    def check_health(self) -> None:
        return None

    def close(self) -> None:
        self._worker.close()
