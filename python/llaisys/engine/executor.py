from __future__ import annotations

from .sampling import Sampler
from .types import BatchPlan, SamplingParams, StepResult
from .worker import Worker


class Executor:
    """Coordinates one engine step: forward + sampling."""

    def __init__(self, worker: Worker, sampler: Sampler | None = None):
        self._worker = worker
        self._sampler = sampler if sampler is not None else Sampler()

    def execute_step(self, plan: BatchPlan, sampling_params: SamplingParams) -> StepResult:
        output_ids, logits_rows = self._worker.execute(plan)
        sampled = self._sampler.sample(logits_rows, sampling_params)
        return StepResult(sampled_token_ids=sampled, output_ids=output_ids)
