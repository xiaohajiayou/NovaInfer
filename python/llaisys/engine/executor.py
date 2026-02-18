from __future__ import annotations

from .sampling import Sampler
from .scheduler import SchedulerOutputs
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

    def execute_scheduler_step(self, outputs: SchedulerOutputs, sampling_params: SamplingParams) -> tuple[list[int], list[str]]:
        plan, token_idx_to_req_id = self._flatten(outputs)
        output_ids, logits_rows = self._worker.execute(plan)
        sampled = [int(t) for t in self._sampler.sample(logits_rows, sampling_params)]

        # Keep compatibility with runners that do not return explicit output_ids.
        if not output_ids:
            output_ids = list(token_idx_to_req_id.keys())

        req_ids: list[str] = []
        for out_idx in output_ids:
            rid = token_idx_to_req_id.get(int(out_idx))
            if rid is None:
                raise RuntimeError("executor output id cannot be mapped to request")
            req_ids.append(rid)

        if len(sampled) != len(req_ids):
            raise RuntimeError("executor sampled/output mapping size mismatch")
        return sampled, req_ids

    @staticmethod
    def _flatten(outputs: SchedulerOutputs) -> tuple[BatchPlan, dict[int, str]]:
        token_ids: list[int] = []
        logits_mask: list[int] = []
        seq_ids: list[int] = []
        pos_ids: list[int] = []
        token_index_to_request_id: dict[int, str] = {}

        base = 0
        for seq in outputs.scheduled_seqs:
            if outputs.is_prefill:
                seq_tokens = [int(t) for t in seq.prompt_token_ids]
                seq_pos = list(range(len(seq_tokens)))
                seq_mask = [0] * len(seq_tokens)
                if seq_mask:
                    seq_mask[-1] = 1
            else:
                seq_tokens = [int(seq.last_token)]
                seq_pos = [len(seq) - 1]
                seq_mask = [1]

            token_ids.extend(seq_tokens)
            logits_mask.extend(seq_mask)
            pos_ids.extend(seq_pos)
            seq_ids.extend([int(seq.seq_id)] * len(seq_tokens))

            for i, m in enumerate(seq_mask):
                if m != 0:
                    token_index_to_request_id[base + i] = str(seq.request_id)
            base += len(seq_tokens)

        return (
            BatchPlan(token_ids=token_ids, logits_mask=logits_mask, pos_ids=pos_ids, seq_ids=seq_ids),
            token_index_to_request_id,
        )
