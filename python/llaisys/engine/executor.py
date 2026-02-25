from __future__ import annotations

from .scheduler import SchedulerOutputs
from .types import BatchPlan, SamplingParams
from .worker import Worker


class Executor:
    """Coordinates one engine step: forward + native sampling."""

    def __init__(self, worker: Worker):
        self._worker = worker

    def execute_scheduler_step(
        self,
        outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ) -> tuple[list[int], list[str]]:
        plan, token_idx_to_req_id = self._flatten(
            outputs, sampling_params=sampling_params, sampling_params_by_req=sampling_params_by_req
        )
        if not plan.token_ids:
            return [], []
        output_ids, sampled = self._worker.execute(plan)

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
        return [int(x) for x in sampled], req_ids

    @staticmethod
    def _flatten(
        outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ) -> tuple[BatchPlan, dict[int, str]]:
        token_ids: list[int] = []
        logits_mask: list[int] = []
        temperatures: list[float] = []
        top_ps: list[float] = []
        top_ks: list[int] = []
        seeds: list[int] = []
        has_seeds: list[int] = []
        seq_ids: list[int] = []
        pos_ids: list[int] = []
        token_index_to_request_id: dict[int, str] = {}
        slot_mapping: list[int] = []

        base = 0
        for seq in outputs.scheduled_seqs:
            bs = max(1, int(seq.block_size))
            if outputs.is_prefill:
                start = max(0, int(seq.num_cached_tokens))
                prompt = seq.prompt_token_ids
                seq_tokens = [int(t) for t in prompt[start:]]
                seq_pos = list(range(start, start + len(seq_tokens)))
                seq_mask = [0] * len(seq_tokens)
                if seq_mask:
                    seq_mask[-1] = 1
            else:
                seq_tokens = [int(seq.last_token)]
                seq_pos = [len(seq) - 1]
                seq_mask = [1]

            rid = str(seq.request_id)
            params = sampling_params_by_req.get(rid) if sampling_params_by_req is not None else None
            if params is None:
                if sampling_params is None:
                    raise RuntimeError("missing sampling params for request")
                params = sampling_params

            token_ids.extend(seq_tokens)
            logits_mask.extend(seq_mask)
            temperatures.extend([float(params.temperature)] * len(seq_tokens))
            top_ps.extend([float(params.top_p)] * len(seq_tokens))
            top_ks.extend([int(params.top_k)] * len(seq_tokens))
            if params.seed is None:
                has_seeds.extend([0] * len(seq_tokens))
                seeds.extend([0] * len(seq_tokens))
            else:
                has_seeds.extend([1] * len(seq_tokens))
                seeds.extend([int(params.seed)] * len(seq_tokens))
            pos_ids.extend(seq_pos)
            seq_ids.extend([int(seq.seq_id)] * len(seq_tokens))
            if seq.block_table:
                for p in seq_pos:
                    bidx = int(p) // bs
                    boff = int(p) % bs
                    if bidx < 0 or bidx >= len(seq.block_table):
                        raise RuntimeError("executor block table out of range")
                    bid = int(seq.block_table[bidx])
                    slot_mapping.append(bid * bs + boff)

            for i, m in enumerate(seq_mask):
                if m != 0:
                    token_index_to_request_id[base + i] = str(seq.request_id)
            base += len(seq_tokens)

        context_lens: list[int] = []
        batch_seq_ids: list[int] = []
        block_tables: list[int] = []
        block_table_width = 0
        if outputs.scheduled_seqs and all(len(s.block_table) > 0 for s in outputs.scheduled_seqs):
            batch_seq_ids = [int(s.seq_id) for s in outputs.scheduled_seqs]
            context_lens = [len(s) for s in outputs.scheduled_seqs]
            block_table_width = max(len(s.block_table) for s in outputs.scheduled_seqs)
            for s in outputs.scheduled_seqs:
                row = [int(b) for b in s.block_table]
                if len(row) < block_table_width:
                    row.extend([-1] * (block_table_width - len(row)))
                block_tables.extend(row)

        return (
            BatchPlan(
                token_ids=token_ids,
                logits_mask=logits_mask,
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
                seeds=seeds,
                has_seeds=has_seeds,
                pos_ids=pos_ids,
                seq_ids=seq_ids,
                slot_mapping=slot_mapping if slot_mapping else None,
                context_lens=context_lens if context_lens else None,
                batch_seq_ids=batch_seq_ids if batch_seq_ids else None,
                block_tables=block_tables if block_tables else None,
                block_table_width=int(block_table_width),
            ),
            token_index_to_request_id,
        )
