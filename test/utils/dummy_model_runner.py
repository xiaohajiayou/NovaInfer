from __future__ import annotations

from dataclasses import dataclass

import llaisys
from llaisys.engine.scheduler import SchedulerOutputs
from llaisys.engine.types import BatchPlan, SamplingParams
from llaisys.libllaisys.model import KvCacheLayout


@dataclass
class DummyModelRunner:
    max_seq_len: int = 32
    end_token_id: int = 4
    vocab_size: int = 16
    kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK

    def __post_init__(self):
        self._kv_cache_layout = KvCacheLayout(int(self.kv_cache_layout))
        self._request_free_calls: list[int] = []

    @property
    def request_free_calls(self) -> list[int]:
        return list(self._request_free_calls)

    def _prepare_model_input(
        self,
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
        q_seq_rows: list[int] = []
        token_index_to_request_id: dict[int, str] = {}
        slot_mapping: list[int] = []

        base = 0
        for row_idx, seq in enumerate(outputs.scheduled_seqs):
            bs = max(1, int(seq.block_size))
            if outputs.is_prefill:
                start = max(0, int(seq.num_cached_tokens))
                seq_tokens = [int(t) for t in seq.prompt_token_ids[start:]]
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
                params = sampling_params or seq.sampling_params

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
            q_seq_rows.extend([int(row_idx)] * len(seq_tokens))
            if seq.block_table:
                for p in seq_pos:
                    bidx = int(p) // bs
                    boff = int(p) % bs
                    if bidx < 0 or bidx >= len(seq.block_table):
                        raise RuntimeError("dummy runner block table out of range")
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
        block_layout = int(self._kv_cache_layout) == int(KvCacheLayout.BLOCK)
        if block_layout:
            if len(slot_mapping) != len(token_ids):
                raise RuntimeError("BLOCK layout requires slot_mapping for every token")
            batch_seq_ids = [int(s.seq_id) for s in outputs.scheduled_seqs]
            context_lens = [len(s) for s in outputs.scheduled_seqs]
            block_table_width = max((len(s.block_table) for s in outputs.scheduled_seqs), default=0)
            for s in outputs.scheduled_seqs:
                row = [int(b) for b in s.block_table]
                if len(row) < block_table_width:
                    row.extend([-1] * (block_table_width - len(row)))
                block_tables.extend(row)

        plan = BatchPlan(
            token_ids=token_ids,
            logits_mask=logits_mask,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            seeds=seeds,
            has_seeds=has_seeds,
            pos_ids=pos_ids,
            seq_ids=seq_ids,
            q_seq_rows=q_seq_rows if block_layout else None,
            q_pos=pos_ids if block_layout else None,
            slot_mapping=slot_mapping if slot_mapping else None,
            context_lens=context_lens if context_lens else None,
            batch_seq_ids=batch_seq_ids if batch_seq_ids else None,
            block_tables=block_tables if block_tables else None,
            block_table_width=int(block_table_width),
        )
        return plan, token_index_to_request_id

    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ):
        plan, token_idx_to_req_id = self._prepare_model_input(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )
        if not plan.token_ids:
            return None, None, None, {}
        output_ids = [i for i, m in enumerate(plan.logits_mask) if int(m) != 0]
        self.on_plan(plan)
        out = llaisys.Tensor((len(output_ids),), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
        out.copy_from_sequence(output_ids)
        return out, None, plan, token_idx_to_req_id

    def sample_tokens(self, logits_tensor, plan: BatchPlan):
        _ = logits_tensor
        output_ids = [i for i, m in enumerate(plan.logits_mask) if int(m) != 0]
        sampled = [(int(plan.token_ids[i]) + 1) % int(self.vocab_size) for i in output_ids]
        out = llaisys.Tensor((len(sampled),), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
        out.copy_from_sequence(sampled)
        return out

    def execute_step(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ):
        output_ids, logits_tensor, plan, token_idx_to_req_id = self.execute_model(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )
        if output_ids is None or plan is None:
            return None, None, {}
        sampled = self.sample_tokens(logits_tensor, plan)
        return output_ids, sampled, token_idx_to_req_id

    def decode_tokens(self, token_ids):
        return "".join(chr(ord("a") + int(t)) for t in token_ids)

    def request_free(self, seq_id: int) -> int:
        self._request_free_calls.append(int(seq_id))
        return 0

    def kv_stats(self) -> dict:
        return {
            "capacity_tokens": 0,
            "used_tokens": 0,
            "free_tokens": 0,
            "peak_used_tokens": 0,
        }

    def kv_reset_prefix_cache(self) -> int:
        return 0

    def on_plan(self, plan: BatchPlan) -> None:
        _ = plan
