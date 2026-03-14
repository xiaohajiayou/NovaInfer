from __future__ import annotations

from dataclasses import dataclass

import llaisys
from llaisys.engine.scheduler import SchedulerOutputs


@dataclass
class DummyModelRunner:
    max_seq_len: int = 32
    end_token_id: int = 4
    vocab_size: int = 16

    def __post_init__(self):
        self._request_free_calls: list[int] = []
        self._execute_state = None

    @property
    def request_free_calls(self) -> list[int]:
        return list(self._request_free_calls)

    def _prepare_inputs(
        self,
        outputs: SchedulerOutputs,
    ) -> tuple[dict, list[int], list[int]]:
        token_ids: list[int] = []
        output_ids: list[int] = []
        active_seqs = []

        for seq in outputs.scheduled_seqs:
            if outputs.is_prefill:
                start = max(0, int(seq.num_cached_tokens))
                seq_tokens = [int(t) for t in seq.prompt_token_ids[start:]]
            else:
                seq_tokens = [int(seq.last_token)]

            if not seq_tokens:
                continue
            active_seqs.append(seq)

            base = len(token_ids)
            token_ids.extend(seq_tokens)
            output_ids.append(base + len(seq_tokens) - 1)

        req_num_scheduled_tokens: list[int] = []
        req_num_computed_tokens: list[int] = []
        block_table_width = 0
        for seq in active_seqs:
            if not seq.block_table:
                raise RuntimeError("BLOCK layout requires non-empty block_table for every scheduled sequence")
        total_sched = 0
        for seq in active_seqs:
            if outputs.is_prefill:
                n_sched = max(0, int(len(seq.prompt_token_ids) - int(seq.num_cached_tokens)))
                n_comp = max(0, int(seq.num_cached_tokens))
            else:
                n_sched = 1
                n_comp = max(0, int(len(seq) - 1))
            req_num_scheduled_tokens.append(int(n_sched))
            req_num_computed_tokens.append(int(n_comp))
            total_sched += int(n_sched)
        if total_sched != len(token_ids):
            raise RuntimeError("sum(req_num_scheduled_tokens) must equal token_ids length")
        block_table_width = max((len(s.block_table) for s in active_seqs), default=0)

        step = {
            "scheduled_seqs": list(active_seqs),
            "is_prefill": bool(outputs.is_prefill),
            "n_tokens": len(token_ids),
            "req_num_scheduled_tokens": req_num_scheduled_tokens if req_num_scheduled_tokens else None,
            "req_num_computed_tokens": req_num_computed_tokens if req_num_computed_tokens else None,
            "block_table_width": int(block_table_width),
        }
        return step, output_ids, token_ids

    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ):
        step, output_ids, token_ids = self._prepare_inputs(scheduler_outputs)
        self._execute_state = None
        if int(step["n_tokens"]) <= 0:
            return None
        self.on_plan(step)
        self._execute_state = (output_ids, token_ids)
        return None

    def sample_tokens(self, grammar_output=None):
        _ = grammar_output
        if self._execute_state is None:
            return None
        output_indices, token_ids = self._execute_state
        self._execute_state = None
        sampled = [(int(token_ids[i]) + 1) % int(self.vocab_size) for i in output_indices]
        out = llaisys.Tensor((len(sampled),), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
        out.copy_from_sequence(sampled)
        return out

    def execute(
        self,
        scheduler_outputs: SchedulerOutputs,
    ):
        self.execute_model(scheduler_outputs)
        sampled = self.sample_tokens(None)
        if sampled is None:
            return None
        return sampled

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

    def on_plan(self, step: dict) -> None:
        _ = step
