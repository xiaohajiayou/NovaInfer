from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scheduler import SchedulerOutputs
    from .sequence import Sequence


class InputBatch:
    """Persistent batch state holder (vLLM-style shape, no extra wrapper object)."""

    def __init__(self, max_num_reqs: int, max_num_batched_tokens: int, max_block_table_width: int) -> None:
        self.max_num_reqs = max(1, int(max_num_reqs))
        self.max_num_batched_tokens = max(1, int(max_num_batched_tokens))
        self.max_block_table_width = max(1, int(max_block_table_width))

        self.seq_id_to_index: dict[int, int] = {}
        self.index_to_seq_id: list[int | None] = [None] * self.max_num_reqs
        self.requests: dict[int, "Sequence"] = {}

        self.token_ids_cpu = np.zeros((self.max_num_batched_tokens,), dtype=np.int64)
        self.pos_ids_cpu = np.zeros((self.max_num_batched_tokens,), dtype=np.int64)
        self.seq_ids_cpu = np.zeros((self.max_num_batched_tokens,), dtype=np.int64)
        self.logits_indices_cpu = np.zeros((self.max_num_reqs,), dtype=np.int64)
        self.req_num_scheduled_tokens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self.req_num_computed_tokens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self.block_table_cpu = np.full((self.max_num_reqs, self.max_block_table_width), -1, dtype=np.int32)

        self.scheduled_seqs: list["Sequence"] = []
        self.is_prefill = False
        self.num_scheduled_tokens_by_seq: dict[int, int] = {}

        self.n_tokens = 0
        self.n_outputs = 0
        self.output_rows: list[int] = []
        self.seq_ids: list[int] = []
        self.req_num_scheduled_tokens_step: list[int] = []
        self.req_num_computed_tokens_step: list[int] = []
        self.block_table_width = 0

    def _release_finished(self, finished_seq_ids: list[int]) -> None:
        for sid in finished_seq_ids:
            sid = int(sid)
            idx = self.seq_id_to_index.pop(sid, None)
            self.requests.pop(sid, None)
            if idx is None:
                continue
            self.index_to_seq_id[idx] = None
            self.req_num_scheduled_tokens_cpu[idx] = 0
            self.req_num_computed_tokens_cpu[idx] = 0
            self.block_table_cpu[idx, :] = -1

    def _acquire_slot(self, seq_id: int) -> int:
        sid = int(seq_id)
        idx = self.seq_id_to_index.get(sid)
        if idx is not None:
            return idx
        for i, cur in enumerate(self.index_to_seq_id):
            if cur is None:
                self.index_to_seq_id[i] = sid
                self.seq_id_to_index[sid] = i
                return i
        raise RuntimeError("InputBatch request slots exhausted")

    def update_states(self, outputs: "SchedulerOutputs") -> None:
        self._release_finished(list(outputs.finished_seq_ids))
        self.scheduled_seqs = list(outputs.scheduled_seqs)
        self.is_prefill = bool(outputs.is_prefill)
        self.num_scheduled_tokens_by_seq = dict(outputs.num_scheduled_tokens or {})
        for seq in self.scheduled_seqs:
            sid = int(seq.seq_id)
            self.requests[sid] = seq
            self._acquire_slot(sid)

    def prepare_inputs(self, *, is_block_layout: bool) -> None:
        self.n_tokens = 0
        self.n_outputs = 0
        self.output_rows = []
        self.seq_ids = []
        self.req_num_scheduled_tokens_step = []
        self.req_num_computed_tokens_step = []
        self.block_table_width = 0

        if is_block_layout and self.scheduled_seqs:
            self.block_table_width = max(len(seq.block_table) for seq in self.scheduled_seqs)
            if self.block_table_width <= 0:
                raise RuntimeError("BLOCK layout requires non-empty block_table rows")
            if self.block_table_width > self.max_block_table_width:
                raise RuntimeError("block_table_width exceeds configured max capacity")

        cursor = 0
        out_idx = 0
        for seq in self.scheduled_seqs:
            sid = int(seq.seq_id)
            if self.is_prefill:
                n_comp = max(0, int(seq.num_cached_tokens))
                start = n_comp
                end = max(start, int(seq.num_prompt_tokens))
                n_sched = max(0, end - start)
                if n_sched <= 0:
                    continue
                seq_tokens = seq.token_ids[start:end]
                pos_start = start
            else:
                n_comp = max(0, int(len(seq) - 1))
                n_sched = 1
                seq_tokens = [int(seq.last_token)]
                pos_start = n_comp

            row_start = cursor
            row_end = cursor + n_sched
            if row_end > self.max_num_batched_tokens:
                raise RuntimeError("prepared token count exceeds max_num_batched_tokens")
            self.token_ids_cpu[row_start:row_end] = np.asarray(seq_tokens, dtype=np.int64)
            self.pos_ids_cpu[row_start:row_end] = np.arange(pos_start, pos_start + n_sched, dtype=np.int64)
            self.seq_ids_cpu[row_start:row_end] = int(seq.seq_id)

            row_last = row_end - 1
            self.output_rows.append(int(row_last))
            self.logits_indices_cpu[out_idx] = int(row_last)
            self.seq_ids.append(sid)
            out_idx += 1

            if is_block_layout:
                self.req_num_scheduled_tokens_step.append(int(n_sched))
                self.req_num_computed_tokens_step.append(int(n_comp))
                slot = self._acquire_slot(sid)
                self.req_num_scheduled_tokens_cpu[slot] = int(n_sched)
                self.req_num_computed_tokens_cpu[slot] = int(n_comp)
                row_blocks = [int(b) for b in seq.block_table]
                if len(row_blocks) == 0 or len(row_blocks) > self.block_table_width:
                    raise RuntimeError("invalid block_table row width")
                if len(row_blocks) < self.block_table_width:
                    row_blocks.extend([-1] * (self.block_table_width - len(row_blocks)))
                self.block_table_cpu[slot, : self.block_table_width] = np.asarray(row_blocks, dtype=np.int32)

            cursor = row_end

        self.n_tokens = int(cursor)
        self.n_outputs = int(out_idx)
        if is_block_layout and self.req_num_scheduled_tokens_step:
            if sum(self.req_num_scheduled_tokens_step) != self.n_tokens:
                raise RuntimeError("sum(req_num_scheduled_tokens) must equal ntoken")
