from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

SeqIdLike = int | Sequence[int]


@dataclass
class BlockBatchState:
    seq_blocks: dict[int, list[int]] = field(default_factory=dict)
    next_bid: int = 0
    next_pos_by_seq: dict[int, int] = field(default_factory=dict)


@dataclass
class BatchBuildResult:
    token_ids: list[int]
    logits_mask: list[int]
    seq_ids: list[int]
    pos_values: list[int]
    req_num_scheduled_tokens: list[int] | None = None
    req_num_computed_tokens: list[int] | None = None
    block_tables: list[int] | None = None
    block_table_width: int = 0
    invalid: bool = False


def _normalize_seq_ids(n_tokens: int, seq_ids: Sequence[SeqIdLike] | None, default_seq_id: int) -> list[SeqIdLike]:
    if seq_ids is None:
        return [int(default_seq_id)] * n_tokens
    if len(seq_ids) != n_tokens:
        raise ValueError("seq_ids length mismatch")
    return list(seq_ids)


def _resolve_pos_values(
    seq_single: Sequence[int],
    pos_ids: Sequence[int] | None,
    state: BlockBatchState | None,
) -> list[int]:
    local_next: dict[int, int] = {}
    if pos_ids is not None:
        pos_values = [int(p) for p in pos_ids]
        if state is not None:
            for sid, p in zip(seq_single, pos_values):
                state.next_pos_by_seq[sid] = max(int(state.next_pos_by_seq.get(sid, 0)), int(p) + 1)
        return pos_values

    pos_values: list[int] = []
    for sid in seq_single:
        if state is not None:
            p = int(state.next_pos_by_seq.get(sid, 0))
            state.next_pos_by_seq[sid] = p + 1
        else:
            p = int(local_next.get(sid, 0))
            local_next[sid] = p + 1
        pos_values.append(p)
    return pos_values


def build_decode_batch(
    token_ids: Sequence[int],
    *,
    logits_mask: Sequence[int] | None = None,
    seq_ids: Sequence[SeqIdLike] | None = None,
    pos_ids: Sequence[int] | None = None,
    block_size: int = 16,
    block_state: BlockBatchState | None = None,
    default_seq_id: int = 0,
    shared_block_ids_per_batch: bool = False,
) -> BatchBuildResult:
    n = len(token_ids)
    if n <= 0:
        raise ValueError("token_ids must be non-empty")
    if logits_mask is not None and len(logits_mask) != n:
        raise ValueError("logits_mask length mismatch")
    if pos_ids is not None and len(pos_ids) != n:
        raise ValueError("pos_ids length mismatch")

    token_values = [int(t) for t in token_ids]
    logits_values = [int(x) for x in logits_mask] if logits_mask is not None else ([0] * max(0, n - 1) + [1])

    seq_norm = _normalize_seq_ids(n, seq_ids, default_seq_id)
    seq_rows: list[list[int]] = []
    for sid in seq_norm:
        if isinstance(sid, (list, tuple)):
            seq_rows.append([int(x) for x in sid])
        else:
            seq_rows.append([int(sid)])

    # Forward API currently expects one seq id per token.
    if any(len(row) != 1 for row in seq_rows):
        seq_single = [int(row[0]) if row else 0 for row in seq_rows]
        pos_values = _resolve_pos_values(seq_single, pos_ids, block_state)
        return BatchBuildResult(
            token_ids=token_values,
            logits_mask=logits_values,
            seq_ids=seq_single,
            pos_values=pos_values,
            invalid=True,
        )

    seq_single = [row[0] for row in seq_rows]
    pos_values = _resolve_pos_values(seq_single, pos_ids, block_state)

    st = block_state if block_state is not None else BlockBatchState()
    bs = max(1, int(block_size))
    if shared_block_ids_per_batch:
        seq_blocks = {sid: [] for sid in set(seq_single)}
        for sid, p in zip(seq_single, pos_values):
            bidx = int(p) // bs
            table = seq_blocks[sid]
            while len(table) <= bidx:
                table.append(len(table))
    else:
        seq_blocks = st.seq_blocks
        for sid, p in zip(seq_single, pos_values):
            bidx = int(p) // bs
            table = seq_blocks.setdefault(sid, [])
            while len(table) <= bidx:
                table.append(int(st.next_bid))
                st.next_bid += 1

    # vLLM-style descriptors: query_start_loc is derived from req_num_scheduled_tokens.
    # For direct core tests (including interleaved seq ids), use one scheduled token per row.
    req_num_scheduled_tokens = [1] * n
    req_num_computed_tokens = [int(p) for p in pos_values]
    row_width = max(1, max(len(seq_blocks[sid]) for sid in seq_single))
    block_tables: list[int] = []
    for sid in seq_single:
        row = list(seq_blocks[sid])
        if len(row) < row_width:
            row.extend([-1] * (row_width - len(row)))
        block_tables.extend(row)

    return BatchBuildResult(
        token_ids=token_values,
        logits_mask=logits_values,
        seq_ids=seq_single,
        pos_values=pos_values,
        req_num_scheduled_tokens=req_num_scheduled_tokens,
        req_num_computed_tokens=req_num_computed_tokens,
        block_tables=block_tables,
        block_table_width=int(row_width),
    )
