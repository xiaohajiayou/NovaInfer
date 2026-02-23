from __future__ import annotations

from ctypes import POINTER, c_int8, c_int32, c_int64, cast
from dataclasses import dataclass, field
from typing import Sequence

from llaisys.libllaisys.model import KvCacheLayout, LlaisysBatch


SeqIdLike = int | Sequence[int]


@dataclass
class BlockBatchState:
    seq_blocks: dict[int, list[int]] = field(default_factory=dict)
    next_bid: int = 0
    next_pos_by_seq: dict[int, int] = field(default_factory=dict)


@dataclass
class BatchBuildResult:
    batch: LlaisysBatch
    hold: tuple
    pos_values: list[int]


def _normalize_seq_ids(n_tokens: int, seq_ids: Sequence[SeqIdLike] | None, default_seq_id: int) -> list[SeqIdLike]:
    if seq_ids is None:
        return [int(default_seq_id)] * n_tokens
    if len(seq_ids) != n_tokens:
        raise ValueError("seq_ids length mismatch")
    return list(seq_ids)


def _build_seq_ptrs(seq_rows: list[list[int]]) -> tuple[object, object, list[object]]:
    n = len(seq_rows)
    n_seq_buf = (c_int32 * n)()
    seq_ptr_buf = (POINTER(c_int64) * n)()
    raw_rows: list[object] = []
    for i, seq_list in enumerate(seq_rows):
        row = (c_int64 * len(seq_list))(*[int(x) for x in seq_list])
        raw_rows.append(row)
        n_seq_buf[i] = len(seq_list)
        seq_ptr_buf[i] = cast(row, POINTER(c_int64))
    return n_seq_buf, seq_ptr_buf, raw_rows


def build_decode_batch(
    token_ids: Sequence[int],
    *,
    logits_mask: Sequence[int] | None = None,
    seq_ids: Sequence[SeqIdLike] | None = None,
    pos_ids: Sequence[int] | None = None,
    layout: KvCacheLayout | int = KvCacheLayout.SLOT,
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

    is_block = int(layout) == int(KvCacheLayout.BLOCK)

    token_buf = (c_int64 * n)(*[int(t) for t in token_ids])
    logits_buf = None if logits_mask is None else (c_int8 * n)(*[int(x) for x in logits_mask])

    seq_norm = _normalize_seq_ids(n, seq_ids, default_seq_id)
    seq_rows: list[list[int]] = []
    for sid in seq_norm:
        if isinstance(sid, (list, tuple)):
            seq_rows.append([int(x) for x in sid])
        else:
            seq_rows.append([int(sid)])
    n_seq_buf, seq_ptr_buf, raw_rows = _build_seq_ptrs(seq_rows)

    if not is_block:
        pos_buf = None if pos_ids is None else (c_int64 * n)(*[int(p) for p in pos_ids])
        batch = LlaisysBatch(
            n_tokens=c_int32(n),
            token=token_buf,
            embd=None,
            pos=pos_buf,
            n_seq_id=n_seq_buf,
            seq_id=seq_ptr_buf,
            logits=logits_buf,
        )
        return BatchBuildResult(
            batch=batch,
            hold=(token_buf, logits_buf, pos_buf, n_seq_buf, seq_ptr_buf, raw_rows),
            pos_values=[] if pos_ids is None else [int(p) for p in pos_ids],
        )

    # BLOCK path requires one seq per token.
    seq_single: list[int] = []
    for row in seq_rows:
        if len(row) != 1:
            # Keep invalid block metadata absent so runtime path can return proper error.
            pos_buf = None if pos_ids is None else (c_int64 * n)(*[int(p) for p in pos_ids])
            batch = LlaisysBatch(
                n_tokens=c_int32(n),
                token=token_buf,
                embd=None,
                pos=pos_buf,
                n_seq_id=n_seq_buf,
                seq_id=seq_ptr_buf,
                logits=logits_buf,
            )
            return BatchBuildResult(
                batch=batch,
                hold=(token_buf, logits_buf, pos_buf, n_seq_buf, seq_ptr_buf, raw_rows),
                pos_values=[] if pos_ids is None else [int(p) for p in pos_ids],
            )
        seq_single.append(row[0])

    st = block_state if block_state is not None else BlockBatchState()
    if pos_ids is None:
        pos_values = []
        for sid in seq_single:
            p = int(st.next_pos_by_seq.get(sid, 0))
            pos_values.append(p)
            st.next_pos_by_seq[sid] = p + 1
    else:
        pos_values = [int(p) for p in pos_ids]
        for sid, p in zip(seq_single, pos_values):
            st.next_pos_by_seq[sid] = max(int(st.next_pos_by_seq.get(sid, 0)), p + 1)

    bs = max(1, int(block_size))
    if shared_block_ids_per_batch:
        seq_blocks = {sid: [] for sid in set(seq_single)}
        for sid, p in zip(seq_single, pos_values):
            bidx = p // bs
            table = seq_blocks[sid]
            while len(table) <= bidx:
                table.append(len(table))
    else:
        seq_blocks = st.seq_blocks
        for sid, p in zip(seq_single, pos_values):
            bidx = p // bs
            table = seq_blocks.setdefault(sid, [])
            while len(table) <= bidx:
                table.append(int(st.next_bid))
                st.next_bid += 1

    uniq_seq = sorted(set(seq_single))
    context_lens = [int(st.next_pos_by_seq[sid]) for sid in uniq_seq]
    row_width = max(1, max(len(seq_blocks[sid]) for sid in uniq_seq))
    block_tables: list[int] = []
    for sid in uniq_seq:
        row = list(seq_blocks[sid])
        if len(row) < row_width:
            row.extend([-1] * (row_width - len(row)))
        block_tables.extend(row)

    seq_to_row = {sid: i for i, sid in enumerate(uniq_seq)}
    slot_mapping = []
    for sid, p in zip(seq_single, pos_values):
        row = seq_to_row[sid]
        bidx = p // bs
        boff = p % bs
        bid = block_tables[row * row_width + bidx]
        slot_mapping.append(int(bid * bs + boff))

    pos_buf = (c_int64 * n)(*pos_values)
    n_seq1_buf = (c_int32 * n)(*[1] * n)
    seq1_ptr_buf = (POINTER(c_int64) * n)()
    seq1_rows: list[object] = []
    for i, sid in enumerate(seq_single):
        row = (c_int64 * 1)(int(sid))
        seq1_rows.append(row)
        seq1_ptr_buf[i] = cast(row, POINTER(c_int64))

    slot_mapping_buf = (c_int32 * n)(*slot_mapping)
    context_lens_buf = (c_int32 * len(context_lens))(*context_lens)
    batch_seq_ids_buf = (c_int64 * len(uniq_seq))(*uniq_seq)
    block_tables_buf = (c_int32 * len(block_tables))(*block_tables)

    batch = LlaisysBatch(
        n_tokens=c_int32(n),
        token=token_buf,
        embd=None,
        pos=pos_buf,
        n_seq_id=n_seq1_buf,
        seq_id=seq1_ptr_buf,
        logits=logits_buf,
        slot_mapping=slot_mapping_buf,
        context_lens=context_lens_buf,
        batch_seq_ids=batch_seq_ids_buf,
        block_tables=block_tables_buf,
        n_batch_seq=c_int32(len(uniq_seq)),
        block_table_width=c_int32(row_width),
    )
    return BatchBuildResult(
        batch=batch,
        hold=(
            token_buf,
            logits_buf,
            pos_buf,
            n_seq1_buf,
            seq1_ptr_buf,
            seq1_rows,
            slot_mapping_buf,
            context_lens_buf,
            batch_seq_ids_buf,
            block_tables_buf,
        ),
        pos_values=pos_values,
    )
