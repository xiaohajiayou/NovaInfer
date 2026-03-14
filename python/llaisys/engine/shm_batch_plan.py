from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

from .batch_plan import BatchPlan
from .config import EngineConfig

_HEADER_I64 = 16
_HEADER_BYTES = _HEADER_I64 * 8
_VERSION = 1


def estimate_shared_batch_plan_bytes(config: EngineConfig) -> int:
    max_tokens = int(config.max_num_batched_tokens)
    max_seqs = int(config.max_num_seqs)
    block_size = max(1, int(config.kv_cache_block_size))
    max_model_len = max(1, int(config.max_model_len or 4096))
    max_block_width = max(1, (max_model_len + block_size - 1) // block_size)
    bytes_total = _HEADER_BYTES
    bytes_total += max_tokens * 8  # input_ids
    bytes_total += max_tokens * 8  # positions
    bytes_total += max_seqs * 8  # scheduled_token_counts
    bytes_total += (max_seqs + 1) * 4  # cu_seqlens_q
    bytes_total += (max_seqs + 1) * 4  # cu_seqlens_k
    bytes_total += max_tokens * 4  # slot_mapping
    bytes_total += max_seqs * max_block_width * 4  # block tables
    bytes_total += max_seqs * 4  # temperatures
    bytes_total += max_seqs * 4  # top_ps
    bytes_total += max_seqs * 4  # top_ks
    bytes_total += max_seqs * 8  # seeds
    bytes_total += max_seqs * 4  # has_seeds
    return int(bytes_total + 4096)


class SharedBatchPlanBuffer:
    def __init__(self, size_bytes: int, shm: shared_memory.SharedMemory | None = None, name: str | None = None):
        if shm is not None:
            self._shm = shm
        elif name is not None:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
        else:
            self._shm = shared_memory.SharedMemory(create=True, size=int(size_bytes))
        self._size_bytes = int(self._shm.size)
        self._buf = self._shm.buf

    @property
    def name(self) -> str:
        return str(self._shm.name)

    @property
    def size_bytes(self) -> int:
        return int(self._size_bytes)

    def close(self, unlink: bool = False) -> None:
        self._shm.close()
        if unlink:
            self._shm.unlink()

    def write(self, plan: BatchPlan) -> None:
        header = np.ndarray((_HEADER_I64,), dtype=np.int64, buffer=self._buf, offset=0)
        header[:] = 0
        input_ids = np.asarray(plan.input_ids, dtype=np.int64)
        positions = np.asarray(plan.positions, dtype=np.int64)
        scheduled_counts = np.asarray(plan.scheduled_token_counts, dtype=np.int64)
        cu_q = np.asarray(plan.cu_seqlens_q, dtype=np.int32)
        cu_k = np.asarray(plan.cu_seqlens_k, dtype=np.int32)
        slot_mapping = np.asarray(plan.slot_mapping, dtype=np.int32)
        block_rows = np.asarray(plan.block_table_rows, dtype=np.int32)
        temperatures = np.asarray(plan.temperatures, dtype=np.float32)
        top_ps = np.asarray(plan.top_ps, dtype=np.float32)
        top_ks = np.asarray(plan.top_ks, dtype=np.int32)
        seeds = np.asarray(plan.seeds, dtype=np.int64)
        has_seeds = np.asarray(plan.has_seeds, dtype=np.int32)

        header[0] = _VERSION
        header[1] = 1 if bool(plan.is_prefill) else 0
        header[2] = int(plan.n_outputs)
        header[3] = int(input_ids.size)
        header[4] = int(positions.size)
        header[5] = int(scheduled_counts.size)
        header[6] = int(cu_q.size)
        header[7] = int(cu_k.size)
        header[8] = int(plan.max_seqlen_q)
        header[9] = int(plan.max_seqlen_k)
        header[10] = int(slot_mapping.size)
        header[11] = int(block_rows.shape[0])
        header[12] = int(plan.block_table_width)
        header[13] = 1 if bool(plan.incremental_block_table_update) else 0
        header[14] = int(temperatures.size)

        offset = _HEADER_BYTES
        offset = _write_array(self._buf, offset, input_ids)
        offset = _write_array(self._buf, offset, positions)
        offset = _write_array(self._buf, offset, scheduled_counts)
        offset = _write_array(self._buf, offset, cu_q)
        offset = _write_array(self._buf, offset, cu_k)
        offset = _write_array(self._buf, offset, slot_mapping)
        offset = _write_array(self._buf, offset, block_rows.reshape(-1))
        offset = _write_array(self._buf, offset, temperatures)
        offset = _write_array(self._buf, offset, top_ps)
        offset = _write_array(self._buf, offset, top_ks)
        offset = _write_array(self._buf, offset, seeds)
        offset = _write_array(self._buf, offset, has_seeds)
        if offset > self._size_bytes:
            raise RuntimeError("shared batch plan buffer overflow")

    def read(self) -> BatchPlan:
        header = np.ndarray((_HEADER_I64,), dtype=np.int64, buffer=self._buf, offset=0)
        if int(header[0]) != _VERSION:
            raise RuntimeError("invalid shared batch plan version")
        is_prefill = bool(int(header[1]))
        n_outputs = int(header[2])
        input_ids_n = int(header[3])
        positions_n = int(header[4])
        scheduled_counts_n = int(header[5])
        cu_q_n = int(header[6])
        cu_k_n = int(header[7])
        max_seqlen_q = int(header[8])
        max_seqlen_k = int(header[9])
        slot_mapping_n = int(header[10])
        block_rows_n = int(header[11])
        block_table_width = int(header[12])
        incremental = bool(int(header[13]))
        temps_n = int(header[14])

        offset = _HEADER_BYTES
        input_ids, offset = _read_array(self._buf, offset, np.int64, input_ids_n)
        positions, offset = _read_array(self._buf, offset, np.int64, positions_n)
        scheduled_counts, offset = _read_array(self._buf, offset, np.int64, scheduled_counts_n)
        cu_q, offset = _read_array(self._buf, offset, np.int32, cu_q_n)
        cu_k, offset = _read_array(self._buf, offset, np.int32, cu_k_n)
        slot_mapping, offset = _read_array(self._buf, offset, np.int32, slot_mapping_n)
        block_rows_flat, offset = _read_array(
            self._buf, offset, np.int32, block_rows_n * block_table_width
        )
        temperatures, offset = _read_array(self._buf, offset, np.float32, temps_n)
        top_ps, offset = _read_array(self._buf, offset, np.float32, temps_n)
        top_ks, offset = _read_array(self._buf, offset, np.int32, temps_n)
        seeds, offset = _read_array(self._buf, offset, np.int64, temps_n)
        has_seeds, offset = _read_array(self._buf, offset, np.int32, temps_n)
        block_rows = block_rows_flat.reshape((block_rows_n, block_table_width))
        if n_outputs != temps_n:
            raise RuntimeError("shared batch plan n_outputs mismatch")
        return BatchPlan(
            is_prefill=is_prefill,
            input_ids=input_ids,
            positions=positions,
            scheduled_token_counts=scheduled_counts,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_table_rows=block_rows,
            block_table_width=block_table_width,
            incremental_block_table_update=incremental,
            temperatures=temperatures.tolist(),
            top_ps=top_ps.tolist(),
            top_ks=[int(v) for v in top_ks.tolist()],
            seeds=[int(v) for v in seeds.tolist()],
            has_seeds=[int(v) for v in has_seeds.tolist()],
        )


def _write_array(buf, offset: int, arr: np.ndarray) -> int:
    nbytes = int(arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=buf, offset=offset)[:] = arr
    return int(offset + nbytes)


def _read_array(buf, offset: int, dtype, count: int):
    arr = np.ndarray((int(count),), dtype=dtype, buffer=buf, offset=offset)
    return arr, int(offset + arr.nbytes)
