from __future__ import annotations

import os
from ctypes import byref, c_int32, c_int64
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .buffers import CpuGpuBuffer
from .config import EngineConfig
from .model_registry import ModelRegistry, create_default_registry
from .sampler import Sampler
from .scheduler import SchedulerOutputs
from ..nvtx import nvtx_range
from ..libllaisys import LIB_LLAISYS, DataType, DeviceType, llaisysDeviceType_t
from ..libllaisys.model import (
    AttentionMetadata,
    AttentionPhase,
    KvCacheLayout,
    LlaisysKvStats,
    ModelForwardInput,
    ModelForwardOutput,
)
from ..runtime import RuntimeAPI
from ..tensor import Tensor

Buffer = Tensor | CpuGpuBuffer
ExecuteState = tuple[Tensor, int, list[Tensor], list[float], list[float], list[int], list[int], list[int]]


@dataclass
class PreparedTensors:
    input_ids: Tensor
    pos_ids: Tensor
    logits_indices: Tensor
    n_outputs: int
    keepalive: list[Tensor]
    phase: int = int(AttentionPhase.PREFILL)
    # SLOT-only tensors.
    seq_ids: Tensor | None = None
    pos_ids_host: Tensor | None = None
    # BLOCK-only tensors.
    cu_seqlens_q: Tensor | None = None
    cu_seqlens_k: Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Tensor | None = None
    block_tables: Tensor | None = None
    block_table_width: int = 0
    # BLOCK+CUDNN-only tensors (already in CUDNN expected row layout).
    cudnn_seq_lens_q: Tensor | None = None
    cudnn_seq_lens_kv: Tensor | None = None
    cudnn_page_table: Tensor | None = None
    cudnn_qo_ragged_offset: Tensor | None = None
    cudnn_b_exec: int = 0


class GPUModelRunner:
    """Owns model + sampler and executes one scheduler step."""

    def __init__(
        self,
        model,
        config: EngineConfig | None = None,
        model_registry: ModelRegistry | None = None,
    ):
        if config is None:
            raise ValueError("config is required")
        self.model = model
        self._config = config
        self._device = self._config.device
        self._model_registry = model_registry if model_registry is not None else create_default_registry()
        self._model_type = str(self._config.model_type or "qwen2").strip().lower()
        self._model_path = Path(self._config.model_path) if self._config.model_path is not None else None
        meta_info = getattr(self.model, "_meta_info", None)
        self._num_heads = int(getattr(meta_info, "nh", 0))
        self._head_dim = int(getattr(meta_info, "dh", 0))
        self._config.max_model_len = int(self.model.max_seq_len)
        self._config.end_token_id = int(self.model.end_token_id)
        self._runtime = None
        self.allocate_kv_cache()
        self.sampler = Sampler(self._device, self._runtime, config=self._config)

        self._runtime_api = RuntimeAPI(DeviceType.NVIDIA) if self._device == DeviceType.NVIDIA else None
        self._paged_attn_backend = str(os.getenv("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "native")).strip().lower()

        self._compute_streams: dict[int, object] = {}

        self._max_num_reqs = self._config.max_num_seqs
        self._max_num_tokens = self._config.max_num_batched_tokens
        self._max_num_cudnn_rows = self._next_pow2(self._max_num_reqs)
        max_model_len = int(self._config.max_model_len)
        block_size = max(1, int(self._config.kv_cache_block_size))
        self._max_block_table_width = max(1, (max_model_len + block_size - 1) // block_size)

        self._sampled_ids_buf: Buffer = self._make_buffer((self._max_num_reqs,), DataType.I64, pin_memory=True)

        token_shape = (self._max_num_tokens,)
        self._input_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._pos_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._seq_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._output_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)

        if int(self._config.kv_cache_layout) == int(KvCacheLayout.BLOCK):
            self._cu_seqlens_q_buf = self._make_buffer((self._max_num_reqs + 1,), DataType.I32, pin_memory=True)
            self._cu_seqlens_k_buf = self._make_buffer((self._max_num_reqs + 1,), DataType.I32, pin_memory=True)
            self._slot_mapping_buf = self._make_buffer((self._max_num_tokens,), DataType.I32, pin_memory=True)
            self._block_tables_buf = self._make_buffer(
                (self._max_num_reqs * self._max_block_table_width,),
                DataType.I32,
                pin_memory=True,
            )
            # CUDNN BLOCK builder uses per-exec-row metadata buffers.
            self._cudnn_seq_lens_q_buf = self._make_buffer((self._max_num_cudnn_rows,), DataType.I32, pin_memory=True)
            self._cudnn_seq_lens_kv_buf = self._make_buffer((self._max_num_cudnn_rows,), DataType.I32, pin_memory=True)
            self._cudnn_page_table_buf = self._make_buffer(
                (self._max_num_cudnn_rows * self._max_block_table_width,),
                DataType.I32,
                pin_memory=True,
            )
            self._cudnn_qo_ragged_offset_buf = self._make_buffer(
                (self._max_num_cudnn_rows + 1,),
                DataType.I32,
                pin_memory=True,
            )

        self._logits_holder: Tensor = Tensor((1,), DataType.F32, self._device, 0)
        self._execute_model_state: ExecuteState | None = None
        self._closed = False

    def allocate_kv_cache(self) -> None:
        if self._runtime is not None:
            return
        if self._model_path is None:
            raise ValueError("model_path is required for runtime allocation")
        runtime_handle, runtime_info = self._model_registry.create_runtime(
            self._model_type,
            self._model_path,
            self._device,
            kv_cache_layout=self._config.kv_cache_layout,
            kv_cache_block_size=self._config.kv_cache_block_size,
            max_model_len=self._config.max_model_len,
            max_num_seqs=int(self._config.max_num_seqs),
            kv_cache_memory_utilization=self._config.kv_cache_memory_utilization,
        )
        if runtime_handle is None:
            raise RuntimeError("runtime allocation failed")
        self._runtime = runtime_handle
        if runtime_info:
            kv_capacity = runtime_info.get("kv_cache_capacity_tokens")
            if self._config.kv_cache_layout == KvCacheLayout.BLOCK and kv_capacity is not None:
                self._config.num_kvcache_blocks = (
                    int(kv_capacity) + int(self._config.kv_cache_block_size) - 1
                ) // int(self._config.kv_cache_block_size)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._execute_model_state = None
        if self._runtime_api is not None:
            for dev_id, stream in list(self._compute_streams.items()):
                try:
                    self._runtime_api.set_device(int(dev_id))
                    self._runtime_api.stream_synchronize(stream)
                except Exception:
                    pass
        runtime = self._runtime
        self._runtime = None
        close_fn = getattr(self.model, "close", None)
        if callable(close_fn):
            close_fn()
        if runtime is not None:
            LIB_LLAISYS.llaisysRuntimeDestroy(runtime)

    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ):
        with nvtx_range("py/runner/execute_model"):
            self._execute_model_state = None
            seqs = scheduler_outputs.scheduled_seqs
            if not seqs:
                return None
            with nvtx_range("py/runner/execute_model/prepare_inputs"):
                prepared = self.prepare_prefill(seqs) if bool(scheduler_outputs.is_prefill) else self.prepare_decode(seqs)
            if prepared is None:
                return None

            with nvtx_range("py/runner/execute_model/build_forward_io"):
                attn = AttentionMetadata()
                is_block = int(self._config.kv_cache_layout) == int(KvCacheLayout.BLOCK)
                attn.mode = c_int32(int(KvCacheLayout.BLOCK if is_block else KvCacheLayout.SLOT))
                attn.phase = c_int32(int(prepared.phase))
                attn.seq_ids = prepared.seq_ids.lib_tensor() if prepared.seq_ids is not None else None
                attn.pos_ids_host = prepared.pos_ids_host.lib_tensor() if prepared.pos_ids_host is not None else None
                attn.cu_seqlens_q = (
                    prepared.cu_seqlens_q.lib_tensor() if prepared.cu_seqlens_q is not None else None
                )
                attn.cu_seqlens_k = (
                    prepared.cu_seqlens_k.lib_tensor() if prepared.cu_seqlens_k is not None else None
                )
                attn.max_seqlen_q = c_int32(int(prepared.max_seqlen_q))
                attn.max_seqlen_k = c_int32(int(prepared.max_seqlen_k))
                attn.slot_mapping = prepared.slot_mapping.lib_tensor() if prepared.slot_mapping is not None else None
                attn.block_tables = prepared.block_tables.lib_tensor() if prepared.block_tables is not None else None
                attn.block_table_width = c_int32(int(prepared.block_table_width))
                attn.cudnn_seq_lens_q = (
                    prepared.cudnn_seq_lens_q.lib_tensor() if prepared.cudnn_seq_lens_q is not None else None
                )
                attn.cudnn_seq_lens_kv = (
                    prepared.cudnn_seq_lens_kv.lib_tensor() if prepared.cudnn_seq_lens_kv is not None else None
                )
                attn.cudnn_page_table = (
                    prepared.cudnn_page_table.lib_tensor() if prepared.cudnn_page_table is not None else None
                )
                attn.cudnn_qo_ragged_offset = (
                    prepared.cudnn_qo_ragged_offset.lib_tensor() if prepared.cudnn_qo_ragged_offset is not None else None
                )
                attn.cudnn_b_exec = c_int32(int(prepared.cudnn_b_exec))

                fin = ModelForwardInput()
                fin.input_ids = prepared.input_ids.lib_tensor()
                fin.pos_ids = prepared.pos_ids.lib_tensor()
                fin.logits_indices = prepared.logits_indices.lib_tensor()
                fin.attention = attn

                fout = ModelForwardOutput()
                fout.logits = self._logits_holder.lib_tensor()

            with nvtx_range("py/runner/execute_model/model_forward"):
                forward_fn = getattr(self.model, "forward", None)
                if not callable(forward_fn):
                    raise RuntimeError("model wrapper must implement forward(ModelForwardInput, ModelForwardOutput)")
                if self._runtime is None:
                    raise RuntimeError("runtime handle is required")
                status = int(forward_fn(self._runtime, fin, fout))
                if status != 0:
                    raise RuntimeError(f"modelForward failed with status={status}")

            with nvtx_range("py/runner/execute_model/prepare_state"):
                logits_shape = self._logits_holder.shape()
                if len(logits_shape) != 2:
                    raise RuntimeError("modelForward returned invalid logits shape")
                if int(logits_shape[0]) != int(prepared.n_outputs):
                    raise RuntimeError("modelForward logits row count mismatch with logits_indices")

                temperatures, top_ps, top_ks, seeds, has_seeds = self.prepare_sample(seqs)
                if len(temperatures) != int(prepared.n_outputs):
                    raise RuntimeError("sampling params count mismatch with logits rows")
                self._execute_model_state = (
                    self._logits_holder,
                    int(prepared.n_outputs),
                    prepared.keepalive,
                    temperatures,
                    top_ps,
                    top_ks,
                    seeds,
                    has_seeds,
                )
            return None

    def prepare_sample(
        self,
        seqs: list[object],
    ) -> tuple[list[float], list[float], list[int], list[int], list[int]]:
        temperatures: list[float] = []
        top_ps: list[float] = []
        top_ks: list[int] = []
        seeds: list[int] = []
        has_seeds: list[int] = []
        for seq_obj in seqs:
            params = seq_obj.sampling_params
            temperatures.append(float(params.temperature))
            top_ps.append(float(params.top_p))
            top_ks.append(int(params.top_k))
            if params.seed is None:
                has_seeds.append(0)
                seeds.append(0)
            else:
                has_seeds.append(1)
                seeds.append(int(params.seed))
        return temperatures, top_ps, top_ks, seeds, has_seeds

    def sample_tokens(self, grammar_output=None):
        with nvtx_range("py/runner/sample_tokens"):
            del grammar_output
            state = self._execute_model_state
            self._execute_model_state = None
            if state is None:
                return None
            (
                logits,
                n_outputs,
                _keepalive,
                temperatures,
                top_ps,
                top_ks,
                seeds,
                has_seeds,
            ) = state
            if int(n_outputs) == 0:
                return []
            if not isinstance(self._sampled_ids_buf, CpuGpuBuffer):
                raise RuntimeError("GPUModelRunner sampled buffer must be CpuGpuBuffer")
            sampled_ids_dev = self._sampled_ids_buf.gpu.slice(0, 0, int(n_outputs))
            with nvtx_range("py/runner/sample_tokens/sampler"):
                sampled = self.sampler.sample_tokens(
                    logits_tensor=logits,
                    out_ids_dev=sampled_ids_dev,
                    temperatures=temperatures,
                    top_ps=top_ps,
                    top_ks=top_ks,
                    seeds=seeds,
                    has_seeds=has_seeds,
                )
            if sampled is None:
                raise RuntimeError("sampler returned None for non-empty logits")
            if int(sampled.shape()[0]) != int(n_outputs):
                raise RuntimeError("sampler output size mismatch with logits rows")
            n_outputs = int(sampled.shape()[0])
            if n_outputs > self._max_num_reqs:
                raise RuntimeError("sampled outputs exceed configured max_num_reqs")
            with nvtx_range("py/runner/sample_tokens/d2h"):
                sampled_host = self._sampled_ids_buf.cpu.slice(0, 0, n_outputs)
                dev_id = int(sampled.device_id())
                compute_stream = self._get_runtime_compute_stream(dev_id)
                self._runtime_api.set_device(dev_id)
                sampled_host.copy_(sampled, non_blocking=True, stream=compute_stream)
                sync_fn = getattr(self._runtime_api, "stream_synchronize", None)
                if callable(sync_fn):
                    sync_fn(compute_stream)
            with nvtx_range("py/runner/sample_tokens/to_list"):
                return [int(token_id) for token_id in sampled_host.tolist()]

    def _make_buffer(self, shape: tuple[int, ...], dtype: DataType, pin_memory: bool = True) -> Buffer:
        if self._device == DeviceType.CPU:
            raise RuntimeError("GPUModelRunner cannot allocate CPU-only buffers; use CPUModelRunner")
        return CpuGpuBuffer(shape=shape, dtype=dtype, device=self._device, pin_memory=pin_memory)

    def _get_runtime_compute_stream(self, device_id: int):
        stream = self._compute_streams.get(int(device_id))
        if stream is not None:
            return stream
        stream = LIB_LLAISYS.llaisysRuntimeGetComputeStream(
            self._runtime,
            llaisysDeviceType_t(self._device),
            int(device_id),
        )
        if stream is None:
            raise RuntimeError(f"failed to acquire runtime compute stream for device_id={device_id}")
        self._compute_streams[int(device_id)] = stream
        return stream

    def _build_output_rows(self, scheduled_token_counts: list[int], n_outputs: int) -> np.ndarray:
        counts = np.asarray(scheduled_token_counts, dtype=np.int64)
        if counts.size == 0:
            output_rows = np.empty((0,), dtype=np.int64)
        else:
            output_rows = np.cumsum(counts, dtype=np.int64) - 1
        if int(output_rows.size) != int(n_outputs):
            raise RuntimeError("output_rows size mismatch")
        return output_rows

    def _use_cudnn_block_builder(self) -> bool:
        cfg = getattr(self, "_config", None)
        if cfg is None:
            return False
        device = getattr(self, "_device", DeviceType.CPU)
        backend = str(getattr(self, "_paged_attn_backend", "native")).strip().lower()
        return (
            device == DeviceType.NVIDIA
            and int(cfg.kv_cache_layout) == int(KvCacheLayout.BLOCK)
            and backend == "cudnn"
        )

    @staticmethod
    def _next_pow2(v: int) -> int:
        x = max(1, int(v))
        out = 1
        while out < x:
            out <<= 1
        return out

    def _build_packed_block_table_rows(self, seqs: list) -> tuple[list[list[int]], int]:
        if not seqs:
            raise RuntimeError("BLOCK layout requires non-empty seq list")
        widths = [len(getattr(seq, "block_table", [])) for seq in seqs]
        if any(w <= 0 for w in widths):
            raise RuntimeError("BLOCK layout requires non-empty block_table for every scheduled sequence")
        block_table_width = int(max(widths))
        rows: list[list[int]] = []
        for seq in seqs:
            row = [int(b) for b in seq.block_table]
            if any(v < 0 for v in row):
                raise RuntimeError("block_table contains invalid negative block id")
            if len(row) < block_table_width:
                # Packed semantics: pad tail with a valid block id instead of sentinel.
                row.extend([int(row[-1])] * (block_table_width - len(row)))
            rows.append(row)
        return rows, block_table_width

    def _build_cudnn_decode_rows(
        self,
        *,
        cu_seqlens_q: list[int],
        cu_seqlens_k: list[int],
        block_table_rows: list[list[int]],
        block_table_width: int,
        ntoken: int,
    ) -> tuple[list[int], list[int], list[int], int]:
        b_exec = int(ntoken)
        seq_q_rows: list[int] = [1] * b_exec
        seq_kv_rows: list[int] = [0] * b_exec
        page_rows: list[int] = [0] * (b_exec * int(block_table_width))

        row_idx = 0
        for seq_row in range(len(block_table_rows)):
            seq_len = int(cu_seqlens_k[seq_row + 1]) - int(cu_seqlens_k[seq_row])
            row_start = int(cu_seqlens_q[seq_row])
            row_end = int(cu_seqlens_q[seq_row + 1])
            row_scheduled = int(row_end - row_start)
            table_row = block_table_rows[seq_row]
            for local in range(row_scheduled):
                qpos = int(seq_len - row_scheduled + local)
                visible = int(max(0, min(seq_len, qpos + 1)))
                seq_kv_rows[row_idx] = visible
                dst = row_idx * int(block_table_width)
                page_rows[dst: dst + int(block_table_width)] = table_row
                row_idx += 1
        if row_idx != int(ntoken):
            raise RuntimeError("cudnn metadata row count mismatch")
        if any(int(v) < 0 for v in page_rows):
            raise RuntimeError("cudnn decode page_table contains invalid negative block id")
        return seq_q_rows, seq_kv_rows, page_rows, int(b_exec)

    def _build_cudnn_prefill_rows(
        self,
        *,
        cu_seqlens_q: list[int],
        cu_seqlens_k: list[int],
        block_table_rows: list[list[int]],
        block_table_width: int,
        ntoken: int,
    ) -> tuple[list[int], list[int], list[int], int, list[int]]:
        nseq = len(block_table_rows)
        if nseq <= 0:
            raise RuntimeError("cudnn prefill metadata requires non-empty batch")
        if len(cu_seqlens_q) != nseq + 1 or len(cu_seqlens_k) != nseq + 1:
            raise RuntimeError("cudnn prefill metadata cu_seqlens size mismatch")

        seq_q_rows = [int(cu_seqlens_q[i + 1] - cu_seqlens_q[i]) for i in range(nseq)]
        seq_kv_rows = [int(cu_seqlens_k[i + 1] - cu_seqlens_k[i]) for i in range(nseq)]
        if any(v <= 0 for v in seq_q_rows):
            raise RuntimeError("cudnn prefill metadata requires seq_len_q > 0 for every sequence")
        if any(v <= 0 for v in seq_kv_rows):
            raise RuntimeError("cudnn prefill metadata requires seq_len_kv > 0 for every sequence")
        if int(sum(seq_q_rows)) != int(ntoken):
            raise RuntimeError("cudnn prefill metadata token count mismatch")

        b_exec = int(nseq)
        page_rows = [int(v) for row in block_table_rows for v in row]
        if len(page_rows) != int(b_exec) * int(block_table_width):
            raise RuntimeError("cudnn prefill page_table size mismatch")
        if any(int(v) < 0 for v in page_rows):
            raise RuntimeError("cudnn prefill page_table contains invalid negative block id")

        hd = int(self._num_heads) * int(self._head_dim)
        if hd <= 0:
            raise RuntimeError("invalid model heads/head_dim for cudnn ragged prefill")
        token_prefix: list[int] = [0]
        for qlen in seq_q_rows:
            token_prefix.append(int(token_prefix[-1]) + int(qlen))
        ragged_offsets = [int(v) * int(hd) for v in token_prefix]
        return seq_q_rows, seq_kv_rows, page_rows, b_exec, ragged_offsets

    def _build_block_common_tensors(
        self,
        *,
        seqs: list,
        input_ids: list[int],
        positions: list[int],
        scheduled_token_counts: list[int],
    ) -> tuple[PreparedTensors, object]:
        ntoken = len(input_ids)
        n_outputs = len(seqs)
        if ntoken > self._max_num_tokens:
            raise RuntimeError("ntoken exceeds configured max_num_batched_tokens")
        if n_outputs > self._max_num_reqs:
            raise RuntimeError("n_outputs exceeds configured max_num_seqs")

        assert self._input_ids_buf is not None
        assert self._pos_ids_buf is not None
        assert self._output_ids_buf is not None
        if (
            not isinstance(self._input_ids_buf, CpuGpuBuffer)
            or not isinstance(self._pos_ids_buf, CpuGpuBuffer)
            or not isinstance(self._output_ids_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("GPU block common builder requires CpuGpuBuffer tensors")

        output_rows = self._build_output_rows(scheduled_token_counts, n_outputs)
        keepalive: list[Tensor] = []

        input_ids_host_t = self._input_ids_buf.cpu.slice(0, 0, ntoken)
        pos_ids_host_t = self._pos_ids_buf.cpu.slice(0, 0, ntoken)
        input_ids_t = self._input_ids_buf.gpu.slice(0, 0, ntoken)
        pos_ids_t = self._pos_ids_buf.gpu.slice(0, 0, ntoken)
        dev_id = int(input_ids_t.device_id())
        h2d_stream = self._get_runtime_compute_stream(dev_id)
        input_ids_host_t.copy_from_numpy(np.asarray(input_ids, dtype=np.int64))
        pos_ids_host_t.copy_from_numpy(np.asarray(positions, dtype=np.int64))
        input_ids_t.copy_(input_ids_host_t, non_blocking=True, stream=h2d_stream)
        pos_ids_t.copy_(pos_ids_host_t, non_blocking=True, stream=h2d_stream)
        keepalive.extend([input_ids_t, pos_ids_t])

        logits_indices_host_t = self._output_ids_buf.cpu.slice(0, 0, n_outputs)
        logits_indices_t = self._output_ids_buf.gpu.slice(0, 0, n_outputs)
        if n_outputs > 0:
            logits_indices_host_t.copy_from_numpy(output_rows)
            logits_indices_t.copy_(logits_indices_host_t, non_blocking=True, stream=h2d_stream)
        keepalive.append(logits_indices_t)

        prepared = PreparedTensors(
            input_ids=input_ids_t,
            pos_ids=pos_ids_t,
            logits_indices=logits_indices_t,
            n_outputs=n_outputs,
            keepalive=keepalive,
        )
        return prepared, h2d_stream

    def _build_block_native_meta(
        self,
        *,
        prepared: PreparedTensors,
        seqs: list,
        cu_seqlens_q: list[int],
        cu_seqlens_k: list[int],
        max_seqlen_q: int,
        max_seqlen_k: int,
        slot_mapping: list[int],
        block_table_rows: list[list[int]],
        block_table_width: int,
        h2d_stream,
    ) -> None:
        ntoken = int(prepared.input_ids.shape()[0])
        if len(cu_seqlens_q) != len(seqs) + 1:
            raise RuntimeError("cu_seqlens_q size mismatch")
        if len(cu_seqlens_k) != len(seqs) + 1:
            raise RuntimeError("cu_seqlens_k size mismatch")
        if int(cu_seqlens_q[-1]) != int(ntoken):
            raise RuntimeError("cu_seqlens_q[-1] must equal ntoken")
        if len(slot_mapping) != ntoken:
            raise RuntimeError("slot_mapping size mismatch")
        if block_table_width <= 0:
            raise RuntimeError("invalid block_table_width")

        n_block_elems = len(seqs) * block_table_width
        if n_block_elems > (self._max_num_reqs * self._max_block_table_width):
            raise RuntimeError("n_block_elems exceeds configured BLOCK metadata capacity")

        assert self._cu_seqlens_q_buf is not None
        assert self._cu_seqlens_k_buf is not None
        assert self._slot_mapping_buf is not None
        assert self._block_tables_buf is not None
        if (
            not isinstance(self._cu_seqlens_q_buf, CpuGpuBuffer)
            or not isinstance(self._cu_seqlens_k_buf, CpuGpuBuffer)
            or not isinstance(self._slot_mapping_buf, CpuGpuBuffer)
            or not isinstance(self._block_tables_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("GPU block native builder requires CpuGpuBuffer tensors")

        block_rows_flat = np.asarray(block_table_rows, dtype=np.int32).reshape(-1)
        cu_seqlens_q_host_t = self._cu_seqlens_q_buf.cpu.slice(0, 0, len(cu_seqlens_q))
        cu_seqlens_q_t = self._cu_seqlens_q_buf.gpu.slice(0, 0, len(cu_seqlens_q))
        cu_seqlens_k_host_t = self._cu_seqlens_k_buf.cpu.slice(0, 0, len(cu_seqlens_k))
        cu_seqlens_k_t = self._cu_seqlens_k_buf.gpu.slice(0, 0, len(cu_seqlens_k))
        slot_mapping_host_t = self._slot_mapping_buf.cpu.slice(0, 0, ntoken)
        slot_mapping_t = self._slot_mapping_buf.gpu.slice(0, 0, ntoken)
        block_tables_host_t = self._block_tables_buf.cpu.slice(0, 0, n_block_elems)
        block_tables_t = self._block_tables_buf.gpu.slice(0, 0, n_block_elems)

        cu_seqlens_q_host_t.copy_from_numpy(np.asarray(cu_seqlens_q, dtype=np.int32))
        cu_seqlens_k_host_t.copy_from_numpy(np.asarray(cu_seqlens_k, dtype=np.int32))
        slot_mapping_host_t.copy_from_numpy(np.asarray(slot_mapping, dtype=np.int32))
        block_tables_host_t.copy_from_numpy(block_rows_flat)
        cu_seqlens_q_t.copy_(cu_seqlens_q_host_t, non_blocking=True, stream=h2d_stream)
        cu_seqlens_k_t.copy_(cu_seqlens_k_host_t, non_blocking=True, stream=h2d_stream)
        slot_mapping_t.copy_(slot_mapping_host_t, non_blocking=True, stream=h2d_stream)
        block_tables_t.copy_(block_tables_host_t, non_blocking=True, stream=h2d_stream)

        prepared.keepalive.extend([cu_seqlens_q_t, cu_seqlens_k_t, slot_mapping_t, block_tables_t])
        prepared.cu_seqlens_q = cu_seqlens_q_t
        prepared.cu_seqlens_k = cu_seqlens_k_t
        prepared.max_seqlen_q = int(max_seqlen_q)
        prepared.max_seqlen_k = int(max_seqlen_k)
        prepared.slot_mapping = slot_mapping_t
        prepared.block_tables = block_tables_t
        prepared.block_table_width = int(block_table_width)

    def _build_block_cudnn_meta(
        self,
        *,
        prepared: PreparedTensors,
        seqs: list,
        attention_phase: int,
        slot_mapping: list[int],
        cu_seqlens_q: list[int],
        cu_seqlens_k: list[int],
        block_table_rows: list[list[int]],
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_table_width: int,
        h2d_stream,
    ) -> None:
        ntoken = int(prepared.input_ids.shape()[0])
        n_outputs = len(seqs)
        n_block_elems = n_outputs * int(block_table_width)
        if len(slot_mapping) != ntoken:
            raise RuntimeError("slot_mapping size mismatch")
        if n_block_elems > (self._max_num_reqs * self._max_block_table_width):
            raise RuntimeError("n_block_elems exceeds configured BLOCK metadata capacity")

        assert self._slot_mapping_buf is not None
        assert self._block_tables_buf is not None
        assert self._cudnn_seq_lens_q_buf is not None
        assert self._cudnn_seq_lens_kv_buf is not None
        assert self._cudnn_page_table_buf is not None
        assert self._cudnn_qo_ragged_offset_buf is not None
        if (
            not isinstance(self._slot_mapping_buf, CpuGpuBuffer)
            or not isinstance(self._block_tables_buf, CpuGpuBuffer)
            or not isinstance(self._cudnn_seq_lens_q_buf, CpuGpuBuffer)
            or not isinstance(self._cudnn_seq_lens_kv_buf, CpuGpuBuffer)
            or not isinstance(self._cudnn_page_table_buf, CpuGpuBuffer)
            or not isinstance(self._cudnn_qo_ragged_offset_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("GPU block cudnn builder requires CpuGpuBuffer tensors")

        block_rows_flat = np.asarray(block_table_rows, dtype=np.int32).reshape(-1)
        slot_mapping_host_t = self._slot_mapping_buf.cpu.slice(0, 0, ntoken)
        slot_mapping_t = self._slot_mapping_buf.gpu.slice(0, 0, ntoken)
        block_tables_host_t = self._block_tables_buf.cpu.slice(0, 0, n_block_elems)
        block_tables_t = self._block_tables_buf.gpu.slice(0, 0, n_block_elems)
        slot_mapping_host_t.copy_from_numpy(np.asarray(slot_mapping, dtype=np.int32))
        block_tables_host_t.copy_from_numpy(block_rows_flat)
        slot_mapping_t.copy_(slot_mapping_host_t, non_blocking=True, stream=h2d_stream)
        block_tables_t.copy_(block_tables_host_t, non_blocking=True, stream=h2d_stream)
        prepared.keepalive.extend([slot_mapping_t, block_tables_t])
        prepared.slot_mapping = slot_mapping_t
        prepared.block_tables = block_tables_t
        prepared.max_seqlen_q = int(max_seqlen_q)
        prepared.max_seqlen_k = int(max_seqlen_k)
        prepared.block_table_width = int(block_table_width)

        if int(attention_phase) == int(AttentionPhase.PREFILL):
            seq_q_rows, seq_kv_rows, page_rows, b_exec, qo_ragged_offsets = self._build_cudnn_prefill_rows(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                block_table_rows=block_table_rows,
                block_table_width=int(block_table_width),
                ntoken=int(ntoken),
            )
        elif int(attention_phase) == int(AttentionPhase.DECODE):
            seq_q_rows, seq_kv_rows, page_rows, b_exec = self._build_cudnn_decode_rows(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                block_table_rows=block_table_rows,
                block_table_width=int(block_table_width),
                ntoken=int(ntoken),
            )
            qo_ragged_offsets = None
        else:
            raise RuntimeError("invalid attention_phase for cudnn builder")
        if b_exec > int(self._max_num_cudnn_rows):
            raise RuntimeError("cudnn_b_exec exceeds configured max_num_seqs")
        if len(page_rows) > int(self._max_num_cudnn_rows) * int(self._max_block_table_width):
            raise RuntimeError("cudnn page_table exceeds configured metadata capacity")

        cudnn_seq_lens_q_host_t = self._cudnn_seq_lens_q_buf.cpu.slice(0, 0, int(b_exec))
        cudnn_seq_lens_q_t = self._cudnn_seq_lens_q_buf.gpu.slice(0, 0, int(b_exec))
        cudnn_seq_lens_kv_host_t = self._cudnn_seq_lens_kv_buf.cpu.slice(0, 0, int(b_exec))
        cudnn_seq_lens_kv_t = self._cudnn_seq_lens_kv_buf.gpu.slice(0, 0, int(b_exec))
        cudnn_page_table_host_t = self._cudnn_page_table_buf.cpu.slice(0, 0, len(page_rows))
        cudnn_page_table_t = self._cudnn_page_table_buf.gpu.slice(0, 0, len(page_rows))

        cudnn_seq_lens_q_host_t.copy_from_numpy(np.asarray(seq_q_rows, dtype=np.int32))
        cudnn_seq_lens_kv_host_t.copy_from_numpy(np.asarray(seq_kv_rows, dtype=np.int32))
        cudnn_page_table_host_t.copy_from_numpy(np.asarray(page_rows, dtype=np.int32))
        cudnn_seq_lens_q_t.copy_(cudnn_seq_lens_q_host_t, non_blocking=True, stream=h2d_stream)
        cudnn_seq_lens_kv_t.copy_(cudnn_seq_lens_kv_host_t, non_blocking=True, stream=h2d_stream)
        cudnn_page_table_t.copy_(cudnn_page_table_host_t, non_blocking=True, stream=h2d_stream)

        prepared.keepalive.extend([cudnn_seq_lens_q_t, cudnn_seq_lens_kv_t, cudnn_page_table_t])
        prepared.cudnn_seq_lens_q = cudnn_seq_lens_q_t
        prepared.cudnn_seq_lens_kv = cudnn_seq_lens_kv_t
        prepared.cudnn_page_table = cudnn_page_table_t
        prepared.cudnn_qo_ragged_offset = None
        if qo_ragged_offsets is not None:
            ragged_host_t = self._cudnn_qo_ragged_offset_buf.cpu.slice(0, 0, int(b_exec) + 1)
            ragged_t = self._cudnn_qo_ragged_offset_buf.gpu.slice(0, 0, int(b_exec) + 1)
            ragged_host_t.copy_from_numpy(np.asarray(qo_ragged_offsets, dtype=np.int32))
            ragged_t.copy_(ragged_host_t, non_blocking=True, stream=h2d_stream)
            prepared.keepalive.append(ragged_t)
            prepared.cudnn_qo_ragged_offset = ragged_t
        prepared.cudnn_b_exec = int(b_exec)

    def _build_slot_tensors(
        self,
        *,
        seqs: list,
        input_ids: list[int],
        positions: list[int],
        scheduled_token_counts: list[int],
    ) -> PreparedTensors:
        with nvtx_range("py/runner/prepare_inputs/slot/build"):
            ntoken = len(input_ids)
            n_outputs = len(seqs)
            if ntoken > self._max_num_tokens:
                raise RuntimeError("ntoken exceeds configured max_num_batched_tokens")
            if n_outputs > self._max_num_reqs:
                raise RuntimeError("n_outputs exceeds configured max_num_seqs")

            assert self._input_ids_buf is not None
            assert self._pos_ids_buf is not None
            assert self._seq_ids_buf is not None
            assert self._output_ids_buf is not None
            if (
                not isinstance(self._input_ids_buf, CpuGpuBuffer)
                or not isinstance(self._pos_ids_buf, CpuGpuBuffer)
                or not isinstance(self._seq_ids_buf, CpuGpuBuffer)
                or not isinstance(self._output_ids_buf, CpuGpuBuffer)
            ):
                raise RuntimeError("GPU slot builder requires CpuGpuBuffer tensors")

            output_rows = self._build_output_rows(scheduled_token_counts, n_outputs)
            keepalive: list[Tensor] = []

            input_ids_host_t = self._input_ids_buf.cpu.slice(0, 0, ntoken)
            pos_ids_host_t = self._pos_ids_buf.cpu.slice(0, 0, ntoken)
            input_ids_t = self._input_ids_buf.gpu.slice(0, 0, ntoken)
            pos_ids_t = self._pos_ids_buf.gpu.slice(0, 0, ntoken)
            dev_id = int(input_ids_t.device_id())
            h2d_stream = self._get_runtime_compute_stream(dev_id)

            input_ids_host_t.copy_from_numpy(np.asarray(input_ids, dtype=np.int64))
            pos_ids_host_t.copy_from_numpy(np.asarray(positions, dtype=np.int64))
            input_ids_t.copy_(input_ids_host_t, non_blocking=True, stream=h2d_stream)
            pos_ids_t.copy_(pos_ids_host_t, non_blocking=True, stream=h2d_stream)
            keepalive.extend([input_ids_t, pos_ids_t])

            seq_ids_per_req = np.asarray([int(seq_obj.seq_id) for seq_obj in seqs], dtype=np.int64)
            sched_counts = np.asarray(scheduled_token_counts, dtype=np.int64)
            seq_token_ids = np.repeat(seq_ids_per_req, sched_counts)
            seq_ids_host_t = self._seq_ids_buf.cpu.slice(0, 0, ntoken)
            seq_ids_t = self._seq_ids_buf.gpu.slice(0, 0, ntoken)
            seq_ids_host_t.copy_from_numpy(seq_token_ids)
            seq_ids_t.copy_(seq_ids_host_t, non_blocking=True, stream=h2d_stream)
            keepalive.extend([seq_ids_t, pos_ids_host_t])

            logits_indices_host_t = self._output_ids_buf.cpu.slice(0, 0, n_outputs)
            logits_indices_t = self._output_ids_buf.gpu.slice(0, 0, n_outputs)
            if n_outputs > 0:
                logits_indices_host_t.copy_from_numpy(output_rows)
                logits_indices_t.copy_(logits_indices_host_t, non_blocking=True, stream=h2d_stream)
            keepalive.append(logits_indices_t)

            return PreparedTensors(
                input_ids=input_ids_t,
                pos_ids=pos_ids_t,
                logits_indices=logits_indices_t,
                n_outputs=n_outputs,
                keepalive=keepalive,
                seq_ids=seq_ids_t,
                pos_ids_host=pos_ids_host_t,
            )

    def _build_block_tensors(
        self,
        *,
        seqs: list,
        attention_phase: int,
        input_ids: list[int],
        positions: list[int],
        scheduled_token_counts: list[int],
        cu_seqlens_q: list[int],
        cu_seqlens_k: list[int],
        max_seqlen_q: int,
        max_seqlen_k: int,
        slot_mapping: list[int],
        block_table_rows: list[list[int]],
        block_table_width: int,
    ) -> PreparedTensors:
        with nvtx_range("py/runner/prepare_inputs/block/build"):
            prepared, h2d_stream = self._build_block_common_tensors(
                seqs=seqs,
                input_ids=input_ids,
                positions=positions,
                scheduled_token_counts=scheduled_token_counts,
            )
            if self._use_cudnn_block_builder():
                with nvtx_range("py/runner/prepare_inputs/block/cudnn_meta"):
                    self._build_block_cudnn_meta(
                        prepared=prepared,
                        seqs=seqs,
                        attention_phase=int(attention_phase),
                        slot_mapping=slot_mapping,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        block_table_rows=block_table_rows,
                        max_seqlen_q=int(max_seqlen_q),
                        max_seqlen_k=int(max_seqlen_k),
                        block_table_width=int(block_table_width),
                        h2d_stream=h2d_stream,
                    )
            else:
                with nvtx_range("py/runner/prepare_inputs/block/native_meta"):
                    self._build_block_native_meta(
                        prepared=prepared,
                        seqs=seqs,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=int(max_seqlen_q),
                        max_seqlen_k=int(max_seqlen_k),
                        slot_mapping=slot_mapping,
                        block_table_rows=block_table_rows,
                        block_table_width=int(block_table_width),
                        h2d_stream=h2d_stream,
                    )
            return prepared

    def prepare_prefill(
        self,
        seqs: list,
    ) -> PreparedTensors | None:
        with nvtx_range("py/runner/prepare_inputs/prefill"):
            is_block_layout = int(self._config.kv_cache_layout) == int(KvCacheLayout.BLOCK)
            input_ids: list[int] = []
            positions: list[int] = []
            scheduled_token_counts: list[int] = []
            for seq in seqs:
                start = max(0, int(seq.num_cached_tokens))
                end = int(len(seq))
                if end <= start:
                    raise RuntimeError("scheduler invariant violated: prefill sequence has zero scheduled tokens")
                input_ids.extend([int(t) for t in seq[start:end]])
                positions.extend(range(start, end))
                scheduled_token_counts.append(int(end - start))
            if len(input_ids) <= 0:
                return None

            if not is_block_layout:
                prepared = self._build_slot_tensors(
                    seqs=seqs,
                    input_ids=input_ids,
                    positions=positions,
                    scheduled_token_counts=scheduled_token_counts,
                )
                prepared.phase = int(AttentionPhase.PREFILL)
                return prepared

            cu_seqlens_q = [0]
            cu_seqlens_k = [0]
            max_seqlen_q = 0
            max_seqlen_k = 0
            slot_mapping: list[int] = []
            for seq in seqs:
                if not seq.block_table:
                    raise ValueError("BLOCK layout requires non-empty block_table for every scheduled sequence")
                seqlen = int(len(seq))
                seqlen_q = max(0, seqlen - int(seq.num_cached_tokens))
                seqlen_k = seqlen
                cu_seqlens_q.append(int(cu_seqlens_q[-1]) + seqlen_q)
                cu_seqlens_k.append(int(cu_seqlens_k[-1]) + seqlen_k)
                max_seqlen_q = max(max_seqlen_q, seqlen_q)
                max_seqlen_k = max(max_seqlen_k, seqlen_k)
                for i in range(int(seq.num_cached_blocks), int(seq.num_blocks)):
                    start = int(seq.block_table[i]) * int(self._config.kv_cache_block_size)
                    if i != int(seq.num_blocks) - 1:
                        end = start + int(self._config.kv_cache_block_size)
                    else:
                        end = start + int(seq.last_block_num_tokens)
                    slot_mapping.extend(range(start, end))
            if int(cu_seqlens_q[-1]) != int(len(input_ids)):
                raise ValueError("sum(seqlen_q) must equal ntoken")
            if len(slot_mapping) != int(len(input_ids)):
                raise ValueError("slot_mapping length must equal ntoken")

            block_table_rows, block_table_width = self._build_packed_block_table_rows(seqs)
            prepared = self._build_block_tensors(
                seqs=seqs,
                attention_phase=int(AttentionPhase.PREFILL),
                input_ids=input_ids,
                positions=positions,
                scheduled_token_counts=scheduled_token_counts,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=int(max_seqlen_k),
                slot_mapping=slot_mapping,
                block_table_rows=block_table_rows,
                block_table_width=int(block_table_width),
            )
            prepared.phase = int(AttentionPhase.PREFILL)
            return prepared

    def prepare_decode(
        self,
        seqs: list,
    ) -> PreparedTensors | None:
        with nvtx_range("py/runner/prepare_inputs/decode"):
            is_block_layout = int(self._config.kv_cache_layout) == int(KvCacheLayout.BLOCK)
            input_ids: list[int] = [int(seq.last_token) for seq in seqs]
            positions: list[int] = [max(0, int(len(seq) - 1)) for seq in seqs]
            scheduled_token_counts: list[int] = [1] * len(seqs)
            if len(input_ids) <= 0:
                return None

            if not is_block_layout:
                prepared = self._build_slot_tensors(
                    seqs=seqs,
                    input_ids=input_ids,
                    positions=positions,
                    scheduled_token_counts=scheduled_token_counts,
                )
                prepared.phase = int(AttentionPhase.DECODE)
                return prepared

            cu_seqlens_q = [0]
            cu_seqlens_k = [0]
            max_seqlen_q = 1
            max_seqlen_k = 0
            slot_mapping: list[int] = []
            for seq in seqs:
                if not seq.block_table:
                    raise ValueError("BLOCK layout requires non-empty block_table for every scheduled sequence")
                seqlen_k = int(len(seq))
                cu_seqlens_q.append(int(cu_seqlens_q[-1]) + 1)
                cu_seqlens_k.append(int(cu_seqlens_k[-1]) + seqlen_k)
                max_seqlen_k = max(max_seqlen_k, seqlen_k)
                slot_mapping.append(
                    int(seq.block_table[-1]) * int(self._config.kv_cache_block_size)
                    + int(seq.last_block_num_tokens)
                    - 1
                )
            if int(cu_seqlens_q[-1]) != int(len(input_ids)):
                raise ValueError("sum(seqlen_q) must equal ntoken")
            if len(slot_mapping) != int(len(input_ids)):
                raise ValueError("slot_mapping length must equal ntoken")

            block_table_rows, block_table_width = self._build_packed_block_table_rows(seqs)
            prepared = self._build_block_tensors(
                seqs=seqs,
                attention_phase=int(AttentionPhase.DECODE),
                input_ids=input_ids,
                positions=positions,
                scheduled_token_counts=scheduled_token_counts,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=int(max_seqlen_k),
                slot_mapping=slot_mapping,
                block_table_rows=block_table_rows,
                block_table_width=int(block_table_width),
            )
            prepared.phase = int(AttentionPhase.DECODE)
            return prepared
    def request_free(self, seq_id: int) -> None:
        if self._runtime is None:
            return
        LIB_LLAISYS.llaisysRuntimeRequestFree(self._runtime, c_int64(int(seq_id)))

    def kv_stats(self) -> dict[str, int]:
        if self._runtime is None:
            return {
                "capacity_tokens": 0,
                "used_tokens": 0,
                "free_tokens": 0,
                "peak_used_tokens": 0,
            }
        out = LlaisysKvStats()
        rc = int(LIB_LLAISYS.llaisysRuntimeKvStats(self._runtime, byref(out)))
        if rc != 0:
            raise RuntimeError(f"KvStats failed with status={rc}")
        return {
            "capacity_tokens": int(out.capacity_tokens),
            "used_tokens": int(out.used_tokens),
            "free_tokens": int(out.free_tokens),
            "peak_used_tokens": int(out.peak_used_tokens),
        }

    def kv_reset_prefix_cache(self) -> int:
        if self._runtime is None:
            return 5
        return int(LIB_LLAISYS.llaisysRuntimeKvResetPrefixCache(self._runtime))
