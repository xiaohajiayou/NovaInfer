from __future__ import annotations

from dataclasses import dataclass
from ctypes import byref, c_int32, c_int64, c_void_p, cast
from pathlib import Path
import numpy as np

from .config import EngineConfig
from .buffers import CpuGpuBuffer
from .gpu_input_batch import InputBatch
from .model_registry import ModelRegistry, create_default_registry
from .sampler import Sampler
from .scheduler import SchedulerOutputs
from ..libllaisys import LIB_LLAISYS, DataType, DeviceType, llaisysDeviceType_t
from ..libllaisys.model import (
    AttentionMetadata,
    KvCacheLayout,
    LlaisysKvStats,
    ModelForwardInput,
    ModelForwardOutput,
)
from ..runtime import RuntimeAPI
from ..tensor import Tensor


@dataclass
class _ExecuteModelState:
    logits: Tensor
    sampled_seq_ids: list[int]
    keepalive: list[Tensor]
    temperatures: list[float]
    top_ps: list[float]
    top_ks: list[int]
    seeds: list[int]
    has_seeds: list[int]


Buffer = Tensor | CpuGpuBuffer


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
        self._config.max_model_len = int(self.model.max_seq_len)
        self._config.end_token_id = int(self.model.end_token_id)
        self._runtime = None
        self.allocate_kv_cache()
        self.sampler = Sampler(self._device, self._runtime, config=self._config)

        self._runtime_api = RuntimeAPI(DeviceType.NVIDIA) if self._device == DeviceType.NVIDIA else None
        
        self._compute_streams: dict[int, object] = {}
        self._h2d_streams: dict[int, object] = {}
        self._d2h_streams: dict[int, object] = {}
        self._prepare_inputs_events: dict[int, object] = {}
        self._sampler_done_events: dict[int, object] = {}
        
        self._max_num_reqs = self._config.max_num_seqs
        self._max_num_tokens = self._config.max_num_batched_tokens
        max_model_len = int(self._config.max_model_len)
        block_size = max(1, int(self._config.kv_cache_block_size))
        self._max_block_table_width = max(1, (max_model_len + block_size - 1) // block_size)
        self.input_batch = InputBatch(self._max_num_reqs, self._max_num_tokens, self._max_block_table_width)

        # Persistent metadata buffers (vLLM-like): allocated once at runner init.
        self._input_ids_buf: Buffer | None = None
        self._pos_ids_buf: Buffer | None = None
        self._seq_ids_buf: Buffer | None = None
        self._req_sched_buf: Buffer | None = None
        self._req_comp_buf: Buffer | None = None
        self._query_start_loc_buf: Buffer | None = None
        self._seq_lens_buf: Buffer | None = None
        self._slot_mapping_buf: Buffer | None = None
        self._block_tables_buf: Buffer | None = None
        self._output_ids_buf: Buffer | None = None

        # Persistent sampled-token host buffer (one token per request).
        self._sampled_ids_buf: Buffer = self._make_buffer((self._max_num_reqs,), DataType.I64, pin_memory=True)

        # Token-level buffers sized by max_num_batched_tokens.
        token_shape = (self._max_num_tokens,)
        self._input_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._pos_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._seq_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._output_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)

        # BLOCK descriptor buffers sized by request capacities.
        if int(self._config.kv_cache_layout) == int(KvCacheLayout.BLOCK):
            self._req_sched_buf = self._make_buffer((self._max_num_reqs,), DataType.I32, pin_memory=True)
            self._req_comp_buf = self._make_buffer((self._max_num_reqs,), DataType.I32, pin_memory=True)
            self._query_start_loc_buf = self._make_buffer((self._max_num_reqs + 1,), DataType.I32, pin_memory=True)
            self._seq_lens_buf = self._make_buffer((self._max_num_reqs,), DataType.I32, pin_memory=True)
            self._slot_mapping_buf = self._make_buffer((self._max_num_tokens,), DataType.I32, pin_memory=True)
            self._block_tables_buf = self._make_buffer(
                (self._max_num_reqs * self._max_block_table_width,),
                DataType.I32,
                pin_memory=True,
            )
        self._logits_holder: Tensor = Tensor((1,), DataType.F32, self._device, 0)
        self._execute_model_state: _ExecuteModelState | None = None
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
            if (
                self._config.kv_cache_layout == KvCacheLayout.BLOCK
                and kv_capacity is not None
            ):
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
            for dev_id, stream in list(self._h2d_streams.items()):
                try:
                    self._runtime_api.set_device(int(dev_id))
                    self._runtime_api.stream_synchronize(stream)
                except Exception:
                    pass
            for dev_id, stream in list(self._d2h_streams.items()):
                try:
                    self._runtime_api.set_device(int(dev_id))
                    self._runtime_api.stream_synchronize(stream)
                except Exception:
                    pass
            for dev_id, evt in list(self._prepare_inputs_events.items()):
                try:
                    self._runtime_api.set_device(int(dev_id))
                    self._runtime_api.destroy_event(evt)
                except Exception:
                    pass
            for dev_id, evt in list(self._sampler_done_events.items()):
                try:
                    self._runtime_api.set_device(int(dev_id))
                    self._runtime_api.destroy_event(evt)
                except Exception:
                    pass
            for dev_id, stream in list(self._h2d_streams.items()):
                try:
                    self._runtime_api.set_device(int(dev_id))
                    self._runtime_api.destroy_stream(stream)
                except Exception:
                    pass
            for dev_id, stream in list(self._d2h_streams.items()):
                try:
                    self._runtime_api.set_device(int(dev_id))
                    self._runtime_api.destroy_stream(stream)
                except Exception:
                    pass
        self._prepare_inputs_events.clear()
        self._sampler_done_events.clear()
        self._h2d_streams.clear()
        self._d2h_streams.clear()
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
        block_layout = int(self._config.kv_cache_layout) == int(KvCacheLayout.BLOCK)
        self._update_states(scheduler_outputs)
        self._prepare_inputs(is_block_layout=block_layout)
        self._execute_model_state = None
        if int(self.input_batch.n_tokens) <= 0:
            return None

        output_rows, logits_tensor, keepalive = self._forward_step()
        sampled_seq_ids = list(self.input_batch.seq_ids)
        if len(sampled_seq_ids) != len(output_rows):
            raise RuntimeError("sampled seq_ids count mismatch with output rows")
        temperatures: list[float] = []
        top_ps: list[float] = []
        top_ks: list[int] = []
        seeds: list[int] = []
        has_seeds: list[int] = []
        for sid in sampled_seq_ids:
            seq_obj = self.input_batch.requests.get(int(sid))
            if seq_obj is None:
                raise RuntimeError(f"missing sequence for seq_id={sid}")
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
        self._execute_model_state = _ExecuteModelState(
            logits=logits_tensor,
            sampled_seq_ids=sampled_seq_ids,
            keepalive=keepalive,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            seeds=seeds,
            has_seeds=has_seeds,
        )
        return None

    def sample_tokens(self, grammar_output=None):
        del grammar_output
        state = self._execute_model_state
        self._execute_model_state = None
        if state is None:
            return None
        if len(state.sampled_seq_ids) == 0:
            return []
        n_outputs = len(state.sampled_seq_ids)
        if not isinstance(self._sampled_ids_buf, CpuGpuBuffer):
            raise RuntimeError("GPUModelRunner sampled buffer must be CpuGpuBuffer")
        sampled_ids_dev = self._sampled_ids_buf.gpu.slice(0, 0, n_outputs)
        sampled = self.sampler.sample_tokens(
            logits_tensor=state.logits,
            out_ids_dev=sampled_ids_dev,
            temperatures=state.temperatures,
            top_ps=state.top_ps,
            top_ks=state.top_ks,
            seeds=state.seeds,
            has_seeds=state.has_seeds,
        )
        if sampled is None:
            raise RuntimeError("sampler returned None for non-empty logits")
        if int(sampled.shape()[0]) != len(state.sampled_seq_ids):
            raise RuntimeError("sampler output size mismatch with sampled sequence ids")
        n_outputs = int(sampled.shape()[0])
        if n_outputs > self._max_num_reqs:
            raise RuntimeError("sampled outputs exceed configured max_num_reqs")
        sampled_host = self._sampled_ids_buf.cpu.slice(0, 0, n_outputs)
        # Sampler runs on compute stream. Build explicit compute->d2h dependency so
        # D2H can run on a dedicated copy stream without host-side synchronize.
        dev_id = int(sampled.device_id())
        compute_stream = self._get_runtime_compute_stream(dev_id)
        sampler_done_event = self._get_sampler_done_event(dev_id)
        self._runtime_api.set_device(dev_id)
        self._runtime_api.event_record(sampler_done_event, compute_stream)
        d2h_stream = self._get_d2h_stream(dev_id)
        self._runtime_api.stream_wait_event(d2h_stream, sampler_done_event)
        sampled_host.copy_(sampled, non_blocking=True, stream=d2h_stream)
        sync_fn = getattr(self._runtime_api, "stream_synchronize", None)
        if callable(sync_fn):
            sync_fn(d2h_stream)
        token_ids = [int(token_id) for token_id in sampled_host.tolist()]
        return token_ids

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

    def _get_h2d_stream(self, device_id: int):
        stream = self._h2d_streams.get(int(device_id))
        if stream is not None:
            return stream
        if self._runtime_api is None:
            raise RuntimeError("runtime api unavailable for H2D stream")
        self._runtime_api.set_device(int(device_id))
        stream = self._runtime_api.create_stream()
        self._h2d_streams[int(device_id)] = stream
        return stream

    def _get_d2h_stream(self, device_id: int):
        stream = self._d2h_streams.get(int(device_id))
        if stream is not None:
            return stream
        if self._runtime_api is None:
            raise RuntimeError("runtime api unavailable for D2H stream")
        self._runtime_api.set_device(int(device_id))
        stream = self._runtime_api.create_stream()
        self._d2h_streams[int(device_id)] = stream
        return stream

    def _get_prepare_inputs_event(self, device_id: int):
        event = self._prepare_inputs_events.get(int(device_id))
        if event is not None:
            return event
        if self._runtime_api is None:
            raise RuntimeError("runtime api unavailable for prepare_inputs_event")
        self._runtime_api.set_device(int(device_id))
        event = self._runtime_api.create_event()
        self._prepare_inputs_events[int(device_id)] = event
        return event

    def _get_sampler_done_event(self, device_id: int):
        event = self._sampler_done_events.get(int(device_id))
        if event is not None:
            return event
        if self._runtime_api is None:
            raise RuntimeError("runtime api unavailable for sampler_done_event")
        self._runtime_api.set_device(int(device_id))
        event = self._runtime_api.create_event()
        self._sampler_done_events[int(device_id)] = event
        return event

    def _record_prepare_inputs_event(self, stream, device_id: int) -> None:
        if self._runtime_api is None:
            return
        event = self._get_prepare_inputs_event(int(device_id))
        self._runtime_api.set_device(int(device_id))
        self._runtime_api.event_record(event, stream)

    def _wait_compute_for_prepare_inputs(self, stream, device_id: int) -> None:
        if self._runtime_api is None:
            return
        event = self._get_prepare_inputs_event(int(device_id))
        self._runtime_api.set_device(int(device_id))
        self._runtime_api.stream_wait_event(stream, event)

    def _forward_step(self) -> tuple[list[int], Tensor, list[Tensor]]:
        forward_fn = getattr(self.model, "forward", None)
        if not callable(forward_fn):
            raise RuntimeError("model wrapper must implement forward(ModelForwardInput, ModelForwardOutput)")
        if self._runtime is None:
            raise RuntimeError("runtime handle is required")

        fin, fout, n_outputs, output_rows, keepalive = self._build_forward_io()
        if self._device == DeviceType.NVIDIA:
            dev_id = int(self._logits_holder.device_id())
            compute_stream = self._get_runtime_compute_stream(dev_id)
            self._wait_compute_for_prepare_inputs(compute_stream, dev_id)
        status = int(forward_fn(self._runtime, fin, fout))
        if status != 0:
            raise RuntimeError(f"modelForward failed with status={status}")
        logits_shape = self._logits_holder.shape()
        if len(logits_shape) != 2:
            raise RuntimeError("modelForward returned invalid logits shape")
        if int(logits_shape[0]) != int(n_outputs):
            raise RuntimeError("modelForward logits row count mismatch with logits_indices")
        return output_rows, self._logits_holder, keepalive

    def _build_model_forward_input(
        self,
        ntoken: int,
    ) -> tuple[ModelForwardInput, int, list[int], list[Tensor]]:
        is_block_layout = int(self._config.kv_cache_layout) == int(KvCacheLayout.BLOCK)
        n_batch_seq = len(self.input_batch.scheduled_seqs) if is_block_layout else 0
        block_table_width = int(self.input_batch.block_table_width) if is_block_layout else 0
        n_block_elems = (
            n_batch_seq * max(0, block_table_width)
            if is_block_layout
            else 0
        )
        if ntoken > self._max_num_tokens:
            raise RuntimeError("ntoken exceeds configured max_num_batched_tokens")
        if n_batch_seq > self._max_num_reqs:
            raise RuntimeError("n_batch_seq exceeds configured max_num_seqs")
        if n_block_elems > (self._max_num_reqs * self._max_block_table_width):
            raise RuntimeError("n_block_elems exceeds configured BLOCK metadata capacity")
        assert self._input_ids_buf is not None
        assert self._pos_ids_buf is not None
        assert self._output_ids_buf is not None
        if (
            not isinstance(self._input_ids_buf, CpuGpuBuffer)
            or not isinstance(self._pos_ids_buf, CpuGpuBuffer)
            or not isinstance(self._output_ids_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("GPUModelRunner input/pos/output buffers must be CpuGpuBuffer")
        input_ids_host = self._input_ids_buf.cpu.slice(0, 0, ntoken)
        input_ids = self._input_ids_buf.gpu.slice(0, 0, ntoken)
        pos_ids_host = self._pos_ids_buf.cpu.slice(0, 0, ntoken)
        pos_ids = self._pos_ids_buf.gpu.slice(0, 0, ntoken)
        dev_id = int(input_ids.device_id())
        h2d_stream = self._get_h2d_stream(dev_id)

        keepalive: list[Tensor] = [
            input_ids,
            input_ids_host,
            pos_ids,
            pos_ids_host,
        ]
        seq_ids: Tensor | None = None
        seq_ids_host: Tensor | None = None

        if not is_block_layout:
            assert self._seq_ids_buf is not None
            if not isinstance(self._seq_ids_buf, CpuGpuBuffer):
                raise RuntimeError("GPUModelRunner non-block metadata buffers must be CpuGpuBuffer")
            seq_ids_host = self._seq_ids_buf.cpu.slice(0, 0, ntoken)
            seq_ids = self._seq_ids_buf.gpu.slice(0, 0, ntoken)
            keepalive.extend([
                seq_ids,
                seq_ids_host,
            ])

        if int(self.input_batch.n_tokens) != ntoken:
            raise RuntimeError("input_batch token count mismatch")
        output_rows = list(self.input_batch.output_rows)
        out_idx = int(self.input_batch.n_outputs)
        if out_idx != len(output_rows):
            raise RuntimeError("input_batch output row count mismatch")
        input_ids_host.load(cast(self.input_batch.token_ids_cpu[:ntoken].ctypes.data, c_void_p))
        pos_ids_host.load(cast(self.input_batch.pos_ids_cpu[:ntoken].ctypes.data, c_void_p))
        if seq_ids_host is not None:
            seq_ids_host.load(cast(self.input_batch.seq_ids_cpu[:ntoken].ctypes.data, c_void_p))

        input_ids.copy_(input_ids_host, non_blocking=True, stream=h2d_stream)
        pos_ids.copy_(pos_ids_host, non_blocking=True, stream=h2d_stream)
        if seq_ids is not None and seq_ids_host is not None:
            seq_ids.copy_(seq_ids_host, non_blocking=True, stream=h2d_stream)

        logits_indices_host = self._output_ids_buf.cpu.slice(0, 0, out_idx)
        logits_indices = self._output_ids_buf.gpu.slice(0, 0, out_idx)
        if out_idx > 0:
            logits_indices_host.load(cast(self.input_batch.logits_indices_cpu[:out_idx].ctypes.data, c_void_p))
            logits_indices.copy_(logits_indices_host, non_blocking=True, stream=h2d_stream)
        keepalive.extend([
            logits_indices,
            logits_indices_host,
        ])

        if not is_block_layout:
            assert seq_ids is not None and seq_ids_host is not None
        else:
            seq_ids = None

        attn, attn_keepalive = self._build_attention_metadata(
            ntoken=ntoken,
            seq_ids=seq_ids,
            pos_ids_host=pos_ids_host,
            is_block_layout=is_block_layout,
            h2d_stream=h2d_stream,
        )
        keepalive.extend(attn_keepalive)
        self._record_prepare_inputs_event(h2d_stream, dev_id)

        fin = ModelForwardInput()
        fin.input_ids = input_ids.lib_tensor()
        fin.pos_ids = pos_ids.lib_tensor()
        fin.logits_indices = logits_indices.lib_tensor()
        fin.attention = attn
        return fin, len(output_rows), output_rows, keepalive

    def _build_forward_io(self) -> tuple[ModelForwardInput, ModelForwardOutput, int, list[int], list[Tensor]]:
        ntoken = int(self.input_batch.n_tokens)
        if ntoken <= 0:
            raise RuntimeError("empty step plan")

        fin, n_outputs, output_rows, in_keepalive = self._build_model_forward_input(
            ntoken=ntoken,
        )
        fout = ModelForwardOutput()
        fout.logits = self._logits_holder.lib_tensor()
        keepalive = in_keepalive
        return fin, fout, n_outputs, output_rows, keepalive

    def _build_attention_metadata(
        self,
        ntoken: int,
        seq_ids: Tensor | None,
        pos_ids_host: Tensor | None,
        is_block_layout: bool,
        h2d_stream=None,
    ) -> tuple[AttentionMetadata, list[Tensor]]:
        attn = AttentionMetadata()
        attn.mode = c_int32(int(KvCacheLayout.BLOCK if is_block_layout else KvCacheLayout.SLOT))
        attn.seq_ids = seq_ids.lib_tensor() if seq_ids is not None else None
        attn.pos_ids_host = pos_ids_host.lib_tensor() if pos_ids_host is not None else None
        attn.req_num_scheduled_tokens = None
        attn.req_num_computed_tokens = None
        attn.query_start_loc = None
        attn.seq_lens = None
        attn.slot_mapping = None
        attn.block_tables = None
        attn.block_table_width = c_int32(0)

        if not is_block_layout:
            if seq_ids is None or pos_ids_host is None:
                raise RuntimeError("slot path requires seq_ids and pos_ids_host")
            return attn, [seq_ids, pos_ids_host]

        req_num_scheduled_tokens = list(self.input_batch.req_num_scheduled_tokens_step)
        req_num_computed_tokens = list(self.input_batch.req_num_computed_tokens_step)
        block_table_width = int(self.input_batch.block_table_width)
        n_batch_seq = len(req_num_scheduled_tokens)
        if n_batch_seq == 0 or len(req_num_computed_tokens) != n_batch_seq:
            raise ValueError("BLOCK metadata length mismatch")
        if block_table_width <= 0:
            raise ValueError("block_table_width must be > 0")
        if len(self.input_batch.scheduled_seqs) != n_batch_seq:
            raise ValueError("scheduled_seqs length mismatch with req_num_scheduled_tokens")
        if sum(int(x) for x in req_num_scheduled_tokens) != ntoken:
            raise ValueError("sum(req_num_scheduled_tokens) must equal ntoken")

        if h2d_stream is None:
            raise RuntimeError("BLOCK metadata build requires h2d_stream")
        n_block_elems = n_batch_seq * block_table_width
        if (
            not isinstance(self._req_sched_buf, CpuGpuBuffer)
            or not isinstance(self._req_comp_buf, CpuGpuBuffer)
            or not isinstance(self._query_start_loc_buf, CpuGpuBuffer)
            or not isinstance(self._seq_lens_buf, CpuGpuBuffer)
            or not isinstance(self._slot_mapping_buf, CpuGpuBuffer)
            or not isinstance(self._block_tables_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("BLOCK metadata buffers must be CpuGpuBuffer")
        req_sched_host = self._req_sched_buf.cpu.slice(0, 0, n_batch_seq)
        req_comp_host = self._req_comp_buf.cpu.slice(0, 0, n_batch_seq)
        query_start_loc = self._query_start_loc_buf.gpu.slice(0, 0, n_batch_seq + 1)
        seq_lens = self._seq_lens_buf.gpu.slice(0, 0, n_batch_seq)
        slot_mapping = self._slot_mapping_buf.gpu.slice(0, 0, ntoken)
        block_tables_host = self._block_tables_buf.cpu.slice(0, 0, n_block_elems)
        req_sched = self._req_sched_buf.gpu.slice(0, 0, n_batch_seq)
        req_comp = self._req_comp_buf.gpu.slice(0, 0, n_batch_seq)
        block_tables = self._block_tables_buf.gpu.slice(0, 0, n_block_elems)
        req_sched_host.copy_from_sequence(req_num_scheduled_tokens)
        req_comp_host.copy_from_sequence(req_num_computed_tokens)
        req_indices = np.asarray(
            [
                int(self.input_batch.seq_id_to_index[int(seq_obj.seq_id)])
                for seq_obj in self.input_batch.scheduled_seqs
            ],
            dtype=np.int32,
        )
        block_table_rows = self.input_batch.block_table_cpu[req_indices, :block_table_width]
        block_tables_host.load(cast(block_table_rows.reshape(-1).ctypes.data, c_void_p))
        block_tables.copy_(block_tables_host, non_blocking=True, stream=h2d_stream)
        req_sched.copy_(req_sched_host, non_blocking=True, stream=h2d_stream)
        req_comp.copy_(req_comp_host, non_blocking=True, stream=h2d_stream)
        dev_id = int(req_sched.device_id())
        compute_stream = self._get_runtime_compute_stream(dev_id)
        self._runtime_api.set_device(dev_id)
        evt = self._get_prepare_inputs_event(dev_id)
        self._runtime_api.event_record(evt, h2d_stream)
        self._runtime_api.stream_wait_event(compute_stream, evt)
        rc = int(
            LIB_LLAISYS.llaisysRuntimeBuildBlockAttentionMetadata(
                self._runtime,
                req_sched.lib_tensor(),
                req_comp.lib_tensor(),
                block_tables.lib_tensor(),
                c_int32(block_table_width),
                c_int32(ntoken),
                query_start_loc.lib_tensor(),
                seq_lens.lib_tensor(),
                slot_mapping.lib_tensor(),
            )
        )
        if rc != 0:
            raise RuntimeError(f"RuntimeBuildBlockAttentionMetadata failed with status={rc}")

        attn.req_num_scheduled_tokens = req_sched.lib_tensor()
        attn.req_num_computed_tokens = req_comp.lib_tensor()
        attn.query_start_loc = query_start_loc.lib_tensor()
        attn.seq_lens = seq_lens.lib_tensor()
        attn.slot_mapping = slot_mapping.lib_tensor()
        attn.block_tables = block_tables.lib_tensor()
        attn.block_table_width = c_int32(block_table_width)
        return attn, [
            req_sched_host,
            req_comp_host,
            block_tables_host,
            req_sched,
            req_comp,
            query_start_loc,
            seq_lens,
            slot_mapping,
            block_tables,
        ]

    def _update_states(self, scheduler_outputs: SchedulerOutputs) -> None:
        self.input_batch.update_states(scheduler_outputs)

    def _prepare_inputs(self, *, is_block_layout: bool) -> None:
        self.input_batch.prepare_inputs(is_block_layout=is_block_layout)

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
