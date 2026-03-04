from __future__ import annotations

from dataclasses import dataclass
from ctypes import byref, c_int32, c_int64

from .config import EngineConfig
from .buffers import CpuGpuBuffer
from .sampler import Sampler
from .scheduler import SchedulerOutputs
from .types import BatchPlan, SamplingParams
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
    plan: BatchPlan
    sampled_req_ids: list[str]
    keepalive: list[Tensor]


Buffer = Tensor | CpuGpuBuffer


class GPUModelRunner:
    """Owns model + sampler and executes one scheduler step."""

    def __init__(
        self,
        model,
        device: DeviceType,
        kv_cache_layout: KvCacheLayout,
        max_num_seqs: int | None = None,
        config: EngineConfig | None = None,
        runtime_handle=None,
    ):
        self.model = model
        cfg = (config or EngineConfig(
            device=device,
            kv_cache_layout=kv_cache_layout,
            max_num_seqs=(max(1, int(max_num_seqs)) if max_num_seqs is not None else 8),
        )).normalized()
        self._config = cfg
        self._device = cfg.device
        self._runtime = runtime_handle
        if self._runtime is None:
            raise ValueError("runtime_handle is required by ModelRunner")
        self.sampler = Sampler(self._device, runtime_handle, config=cfg)
        self._owns_runtime = True
        self._compute_streams: dict[int, object] = {}
        self._h2d_streams: dict[int, object] = {}
        self._d2h_streams: dict[int, object] = {}
        self._runtime_api = RuntimeAPI(DeviceType.NVIDIA) if self._device == DeviceType.NVIDIA else None
        self._prepare_inputs_events: dict[int, object] = {}
        self._sampler_done_events: dict[int, object] = {}
        self._kv_cache_layout = KvCacheLayout(int(cfg.kv_cache_layout))
        self._max_num_reqs = max(1, int(cfg.max_num_seqs))
        if cfg.max_num_batched_tokens is not None:
            max_tokens = int(cfg.max_num_batched_tokens)
        elif cfg.max_model_len is not None:
            max_tokens = int(cfg.max_model_len)
        else:
            max_tokens = int(self.max_seq_len)
        self._max_num_tokens = max(1, max_tokens)
        max_model_len = int(cfg.max_model_len) if cfg.max_model_len is not None else int(self.max_seq_len)
        block_size = max(1, int(cfg.kv_cache_block_size))
        self._max_block_table_width = max(1, (max_model_len + block_size - 1) // block_size)
        self._max_block_elems = self._max_num_reqs * self._max_block_table_width
        self._max_block_meta_i32 = (3 * self._max_num_tokens) + self._max_num_reqs + self._max_block_elems

        # Persistent metadata buffers (vLLM-like): allocated once at runner init.
        self._input_ids_buf: Buffer | None = None
        self._pos_ids_buf: Buffer | None = None
        self._seq_ids_buf: Buffer | None = None
        self._batch_seq_ids_buf: Buffer | None = None
        self._block_meta_buf: Buffer | None = None
        self._output_ids_buf: Buffer | None = None

        # Persistent sampled-token host buffer (one token per request).
        self._sampled_ids_buf: Buffer = self._make_buffer((self._max_num_reqs,), DataType.I64, pin_memory=True)

        # Token-level buffers sized by max_num_batched_tokens.
        token_shape = (self._max_num_tokens,)
        self._input_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._pos_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._seq_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)
        self._output_ids_buf = self._make_buffer(token_shape, DataType.I64, pin_memory=True)

        # Request-level buffers sized by max_num_seqs.
        self._batch_seq_ids_buf = self._make_buffer((self._max_num_reqs,), DataType.I64, pin_memory=True)

        # BLOCK metadata buffer sized by max token/request capacities.
        if int(self._kv_cache_layout) == int(KvCacheLayout.BLOCK):
            self._block_meta_buf = self._make_buffer((self._max_block_meta_i32,), DataType.I32, pin_memory=True)
        self._logits_holder: Tensor = Tensor((1,), DataType.F32, self._device, 0)
        self._execute_model_state: _ExecuteModelState | None = None

    @property
    def max_seq_len(self) -> int:
        if hasattr(self.model, "max_seq_len"):
            return int(self.model.max_seq_len)
        return int(self.model._meta_info.maxseq)

    @property
    def end_token_id(self) -> int:
        if hasattr(self.model, "end_token_id"):
            return int(self.model.end_token_id)
        return int(self.model._meta_info.end_token)

    @property
    def kv_cache_capacity_tokens(self) -> int | None:
        return None

    def close(self) -> None:
        self._execute_model_state = None
        if self._runtime_api is not None:
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
        if runtime is not None and self._owns_runtime:
            LIB_LLAISYS.llaisysRuntimeDestroy(runtime)

    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ):
        plan, token_idx_to_req_id = self.prepare_model_input(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )
        self._execute_model_state = None
        if not plan.token_ids:
            return None

        output_rows, logits_tensor, keepalive = self._forward_step(plan)
        sampled_req_ids: list[str] = []
        for out_idx in output_rows:
            rid = token_idx_to_req_id.get(int(out_idx))
            if rid is None:
                raise RuntimeError("executor output id cannot be mapped to request")
            sampled_req_ids.append(rid)
        self._execute_model_state = _ExecuteModelState(
            logits=logits_tensor,
            plan=plan,
            sampled_req_ids=sampled_req_ids,
            keepalive=keepalive,
        )
        return None

    def sample_tokens(self, grammar_output=None):
        del grammar_output
        state = self._execute_model_state
        self._execute_model_state = None
        if state is None:
            return None
        if len(state.sampled_req_ids) == 0:
            empty = Tensor((0,), DataType.I64, DeviceType.CPU, 0)
            return empty, []
        n_outputs = len(state.sampled_req_ids)
        if not isinstance(self._sampled_ids_buf, CpuGpuBuffer):
            raise RuntimeError("GPUModelRunner sampled buffer must be CpuGpuBuffer")
        sampled_ids_dev = self._sampled_ids_buf.gpu.slice(0, 0, n_outputs)
        sampled = self.sampler.sample_tokens(
            logits_tensor=state.logits,
            out_ids_dev=sampled_ids_dev,
            temperatures=state.plan.temperatures,
            top_ps=state.plan.top_ps,
            top_ks=state.plan.top_ks,
            seeds=state.plan.seeds,
            has_seeds=state.plan.has_seeds,
        )
        if sampled is None:
            raise RuntimeError("sampler returned None for non-empty logits")
        if int(sampled.shape()[0]) != len(state.sampled_req_ids):
            raise RuntimeError("sampler output size mismatch with sampled request ids")
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
        sampled = sampled_host
        return sampled, list(state.sampled_req_ids)

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

    def _forward_step(self, plan: BatchPlan) -> tuple[list[int], Tensor, list[Tensor]]:
        forward_fn = getattr(self.model, "forward", None)
        if not callable(forward_fn):
            raise RuntimeError("model wrapper must implement forward(ModelForwardInput, ModelForwardOutput)")
        if self._runtime is None:
            raise RuntimeError("runtime handle is required")

        fin, fout, n_outputs, output_rows, keepalive = self._build_forward_io(plan)
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

    def _normalize_step_inputs(self, plan: BatchPlan, ntoken: int) -> tuple[bool, list[int], list[int], list[int]]:
        is_block_layout = int(self._kv_cache_layout) == int(KvCacheLayout.BLOCK)
        pos = (
            list(plan.pos_ids)
            if plan.pos_ids is not None
            else ([0] * ntoken if not is_block_layout else None)
        )
        if pos is None or len(pos) != ntoken:
            raise ValueError("pos_ids length mismatch")
        seq = list(plan.seq_ids) if plan.seq_ids is not None else [0] * ntoken
        if len(seq) != ntoken:
            raise ValueError("seq_ids length mismatch")
        mask = list(plan.logits_mask) if plan.logits_mask is not None else [0] * ntoken
        if plan.logits_mask is None:
            mask[-1] = 1
        if len(mask) != ntoken:
            raise ValueError("logits_mask length mismatch")
        return is_block_layout, pos, seq, mask

    def _build_model_forward_input(
        self,
        plan: BatchPlan,
        ntoken: int,
        is_block_layout: bool,
        pos: list[int],
        seq: list[int],
        output_rows: list[int],
    ) -> tuple[ModelForwardInput, int, list[Tensor]]:
        n_batch_seq = len(plan.batch_seq_ids) if plan.batch_seq_ids is not None else 0
        n_block_elems = len(plan.block_tables) if plan.block_tables is not None else 0
        if ntoken > self._max_num_tokens:
            raise RuntimeError("ntoken exceeds configured max_num_batched_tokens")
        if n_batch_seq > self._max_num_reqs:
            raise RuntimeError("n_batch_seq exceeds configured max_num_seqs")
        if n_block_elems > self._max_block_elems:
            raise RuntimeError("n_block_elems exceeds configured BLOCK metadata capacity")
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
            raise RuntimeError("GPUModelRunner metadata buffers must be CpuGpuBuffer")
        input_ids_host = self._input_ids_buf.cpu.slice(0, 0, ntoken)
        input_ids = self._input_ids_buf.gpu.slice(0, 0, ntoken)
        pos_ids_host = self._pos_ids_buf.cpu.slice(0, 0, ntoken)
        pos_ids = self._pos_ids_buf.gpu.slice(0, 0, ntoken)
        seq_ids_host = self._seq_ids_buf.cpu.slice(0, 0, ntoken)
        seq_ids = self._seq_ids_buf.gpu.slice(0, 0, ntoken)
        logits_indices_host = self._output_ids_buf.cpu.slice(0, 0, len(output_rows))
        logits_indices = self._output_ids_buf.gpu.slice(0, 0, len(output_rows))
        dev_id = int(input_ids.device_id())
        h2d_stream = self._get_h2d_stream(dev_id)
        input_ids_host.copy_from_sequence(plan.token_ids)
        pos_ids_host.copy_from_sequence(pos)
        seq_ids_host.copy_from_sequence(seq)
        if output_rows:
            logits_indices_host.copy_from_sequence(output_rows)
        input_ids.copy_(input_ids_host, non_blocking=True, stream=h2d_stream)
        pos_ids.copy_(pos_ids_host, non_blocking=True, stream=h2d_stream)
        seq_ids.copy_(seq_ids_host, non_blocking=True, stream=h2d_stream)
        if output_rows:
            logits_indices.copy_(logits_indices_host, non_blocking=True, stream=h2d_stream)

        keepalive: list[Tensor] = [
            input_ids,
            pos_ids,
            seq_ids,
            logits_indices,
            input_ids_host,
            pos_ids_host,
            seq_ids_host,
            logits_indices_host,
        ]
        attn, attn_keepalive = self._build_attention_metadata(
            plan=plan,
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
        return fin, len(output_rows), keepalive

    def _build_forward_io(
        self,
        plan: BatchPlan,
    ) -> tuple[ModelForwardInput, ModelForwardOutput, int, list[int], list[Tensor]]:
        ntoken = len(plan.token_ids)
        if ntoken <= 0:
            raise RuntimeError("empty step plan")

        is_block_layout, pos, seq, mask = self._normalize_step_inputs(plan, ntoken)
        output_rows = [i for i, m in enumerate(mask) if int(m) != 0]
        fin, n_outputs, in_keepalive = self._build_model_forward_input(
            plan=plan,
            ntoken=ntoken,
            is_block_layout=is_block_layout,
            pos=pos,
            seq=seq,
            output_rows=output_rows,
        )
        fout = ModelForwardOutput()
        fout.logits = self._logits_holder.lib_tensor()
        keepalive = in_keepalive
        return fin, fout, n_outputs, output_rows, keepalive

    def _build_attention_metadata(
        self,
        plan: BatchPlan,
        ntoken: int,
        seq_ids: Tensor,
        pos_ids_host: Tensor,
        is_block_layout: bool,
        h2d_stream=None,
    ) -> tuple[AttentionMetadata, list[Tensor]]:
        attn = AttentionMetadata()
        attn.mode = c_int32(int(KvCacheLayout.BLOCK if is_block_layout else KvCacheLayout.SLOT))
        attn.seq_ids = seq_ids.lib_tensor()
        attn.pos_ids_host = pos_ids_host.lib_tensor()
        attn.q_seq_rows = None
        attn.q_pos = None
        attn.slot_mapping = None
        attn.context_lens = None
        attn.batch_seq_ids = None
        attn.block_tables = None
        attn.block_table_width = c_int32(0)

        if not is_block_layout:
            return attn, [seq_ids, pos_ids_host]

        if (
            plan.q_seq_rows is None
            or plan.q_pos is None
            or plan.slot_mapping is None
            or plan.context_lens is None
            or plan.batch_seq_ids is None
            or plan.block_tables is None
        ):
            raise ValueError(
                "incomplete BLOCK metadata: q_seq_rows/q_pos/slot_mapping/context_lens/batch_seq_ids/block_tables must all be set"
            )

        block_table_width = int(plan.block_table_width)
        if (
            len(plan.q_seq_rows) != ntoken
            or len(plan.q_pos) != ntoken
            or len(plan.slot_mapping) != ntoken
            or len(plan.context_lens) != len(plan.batch_seq_ids)
        ):
            raise ValueError("BLOCK metadata length mismatch")
        if block_table_width <= 0 or len(plan.block_tables) != len(plan.batch_seq_ids) * block_table_width:
            raise ValueError("block_tables length mismatch")

        assert self._batch_seq_ids_buf is not None
        n_batch_seq = len(plan.context_lens)
        n_block_elems = len(plan.block_tables)
        n_block_meta_i32 = (3 * ntoken) + n_batch_seq + n_block_elems
        if n_block_meta_i32 > self._max_block_meta_i32:
            raise RuntimeError("BLOCK metadata exceeds configured max_block_meta_i32")
        assert self._block_meta_buf is not None
        if not isinstance(self._block_meta_buf, CpuGpuBuffer):
            raise RuntimeError("GPUModelRunner block metadata buffer must be CpuGpuBuffer")
        block_meta_host = self._block_meta_buf.cpu.slice(0, 0, n_block_meta_i32)
        block_meta_dev = self._block_meta_buf.gpu.slice(0, 0, n_block_meta_i32)
        packed_i32 = [
            *plan.q_seq_rows,
            *plan.q_pos,
            *plan.slot_mapping,
            *plan.context_lens,
            *plan.block_tables,
        ]
        block_meta_host.copy_from_sequence(packed_i32)
        block_meta_dev.copy_(block_meta_host, non_blocking=True, stream=h2d_stream)

        off = 0
        q_seq_rows = block_meta_dev.slice(0, off, off + ntoken)
        off += ntoken
        q_pos = block_meta_dev.slice(0, off, off + ntoken)
        off += ntoken
        slot_mapping = block_meta_dev.slice(0, off, off + ntoken)
        off += ntoken
        context_lens = block_meta_dev.slice(0, off, off + n_batch_seq)
        off += n_batch_seq
        block_tables = block_meta_dev.slice(0, off, off + n_block_elems)

        if not isinstance(self._batch_seq_ids_buf, CpuGpuBuffer):
            raise RuntimeError("GPUModelRunner batch_seq_ids buffer must be CpuGpuBuffer")
        batch_seq_ids_host = self._batch_seq_ids_buf.cpu.slice(0, 0, len(plan.batch_seq_ids))
        batch_seq_ids = self._batch_seq_ids_buf.gpu.slice(0, 0, len(plan.batch_seq_ids))
        batch_seq_ids_host.copy_from_sequence(plan.batch_seq_ids)
        batch_seq_ids.copy_(batch_seq_ids_host, non_blocking=True, stream=h2d_stream)

        attn.q_seq_rows = q_seq_rows.lib_tensor()
        attn.q_pos = q_pos.lib_tensor()
        attn.slot_mapping = slot_mapping.lib_tensor()
        attn.context_lens = context_lens.lib_tensor()
        attn.batch_seq_ids = batch_seq_ids.lib_tensor()
        attn.block_tables = block_tables.lib_tensor()
        attn.block_table_width = c_int32(block_table_width)
        return attn, [
            seq_ids,
            pos_ids_host,
            q_seq_rows,
            q_pos,
            slot_mapping,
            context_lens,
            batch_seq_ids,
            block_tables,
            block_meta_host,
            block_meta_dev,
            batch_seq_ids_host,
        ]

    def _collect_token_inputs(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None,
        sampling_params_by_req: dict[str, SamplingParams] | None,
    ) -> tuple[
        list[int],
        list[int],
        list[float],
        list[float],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        dict[int, str],
    ]:
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
        slot_mapping: list[int] = []
        token_idx_to_req_id: dict[int, str] = {}

        base = 0
        for row_idx, seq in enumerate(scheduler_outputs.scheduled_seqs):
            bs = max(1, int(seq.block_size))
            if scheduler_outputs.is_prefill:
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
            if sampling_params_by_req is None and sampling_params is None:
                raise RuntimeError("missing sampling params for request")
            params = sampling_params
            if params is None:
                params = sampling_params_by_req.get(rid) if sampling_params_by_req is not None else None

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
                        raise RuntimeError("executor block table out of range")
                    bid = int(seq.block_table[bidx])
                    slot_mapping.append(bid * bs + boff)

            for i, m in enumerate(seq_mask):
                if m != 0:
                    token_idx_to_req_id[base + i] = rid
            base += len(seq_tokens)

        return (
            token_ids,
            logits_mask,
            temperatures,
            top_ps,
            top_ks,
            seeds,
            has_seeds,
            seq_ids,
            pos_ids,
            q_seq_rows,
            slot_mapping,
            token_idx_to_req_id,
        )

    def _build_block_metadata_inputs(
        self,
        outputs: SchedulerOutputs,
        token_ids: list[int],
        slot_mapping: list[int],
    ) -> tuple[list[int] | None, list[int] | None, list[int] | None, int]:
        block_layout = int(self._kv_cache_layout) == int(KvCacheLayout.BLOCK)
        if not block_layout:
            return None, None, None, 0

        for seq in outputs.scheduled_seqs:
            if not seq.block_table:
                raise RuntimeError("BLOCK layout requires non-empty block_table for every scheduled sequence")
        if len(slot_mapping) != len(token_ids):
            raise RuntimeError("BLOCK layout requires slot_mapping for every token")

        batch_seq_ids = [int(s.seq_id) for s in outputs.scheduled_seqs]
        context_lens = [len(s) for s in outputs.scheduled_seqs]
        block_table_width = max(len(s.block_table) for s in outputs.scheduled_seqs)
        block_tables: list[int] = []
        for s in outputs.scheduled_seqs:
            row = [int(b) for b in s.block_table]
            if len(row) < block_table_width:
                row.extend([-1] * (block_table_width - len(row)))
            block_tables.extend(row)

        return (
            context_lens if context_lens else None,
            batch_seq_ids if batch_seq_ids else None,
            block_tables if block_tables else None,
            int(block_table_width),
        )

    def prepare_model_input(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ) -> tuple[BatchPlan, dict[int, str]]:
        (
            token_ids,
            logits_mask,
            temperatures,
            top_ps,
            top_ks,
            seeds,
            has_seeds,
            seq_ids,
            pos_ids,
            q_seq_rows,
            slot_mapping,
            token_idx_to_req_id,
        ) = self._collect_token_inputs(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )

        context_lens, batch_seq_ids, block_tables, block_table_width = self._build_block_metadata_inputs(
            scheduler_outputs,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
        )
        block_layout = int(self._kv_cache_layout) == int(KvCacheLayout.BLOCK)

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
                q_seq_rows=q_seq_rows if block_layout else None,
                q_pos=pos_ids if block_layout else None,
                slot_mapping=slot_mapping if block_layout else None,
                context_lens=context_lens,
                batch_seq_ids=batch_seq_ids,
                block_tables=block_tables,
                block_table_width=int(block_table_width),
            ),
            token_idx_to_req_id,
        )

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
