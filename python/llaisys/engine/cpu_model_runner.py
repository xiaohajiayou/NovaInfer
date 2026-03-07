from __future__ import annotations

from ctypes import c_int32

from .buffers import CpuGpuBuffer
from .config import EngineConfig
from .gpu_model_runner import GPUModelRunner
from .model_registry import ModelRegistry
from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.model import AttentionMetadata, KvCacheLayout, ModelForwardInput
from ..tensor import Tensor


class CPUModelRunner(GPUModelRunner):
    """CPU specialization that reuses GPUModelRunner control flow."""

    def __init__(
        self,
        model,
        config: EngineConfig | None = None,
        model_registry: ModelRegistry | None = None,
    ):
        if config is None:
            raise ValueError("config is required")
        config.device = DeviceType.CPU
        super().__init__(
            model=model,
            config=config,
            model_registry=model_registry,
        )

    def _make_buffer(self, shape: tuple[int, ...], dtype: DataType, pin_memory: bool = True) -> Tensor:
        return Tensor(shape, dtype, DeviceType.CPU, 0, pin_memory=pin_memory)

    def sample_tokens(self, grammar_output=None):
        del grammar_output
        state = self._execute_model_state
        self._execute_model_state = None
        if state is None:
            return None
        if len(state.sampled_seq_ids) == 0:
            return []
        n_outputs = len(state.sampled_seq_ids)
        if isinstance(self._sampled_ids_buf, CpuGpuBuffer):
            raise RuntimeError("CPUModelRunner sampled buffer must be CPU Tensor")
        sampled_ids_dev = self._sampled_ids_buf.slice(0, 0, n_outputs)
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
        return [int(token_id) for token_id in sampled.tolist()]

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
        assert self._seq_ids_buf is not None
        assert self._output_ids_buf is not None
        if (
            isinstance(self._input_ids_buf, CpuGpuBuffer)
            or isinstance(self._pos_ids_buf, CpuGpuBuffer)
            or isinstance(self._seq_ids_buf, CpuGpuBuffer)
            or isinstance(self._output_ids_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("CPUModelRunner metadata buffers must be CPU Tensor")

        input_ids = self._input_ids_buf.slice(0, 0, ntoken)
        pos_ids = self._pos_ids_buf.slice(0, 0, ntoken)
        seq_ids = self._seq_ids_buf.slice(0, 0, ntoken)
        output_rows = list(self.input_batch.output_rows)
        out_idx = int(self.input_batch.n_outputs)
        if out_idx != len(output_rows):
            raise RuntimeError("input_batch output row count mismatch")
        if int(self.input_batch.n_tokens) != ntoken:
            raise RuntimeError("input_batch token count mismatch")
        input_ids.copy_from_sequence(self.input_batch.token_ids_cpu[:ntoken])
        pos_ids.copy_from_sequence(self.input_batch.pos_ids_cpu[:ntoken])
        seq_ids.copy_from_sequence(self.input_batch.seq_ids_cpu[:ntoken])
        logits_indices = self._output_ids_buf.slice(0, 0, out_idx)
        if out_idx > 0:
            logits_indices.copy_from_sequence(self.input_batch.logits_indices_cpu[:out_idx])

        keepalive: list[Tensor] = [
            input_ids,
            pos_ids,
            seq_ids,
            logits_indices,
        ]
        attn, attn_keepalive = self._build_attention_metadata(
            ntoken=ntoken,
            seq_ids=seq_ids,
            pos_ids_host=pos_ids,
            is_block_layout=is_block_layout,
        )
        keepalive.extend(attn_keepalive)

        fin = ModelForwardInput()
        fin.input_ids = input_ids.lib_tensor()
        fin.pos_ids = pos_ids.lib_tensor()
        fin.logits_indices = logits_indices.lib_tensor()
        fin.attention = attn
        return fin, len(output_rows), output_rows, keepalive

    def _build_attention_metadata(
        self,
        ntoken: int,
        seq_ids: Tensor,
        pos_ids_host: Tensor,
        is_block_layout: bool,
    ) -> tuple[AttentionMetadata, list[Tensor]]:
        attn = AttentionMetadata()
        attn.mode = c_int32(int(KvCacheLayout.BLOCK if is_block_layout else KvCacheLayout.SLOT))
        attn.seq_ids = seq_ids.lib_tensor()
        attn.pos_ids_host = pos_ids_host.lib_tensor()
        attn.req_num_scheduled_tokens = None
        attn.req_num_computed_tokens = None
        attn.query_start_loc = None
        attn.seq_lens = None
        attn.slot_mapping = None
        attn.block_tables = None
        attn.block_table_width = c_int32(0)

        if not is_block_layout:
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

        n_block_elems = n_batch_seq * block_table_width
        assert self._req_sched_buf is not None
        assert self._req_comp_buf is not None
        assert self._query_start_loc_buf is not None
        assert self._seq_lens_buf is not None
        assert self._slot_mapping_buf is not None
        assert self._block_tables_buf is not None
        if (
            isinstance(self._req_sched_buf, CpuGpuBuffer)
            or isinstance(self._req_comp_buf, CpuGpuBuffer)
            or isinstance(self._query_start_loc_buf, CpuGpuBuffer)
            or isinstance(self._seq_lens_buf, CpuGpuBuffer)
            or isinstance(self._slot_mapping_buf, CpuGpuBuffer)
            or isinstance(self._block_tables_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("CPUModelRunner BLOCK metadata buffers must be CPU Tensor")
        req_sched = self._req_sched_buf.slice(0, 0, n_batch_seq)
        req_comp = self._req_comp_buf.slice(0, 0, n_batch_seq)
        query_start_loc = self._query_start_loc_buf.slice(0, 0, n_batch_seq + 1)
        seq_lens = self._seq_lens_buf.slice(0, 0, n_batch_seq)
        slot_mapping = self._slot_mapping_buf.slice(0, 0, ntoken)
        block_tables = self._block_tables_buf.slice(0, 0, n_block_elems)
        req_sched.copy_from_sequence(req_num_scheduled_tokens)
        req_comp.copy_from_sequence(req_num_computed_tokens)
        req_indices = [
            int(self.input_batch.seq_id_to_index[int(seq_obj.seq_id)])
            for seq_obj in self.input_batch.scheduled_seqs
        ]
        block_rows = self.input_batch.block_table_cpu[req_indices, :block_table_width]
        block_tables.copy_from_sequence(block_rows.reshape(-1))
        if self._runtime is None:
            raise RuntimeError("runtime handle is required")
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
            seq_ids,
            pos_ids_host,
            req_sched,
            req_comp,
            query_start_loc,
            seq_lens,
            slot_mapping,
            block_tables,
        ]
