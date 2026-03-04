from __future__ import annotations

from ctypes import c_int32

from .buffers import CpuGpuBuffer
from .config import EngineConfig
from .gpu_model_runner import GPUModelRunner
from .types import BatchPlan
from ..libllaisys import DataType, DeviceType
from ..libllaisys.model import AttentionMetadata, KvCacheLayout, ModelForwardInput
from ..tensor import Tensor


class CPUModelRunner(GPUModelRunner):
    """CPU specialization that reuses GPUModelRunner control flow."""

    def __init__(
        self,
        model,
        device: DeviceType = DeviceType.CPU,
        kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
        max_num_seqs: int | None = None,
        config: EngineConfig | None = None,
        runtime_handle=None,
    ):
        if config is None:
            config = EngineConfig(
                device=DeviceType.CPU,
                kv_cache_layout=kv_cache_layout,
                max_num_seqs=(max(1, int(max_num_seqs)) if max_num_seqs is not None else 8),
            )
        else:
            config.device = DeviceType.CPU
        super().__init__(
            model=model,
            device=DeviceType.CPU,
            kv_cache_layout=kv_cache_layout,
            max_num_seqs=max_num_seqs,
            config=config,
            runtime_handle=runtime_handle,
        )

    def _make_buffer(self, shape: tuple[int, ...], dtype: DataType, pin_memory: bool = True) -> Tensor:
        return Tensor(shape, dtype, DeviceType.CPU, 0, pin_memory=pin_memory)

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
        if isinstance(self._sampled_ids_buf, CpuGpuBuffer):
            raise RuntimeError("CPUModelRunner sampled buffer must be CPU Tensor")
        sampled_ids_dev = self._sampled_ids_buf.slice(0, 0, n_outputs)
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
        return sampled, list(state.sampled_req_ids)

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
            isinstance(self._input_ids_buf, CpuGpuBuffer)
            or isinstance(self._pos_ids_buf, CpuGpuBuffer)
            or isinstance(self._seq_ids_buf, CpuGpuBuffer)
            or isinstance(self._output_ids_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("CPUModelRunner metadata buffers must be CPU Tensor")

        input_ids = self._input_ids_buf.slice(0, 0, ntoken)
        pos_ids = self._pos_ids_buf.slice(0, 0, ntoken)
        seq_ids = self._seq_ids_buf.slice(0, 0, ntoken)
        logits_indices = self._output_ids_buf.slice(0, 0, len(output_rows))
        input_ids.copy_from_sequence(plan.token_ids)
        pos_ids.copy_from_sequence(pos)
        seq_ids.copy_from_sequence(seq)
        if output_rows:
            logits_indices.copy_from_sequence(output_rows)

        keepalive: list[Tensor] = [
            input_ids,
            pos_ids,
            seq_ids,
            logits_indices,
        ]
        attn, attn_keepalive = self._build_attention_metadata(
            plan=plan,
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
        return fin, len(output_rows), keepalive

    def _build_attention_metadata(
        self,
        plan: BatchPlan,
        ntoken: int,
        seq_ids: Tensor,
        pos_ids_host: Tensor,
        is_block_layout: bool,
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
        if isinstance(self._block_meta_buf, CpuGpuBuffer):
            raise RuntimeError("CPUModelRunner block metadata buffer must be CPU Tensor")
        block_meta = self._block_meta_buf.slice(0, 0, n_block_meta_i32)

        packed_i32 = [
            *plan.q_seq_rows,
            *plan.q_pos,
            *plan.slot_mapping,
            *plan.context_lens,
            *plan.block_tables,
        ]
        block_meta.copy_from_sequence(packed_i32)

        off = 0
        q_seq_rows = block_meta.slice(0, off, off + ntoken)
        off += ntoken
        q_pos = block_meta.slice(0, off, off + ntoken)
        off += ntoken
        slot_mapping = block_meta.slice(0, off, off + ntoken)
        off += ntoken
        context_lens = block_meta.slice(0, off, off + n_batch_seq)
        off += n_batch_seq
        block_tables = block_meta.slice(0, off, off + n_block_elems)

        if isinstance(self._batch_seq_ids_buf, CpuGpuBuffer):
            raise RuntimeError("CPUModelRunner batch_seq_ids buffer must be CPU Tensor")
        batch_seq_ids = self._batch_seq_ids_buf.slice(0, 0, len(plan.batch_seq_ids))
        batch_seq_ids.copy_from_sequence(plan.batch_seq_ids)

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
            block_meta,
        ]
