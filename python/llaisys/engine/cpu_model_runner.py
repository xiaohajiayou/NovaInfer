from __future__ import annotations

from .buffers import CpuGpuBuffer
from .config import EngineConfig
from .gpu_model_runner import GPUModelRunner, PreparedTensors
from .model_registry import ModelRegistry
from ..libllaisys import DataType, DeviceType
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

    def _build_slot_tensors(
        self,
        *,
        seqs: list,
        input_ids: list[int],
        positions: list[int],
        scheduled_token_counts: list[int],
    ) -> PreparedTensors:
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
            isinstance(self._input_ids_buf, CpuGpuBuffer)
            or isinstance(self._pos_ids_buf, CpuGpuBuffer)
            or isinstance(self._seq_ids_buf, CpuGpuBuffer)
            or isinstance(self._output_ids_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("CPU slot builder requires CPU tensors")

        output_rows = self._build_output_rows(scheduled_token_counts, n_outputs)
        keepalive: list[Tensor] = []

        input_ids_t = self._input_ids_buf.slice(0, 0, ntoken)
        pos_ids_t = self._pos_ids_buf.slice(0, 0, ntoken)
        input_ids_t.copy_from_sequence(input_ids)
        pos_ids_t.copy_from_sequence(positions)
        keepalive.extend([input_ids_t, pos_ids_t])

        seq_token_ids: list[int] = []
        for seq_obj, n_sched in zip(seqs, scheduled_token_counts):
            seq_token_ids.extend([int(seq_obj.seq_id)] * int(n_sched))
        seq_ids_t = self._seq_ids_buf.slice(0, 0, ntoken)
        seq_ids_t.copy_from_sequence(seq_token_ids)
        keepalive.append(seq_ids_t)

        logits_indices_t = self._output_ids_buf.slice(0, 0, n_outputs)
        if n_outputs > 0:
            logits_indices_t.copy_from_sequence(output_rows)
        keepalive.append(logits_indices_t)

        return PreparedTensors(
            input_ids=input_ids_t,
            pos_ids=pos_ids_t,
            logits_indices=logits_indices_t,
            n_outputs=n_outputs,
            keepalive=keepalive,
            seq_ids=seq_ids_t,
            pos_ids_host=pos_ids_t,
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
        cudnn_seq_lens_q: list[int] | None = None,
        cudnn_seq_lens_kv: list[int] | None = None,
        cudnn_page_table: list[int] | None = None,
        cudnn_b_exec: int = 0,
        incremental_block_table_update: bool = False,
    ) -> PreparedTensors:
        del (
            attention_phase,
            cudnn_seq_lens_q,
            cudnn_seq_lens_kv,
            cudnn_page_table,
            cudnn_b_exec,
            incremental_block_table_update,
        )
        ntoken = len(input_ids)
        n_outputs = len(seqs)
        if ntoken > self._max_num_tokens:
            raise RuntimeError("ntoken exceeds configured max_num_batched_tokens")
        if n_outputs > self._max_num_reqs:
            raise RuntimeError("n_outputs exceeds configured max_num_seqs")
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

        assert self._input_ids_buf is not None
        assert self._pos_ids_buf is not None
        assert self._output_ids_buf is not None
        assert self._cu_seqlens_q_buf is not None
        assert self._cu_seqlens_k_buf is not None
        assert self._slot_mapping_buf is not None
        assert self._block_tables_buf is not None
        if (
            isinstance(self._input_ids_buf, CpuGpuBuffer)
            or isinstance(self._pos_ids_buf, CpuGpuBuffer)
            or isinstance(self._output_ids_buf, CpuGpuBuffer)
            or isinstance(self._cu_seqlens_q_buf, CpuGpuBuffer)
            or isinstance(self._cu_seqlens_k_buf, CpuGpuBuffer)
            or isinstance(self._slot_mapping_buf, CpuGpuBuffer)
            or isinstance(self._block_tables_buf, CpuGpuBuffer)
        ):
            raise RuntimeError("CPU block builder requires CPU tensors")

        output_rows = self._build_output_rows(scheduled_token_counts, n_outputs)
        keepalive: list[Tensor] = []

        input_ids_t = self._input_ids_buf.slice(0, 0, ntoken)
        pos_ids_t = self._pos_ids_buf.slice(0, 0, ntoken)
        input_ids_t.copy_from_sequence(input_ids)
        pos_ids_t.copy_from_sequence(positions)
        keepalive.extend([input_ids_t, pos_ids_t])

        logits_indices_t = self._output_ids_buf.slice(0, 0, n_outputs)
        if n_outputs > 0:
            logits_indices_t.copy_from_sequence(output_rows)
        keepalive.append(logits_indices_t)

        cu_seqlens_q_t = self._cu_seqlens_q_buf.slice(0, 0, len(cu_seqlens_q))
        cu_seqlens_k_t = self._cu_seqlens_k_buf.slice(0, 0, len(cu_seqlens_k))
        slot_mapping_t = self._slot_mapping_buf.slice(0, 0, ntoken)
        block_rows_t = self._block_tables_buf.slice(0, 0, n_block_elems)
        cu_seqlens_q_t.copy_from_sequence(cu_seqlens_q)
        cu_seqlens_k_t.copy_from_sequence(cu_seqlens_k)
        slot_mapping_t.copy_from_sequence(slot_mapping)
        block_rows_t.copy_from_sequence([int(v) for row in block_table_rows for v in row])
        keepalive.extend([cu_seqlens_q_t, cu_seqlens_k_t, slot_mapping_t, block_rows_t])

        return PreparedTensors(
            input_ids=input_ids_t,
            pos_ids=pos_ids_t,
            logits_indices=logits_indices_t,
            n_outputs=n_outputs,
            keepalive=keepalive,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_k=cu_seqlens_k_t,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            slot_mapping=slot_mapping_t,
            block_tables=block_rows_t,
            block_table_width=int(block_table_width),
        )

    def sample_tokens(self, grammar_output=None):
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
        if isinstance(self._sampled_ids_buf, CpuGpuBuffer):
            raise RuntimeError("CPUModelRunner sampled buffer must be CPU Tensor")
        sampled_ids_dev = self._sampled_ids_buf.slice(0, 0, int(n_outputs))
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
        return [int(token_id) for token_id in sampled.tolist()]
