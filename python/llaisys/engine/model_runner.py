from __future__ import annotations

from ctypes import byref, c_int32, c_int64

from .sampler import Sampler
from .scheduler import SchedulerOutputs
from .types import BatchPlan, SamplingParams
from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.model import (
    AttentionMetadata,
    KvCacheLayout,
    LlaisysKvStats,
    ModelForwardInput,
    ModelForwardOutput,
)
from ..tensor import Tensor


class ModelRunner:
    """Owns model + sampler and executes one scheduler step."""

    def __init__(self, model, device: DeviceType, kv_cache_layout: KvCacheLayout | None = None):
        self.model = model
        self.sampler = Sampler(device)
        self._device = device
        self._runtime = getattr(model, "runtime_handle", None)
        if kv_cache_layout is not None:
            self._kv_cache_layout = KvCacheLayout(int(kv_cache_layout))
        else:
            raw_layout = getattr(model, "kv_cache_layout", getattr(model, "_kv_cache_layout", KvCacheLayout.BLOCK))
            self._kv_cache_layout = KvCacheLayout(int(raw_layout))

        self._step_capacity_tokens = 0
        self._step_capacity_batch_seq = 0
        self._step_capacity_block_elems = 0
        self._input_ids_buf: Tensor | None = None
        self._pos_ids_buf: Tensor | None = None
        self._pos_ids_host_buf: Tensor | None = None
        self._seq_ids_buf: Tensor | None = None
        self._q_seq_rows_buf: Tensor | None = None
        self._q_pos_buf: Tensor | None = None
        self._logits_mask_buf: Tensor | None = None
        self._slot_mapping_buf: Tensor | None = None
        self._context_lens_buf: Tensor | None = None
        self._batch_seq_ids_buf: Tensor | None = None
        self._block_tables_buf: Tensor | None = None
        self._output_ids_buf: Tensor | None = None
        self._logits_holder: Tensor = Tensor((1,), DataType.F32, self._device, 0)

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
        if hasattr(self.model, "kv_cache_capacity_tokens"):
            try:
                return int(self.model.kv_cache_capacity_tokens)
            except Exception:
                return None
        return None

    def close(self) -> None:
        close_fn = getattr(self.model, "close", None)
        if callable(close_fn):
            close_fn()

    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ):
        plan, token_idx_to_req_id = self._prepare_inputs(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )
        if not plan.token_ids:
            return None, None, None, {}

        output_ids, logits_tensor = self._forward_step(plan)
        return output_ids, logits_tensor, plan, token_idx_to_req_id

    def sample_tokens(self, logits_tensor, plan: BatchPlan):
        return self.sampler.sample_tokens(
            logits_tensor=logits_tensor,
            temperatures=plan.temperatures,
            top_ps=plan.top_ps,
            top_ks=plan.top_ks,
            seeds=plan.seeds,
            has_seeds=plan.has_seeds,
        )

    def execute_step(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampling_params: SamplingParams | None = None,
        sampling_params_by_req: dict[str, SamplingParams] | None = None,
    ):
        output_ids, logits_tensor, plan, token_idx_to_req_id = self.execute_model(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )
        if output_ids is None or plan is None:
            return None, None, {}
        if int(output_ids.shape()[0]) == 0:
            return output_ids, output_ids, token_idx_to_req_id
        sampled = self.sample_tokens(logits_tensor, plan)
        return output_ids, sampled, token_idx_to_req_id

    @staticmethod
    def _has_complete_block_metadata(plan: BatchPlan) -> bool:
        has_any_block_meta = (
            plan.q_seq_rows is not None
            or plan.q_pos is not None
            or plan.slot_mapping is not None
            or plan.context_lens is not None
            or plan.batch_seq_ids is not None
            or plan.block_tables is not None
            or int(plan.block_table_width) > 0
        )
        if not has_any_block_meta:
            return False
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
        return True

    def _forward_step(self, plan: BatchPlan) -> tuple[Tensor, Tensor]:
        forward_fn = getattr(self.model, "forward", None)
        if not callable(forward_fn):
            raise RuntimeError("model wrapper must implement forward(ModelForwardInput, ModelForwardOutput)")

        fin, fout, output_ids, ntoken, _keepalive = self._build_forward_io(plan)
        status = int(forward_fn(fin, fout))
        if status != 0:
            raise RuntimeError(f"modelForward failed with status={status}")
        n_outputs = int(fout.n_outputs)
        if n_outputs < 0 or n_outputs > ntoken:
            raise RuntimeError("modelForward returned invalid n_outputs")
        return output_ids.slice(0, 0, n_outputs), self._logits_holder

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
        mask: list[int],
    ) -> tuple[ModelForwardInput, list[Tensor]]:
        n_batch_seq = len(plan.batch_seq_ids) if plan.batch_seq_ids is not None else 0
        n_block_elems = len(plan.block_tables) if plan.block_tables is not None else 0
        self._ensure_token_buffers(ntoken)
        self._ensure_batch_buffers(n_batch_seq, n_block_elems)
        assert self._input_ids_buf is not None and self._pos_ids_buf is not None
        assert self._pos_ids_host_buf is not None
        assert self._seq_ids_buf is not None and self._logits_mask_buf is not None

        input_ids = self._input_ids_buf.slice(0, 0, ntoken)
        pos_ids = self._pos_ids_buf.slice(0, 0, ntoken)
        pos_ids_host = self._pos_ids_host_buf.slice(0, 0, ntoken)
        seq_ids = self._seq_ids_buf.slice(0, 0, ntoken)
        logits_mask = self._logits_mask_buf.slice(0, 0, ntoken)
        input_ids.copy_from_sequence(plan.token_ids)
        pos_ids.copy_from_sequence(pos)
        pos_ids_host.copy_from_sequence(pos)
        seq_ids.copy_from_sequence(seq)
        logits_mask.copy_from_sequence(mask)

        keepalive: list[Tensor] = [input_ids, pos_ids, pos_ids_host, seq_ids, logits_mask]
        attn, attn_keepalive = self._build_attention_metadata(
            plan=plan,
            ntoken=ntoken,
            seq_ids=seq_ids,
            pos_ids_host=pos_ids_host,
            is_block_layout=is_block_layout,
        )
        keepalive.extend(attn_keepalive)

        fin = ModelForwardInput()
        fin.input_ids = input_ids.lib_tensor()
        fin.pos_ids = pos_ids.lib_tensor()
        fin.logits_mask = logits_mask.lib_tensor()
        fin.attention = attn
        return fin, keepalive

    def _build_model_forward_output(self, ntoken: int) -> tuple[ModelForwardOutput, Tensor, list[Tensor]]:
        assert self._output_ids_buf is not None
        output_ids = self._output_ids_buf.slice(0, 0, ntoken)
        keepalive: list[Tensor] = [output_ids]

        fout = ModelForwardOutput()
        fout.output_ids = output_ids.lib_tensor()
        fout.logits = self._logits_holder.lib_tensor()
        fout.n_outputs = c_int32(0)
        return fout, output_ids, keepalive

    def _build_forward_io(
        self,
        plan: BatchPlan,
    ) -> tuple[ModelForwardInput, ModelForwardOutput, Tensor, int, list[Tensor]]:
        ntoken = len(plan.token_ids)
        if ntoken <= 0:
            raise RuntimeError("empty step plan")

        is_block_layout, pos, seq, mask = self._normalize_step_inputs(plan, ntoken)
        fin, in_keepalive = self._build_model_forward_input(
            plan=plan,
            ntoken=ntoken,
            is_block_layout=is_block_layout,
            pos=pos,
            seq=seq,
            mask=mask,
        )
        fout, output_ids, out_keepalive = self._build_model_forward_output(ntoken)
        keepalive = in_keepalive + out_keepalive
        return fin, fout, output_ids, ntoken, keepalive

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

        self._has_complete_block_metadata(plan)
        assert plan.q_seq_rows is not None
        assert plan.q_pos is not None
        assert plan.slot_mapping is not None
        assert plan.context_lens is not None
        assert plan.batch_seq_ids is not None
        assert plan.block_tables is not None

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

        assert self._q_seq_rows_buf is not None
        assert self._q_pos_buf is not None
        assert self._slot_mapping_buf is not None
        assert self._context_lens_buf is not None
        assert self._batch_seq_ids_buf is not None
        assert self._block_tables_buf is not None
        q_seq_rows = self._q_seq_rows_buf.slice(0, 0, ntoken)
        q_pos = self._q_pos_buf.slice(0, 0, ntoken)
        slot_mapping = self._slot_mapping_buf.slice(0, 0, ntoken)
        context_lens = self._context_lens_buf.slice(0, 0, len(plan.context_lens))
        batch_seq_ids = self._batch_seq_ids_buf.slice(0, 0, len(plan.batch_seq_ids))
        block_tables = self._block_tables_buf.slice(0, 0, len(plan.block_tables))
        q_seq_rows.copy_from_sequence(plan.q_seq_rows)
        q_pos.copy_from_sequence(plan.q_pos)
        slot_mapping.copy_from_sequence(plan.slot_mapping)
        context_lens.copy_from_sequence(plan.context_lens)
        batch_seq_ids.copy_from_sequence(plan.batch_seq_ids)
        block_tables.copy_from_sequence(plan.block_tables)

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
        ]

    def _ensure_token_buffers(self, n_tokens: int) -> None:
        if n_tokens <= self._step_capacity_tokens:
            return
        self._input_ids_buf = Tensor((n_tokens,), DataType.I64, self._device, 0)
        self._pos_ids_buf = Tensor((n_tokens,), DataType.I64, self._device, 0)
        self._pos_ids_host_buf = Tensor((n_tokens,), DataType.I64, DeviceType.CPU, 0)
        self._seq_ids_buf = Tensor((n_tokens,), DataType.I64, DeviceType.CPU, 0)
        self._q_seq_rows_buf = Tensor((n_tokens,), DataType.I32, self._device, 0)
        self._q_pos_buf = Tensor((n_tokens,), DataType.I32, self._device, 0)
        self._logits_mask_buf = Tensor((n_tokens,), DataType.I8, DeviceType.CPU, 0)
        self._slot_mapping_buf = Tensor((n_tokens,), DataType.I32, self._device, 0)
        self._output_ids_buf = Tensor((n_tokens,), DataType.I64, self._device, 0)
        self._step_capacity_tokens = n_tokens

    def _ensure_batch_buffers(self, n_batch_seq: int, n_block_elems: int) -> None:
        if n_batch_seq > self._step_capacity_batch_seq:
            self._context_lens_buf = Tensor((n_batch_seq,), DataType.I32, self._device, 0)
            self._batch_seq_ids_buf = Tensor((n_batch_seq,), DataType.I64, DeviceType.CPU, 0)
            self._step_capacity_batch_seq = n_batch_seq
        if n_block_elems > self._step_capacity_block_elems:
            self._block_tables_buf = Tensor((n_block_elems,), DataType.I32, self._device, 0)
            self._step_capacity_block_elems = n_block_elems

    def _collect_token_inputs(
        self,
        outputs: SchedulerOutputs,
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
        for row_idx, seq in enumerate(outputs.scheduled_seqs):
            bs = max(1, int(seq.block_size))
            if outputs.is_prefill:
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
            params = sampling_params_by_req.get(rid) if sampling_params_by_req is not None else None
            if params is None:
                if sampling_params is None:
                    raise RuntimeError("missing sampling params for request")
                params = sampling_params

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

    def _prepare_inputs(
        self,
        outputs: SchedulerOutputs,
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
            outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )

        context_lens, batch_seq_ids, block_tables, block_table_width = self._build_block_metadata_inputs(
            outputs,
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
