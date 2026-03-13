from __future__ import annotations

import os
from ctypes import POINTER, byref, c_int, c_size_t, c_void_p, cast
from dataclasses import dataclass

import numpy as np

import llaisys
from llaisys.engine.sampler import Sampler
from llaisys.libllaisys import LIB_LLAISYS, DataType, DeviceType
from llaisys.libllaisys.model import (
    AttentionMetadata,
    AttentionPhase,
    LlaisysModelCreateParams,
    LlaisysKvStateCreateParams,
    ModelForwardInput,
    ModelForwardOutput,
    ModelType,
)
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from llaisys.bindings.tensor import Tensor

from .batch_builders import BatchBuildResult


@dataclass(frozen=True)
class TinyMeta:
    nlayer: int = 1
    hs: int = 8
    nh: int = 2
    nkvh: int = 2
    dh: int = 4
    di: int = 16
    maxseq: int = 64
    voc: int = 32
    epsilon: float = 1e-6
    theta: float = 10000.0
    end_token: int = 1


@dataclass
class ForwardRunResult:
    status: int
    n_outputs: int
    output_ids: list[int]
    output_ids_tensor: Tensor
    logits_tensor: Tensor
    runtime: object


def _detach_tensor_handle(tensor: Tensor):
    handle = tensor.lib_tensor()
    tensor._tensor = None  # type: ignore[attr-defined]
    return handle


def _make_weight_tensor_handle(shape: tuple[int, ...], rng: np.random.Generator):
    arr = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
    t = Tensor(shape=shape, dtype=DataType.F32, device=DeviceType.CPU, device_id=0)
    t.load(arr.ctypes.data_as(c_void_p))
    return _detach_tensor_handle(t)


def build_meta(meta: TinyMeta) -> LlaisysQwen2Meta:
    return LlaisysQwen2Meta(
        DataType.F32,
        int(meta.nlayer),
        int(meta.hs),
        int(meta.nh),
        int(meta.nkvh),
        int(meta.dh),
        int(meta.di),
        int(meta.maxseq),
        int(meta.voc),
        float(meta.epsilon),
        float(meta.theta),
        int(meta.end_token),
    )


def create_runtime(
    *,
    block_size: int = 16,
    max_model_len: int = 0,
    kv_capacity_tokens: int = 0,
):
    params = LlaisysKvStateCreateParams(
        int(block_size),
        int(max_model_len),
        int(kv_capacity_tokens),
    )
    runtime = LIB_LLAISYS.llaisysKvStateCreate(byref(params))
    if not runtime:
        raise RuntimeError("Failed to create kv_state")
    return runtime


def create_tiny_qwen2_model(
    meta: TinyMeta = TinyMeta(),
    *,
    block_size: int = 16,
):
    runtime = create_runtime(
        block_size=block_size,
        max_model_len=int(meta.maxseq),
        kv_capacity_tokens=int(meta.maxseq),
    )
    meta_struct = build_meta(meta)
    dev_ids = (c_int * 1)(0)
    params = LlaisysModelCreateParams(
        int(ModelType.QWEN2),
        cast(byref(meta_struct), c_void_p),
        DeviceType.CPU,
        dev_ids,
        1,
    )
    model = LIB_LLAISYS.llaisysModelCreate(byref(params))
    if not model:
        LIB_LLAISYS.llaisysKvStateDestroy(runtime)
        raise RuntimeError("Failed to create tiny Qwen2 model")

    weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(model), POINTER(LlaisysQwen2Weights))
    if not weights_ptr:
        LIB_LLAISYS.llaisysModelDestroy(model)
        LIB_LLAISYS.llaisysKvStateDestroy(runtime)
        raise RuntimeError("Failed to fetch model weights")

    weights = weights_ptr.contents
    rng = np.random.default_rng(7)
    weights.in_embed = _make_weight_tensor_handle((meta.voc, meta.hs), rng)
    weights.out_embed = _make_weight_tensor_handle((meta.voc, meta.hs), rng)
    weights.out_norm_w = _make_weight_tensor_handle((meta.hs,), rng)
    for i in range(meta.nlayer):
        weights.attn_norm_w[i] = _make_weight_tensor_handle((meta.hs,), rng)
        weights.attn_q_w[i] = _make_weight_tensor_handle((meta.nh * meta.dh, meta.hs), rng)
        weights.attn_k_w[i] = _make_weight_tensor_handle((meta.nkvh * meta.dh, meta.hs), rng)
        weights.attn_v_w[i] = _make_weight_tensor_handle((meta.nkvh * meta.dh, meta.hs), rng)
        weights.attn_o_w[i] = _make_weight_tensor_handle((meta.hs, meta.nh * meta.dh), rng)
        weights.mlp_norm_w[i] = _make_weight_tensor_handle((meta.hs,), rng)
        weights.mlp_gate_w[i] = _make_weight_tensor_handle((meta.di, meta.hs), rng)
        weights.mlp_up_w[i] = _make_weight_tensor_handle((meta.di, meta.hs), rng)
        weights.mlp_down_w[i] = _make_weight_tensor_handle((meta.hs, meta.di), rng)

    return runtime, model, meta


def create_mock_model(*, block_size: int = 16):
    runtime = create_runtime(block_size=block_size)
    params = LlaisysModelCreateParams(
        int(ModelType.MOCK),
        None,
        DeviceType.CPU,
        None,
        0,
    )
    model = LIB_LLAISYS.llaisysModelCreate(byref(params))
    if not model:
        LIB_LLAISYS.llaisysKvStateDestroy(runtime)
        raise RuntimeError("Failed to create mock model")
    return runtime, model


def destroy_model_runtime(model, runtime) -> None:
    if model:
        LIB_LLAISYS.llaisysModelDestroy(model)
    if runtime:
        LIB_LLAISYS.llaisysKvStateDestroy(runtime)


def _make_tensor_1d(values, dtype: DataType, device: DeviceType) -> Tensor:
    n = int(len(values))
    if n == 0:
        # Runtime tensor backend does not guarantee 0-sized allocation on all devices.
        # Keep API-level shape as [0] via slice of a small backing allocation.
        return Tensor((1,), dtype, device, 0).slice(0, 0, 0)
    t = Tensor((n,), dtype, device, 0)
    t.copy_from_sequence(values)
    return t


def _is_cudnn_backend_requested() -> bool:
    return str(os.getenv("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "")).strip().lower() == "cudnn"


def _tensor_shape_raw(tensor_handle) -> tuple[int, ...]:
    if not tensor_handle:
        return ()
    ndim = int(LIB_LLAISYS.tensorGetNdim(tensor_handle))
    if ndim <= 0:
        return ()
    buf = (c_size_t * ndim)()
    LIB_LLAISYS.tensorGetShape(tensor_handle, buf)
    return tuple(int(buf[i]) for i in range(ndim))


def _qwen2_local_qk_dim(model_handle) -> int:
    weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(model_handle), POINTER(LlaisysQwen2Weights))
    if not weights_ptr:
        return 0
    weights = weights_ptr.contents
    if not weights.attn_q_w:
        return 0
    shape = _tensor_shape_raw(weights.attn_q_w[0])
    if len(shape) < 1:
        return 0
    return int(shape[0])


def _pack_cudnn_page_table_from_block_tables(
    flat_block_tables: list[int],
    *,
    n_batch_seq: int,
    block_table_width: int,
) -> np.ndarray:
    rows = np.asarray(flat_block_tables, dtype=np.int32).reshape(int(n_batch_seq), int(block_table_width))
    packed = rows.copy()
    for i in range(int(n_batch_seq)):
        row = packed[i]
        neg = np.flatnonzero(row < 0)
        if neg.size == 0:
            continue
        first_neg = int(neg[0])
        if first_neg == 0:
            raise RuntimeError("invalid block table row for cudnn: no valid block id")
        row[first_neg:] = row[first_neg - 1]
    return packed.reshape(-1)


def run_model_forward(model, runtime, batch: BatchBuildResult, *, device: DeviceType = DeviceType.CPU) -> ForwardRunResult:
    ntoken = len(batch.token_ids)
    if ntoken <= 0:
        raise ValueError("empty batch")

    input_ids_t = _make_tensor_1d(batch.token_ids, DataType.I64, device)
    pos_ids_t = _make_tensor_1d(batch.pos_values, DataType.I64, device)
    output_ids = [i for i, m in enumerate(batch.logits_mask) if int(m) != 0]
    output_ids_t = _make_tensor_1d(output_ids, DataType.I64, device)
    attn = AttentionMetadata()
    attn.phase = int(AttentionPhase.PREFILL)
    attn.cu_seqlens_q = None
    attn.cu_seqlens_k = None
    attn.max_seqlen_q = 0
    attn.max_seqlen_k = 0
    attn.slot_mapping = None
    attn.block_tables = None
    attn.block_table_width = 0
    attn.cudnn_seq_lens_q = None
    attn.cudnn_seq_lens_kv = None
    attn.cudnn_page_table = None
    attn.cudnn_qo_ragged_offset = None
    attn.cudnn_b_exec = 0

    if not batch.invalid:
        if (
            batch.req_num_scheduled_tokens is None
            or batch.req_num_computed_tokens is None
            or batch.block_tables is None
        ):
            raise RuntimeError("incomplete BLOCK metadata")
        block_tables_t = _make_tensor_1d(batch.block_tables, DataType.I32, device)
        n_batch_seq = len(batch.req_num_scheduled_tokens)
        block_table_width = int(batch.block_table_width)
        if block_table_width <= 0:
            raise RuntimeError("invalid block_table_width")
        if int(len(batch.block_tables)) != int(n_batch_seq) * block_table_width:
            raise RuntimeError("block_tables size mismatch")
        block_rows = np.asarray(batch.block_tables, dtype=np.int32).reshape(n_batch_seq, block_table_width)
        req_sched = np.asarray(batch.req_num_scheduled_tokens, dtype=np.int32)
        req_comp = np.asarray(batch.req_num_computed_tokens, dtype=np.int32)
        if np.any(req_sched < 0) or np.any(req_comp < 0):
            raise RuntimeError("invalid req_num_* metadata")
        cu_q_vals = [0]
        cu_k_vals = [0]
        max_q = 0
        max_k = 0
        for n_sched, n_comp in zip(req_sched.tolist(), req_comp.tolist()):
            seqlen_q = int(n_sched)
            seqlen_k = int(n_sched) + int(n_comp)
            cu_q_vals.append(int(cu_q_vals[-1]) + seqlen_q)
            cu_k_vals.append(int(cu_k_vals[-1]) + seqlen_k)
            max_q = max(max_q, seqlen_q)
            max_k = max(max_k, seqlen_k)
        if int(cu_q_vals[-1]) != int(ntoken):
            raise RuntimeError("ntoken mismatch with req_num_scheduled_tokens")
        block_size = 16
        slot_mapping = np.empty((ntoken,), dtype=np.int32)
        q_cursor = 0
        for row in range(n_batch_seq):
            n_sched = int(req_sched[row])
            seqlen = int(req_sched[row] + req_comp[row])
            q_base = seqlen - n_sched
            for local in range(n_sched):
                qpos = q_base + local
                bidx = int(qpos // block_size)
                boff = int(qpos % block_size)
                if bidx < 0 or bidx >= block_table_width:
                    raise RuntimeError("slot_mapping block index out of range")
                bid = int(block_rows[row, bidx])
                if bid < 0:
                    raise RuntimeError("slot_mapping resolved invalid negative block id")
                slot_mapping[q_cursor + local] = bid * block_size + boff
            q_cursor += n_sched
        slot_mapping_t = _make_tensor_1d(slot_mapping, DataType.I32, device)
        cu_seqlens_q_t = _make_tensor_1d(cu_q_vals, DataType.I32, device)
        cu_seqlens_k_t = _make_tensor_1d(cu_k_vals, DataType.I32, device)
        attn.cu_seqlens_q = cu_seqlens_q_t.lib_tensor()
        attn.cu_seqlens_k = cu_seqlens_k_t.lib_tensor()
        attn.max_seqlen_q = int(max_q)
        attn.max_seqlen_k = int(max_k)
        attn.slot_mapping = slot_mapping_t.lib_tensor()
        attn.block_tables = block_tables_t.lib_tensor()
        attn.block_table_width = int(batch.block_table_width)
        is_decode = (
            len(batch.req_num_scheduled_tokens) == ntoken
            and all(int(v) == 1 for v in batch.req_num_scheduled_tokens)
        )
        attn.phase = int(AttentionPhase.DECODE if is_decode else AttentionPhase.PREFILL)

        # Keep test helper compatible with CUDNN BLOCK path by attaching
        # CUDNN metadata when backend=cudnn.
        if device == DeviceType.NVIDIA and _is_cudnn_backend_requested():
            seq_q_rows = np.diff(np.asarray(cu_q_vals, dtype=np.int32))
            seq_kv_rows = np.diff(np.asarray(cu_k_vals, dtype=np.int32))
            if np.any(seq_q_rows <= 0):
                raise RuntimeError("invalid cudnn seq_len_q rows")
            if np.any(seq_kv_rows <= 0):
                raise RuntimeError("invalid cudnn seq_len_kv rows")
            b_exec = int(n_batch_seq)
            if int(len(batch.block_tables)) != int(b_exec) * int(batch.block_table_width):
                raise RuntimeError("cudnn page_table size mismatch")

            cudnn_seq_lens_q_t = _make_tensor_1d(seq_q_rows, DataType.I32, device)
            cudnn_seq_lens_kv_t = _make_tensor_1d(seq_kv_rows, DataType.I32, device)
            packed_page_table = _pack_cudnn_page_table_from_block_tables(
                batch.block_tables,
                n_batch_seq=b_exec,
                block_table_width=int(batch.block_table_width),
            )
            cudnn_page_table_t = _make_tensor_1d(packed_page_table, DataType.I32, device)
            attn.cudnn_seq_lens_q = cudnn_seq_lens_q_t.lib_tensor()
            attn.cudnn_seq_lens_kv = cudnn_seq_lens_kv_t.lib_tensor()
            attn.cudnn_page_table = cudnn_page_table_t.lib_tensor()
            attn.cudnn_b_exec = int(b_exec)

            if int(attn.phase) == int(AttentionPhase.PREFILL):
                hd = _qwen2_local_qk_dim(model)
                if hd <= 0:
                    raise RuntimeError("failed to infer q head_dim product for cudnn ragged offsets")
                token_prefix = np.concatenate(
                    (
                        np.zeros((1,), dtype=np.int64),
                        np.cumsum(seq_q_rows, dtype=np.int64),
                    )
                )
                ragged = (token_prefix * np.int64(hd)).astype(np.int32, copy=False)
                cudnn_ragged_t = _make_tensor_1d(ragged, DataType.I32, device)
                attn.cudnn_qo_ragged_offset = cudnn_ragged_t.lib_tensor()

    logits_holder_t = Tensor((1,), DataType.F32, device, 0)

    fin = ModelForwardInput()
    fin.input_ids = input_ids_t.lib_tensor()
    fin.pos_ids = pos_ids_t.lib_tensor()
    fin.logits_indices = output_ids_t.lib_tensor()
    fin.attention = attn

    fout = ModelForwardOutput()
    fout.logits = logits_holder_t.lib_tensor()

    status = int(LIB_LLAISYS.llaisysModelForward(model, runtime, byref(fin), byref(fout)))
    n_outputs = len(output_ids) if status == 0 else 0
    out_view = output_ids_t
    return ForwardRunResult(
        status=status,
        n_outputs=n_outputs,
        output_ids=output_ids,
        output_ids_tensor=out_view,
        logits_tensor=logits_holder_t,
        runtime=runtime,
    )


def sample_from_forward(result: ForwardRunResult, *, device: DeviceType = DeviceType.CPU) -> list[int]:
    if result.n_outputs <= 0:
        return []
    sampler = Sampler(device)
    out_ids_dev = llaisys.Tensor((result.n_outputs,), llaisys.DataType.I64, device, 0)
    sampled = sampler.sample_tokens(
        logits_tensor=result.logits_tensor,
        out_ids_dev=out_ids_dev,
    )
    if sampled is None:
        return []
    sampled_cpu = sampled if sampled.device_type() == DeviceType.CPU else sampled.to(DeviceType.CPU)
    return [int(x) for x in sampled_cpu.tolist()]
