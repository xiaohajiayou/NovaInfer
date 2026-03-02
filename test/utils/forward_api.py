from __future__ import annotations

from ctypes import POINTER, byref, c_int, c_void_p, cast
from dataclasses import dataclass

import numpy as np

import llaisys
from llaisys.engine.sampler import Sampler
from llaisys.libllaisys import LIB_LLAISYS, DataType, DeviceType
from llaisys.libllaisys.model import (
    AttentionMetadata,
    KvCacheLayout,
    LlaisysModelCreateParams,
    LlaisysRuntimeCreateParams,
    ModelForwardInput,
    ModelForwardOutput,
    ModelType,
)
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from llaisys.tensor import Tensor

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
    layout: KvCacheLayout | int = KvCacheLayout.BLOCK,
    block_size: int = 16,
    max_model_len: int = 0,
    kv_capacity_tokens: int = 0,
):
    params = LlaisysRuntimeCreateParams(
        int(layout),
        int(block_size),
        int(max_model_len),
        int(kv_capacity_tokens),
    )
    runtime = LIB_LLAISYS.llaisysRuntimeCreate(byref(params))
    if not runtime:
        raise RuntimeError("Failed to create runtime")
    return runtime


def create_tiny_qwen2_model(
    meta: TinyMeta = TinyMeta(),
    *,
    layout: KvCacheLayout | int = KvCacheLayout.BLOCK,
    block_size: int = 16,
):
    runtime = create_runtime(
        layout=layout,
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
    model = LIB_LLAISYS.llaisysModelCreate(byref(params), runtime)
    if not model:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise RuntimeError("Failed to create tiny Qwen2 model")

    weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(model), POINTER(LlaisysQwen2Weights))
    if not weights_ptr:
        LIB_LLAISYS.llaisysModelDestroy(model)
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
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


def create_mock_model(*, layout: KvCacheLayout | int = KvCacheLayout.BLOCK, block_size: int = 16):
    runtime = create_runtime(layout=layout, block_size=block_size)
    params = LlaisysModelCreateParams(
        int(ModelType.MOCK),
        None,
        DeviceType.CPU,
        None,
        0,
    )
    model = LIB_LLAISYS.llaisysModelCreate(byref(params), runtime)
    if not model:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise RuntimeError("Failed to create mock model")
    return runtime, model


def destroy_model_runtime(model, runtime) -> None:
    if model:
        LIB_LLAISYS.llaisysModelDestroy(model)
    if runtime:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)


def _make_tensor_1d(values, dtype: DataType, device: DeviceType) -> Tensor:
    t = Tensor((len(values),), dtype, device, 0)
    t.copy_from_sequence(values)
    return t


def run_model_forward(model, batch: BatchBuildResult, *, device: DeviceType = DeviceType.CPU) -> ForwardRunResult:
    ntoken = len(batch.token_ids)
    if ntoken <= 0:
        raise ValueError("empty batch")

    input_ids_t = _make_tensor_1d(batch.token_ids, DataType.I64, device)
    pos_ids_t = _make_tensor_1d(batch.pos_values, DataType.I64, device)
    logits_mask_t = _make_tensor_1d(batch.logits_mask, DataType.I8, DeviceType.CPU)
    seq_ids_t = _make_tensor_1d(batch.seq_ids, DataType.I64, DeviceType.CPU)
    pos_ids_host_t = _make_tensor_1d(batch.pos_values, DataType.I64, DeviceType.CPU)

    attn = AttentionMetadata()
    attn.mode = int(batch.mode)
    attn.seq_ids = seq_ids_t.lib_tensor()
    attn.pos_ids_host = pos_ids_host_t.lib_tensor()
    attn.q_seq_rows = None
    attn.q_pos = None
    attn.slot_mapping = None
    attn.context_lens = None
    attn.batch_seq_ids = None
    attn.block_tables = None
    attn.block_table_width = 0

    if not batch.invalid and int(batch.mode) == int(KvCacheLayout.BLOCK):
        if (
            batch.q_seq_rows is None
            or batch.q_pos is None
            or batch.slot_mapping is None
            or batch.context_lens is None
            or batch.batch_seq_ids is None
            or batch.block_tables is None
        ):
            raise RuntimeError("incomplete BLOCK metadata")
        q_seq_rows_t = _make_tensor_1d(batch.q_seq_rows, DataType.I32, device)
        q_pos_t = _make_tensor_1d(batch.q_pos, DataType.I32, device)
        slot_mapping_t = _make_tensor_1d(batch.slot_mapping, DataType.I32, device)
        context_lens_t = _make_tensor_1d(batch.context_lens, DataType.I32, device)
        batch_seq_ids_t = _make_tensor_1d(batch.batch_seq_ids, DataType.I64, DeviceType.CPU)
        block_tables_t = _make_tensor_1d(batch.block_tables, DataType.I32, device)
        attn.q_seq_rows = q_seq_rows_t.lib_tensor()
        attn.q_pos = q_pos_t.lib_tensor()
        attn.slot_mapping = slot_mapping_t.lib_tensor()
        attn.context_lens = context_lens_t.lib_tensor()
        attn.batch_seq_ids = batch_seq_ids_t.lib_tensor()
        attn.block_tables = block_tables_t.lib_tensor()
        attn.block_table_width = int(batch.block_table_width)

    output_ids_t = Tensor((ntoken,), DataType.I64, device, 0)
    logits_holder_t = Tensor((1,), DataType.F32, device, 0)

    fin = ModelForwardInput()
    fin.input_ids = input_ids_t.lib_tensor()
    fin.pos_ids = pos_ids_t.lib_tensor()
    fin.logits_mask = logits_mask_t.lib_tensor()
    fin.attention = attn

    fout = ModelForwardOutput()
    fout.output_ids = output_ids_t.lib_tensor()
    fout.logits = logits_holder_t.lib_tensor()
    fout.n_outputs = 0

    status = int(LIB_LLAISYS.llaisysModelForward(model, byref(fin), byref(fout)))
    n_outputs = int(fout.n_outputs) if status == 0 else 0
    if n_outputs < 0 or n_outputs > ntoken:
        raise RuntimeError("invalid n_outputs from modelForward")

    out_view = output_ids_t.slice(0, 0, n_outputs)
    out_cpu = out_view if out_view.device_type() == DeviceType.CPU else out_view.to(DeviceType.CPU)
    output_ids = [int(x) for x in out_cpu.tolist()]
    return ForwardRunResult(
        status=status,
        n_outputs=n_outputs,
        output_ids=output_ids,
        output_ids_tensor=out_view,
        logits_tensor=logits_holder_t,
    )


def sample_from_forward(result: ForwardRunResult, *, device: DeviceType = DeviceType.CPU) -> list[int]:
    if result.n_outputs <= 0:
        return []
    sampler = Sampler(device)
    sampled = sampler.sample_tokens(
        logits_tensor=result.logits_tensor,
    )
    if sampled is None:
        return []
    sampled_cpu = sampled if sampled.device_type() == DeviceType.CPU else sampled.to(DeviceType.CPU)
    return [int(x) for x in sampled_cpu.tolist()]
