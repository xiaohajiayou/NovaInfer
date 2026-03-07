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
    model = LIB_LLAISYS.llaisysModelCreate(byref(params))
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
    model = LIB_LLAISYS.llaisysModelCreate(byref(params))
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
    n = int(len(values))
    if n == 0:
        # Runtime tensor backend does not guarantee 0-sized allocation on all devices.
        # Keep API-level shape as [0] via slice of a small backing allocation.
        return Tensor((1,), dtype, device, 0).slice(0, 0, 0)
    t = Tensor((n,), dtype, device, 0)
    t.copy_from_sequence(values)
    return t


def run_model_forward(model, runtime, batch: BatchBuildResult, *, device: DeviceType = DeviceType.CPU) -> ForwardRunResult:
    ntoken = len(batch.token_ids)
    if ntoken <= 0:
        raise ValueError("empty batch")

    input_ids_t = _make_tensor_1d(batch.token_ids, DataType.I64, device)
    pos_ids_t = _make_tensor_1d(batch.pos_values, DataType.I64, device)
    output_ids = [i for i, m in enumerate(batch.logits_mask) if int(m) != 0]
    output_ids_t = _make_tensor_1d(output_ids, DataType.I64, device)
    seq_ids_t = _make_tensor_1d(batch.seq_ids, DataType.I64, DeviceType.CPU)
    pos_ids_host_t = _make_tensor_1d(batch.pos_values, DataType.I64, DeviceType.CPU)

    attn = AttentionMetadata()
    attn.mode = int(batch.mode)
    attn.seq_ids = seq_ids_t.lib_tensor()
    attn.pos_ids_host = pos_ids_host_t.lib_tensor()
    attn.req_num_scheduled_tokens = None
    attn.req_num_computed_tokens = None
    attn.query_start_loc = None
    attn.seq_lens = None
    attn.slot_mapping = None
    attn.block_tables = None
    attn.block_table_width = 0

    if not batch.invalid and int(batch.mode) == int(KvCacheLayout.BLOCK):
        if (
            batch.req_num_scheduled_tokens is None
            or batch.req_num_computed_tokens is None
            or batch.block_tables is None
        ):
            raise RuntimeError("incomplete BLOCK metadata")
        req_sched_t = _make_tensor_1d(batch.req_num_scheduled_tokens, DataType.I32, device)
        req_comp_t = _make_tensor_1d(batch.req_num_computed_tokens, DataType.I32, device)
        block_tables_t = _make_tensor_1d(batch.block_tables, DataType.I32, device)
        n_batch_seq = len(batch.req_num_scheduled_tokens)
        query_start_loc_t = Tensor((n_batch_seq + 1,), DataType.I32, device, 0)
        seq_lens_t = Tensor((n_batch_seq,), DataType.I32, device, 0)
        slot_mapping_t = Tensor((ntoken,), DataType.I32, device, 0)
        rc = int(
            LIB_LLAISYS.llaisysRuntimeBuildBlockAttentionMetadata(
                runtime,
                req_sched_t.lib_tensor(),
                req_comp_t.lib_tensor(),
                block_tables_t.lib_tensor(),
                int(batch.block_table_width),
                int(ntoken),
                query_start_loc_t.lib_tensor(),
                seq_lens_t.lib_tensor(),
                slot_mapping_t.lib_tensor(),
            )
        )
        if rc != 0:
            raise RuntimeError(f"RuntimeBuildBlockAttentionMetadata failed with status={rc}")
        attn.req_num_scheduled_tokens = req_sched_t.lib_tensor()
        attn.req_num_computed_tokens = req_comp_t.lib_tensor()
        attn.query_start_loc = query_start_loc_t.lib_tensor()
        attn.seq_lens = seq_lens_t.lib_tensor()
        attn.slot_mapping = slot_mapping_t.lib_tensor()
        attn.block_tables = block_tables_t.lib_tensor()
        attn.block_table_width = int(batch.block_table_width)

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
    sampler = Sampler(device, result.runtime)
    out_ids_dev = llaisys.Tensor((result.n_outputs,), llaisys.DataType.I64, device, 0)
    sampled = sampler.sample_tokens(
        logits_tensor=result.logits_tensor,
        out_ids_dev=out_ids_dev,
    )
    if sampled is None:
        return []
    sampled_cpu = sampled if sampled.device_type() == DeviceType.CPU else sampled.to(DeviceType.CPU)
    return [int(x) for x in sampled_cpu.tolist()]
