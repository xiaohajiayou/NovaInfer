from __future__ import annotations

from ctypes import POINTER, byref, c_int, c_int8, c_int32, c_int64, c_void_p, cast
from dataclasses import dataclass

import numpy as np

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout, LlaisysBatch, LlaisysModelCreateParams, ModelType
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights

TEST_KV_LAYOUT = int(KvCacheLayout.BLOCK)
TEST_KV_BLOCK_SIZE = 16


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


def _detach_tensor_handle(tensor: llaisys.Tensor):
    handle = tensor.lib_tensor()
    tensor._tensor = None  # type: ignore[attr-defined]
    return handle


def _make_tensor(shape: tuple[int, ...], rng: np.random.Generator):
    arr = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
    t = llaisys.Tensor(
        shape=shape,
        dtype=llaisys.DataType.F32,
        device=llaisys.DeviceType.CPU,
        device_id=0,
    )
    t.load(arr.ctypes.data_as(c_void_p))
    return _detach_tensor_handle(t)


def _build_meta(meta: TinyMeta) -> LlaisysQwen2Meta:
    return LlaisysQwen2Meta(
        llaisys.DataType.F32,
        meta.nlayer,
        meta.hs,
        meta.nh,
        meta.nkvh,
        meta.dh,
        meta.di,
        meta.maxseq,
        meta.voc,
        meta.epsilon,
        meta.theta,
        meta.end_token,
    )


def create_tiny_qwen2_model(meta: TinyMeta = TinyMeta()):
    meta_struct = _build_meta(meta)
    dev_ids = (c_int * 1)(0)
    params = LlaisysModelCreateParams(
        int(ModelType.QWEN2),
        cast(byref(meta_struct), c_void_p),
        llaisys.DeviceType.CPU,
        dev_ids,
        1,
        TEST_KV_LAYOUT,
        TEST_KV_BLOCK_SIZE,
    )
    model = LIB_LLAISYS.llaisysModelCreate(byref(params))
    if not model:
        raise RuntimeError("Failed to create tiny Qwen2 model")

    weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(model), POINTER(LlaisysQwen2Weights))
    if not weights_ptr:
        LIB_LLAISYS.llaisysModelDestroy(model)
        raise RuntimeError("Failed to fetch model weights")

    weights = weights_ptr.contents
    rng = np.random.default_rng(7)
    weights.in_embed = _make_tensor((meta.voc, meta.hs), rng)
    weights.out_embed = _make_tensor((meta.voc, meta.hs), rng)
    weights.out_norm_w = _make_tensor((meta.hs,), rng)

    for i in range(meta.nlayer):
        weights.attn_norm_w[i] = _make_tensor((meta.hs,), rng)
        weights.attn_q_w[i] = _make_tensor((meta.nh * meta.dh, meta.hs), rng)
        weights.attn_k_w[i] = _make_tensor((meta.nkvh * meta.dh, meta.hs), rng)
        weights.attn_v_w[i] = _make_tensor((meta.nkvh * meta.dh, meta.hs), rng)
        weights.attn_o_w[i] = _make_tensor((meta.hs, meta.nh * meta.dh), rng)
        weights.mlp_norm_w[i] = _make_tensor((meta.hs,), rng)
        weights.mlp_gate_w[i] = _make_tensor((meta.di, meta.hs), rng)
        weights.mlp_up_w[i] = _make_tensor((meta.di, meta.hs), rng)
        weights.mlp_down_w[i] = _make_tensor((meta.hs, meta.di), rng)

    return model, meta


def _make_batch(token_ids: list[int], logits_mask: list[int]) -> LlaisysBatch:
    n = len(token_ids)
    token_buf = (c_int64 * n)(*token_ids)
    logits_buf = (c_int8 * n)(*logits_mask)
    return LlaisysBatch(
        n_tokens=c_int32(n),
        token=token_buf,
        embd=None,
        pos=None,
        n_seq_id=None,
        seq_id=None,
        logits=logits_buf,
    )


def test_model_create_decode_logits_and_kv_api():
    model, meta = create_tiny_qwen2_model()
    try:
        assert int(LIB_LLAISYS.llaisysModelType(model)) == int(ModelType.QWEN2)

        batch = _make_batch([1, 2, 3], [0, 0, 1])
        status = int(LIB_LLAISYS.llaisysModelDecode(model, batch))
        assert status == 0

        n_outputs = int(LIB_LLAISYS.llaisysModelNOutputs(model))
        assert n_outputs == 1
        assert bool(LIB_LLAISYS.llaisysModelGetLogits(model))
        assert bool(LIB_LLAISYS.llaisysModelGetLogitsIth(model, c_int32(0)))
        assert not bool(LIB_LLAISYS.llaisysModelGetLogitsIth(model, c_int32(1)))

        assert int(LIB_LLAISYS.llaisysModelKvSeqKeep(model, c_int64(0))) == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqKeep(model, c_int64(1))) == 2
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(0))) == 2

        rm_status = int(LIB_LLAISYS.llaisysModelKvSeqRm(model, c_int64(0), c_int64(2), c_int64(3)))
        assert rm_status == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(0))) == 1
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


def test_model_decode_reports_oom_when_exceeding_maxseq():
    model, meta = create_tiny_qwen2_model()
    try:
        token_ids = [1] * (meta.maxseq + 1)
        batch = _make_batch(token_ids, [0] * meta.maxseq + [1])
        status = int(LIB_LLAISYS.llaisysModelDecode(model, batch))
        assert status == 1
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


if __name__ == "__main__":
    test_model_create_decode_logits_and_kv_api()
    test_model_decode_reports_oom_when_exceeding_maxseq()
    print("\033[92mtest_core_model_api passed!\033[0m")
