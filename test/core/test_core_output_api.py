from __future__ import annotations

from ctypes import POINTER, byref, c_int, c_int32, c_void_p, cast
from dataclasses import dataclass

import numpy as np

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout, LlaisysModelCreateParams, ModelType
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from test.utils.batch_builders import build_decode_batch

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


def _decode(model, token_ids: list[int], logits_mask: list[int]) -> None:
    built = build_decode_batch(
        token_ids,
        logits_mask=logits_mask,
        seq_ids=[0] * len(token_ids),
        pos_ids=[int(i) for i in range(len(token_ids))],
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
    )
    status = int(LIB_LLAISYS.llaisysModelDecode(model, built.batch))
    assert status == 0


def test_output_rows_match_logits_mask_and_ith_access():
    model, meta = create_tiny_qwen2_model()
    try:
        _decode(model, [2, 4, 6, 8], [1, 0, 1, 0])

        n_outputs = int(LIB_LLAISYS.llaisysModelNOutputs(model))
        assert n_outputs == 2

        output_ids_ptr = LIB_LLAISYS.llaisysModelOutputIds(model)
        assert bool(output_ids_ptr)
        output_ids = np.ctypeslib.as_array(output_ids_ptr, shape=(n_outputs,))
        assert output_ids.tolist() == [0, 2]

        logits_ptr = LIB_LLAISYS.llaisysModelGetLogits(model)
        assert bool(logits_ptr)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_outputs * meta.voc,)).reshape(n_outputs, meta.voc)

        row0_ptr = LIB_LLAISYS.llaisysModelGetLogitsIth(model, c_int32(0))
        row1_ptr = LIB_LLAISYS.llaisysModelGetLogitsIth(model, c_int32(1))
        assert bool(row0_ptr)
        assert bool(row1_ptr)

        row0 = np.ctypeslib.as_array(row0_ptr, shape=(meta.voc,))
        row1 = np.ctypeslib.as_array(row1_ptr, shape=(meta.voc,))
        assert np.allclose(logits[0], row0)
        assert np.allclose(logits[1], row1)
        assert not bool(LIB_LLAISYS.llaisysModelGetLogitsIth(model, c_int32(2)))
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


def test_default_logits_behavior_returns_last_row_only():
    model, meta = create_tiny_qwen2_model()
    try:
        _decode(model, [1, 3, 5], [0, 0, 0])
        assert int(LIB_LLAISYS.llaisysModelNOutputs(model)) == 0

        # Null logits mask means "return last token row".
        built = build_decode_batch(
            [1, 3, 5],
            logits_mask=None,
            seq_ids=[0, 0, 0],
            pos_ids=[0, 1, 2],
            layout=KvCacheLayout(TEST_KV_LAYOUT),
            block_size=TEST_KV_BLOCK_SIZE,
        )
        status = int(LIB_LLAISYS.llaisysModelDecode(model, built.batch))
        assert status == 0

        assert int(LIB_LLAISYS.llaisysModelNOutputs(model)) == 1
        output_ids = np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(model), shape=(1,))
        assert output_ids.tolist() == [2]
        last_row = LIB_LLAISYS.llaisysModelGetLogitsIth(model, c_int32(0))
        assert bool(last_row)
        row = np.ctypeslib.as_array(last_row, shape=(meta.voc,))
        assert row.shape[0] == meta.voc
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


if __name__ == "__main__":
    test_output_rows_match_logits_mask_and_ith_access()
    test_default_logits_behavior_returns_last_row_only()
    print("\033[92mtest_core_output_api passed!\033[0m")
