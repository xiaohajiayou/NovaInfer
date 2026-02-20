from ctypes import POINTER, byref, c_int, c_int32, c_int64, c_void_p, cast

import numpy as np

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout, LlaisysModelCreateParams, ModelType
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from test.utils.batch_builders import build_decode_batch

TEST_KV_LAYOUT = int(KvCacheLayout.BLOCK)
TEST_KV_BLOCK_SIZE = 16



def _detach_tensor_handle(tensor: llaisys.Tensor):
    handle = tensor.lib_tensor()
    tensor._tensor = None  # type: ignore[attr-defined]
    return handle


def _make_tensor(shape, rng):
    arr = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
    t = llaisys.Tensor(
        shape=shape,
        dtype=llaisys.DataType.F32,
        device=llaisys.DeviceType.CPU,
        device_id=0,
    )
    t.load(arr.ctypes.data_as(c_void_p))
    return _detach_tensor_handle(t)


def _make_tiny_qwen2():
    meta = LlaisysQwen2Meta(
        llaisys.DataType.F32,
        1,
        8,
        2,
        2,
        4,
        16,
        16,
        32,
        1e-6,
        10000.0,
        1,
    )
    dev_ids = (c_int * 1)(0)
    params = LlaisysModelCreateParams(
        int(ModelType.QWEN2),
        cast(byref(meta), c_void_p),
        llaisys.DeviceType.CPU,
        dev_ids,
        1,
        TEST_KV_LAYOUT,
        TEST_KV_BLOCK_SIZE,
    )
    model = LIB_LLAISYS.llaisysModelCreate(byref(params))
    assert model

    weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(model), POINTER(LlaisysQwen2Weights))
    assert weights_ptr
    w = weights_ptr.contents
    rng = np.random.default_rng(7)
    w.in_embed = _make_tensor((32, 8), rng)
    w.out_embed = _make_tensor((32, 8), rng)
    w.out_norm_w = _make_tensor((8,), rng)
    w.attn_norm_w[0] = _make_tensor((8,), rng)
    w.attn_q_w[0] = _make_tensor((8, 8), rng)
    w.attn_k_w[0] = _make_tensor((8, 8), rng)
    w.attn_v_w[0] = _make_tensor((8, 8), rng)
    w.attn_o_w[0] = _make_tensor((8, 8), rng)
    w.mlp_norm_w[0] = _make_tensor((8,), rng)
    w.mlp_gate_w[0] = _make_tensor((16, 8), rng)
    w.mlp_up_w[0] = _make_tensor((16, 8), rng)
    w.mlp_down_w[0] = _make_tensor((8, 16), rng)
    return model


def _make_mock():
    params = LlaisysModelCreateParams(
        int(ModelType.MOCK),
        None,
        llaisys.DeviceType.CPU,
        None,
        0,
        -1,
        0,
    )
    model = LIB_LLAISYS.llaisysModelCreate(byref(params))
    assert model
    return model


def _decode_mock(model):
    built = build_decode_batch(
        [10, 20, 30],
        logits_mask=[0, 1, 1],
        seq_ids=[5, 6, 5],
        layout=KvCacheLayout.SLOT,
    )
    return int(LIB_LLAISYS.llaisysModelDecode(model, built.batch))


def test_model_registry_qwen2_and_mock():
    qwen2 = _make_tiny_qwen2()
    mock = _make_mock()
    try:
        assert int(LIB_LLAISYS.llaisysModelType(qwen2)) == int(ModelType.QWEN2)
        assert int(LIB_LLAISYS.llaisysModelType(mock)) == int(ModelType.MOCK)

        assert LIB_LLAISYS.llaisysModelWeights(qwen2)
        assert not LIB_LLAISYS.llaisysModelWeights(mock)

        status = _decode_mock(mock)
        assert status == 0
        assert int(LIB_LLAISYS.llaisysModelNOutputs(mock)) == 2
        ids = np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(mock), shape=(2,))
        assert ids.tolist() == [1, 2]
        assert bool(LIB_LLAISYS.llaisysModelGetLogitsIth(mock, c_int32(0)))
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(mock, c_int64(5))) == 1
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(mock, c_int64(6))) == 0
    finally:
        LIB_LLAISYS.llaisysModelDestroy(qwen2)
        LIB_LLAISYS.llaisysModelDestroy(mock)


if __name__ == "__main__":
    test_model_registry_qwen2_and_mock()
    print("\033[92mtest_model_registry passed!\033[0m")
