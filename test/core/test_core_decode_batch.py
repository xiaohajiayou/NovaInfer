from ctypes import POINTER, byref, c_int, c_int32, c_int64, c_void_p, cast
from dataclasses import dataclass

import numpy as np

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout, LlaisysModelCreateParams, ModelType
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from test.utils.batch_builders import BlockBatchState, build_decode_batch

TEST_KV_LAYOUT = int(KvCacheLayout.BLOCK)
TEST_KV_BLOCK_SIZE = 16
IS_BLOCK_LAYOUT = TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK)



@dataclass(frozen=True)
class TinyMeta:
    nlayer: int = 1
    hs: int = 8
    nh: int = 2
    nkvh: int = 2
    dh: int = 4
    di: int = 16
    maxseq: int = 32
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


def _create_model(meta: TinyMeta = TinyMeta()):
    meta_struct = LlaisysQwen2Meta(
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
        raise RuntimeError("Failed to create tiny model")

    weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(model), POINTER(LlaisysQwen2Weights))
    if not weights_ptr:
        raise RuntimeError("Failed to fetch weights")
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

    return model


def _make_batch(tokens, logits_mask, seq_ids=None, pos=None, block_state: BlockBatchState | None = None):
    return build_decode_batch(
        tokens,
        logits_mask=logits_mask,
        seq_ids=seq_ids,
        pos_ids=pos,
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
        block_state=block_state,
    )


def test_single_seq_decode_mask():
    model = _create_model()
    try:
        built = _make_batch([1, 2, 3], [0, 1, 1])
        status = int(LIB_LLAISYS.llaisysModelDecode(model, built.batch))
        assert status == 0
        assert int(LIB_LLAISYS.llaisysModelNOutputs(model)) == 2
        output_ids = np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(model), shape=(2,))
        assert output_ids.tolist() == [1, 2]
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


def test_multi_seq_interleaved_decode():
    model = _create_model()
    try:
        block_state = BlockBatchState()
        batch1 = _make_batch(
            tokens=[10, 11, 12, 13],
            logits_mask=[1, 1, 1, 1],
            seq_ids=[100, 200, 100, 200],
            pos=[0, 0, 1, 1],
            block_state=block_state,
        )
        status = int(LIB_LLAISYS.llaisysModelDecode(model, batch1.batch))
        assert status == 0
        if IS_BLOCK_LAYOUT:
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(100))) == -1
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(200))) == -1
        else:
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(100))) == 1
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(200))) == 1
        output_ids = np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(model), shape=(4,))
        assert output_ids.tolist() == [0, 1, 2, 3]

        batch2 = _make_batch(
            tokens=[14, 15],
            logits_mask=[1, 1],
            seq_ids=[100, 200],
            pos=[2, 2],
            block_state=block_state,
        )
        status = int(LIB_LLAISYS.llaisysModelDecode(model, batch2.batch))
        assert status == 0
        if IS_BLOCK_LAYOUT:
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(100))) == -1
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(200))) == -1
        else:
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(100))) == 2
            assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(200))) == 2
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


def test_multi_seq_set_decode():
    model = _create_model()
    try:
        built = _make_batch(
            tokens=[10, 11, 12],
            logits_mask=[1, 1, 1],
            seq_ids=[1, 2, (1, 2)],
            pos=[0, 0, 1],
        )
        status = int(LIB_LLAISYS.llaisysModelDecode(model, built.batch))
        if IS_BLOCK_LAYOUT:
            assert status == -1
            return
        assert status == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(1))) == 1
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(2))) == 1
        assert int(LIB_LLAISYS.llaisysModelNOutputs(model)) == 3
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


if __name__ == "__main__":
    test_single_seq_decode_mask()
    test_multi_seq_interleaved_decode()
    test_multi_seq_set_decode()
    print("\033[92mtest_core_decode_batch passed!\033[0m")
