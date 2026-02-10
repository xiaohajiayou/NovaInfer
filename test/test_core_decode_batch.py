from ctypes import POINTER, byref, c_int, c_int8, c_int32, c_int64, c_void_p, cast
from dataclasses import dataclass

import numpy as np

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import LlaisysBatch, LlaisysModelCreateParams, ModelType
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights



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


def _make_batch(tokens, logits_mask, seq_ids=None, pos=None):
    n = len(tokens)
    token_buf = (c_int64 * n)(*tokens)
    logits_buf = None if logits_mask is None else (c_int8 * n)(*logits_mask)
    pos_buf = None if pos is None else (c_int64 * n)(*pos)

    n_seq_buf = None
    seq_ptr_buf = None
    if seq_ids is not None:
        n_seq_buf = (c_int32 * n)()
        seq_ptr_buf = (POINTER(c_int64) * n)()
        raw_rows = []
        for i, sid in enumerate(seq_ids):
            row = (c_int64 * 1)(sid)
            raw_rows.append(row)
            n_seq_buf[i] = 1
            seq_ptr_buf[i] = cast(row, POINTER(c_int64))
        _make_batch._raw_rows = raw_rows  # keep refs alive

    return LlaisysBatch(
        n_tokens=c_int32(n),
        token=token_buf,
        embd=None,
        pos=pos_buf,
        n_seq_id=n_seq_buf,
        seq_id=seq_ptr_buf,
        logits=logits_buf,
    )


def test_single_seq_decode_mask():
    model = _create_model()
    try:
        batch = _make_batch([1, 2, 3], [0, 1, 1])
        status = int(LIB_LLAISYS.llaisysModelDecode(model, batch))
        assert status == 0
        assert int(LIB_LLAISYS.llaisysModelNOutputs(model)) == 2
        output_ids = np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(model), shape=(2,))
        assert output_ids.tolist() == [1, 2]
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


def test_multi_seq_interleaved_decode():
    model = _create_model()
    try:
        batch1 = _make_batch(
            tokens=[10, 11, 12, 13],
            logits_mask=[1, 1, 1, 1],
            seq_ids=[100, 200, 100, 200],
            pos=[0, 0, 1, 1],
        )
        status = int(LIB_LLAISYS.llaisysModelDecode(model, batch1))
        assert status == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(100))) == 1
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(200))) == 1
        output_ids = np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(model), shape=(4,))
        assert output_ids.tolist() == [0, 1, 2, 3]

        batch2 = _make_batch(
            tokens=[14, 15],
            logits_mask=[1, 1],
            seq_ids=[100, 200],
            pos=[2, 2],
        )
        status = int(LIB_LLAISYS.llaisysModelDecode(model, batch2))
        assert status == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(100))) == 2
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(200))) == 2
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


if __name__ == "__main__":
    test_single_seq_decode_mask()
    test_multi_seq_interleaved_decode()
    print("\033[92mtest_core_decode_batch passed!\033[0m")
