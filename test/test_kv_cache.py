import io
import sys
from ctypes import POINTER, byref, c_int, c_int8, c_int32, c_int64, c_void_p, cast
from dataclasses import dataclass

import numpy as np

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import LlaisysBatch, LlaisysModelCreateParams, ModelType
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


@dataclass(frozen=True)
class TinyMeta:
    nlayer: int = 1
    hs: int = 8
    nh: int = 2
    nkvh: int = 2
    dh: int = 4
    di: int = 16
    maxseq: int = 8
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


def _decode(model, tokens, seq_ids):
    n = len(tokens)
    token_buf = (c_int64 * n)(*tokens)
    logits_buf = (c_int8 * n)(*([1] * n))
    n_seq_buf = (c_int32 * n)()
    seq_ptr_buf = (POINTER(c_int64) * n)()
    rows = []
    for i, sid in enumerate(seq_ids):
        row = (c_int64 * 1)(sid)
        rows.append(row)
        n_seq_buf[i] = 1
        seq_ptr_buf[i] = cast(row, POINTER(c_int64))
    _decode._rows = rows

    batch = LlaisysBatch(
        n_tokens=c_int32(n),
        token=token_buf,
        embd=None,
        pos=None,
        n_seq_id=n_seq_buf,
        seq_id=seq_ptr_buf,
        logits=logits_buf,
    )
    return int(LIB_LLAISYS.llaisysModelDecode(model, batch))


def test_kv_seq_basic_ops():
    model = _create_model()
    try:
        assert _decode(model, [1, 2, 3], [10, 10, 20]) == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(10))) == 1
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(20))) == 0

        assert int(LIB_LLAISYS.llaisysModelKvSeqCp(model, c_int64(30), c_int64(10), c_int64(0), c_int64(2))) == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(30))) == 1

        assert int(LIB_LLAISYS.llaisysModelKvSeqRm(model, c_int64(10), c_int64(1), c_int64(2))) == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(10))) == 0

        assert int(LIB_LLAISYS.llaisysModelKvSeqAdd(model, c_int64(20), c_int64(0), c_int64(1), c_int64(0))) == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqAdd(model, c_int64(20), c_int64(0), c_int64(1), c_int64(1))) == 3

        assert int(LIB_LLAISYS.llaisysModelKvSeqKeep(model, c_int64(20))) == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(10))) == -1
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(20))) == 0
        assert int(LIB_LLAISYS.llaisysModelKvSeqPosMax(model, c_int64(30))) == -1
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


def test_kv_slot_exhaustion_returns_decode_oom():
    model = _create_model(TinyMeta(maxseq=4))
    try:
        assert _decode(model, [1, 2, 3, 4], [1, 2, 3, 4]) == 0
        status = _decode(model, [5], [9])
        assert status == 1
    finally:
        LIB_LLAISYS.llaisysModelDestroy(model)


if __name__ == "__main__":
    test_kv_seq_basic_ops()
    test_kv_slot_exhaustion_returns_decode_oom()
    print("\033[92mtest_kv_cache passed!\033[0m")
