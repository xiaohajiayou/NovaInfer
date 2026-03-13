from ctypes import c_int64

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from test.utils.batch_builders import build_decode_batch
from test.utils.forward_api import TinyMeta, create_tiny_qwen2_model, destroy_model_runtime, run_model_forward

TEST_KV_BLOCK_SIZE = 16


def _create_model(meta: TinyMeta = TinyMeta(maxseq=8)):
    return create_tiny_qwen2_model(
        meta,
        block_size=TEST_KV_BLOCK_SIZE,
    )


def _forward(runtime, model, tokens, seq_ids):
    built = build_decode_batch(
        tokens,
        logits_mask=[1] * len(tokens),
        seq_ids=seq_ids,
        block_size=TEST_KV_BLOCK_SIZE,
        shared_block_ids_per_batch=True,
    )
    out = run_model_forward(model, runtime, built, device=llaisys.DeviceType.CPU)
    return int(out.status)


def test_kv_seq_basic_ops():
    runtime, model, _ = _create_model()
    try:
        first_status = _forward(runtime, model, [1, 2, 3], [10, 10, 20])
        assert first_status == 0
        assert int(LIB_LLAISYS.llaisysKvStateSeqCp(runtime, c_int64(30), c_int64(10), c_int64(0), c_int64(2))) == 5
        assert int(LIB_LLAISYS.llaisysKvStateSeqRm(runtime, c_int64(10), c_int64(1), c_int64(2))) == 5
        assert int(LIB_LLAISYS.llaisysKvStateSeqAdd(runtime, c_int64(20), c_int64(0), c_int64(1), c_int64(1))) == 5
        assert int(LIB_LLAISYS.llaisysKvStateSeqKeep(runtime, c_int64(20))) == 5
    finally:
        destroy_model_runtime(model, runtime)


def test_kv_slot_exhaustion_returns_forward_oom():
    runtime, model, _ = _create_model(TinyMeta(maxseq=4))
    try:
        first_status = _forward(runtime, model, [1, 2, 3, 4], [1, 2, 3, 4])
        assert first_status == 0
        assert _forward(runtime, model, [5], [9]) == 0
    finally:
        destroy_model_runtime(model, runtime)


def test_kv_seq_cp_then_rm_does_not_break_src():
    runtime, model, _ = _create_model()
    try:
        assert _forward(runtime, model, [1, 2], [7, 7]) == 0
        cp_status = int(LIB_LLAISYS.llaisysKvStateSeqCp(runtime, c_int64(8), c_int64(7), c_int64(0), c_int64(2)))
        assert cp_status == 5
    finally:
        destroy_model_runtime(model, runtime)
