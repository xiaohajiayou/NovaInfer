from ctypes import c_int64

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout
from test.utils.batch_builders import BlockBatchState, build_decode_batch
from test.utils.forward_api import TinyMeta, create_tiny_qwen2_model, destroy_model_runtime, run_model_forward

TEST_KV_LAYOUT = int(KvCacheLayout.BLOCK)
TEST_KV_BLOCK_SIZE = 16
IS_BLOCK_LAYOUT = TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK)


def _create_model(meta: TinyMeta = TinyMeta(maxseq=32)):
    return create_tiny_qwen2_model(
        meta,
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
    )


def _forward(model, *, tokens, logits_mask, seq_ids=None, pos=None, block_state: BlockBatchState | None = None):
    built = build_decode_batch(
        tokens,
        logits_mask=logits_mask,
        seq_ids=seq_ids,
        pos_ids=pos,
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
        block_state=block_state,
    )
    return run_model_forward(model, built, device=llaisys.DeviceType.CPU)


def test_single_seq_decode_mask():
    runtime, model, _ = _create_model()
    try:
        out = _forward(model, tokens=[1, 2, 3], logits_mask=[0, 1, 1])
        assert out.status == 0
        assert out.n_outputs == 2
        assert out.output_ids == [1, 2]
    finally:
        destroy_model_runtime(model, runtime)


def test_multi_seq_interleaved_decode():
    runtime, model, _ = _create_model()
    try:
        block_state = BlockBatchState()
        out1 = _forward(
            model,
            tokens=[10, 11, 12, 13],
            logits_mask=[1, 1, 1, 1],
            seq_ids=[100, 200, 100, 200],
            pos=[0, 0, 1, 1],
            block_state=block_state,
        )
        assert out1.status == 0
        if IS_BLOCK_LAYOUT:
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(100))) == -1
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(200))) == -1
        else:
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(100))) == 1
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(200))) == 1
        assert out1.output_ids == [0, 1, 2, 3]

        out2 = _forward(
            model,
            tokens=[14, 15],
            logits_mask=[1, 1],
            seq_ids=[100, 200],
            pos=[2, 2],
            block_state=block_state,
        )
        assert out2.status == 0
        if IS_BLOCK_LAYOUT:
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(100))) == -1
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(200))) == -1
        else:
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(100))) == 2
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(200))) == 2
    finally:
        destroy_model_runtime(model, runtime)


def test_multi_seq_set_decode():
    runtime, model, _ = _create_model()
    try:
        out = _forward(
            model,
            tokens=[10, 11, 12],
            logits_mask=[1, 1, 1],
            seq_ids=[1, 2, (1, 2)],
            pos=[0, 0, 1],
        )
        # Forward metadata contract is one seq-id per token.
        assert out.status != 0
        if not IS_BLOCK_LAYOUT:
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(1))) in (-1, 0, 1)
    finally:
        destroy_model_runtime(model, runtime)
