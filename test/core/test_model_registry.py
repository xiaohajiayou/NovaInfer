from ctypes import c_int64

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout, ModelType
from test.utils.batch_builders import build_decode_batch
from test.utils.forward_api import (
    TinyMeta,
    create_mock_model,
    create_tiny_qwen2_model,
    destroy_model_runtime,
    run_model_forward,
)

TEST_KV_LAYOUT = int(KvCacheLayout.BLOCK)
TEST_KV_BLOCK_SIZE = 16


def test_model_registry_qwen2_and_mock():
    qwen2_runtime, qwen2 = None, None
    mock_runtime, mock = None, None
    try:
        qwen2_runtime, qwen2, _ = create_tiny_qwen2_model(
            TinyMeta(maxseq=64),
            layout=KvCacheLayout(TEST_KV_LAYOUT),
            block_size=TEST_KV_BLOCK_SIZE,
        )
        mock_runtime, mock = create_mock_model(
            layout=KvCacheLayout(TEST_KV_LAYOUT),
            block_size=TEST_KV_BLOCK_SIZE,
        )

        assert int(LIB_LLAISYS.llaisysModelType(qwen2)) == int(ModelType.QWEN2)
        assert int(LIB_LLAISYS.llaisysModelType(mock)) == int(ModelType.MOCK)

        assert LIB_LLAISYS.llaisysModelWeights(qwen2)
        assert not LIB_LLAISYS.llaisysModelWeights(mock)

        built = build_decode_batch(
            [10, 20, 30],
            logits_mask=[0, 1, 1],
            seq_ids=[5, 6, 5],
            pos_ids=[0, 0, 1],
            layout=KvCacheLayout(TEST_KV_LAYOUT),
            block_size=TEST_KV_BLOCK_SIZE,
        )
        out = run_model_forward(qwen2, built, device=llaisys.DeviceType.CPU)
        assert out.status == 0
        assert out.n_outputs == 2
        assert out.output_ids == [1, 2]

        # MOCK does not provide model forward graph.
        mock_out = run_model_forward(mock, built, device=llaisys.DeviceType.CPU)
        assert mock_out.status == -2

        assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(mock_runtime, c_int64(5))) == -1
        assert int(LIB_LLAISYS.llaisysRuntimeRequestFree(mock_runtime, c_int64(5))) == 2
    finally:
        destroy_model_runtime(qwen2, qwen2_runtime)
        destroy_model_runtime(mock, mock_runtime)
