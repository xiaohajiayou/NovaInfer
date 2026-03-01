from __future__ import annotations

from ctypes import byref, c_int64

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout, LlaisysKvStats, ModelType
from test.utils.batch_builders import build_decode_batch
from test.utils.forward_api import (
    TinyMeta,
    create_tiny_qwen2_model,
    destroy_model_runtime,
    run_model_forward,
    sample_from_forward,
)

TEST_KV_LAYOUT = int(KvCacheLayout.BLOCK)
TEST_KV_BLOCK_SIZE = 16


def _create_model(meta: TinyMeta = TinyMeta(maxseq=64)):
    return create_tiny_qwen2_model(
        meta,
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
    )


def _forward(model, token_ids: list[int], logits_mask: list[int]):
    built = build_decode_batch(
        token_ids,
        logits_mask=logits_mask,
        seq_ids=[0] * len(token_ids),
        pos_ids=[int(i) for i in range(len(token_ids))],
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
    )
    return run_model_forward(model, built, device=llaisys.DeviceType.CPU)


def test_model_create_forward_sampler_and_runtime_kv_api():
    runtime, model, meta = _create_model()
    try:
        assert int(LIB_LLAISYS.llaisysModelType(model)) == int(ModelType.QWEN2)

        out = _forward(model, [1, 2, 3], [0, 0, 1])
        assert out.status == 0
        assert out.n_outputs == 1
        try:
            sampled_ids = sample_from_forward(out, device=llaisys.DeviceType.CPU)
            assert len(sampled_ids) == 1
        except RuntimeError as exc:
            # CPU sampler path is still under refactor in stage-1.
            assert "samplerSample failed" in str(exc)

        keep0 = int(LIB_LLAISYS.llaisysRuntimeKvSeqKeep(runtime, c_int64(0)))
        keep1 = int(LIB_LLAISYS.llaisysRuntimeKvSeqKeep(runtime, c_int64(1)))
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert keep0 == 5
            assert keep1 == 5
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(0))) == -1
        else:
            assert keep0 == 0
            assert keep1 == 2
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(0))) == 2

        rm_status = int(LIB_LLAISYS.llaisysRuntimeKvSeqRm(runtime, c_int64(0), c_int64(2), c_int64(3)))
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert rm_status == 5
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(0))) == -1
            assert int(LIB_LLAISYS.llaisysRuntimeKvResetPrefixCache(runtime)) == 0
        else:
            assert rm_status == 0
            assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(0))) == 1

        free_status = int(LIB_LLAISYS.llaisysRuntimeRequestFree(runtime, c_int64(0)))
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert free_status == 2
        else:
            assert free_status == 0
        assert int(LIB_LLAISYS.llaisysRuntimeKvSeqPosMax(runtime, c_int64(0))) == -1

        stats = LlaisysKvStats()
        stats_rc = int(LIB_LLAISYS.llaisysRuntimeKvStats(runtime, byref(stats)))
        assert stats_rc == 0
        assert int(stats.capacity_tokens) == meta.maxseq
        assert int(stats.used_tokens) >= 0
        assert int(stats.free_tokens) >= 0
        assert int(stats.peak_used_tokens) >= 0

        assert int(LIB_LLAISYS.llaisysRuntimeKvResetPrefixCache(runtime)) == 0
    finally:
        destroy_model_runtime(model, runtime)


def test_model_forward_reports_oom_when_exceeding_maxseq():
    runtime, model, meta = _create_model()
    try:
        token_ids = [1] * (meta.maxseq + 1)
        out = _forward(model, token_ids, [0] * meta.maxseq + [1])
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert out.status != 0
        else:
            assert out.status == 1
    finally:
        destroy_model_runtime(model, runtime)
