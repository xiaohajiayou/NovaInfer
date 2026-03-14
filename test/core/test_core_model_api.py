from __future__ import annotations

from ctypes import byref, c_int64

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import LlaisysKvStats, ModelType
from test.utils.batch_builders import build_decode_batch
from test.utils.forward_api import (
    TinyMeta,
    create_runtime,
    create_tiny_qwen2_model,
    destroy_model_runtime,
    run_model_forward,
    sample_from_forward,
)

TEST_KV_BLOCK_SIZE = 16


def _create_model(meta: TinyMeta = TinyMeta(maxseq=64)):
    return create_tiny_qwen2_model(meta, block_size=TEST_KV_BLOCK_SIZE)


def _forward(runtime, model, token_ids: list[int], logits_mask: list[int]):
    built = build_decode_batch(
        token_ids,
        logits_mask=logits_mask,
        seq_ids=[0] * len(token_ids),
        pos_ids=[int(i) for i in range(len(token_ids))],
        block_size=TEST_KV_BLOCK_SIZE,
    )
    return run_model_forward(model, runtime, built, device=llaisys.DeviceType.CPU)


def test_model_create_forward_sampler_and_runtime_kv_api():
    runtime, model, meta = _create_model()
    try:
        assert int(LIB_LLAISYS.llaisysModelType(model)) == int(ModelType.QWEN2)

        out = _forward(runtime, model, [1, 2, 3], [0, 0, 1])
        assert out.status == 0
        assert out.n_outputs == 1
        try:
            sampled_ids = sample_from_forward(out, device=llaisys.DeviceType.CPU)
            assert len(sampled_ids) == 1
        except RuntimeError as exc:
            assert "samplerSample failed" in str(exc)

        assert int(LIB_LLAISYS.llaisysKvStateResetPrefixCache(runtime)) == 0

        free_status = int(LIB_LLAISYS.llaisysKvStateRequestFree(runtime, c_int64(0)))
        assert free_status == 2

        stats = LlaisysKvStats()
        stats_rc = int(LIB_LLAISYS.llaisysKvStateStats(runtime, byref(stats)))
        assert stats_rc == 0
        assert int(stats.capacity_tokens) == meta.maxseq
        assert int(stats.used_tokens) >= 0
        assert int(stats.free_tokens) >= 0
        assert int(stats.peak_used_tokens) >= 0

        assert int(LIB_LLAISYS.llaisysKvStateResetPrefixCache(runtime)) == 0
    finally:
        destroy_model_runtime(model, runtime)


def test_model_forward_reports_oom_when_exceeding_maxseq():
    runtime, model, meta = _create_model()
    try:
        token_ids = [1] * (meta.maxseq + 1)
        out = _forward(runtime, model, token_ids, [0] * meta.maxseq + [1])
        assert out.status != 0
    finally:
        destroy_model_runtime(model, runtime)


def test_model_forward_fails_fast_on_runtime_handle_change():
    runtime_a, model, meta = _create_model()
    runtime_b = create_runtime(
        block_size=TEST_KV_BLOCK_SIZE,
        max_model_len=int(meta.maxseq),
        kv_capacity_tokens=int(meta.maxseq),
    )
    try:
        out_a = _forward(runtime_a, model, [1], [1])
        assert out_a.status == 0

        out_b = _forward(runtime_b, model, [2], [1])
        assert out_b.status == -1
    finally:
        if runtime_b:
            LIB_LLAISYS.llaisysKvStateDestroy(runtime_b)
        destroy_model_runtime(model, runtime_a)
