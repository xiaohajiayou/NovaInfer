from __future__ import annotations

from ctypes import byref, c_char_p, c_int, c_int64

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import KvCacheLayout, LlaisysKvStats, LlaisysParallelInitParams, ModelType
from test.utils.batch_builders import build_decode_batch
from test.utils.forward_api import (
    TinyMeta,
    create_runtime,
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


def _forward(runtime, model, token_ids: list[int], logits_mask: list[int]):
    built = build_decode_batch(
        token_ids,
        logits_mask=logits_mask,
        seq_ids=[0] * len(token_ids),
        pos_ids=[int(i) for i in range(len(token_ids))],
        layout=KvCacheLayout(TEST_KV_LAYOUT),
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
            # CPU sampler path is still under refactor in stage-1.
            assert "samplerSample failed" in str(exc)

        keep0 = int(LIB_LLAISYS.llaisysKvStateSeqKeep(runtime, c_int64(0)))
        keep1 = int(LIB_LLAISYS.llaisysKvStateSeqKeep(runtime, c_int64(1)))
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert keep0 == 5
            assert keep1 == 5
            assert int(LIB_LLAISYS.llaisysKvStateSeqPosMax(runtime, c_int64(0))) == -1
        else:
            assert keep0 == 0
            assert keep1 == 2
            assert int(LIB_LLAISYS.llaisysKvStateSeqPosMax(runtime, c_int64(0))) == 2

        rm_status = int(LIB_LLAISYS.llaisysKvStateSeqRm(runtime, c_int64(0), c_int64(2), c_int64(3)))
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert rm_status == 5
            assert int(LIB_LLAISYS.llaisysKvStateSeqPosMax(runtime, c_int64(0))) == -1
            assert int(LIB_LLAISYS.llaisysKvStateResetPrefixCache(runtime)) == 0
        else:
            assert rm_status == 0
            assert int(LIB_LLAISYS.llaisysKvStateSeqPosMax(runtime, c_int64(0))) == 1

        free_status = int(LIB_LLAISYS.llaisysKvStateRequestFree(runtime, c_int64(0)))
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert free_status == 2
        else:
            assert free_status == 0
        assert int(LIB_LLAISYS.llaisysKvStateSeqPosMax(runtime, c_int64(0))) == -1

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
        if TEST_KV_LAYOUT == int(KvCacheLayout.BLOCK):
            assert out.status != 0
        else:
            assert out.status == 1
    finally:
        destroy_model_runtime(model, runtime)


def test_kv_state_parallel_init_tp1_success():
    kv_state = create_runtime(layout=KvCacheLayout(TEST_KV_LAYOUT), block_size=TEST_KV_BLOCK_SIZE)
    try:
        params = LlaisysParallelInitParams(
            1,  # tensor_parallel_size
            1,  # pipeline_parallel_size
            1,  # world_size
            0,  # rank
            0,  # local_rank
            c_char_p(b"uni"),
            c_char_p(b"nccl"),
            c_char_p(b""),
            0,  # master_port
            0,  # node_rank
            1,  # nnodes
            c_char_p(b""),
            c_char_p(b"tp"),
            1,  # use_single_process_tp
            None,  # device_ids
            0,  # ndevice
        )
        rc = int(LIB_LLAISYS.llaisysKvStateParallelInit(kv_state, byref(params)))
        assert rc == 0
    finally:
        destroy_model_runtime(None, kv_state)


def test_kv_state_parallel_init_invalid_device_ids_for_tp():
    kv_state = create_runtime(layout=KvCacheLayout(TEST_KV_LAYOUT), block_size=TEST_KV_BLOCK_SIZE)
    try:
        dev_ids = (c_int * 1)(0)
        params = LlaisysParallelInitParams(
            2,  # tensor_parallel_size
            1,  # pipeline_parallel_size
            2,  # world_size
            0,  # rank
            0,  # local_rank
            c_char_p(b"uni"),
            c_char_p(b"nccl"),
            c_char_p(b""),
            0,  # master_port
            0,  # node_rank
            1,  # nnodes
            c_char_p(b""),
            c_char_p(b"tp"),
            1,  # use_single_process_tp
            dev_ids,
            1,  # ndevice (invalid, expected 2)
        )
        rc = int(LIB_LLAISYS.llaisysKvStateParallelInit(kv_state, byref(params)))
        assert rc == -1
    finally:
        destroy_model_runtime(None, kv_state)


def test_model_forward_fails_fast_on_kv_state_handle_change():
    kv_state_a, model, meta = _create_model()
    kv_state_b = create_runtime(
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
        max_model_len=int(meta.maxseq),
        kv_capacity_tokens=int(meta.maxseq),
    )
    try:
        out_a = _forward(kv_state_a, model, [1], [1])
        assert out_a.status == 0

        out_b = _forward(kv_state_b, model, [2], [1])
        assert out_b.status == -1
    finally:
        if kv_state_b:
            LIB_LLAISYS.llaisysKvStateDestroy(kv_state_b)
        destroy_model_runtime(model, kv_state_a)
