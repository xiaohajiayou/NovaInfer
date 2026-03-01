from __future__ import annotations

import llaisys
from llaisys.libllaisys.model import KvCacheLayout
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


def _forward(model, token_ids: list[int], logits_mask: list[int] | None):
    built = build_decode_batch(
        token_ids,
        logits_mask=logits_mask,
        seq_ids=[0] * len(token_ids),
        pos_ids=[int(i) for i in range(len(token_ids))],
        layout=KvCacheLayout(TEST_KV_LAYOUT),
        block_size=TEST_KV_BLOCK_SIZE,
    )
    return run_model_forward(model, built, device=llaisys.DeviceType.CPU)


def test_output_rows_match_logits_mask_and_sampler():
    runtime, model, _ = _create_model()
    try:
        out = _forward(model, [2, 4, 6, 8], [1, 0, 1, 0])
        assert out.status == 0
        assert out.n_outputs == 2
        assert out.output_ids == [0, 2]

        try:
            sampled_ids = sample_from_forward(out, device=llaisys.DeviceType.CPU)
            assert len(sampled_ids) == 2
        except RuntimeError as exc:
            assert "samplerSample failed" in str(exc)
    finally:
        destroy_model_runtime(model, runtime)


def test_default_logits_behavior_returns_last_row_only():
    runtime, model, _ = _create_model()
    try:
        no_out = _forward(model, [1, 3, 5], [0, 0, 0])
        assert no_out.status == 0
        assert no_out.n_outputs == 0

        # Null logits mask means "return last token row".
        out = _forward(model, [1, 3, 5], None)
        assert out.status == 0
        assert out.n_outputs == 1
        assert out.output_ids == [2]
        try:
            sampled_ids = sample_from_forward(out, device=llaisys.DeviceType.CPU)
            assert len(sampled_ids) == 1
        except RuntimeError as exc:
            assert "samplerSample failed" in str(exc)
    finally:
        destroy_model_runtime(model, runtime)
