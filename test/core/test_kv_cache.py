import llaisys
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


def test_kv_slot_exhaustion_returns_forward_oom():
    runtime, model, _ = _create_model(TinyMeta(maxseq=4))
    try:
        first_status = _forward(runtime, model, [1, 2, 3, 4], [1, 2, 3, 4])
        assert first_status == 0
        assert _forward(runtime, model, [5], [9]) == 0
    finally:
        destroy_model_runtime(model, runtime)
