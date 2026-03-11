import tempfile
from pathlib import Path

import json
import numpy as np

from llaisys.models import qwen2 as qwen2_mod



def test_qwen2_adapter_meta_parse():
    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td)
        cfg = {
            "torch_dtype": "float32",
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "max_position_embeddings": 64,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "eos_token_id": 1,
        }
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        meta = qwen2_mod._parse_meta(model_dir)
        assert meta.nlayer == 1
        assert meta.hs == 8
        assert meta.nh == 2
        assert meta.nkvh == 2
        assert meta.di == 16
        assert meta.voc == 32
        assert meta.end_token == 1


def test_qwen2_adapter_uses_generic_model_api_symbols():
    source = Path("python/llaisys/models/qwen2.py").read_text(encoding="utf-8")
    assert "llaisysRuntimeCreate" not in source
    assert "llaisysModelCreate" in source
    assert "llaisysModelWeights" in source
    assert "llaisysModelForward" in source
    assert "def forward(self, runtime, fin: ModelForwardInput, fout: ModelForwardOutput)" in source


def test_qwen2_tp_shard_column_and_row():
    obj = qwen2_mod.Qwen2.__new__(qwen2_mod.Qwen2)
    obj._tp_size = 2
    obj._tp_rank = 1
    obj._np_dtype = np.dtype(np.float32)

    q_w = np.arange(24, dtype=np.float32).reshape(6, 4)
    shard_q = obj._maybe_shard_array("attn_q_w", q_w)
    assert shard_q.shape == (3, 4)
    np.testing.assert_array_equal(shard_q, q_w[3:6, :])

    o_w = np.arange(30, dtype=np.float32).reshape(5, 6)
    shard_o = obj._maybe_shard_array("attn_o_w", o_w)
    assert shard_o.shape == (5, 3)
    np.testing.assert_array_equal(shard_o, o_w[:, 3:6])

    # Replicated field: unchanged.
    norm = np.arange(8, dtype=np.float32)
    shard_norm = obj._maybe_shard_array("out_norm_w", norm)
    np.testing.assert_array_equal(shard_norm, norm)


def test_qwen2_tp_shard_requires_divisible_dim():
    obj = qwen2_mod.Qwen2.__new__(qwen2_mod.Qwen2)
    obj._tp_size = 2
    obj._tp_rank = 0
    obj._np_dtype = np.dtype(np.float32)

    bad = np.arange(10, dtype=np.float32).reshape(5, 2)  # dim0=5 cannot split by tp=2
    try:
        obj._maybe_shard_array("attn_q_w", bad)
        raise AssertionError("expected ValueError for non-divisible shard dim")
    except ValueError as exc:
        assert "divisible" in str(exc)


if __name__ == "__main__":
    test_qwen2_adapter_meta_parse()
    test_qwen2_adapter_uses_generic_model_api_symbols()
    test_qwen2_tp_shard_column_and_row()
    test_qwen2_tp_shard_requires_divisible_dim()
    print("\033[92mtest_qwen2_adapter passed!\033[0m")
