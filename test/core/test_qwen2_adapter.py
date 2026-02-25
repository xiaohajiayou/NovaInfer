import tempfile
from pathlib import Path

import json

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
    assert "llaisysModelCreate" in source
    assert "llaisysModelWeights" in source
    assert "llaisysModelDecode" in source
    assert "llaisysModelSampledIds" in source


if __name__ == "__main__":
    test_qwen2_adapter_meta_parse()
    test_qwen2_adapter_uses_generic_model_api_symbols()
    print("\033[92mtest_qwen2_adapter passed!\033[0m")
