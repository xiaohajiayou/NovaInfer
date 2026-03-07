from __future__ import annotations

from unittest.mock import patch

from llaisys.engine.config import EngineConfig
from llaisys.engine.llm_engine import LLMEngine
from llaisys.libllaisys.model import KvCacheLayout


class InjectedWorker:
    def __init__(self, runner) -> None:
        self._runner = runner

    @property
    def model_runner(self):
        return self._runner

    def close(self) -> None:
        return None

    def execute_model(self, scheduler_outputs):
        return self._runner.execute_model(scheduler_outputs)

    def sample_tokens(self):
        # Unit path: avoid allocating native tensors from dummy runner.
        state = getattr(self._runner, "_execute_state", None)
        if state is not None:
            output_indices, token_ids = state
            self._runner._execute_state = None
            vocab_size = int(getattr(self._runner, "vocab_size", 32000))
            return [int((int(token_ids[i]) + 1) % vocab_size) for i in output_indices]

        sampled = self._runner.sample_tokens()
        if sampled is None:
            return None
        tolist = getattr(sampled, "tolist", None)
        if callable(tolist):
            return [int(t) for t in tolist()]
        return [int(t) for t in sampled]

    def free_request(self, seq_id: int) -> None:
        self._runner.request_free(int(seq_id))

    def decode_tokens(self, token_ids: list[int]) -> str | None:
        _ = token_ids
        return None

    def encode_chat_messages(self, messages: list[dict]) -> list[int]:
        text = "\n".join(str(m.get("content", "")) for m in messages if m.get("content"))
        return [int(b) for b in text.encode("utf-8")]

    def get_default_sampling_params(self) -> dict:
        return {"temperature": 1.0, "top_p": 1.0, "top_k": 0}


def make_engine_with_runner(runner, **kwargs) -> LLMEngine:
    cfg_kwargs = dict(kwargs)
    if "max_num_seqs" not in cfg_kwargs and "max_batch_size" in cfg_kwargs:
        cfg_kwargs["max_num_seqs"] = int(cfg_kwargs.pop("max_batch_size"))
    else:
        cfg_kwargs.pop("max_batch_size", None)

    cfg_kwargs.setdefault("max_model_len", int(getattr(runner, "max_seq_len", 4096)))
    cfg_kwargs.setdefault("end_token_id", int(getattr(runner, "end_token_id", 0)))
    cfg_kwargs.setdefault(
        "kv_cache_layout",
        KvCacheLayout(int(getattr(runner, "kv_cache_layout", KvCacheLayout.BLOCK))),
    )
    if (
        KvCacheLayout(int(cfg_kwargs["kv_cache_layout"])) == KvCacheLayout.BLOCK
        and int(cfg_kwargs.get("num_kvcache_blocks", 0) or 0) <= 0
    ):
        block_size = max(1, int(cfg_kwargs.get("kv_cache_block_size", 256)))
        max_model_len = max(1, int(cfg_kwargs["max_model_len"]))
        cfg_kwargs["num_kvcache_blocks"] = (max_model_len + block_size - 1) // block_size

    cfg = EngineConfig(**cfg_kwargs)
    fake_worker = InjectedWorker(runner)
    with patch("llaisys.engine.llm_engine.Worker", side_effect=lambda *args, **_kw: fake_worker):
        return LLMEngine(config=cfg)
