from __future__ import annotations

import json
from pathlib import Path

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout
from .config import EngineConfig
from .input_processor import InputProcessor
from .model_runner import ModelRunner
from .model_registry import ModelRegistry, create_default_registry
class Worker:
    """Executes model forward for a batch plan."""

    def __init__(
        self,
        model_type: str = "qwen2",
        model_path: Path | str | None = None,
        device: DeviceType = DeviceType.CPU,
        kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
        kv_cache_block_size: int = 16,
        max_model_len: int | None = None,
        kv_cache_capacity_tokens: int | None = None,
        kv_cache_auto_capacity: bool = False,
        kv_cache_memory_utilization: float = 0.9,
        config: EngineConfig | None = None,
        model_runner=None,
        model_registry: ModelRegistry | None = None,
    ):
        cfg = (config or EngineConfig(
            model_type=model_type,
            model_path=model_path,
            device=device,
            kv_cache_layout=kv_cache_layout,
            kv_cache_block_size=kv_cache_block_size,
            max_model_len=max_model_len,
            kv_cache_capacity_tokens=kv_cache_capacity_tokens,
            kv_cache_auto_capacity=kv_cache_auto_capacity,
            kv_cache_memory_utilization=kv_cache_memory_utilization,
        )).normalized()
        self._config = cfg
        self._model_type = cfg.model_type
        self._model_path = Path(cfg.model_path) if cfg.model_path is not None else None
        self._device = cfg.device
        self._kv_cache_layout = cfg.kv_cache_layout
        self._kv_cache_block_size = int(cfg.kv_cache_block_size)
        self._max_model_len = int(cfg.max_model_len) if cfg.max_model_len is not None else None
        self._kv_cache_capacity_tokens = (
            int(cfg.kv_cache_capacity_tokens) if cfg.kv_cache_capacity_tokens is not None else None
        )
        self._kv_cache_auto_capacity = bool(cfg.kv_cache_auto_capacity)
        self._kv_cache_memory_utilization = float(cfg.kv_cache_memory_utilization)
        self._model_registry = model_registry if model_registry is not None else create_default_registry()
        runner_obj = model_runner if model_runner is not None else self._create_model_wrapper()
        if hasattr(runner_obj, "execute_model") and hasattr(runner_obj, "sample_tokens"):
            self._model_runner = runner_obj
        else:
            self._model_runner = ModelRunner(runner_obj, self._device, kv_cache_layout=self._kv_cache_layout)
        self._input_processor = InputProcessor(self._model_path)

    def _create_model_wrapper(self):
        if self._model_path is None:
            raise ValueError("model_path is required when model_runner is not provided")
        return self._model_registry.create(
            self._model_type,
            self._model_path,
            self._device,
            kv_cache_layout=self._kv_cache_layout,
            kv_cache_block_size=self._kv_cache_block_size,
            max_model_len=self._max_model_len,
            max_num_seqs=int(self._config.max_num_seqs),
            kv_cache_capacity_tokens=self._kv_cache_capacity_tokens,
            kv_cache_auto_capacity=self._kv_cache_auto_capacity,
            kv_cache_memory_utilization=self._kv_cache_memory_utilization,
        )

    @property
    def model_runner(self):
        return self._model_runner

    def close(self) -> None:
        runner = getattr(self, "_model_runner", None)
        if runner is None:
            return
        close_fn = getattr(runner, "close", None)
        if callable(close_fn):
            close_fn()

    @property
    def max_seq_len(self) -> int:
        if hasattr(self._model_runner, "max_seq_len"):
            return int(self._model_runner.max_seq_len)
        return int(self._model_runner.model._meta_info.maxseq)

    @property
    def end_token_id(self) -> int:
        if hasattr(self._model_runner, "end_token_id"):
            return int(self._model_runner.end_token_id)
        return int(self._model_runner.model._meta_info.end_token)

    @property
    def kv_cache_capacity_tokens(self) -> int | None:
        if hasattr(self._model_runner, "kv_cache_capacity_tokens"):
            try:
                return int(self._model_runner.kv_cache_capacity_tokens)
            except Exception:
                return self._kv_cache_capacity_tokens
        return self._kv_cache_capacity_tokens

    def execute_model(self, scheduler_outputs, sampling_params=None, sampling_params_by_req=None):
        return self._model_runner.execute_model(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )

    def sample_tokens(self, logits_handle, plan):
        return self._model_runner.sample_tokens(logits_handle, plan)

    def execute(self, scheduler_outputs, sampling_params=None, sampling_params_by_req=None):
        return self._model_runner.execute_step(
            scheduler_outputs,
            sampling_params=sampling_params,
            sampling_params_by_req=sampling_params_by_req,
        )

    def free_request(self, seq_id: int) -> None:
        fn = getattr(self._model_runner, "request_free", None)
        if callable(fn):
            try:
                fn(int(seq_id))
            except Exception:
                pass

    def decode_tokens(self, token_ids: list[int]) -> str | None:
        try:
            return self._input_processor.decode_tokens(token_ids)
        except Exception:
            return None

    def encode_chat_messages(self, messages: list[dict]) -> list[int]:
        try:
            return [int(t) for t in self._input_processor.encode_chat_messages(messages)]
        except Exception:
            text = "\n".join(str(m.get("content", "")) for m in messages if m.get("content"))
            return [int(b) for b in text.encode("utf-8")]

    def get_default_sampling_params(self) -> dict:
        # vLLM-like neutral defaults. Note: for OpenAI chat, max_tokens default
        # is derived from context window (not a fixed 16).
        out: dict[str, int | float] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
        }

        if self._model_path is None:
            return out
        gen_cfg_path = self._model_path / "generation_config.json"
        if not gen_cfg_path.exists():
            return out
        try:
            with gen_cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return out

        if cfg.get("temperature") is not None:
            out["temperature"] = float(cfg["temperature"])
        if cfg.get("top_p") is not None:
            out["top_p"] = float(cfg["top_p"])
        if cfg.get("top_k") is not None:
            out["top_k"] = int(cfg["top_k"])
        # HF max_new_tokens corresponds to vLLM/NovaInfer max_tokens.
        if cfg.get("max_new_tokens") is not None:
            out["max_tokens"] = int(cfg["max_new_tokens"])
        return out
