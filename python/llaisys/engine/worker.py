from __future__ import annotations

import json
from pathlib import Path

from ..libllaisys import DeviceType
from ..utils.nvtx import nvtx_range
from .config import EngineConfig
from .cpu_model_runner import CPUModelRunner
from .gpu_model_runner import GPUModelRunner
from .input_processor import InputProcessor
from .model_registry import ModelRegistry, create_default_registry
from .runtime_factory import select_tp_device_ids


class Worker:
    """Executes model forward for a batch plan."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        model_registry: ModelRegistry | None = None,
        **kwargs,
    ):
        cfg = config or EngineConfig(**kwargs)
        self._config = cfg
        self._model_path = Path(cfg.model_path) if cfg.model_path is not None else None
        if int(getattr(cfg, "tensor_parallel_size", 1)) > 1:
            if cfg.device != DeviceType.NVIDIA:
                raise RuntimeError("tensor parallel requires NVIDIA device")
            selected = select_tp_device_ids(
                int(cfg.tensor_parallel_size),
                tuple(cfg.tensor_parallel_device_ids) if cfg.tensor_parallel_device_ids is not None else None,
            )
            cfg.tensor_parallel_device_ids = tuple(int(v) for v in selected)
            cfg.tp_local_rank = int(getattr(cfg, "tp_rank", 0))
        self._model_registry = model_registry if model_registry is not None else create_default_registry()
        model_obj = self._create_model_wrapper()
        runner_cls = CPUModelRunner if cfg.device == DeviceType.CPU else GPUModelRunner
        self._model_runner = runner_cls(
            model_obj,
            config=self._config,
            model_registry=self._model_registry,
        )
        self._input_processor = InputProcessor(self._model_path)

    def _create_model_wrapper(self):
        if self._model_path is None:
            raise ValueError("model_path is required")
        return self._model_registry.create(
            self._config.model_type,
            self._model_path,
            self._config.device,
            max_model_len=self._config.max_model_len,
            tensor_parallel_device_ids=getattr(self._config, "tensor_parallel_device_ids", None),
            tensor_parallel_size=int(getattr(self._config, "tensor_parallel_size", 1)),
            tp_rank=int(getattr(self._config, "tp_rank", 0)),
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

    def execute_model(self, scheduler_outputs):
        with nvtx_range("py/worker/execute_model"):
            return self._model_runner.execute_model(scheduler_outputs)

    def sample_tokens(self):
        with nvtx_range("py/worker/sample_tokens"):
            return self._model_runner.sample_tokens()

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
