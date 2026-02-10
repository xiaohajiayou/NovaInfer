from __future__ import annotations

from pathlib import Path

from ..libllaisys import DeviceType
from .model_registry import ModelRegistry, create_default_registry
from .types import BatchPlan


class Worker:
    """Executes model forward for a batch plan."""

    def __init__(
        self,
        model_type: str = "qwen2",
        model_path: Path | str | None = None,
        device: DeviceType = DeviceType.CPU,
        model_runner=None,
        model_registry: ModelRegistry | None = None,
    ):
        self._model_type = model_type
        self._model_path = Path(model_path) if model_path is not None else None
        self._device = device
        self._model_registry = model_registry if model_registry is not None else create_default_registry()
        self._model_runner = model_runner if model_runner is not None else self._create_model_runner()

    def _create_model_runner(self):
        if self._model_path is None:
            raise ValueError("model_path is required when model_runner is not provided")
        return self._model_registry.create(self._model_type, self._model_path, self._device)

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
        return int(self._model_runner._meta_info.maxseq)

    @property
    def end_token_id(self) -> int:
        if hasattr(self._model_runner, "end_token_id"):
            return int(self._model_runner.end_token_id)
        return int(self._model_runner._meta_info.end_token)

    def execute(self, plan: BatchPlan):
        return self._model_runner.decode_batch(
            token_ids=plan.token_ids,
            pos_ids=plan.pos_ids,
            seq_ids=plan.seq_ids,
            logits_mask=plan.logits_mask,
        )

    def decode_tokens(self, token_ids: list[int]) -> str | None:
        if hasattr(self._model_runner, "decode_tokens"):
            try:
                return self._model_runner.decode_tokens(token_ids)
            except Exception:
                return None
        return None

    def encode_chat_messages(self, messages: list[dict]) -> list[int]:
        if hasattr(self._model_runner, "encode_chat_messages"):
            try:
                return [int(t) for t in self._model_runner.encode_chat_messages(messages)]
            except Exception:
                pass
        # Fallback for dummy runners in tests.
        text = "\n".join(str(m.get("content", "")) for m in messages if m.get("content"))
        return [int(b) for b in text.encode("utf-8")]
