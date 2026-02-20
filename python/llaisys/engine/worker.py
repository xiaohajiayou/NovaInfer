from __future__ import annotations

from pathlib import Path

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout
from .config import EngineConfig
from .model_registry import ModelRegistry, create_default_registry
from .types import BatchPlan


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
        self._model_runner = model_runner if model_runner is not None else self._create_model_runner()

    def _create_model_runner(self):
        if self._model_path is None:
            raise ValueError("model_path is required when model_runner is not provided")
        return self._model_registry.create(
            self._model_type,
            self._model_path,
            self._device,
            kv_cache_layout=self._kv_cache_layout,
            kv_cache_block_size=self._kv_cache_block_size,
            max_model_len=self._max_model_len,
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
        return int(self._model_runner._meta_info.maxseq)

    @property
    def end_token_id(self) -> int:
        if hasattr(self._model_runner, "end_token_id"):
            return int(self._model_runner.end_token_id)
        return int(self._model_runner._meta_info.end_token)

    @property
    def kv_cache_capacity_tokens(self) -> int | None:
        if hasattr(self._model_runner, "kv_cache_capacity_tokens"):
            try:
                return int(self._model_runner.kv_cache_capacity_tokens)
            except Exception:
                return self._kv_cache_capacity_tokens
        return self._kv_cache_capacity_tokens

    def execute(self, plan: BatchPlan):
        try:
            return self._model_runner.decode_batch(
                token_ids=plan.token_ids,
                pos_ids=plan.pos_ids,
                seq_ids=plan.seq_ids,
                logits_mask=plan.logits_mask,
                slot_mapping=plan.slot_mapping,
                context_lens=plan.context_lens,
                batch_seq_ids=plan.batch_seq_ids,
                block_tables=plan.block_tables,
                block_table_width=plan.block_table_width,
            )
        except TypeError:
            # Backward-compatible fallback for dummy/stub runners in tests.
            return self._model_runner.decode_batch(
                token_ids=plan.token_ids,
                pos_ids=plan.pos_ids,
                seq_ids=plan.seq_ids,
                logits_mask=plan.logits_mask,
            )

    def free_request(self, seq_id: int) -> None:
        # Capability probing to avoid coupling engine lifecycle to one runner API.
        for fn_name in ("request_free", "free_request", "kv_request_free"):
            fn = getattr(self._model_runner, fn_name, None)
            if callable(fn):
                try:
                    fn(int(seq_id))
                except Exception:
                    pass
                return

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
