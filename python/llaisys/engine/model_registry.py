from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout


ModelFactory = Callable[..., object]


@dataclass
class ModelRegistry:
    _factories: Dict[str, ModelFactory]

    def register(self, model_type: str, factory: ModelFactory) -> None:
        key = model_type.strip().lower()
        if not key:
            raise ValueError("model_type must be non-empty")
        self._factories[key] = factory

    def create(self, model_type: str, model_path: Path | str, device: DeviceType, **model_kwargs):
        key = model_type.strip().lower()
        factory = self._factories.get(key)
        if factory is None:
            raise ValueError(f"Unsupported model_type: {model_type}")
        try:
            return factory(model_path, device, **model_kwargs)
        except TypeError:
            # Backward compatibility for old 2-arg factories used by tests/custom code.
            return factory(model_path, device)


def _create_qwen2(
    model_path: Path | str,
    device: DeviceType,
    kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
    kv_cache_block_size: int = 16,
    max_model_len: int | None = None,
    kv_cache_capacity_tokens: int | None = None,
):
    from ..models.qwen2 import Qwen2

    return Qwen2(
        model_path=model_path,
        device=device,
        kv_cache_layout=kv_cache_layout,
        kv_cache_block_size=kv_cache_block_size,
        max_model_len=max_model_len,
        kv_cache_capacity_tokens=kv_cache_capacity_tokens,
    )


def create_default_registry() -> ModelRegistry:
    registry = ModelRegistry(_factories={})
    registry.register("qwen2", _create_qwen2)
    return registry
