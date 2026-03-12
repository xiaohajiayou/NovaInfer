from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout


ModelFactory = Callable[..., object]
KvStateFactory = Callable[..., Tuple[object, dict]]


@dataclass
class ModelRegistry:
    _factories: Dict[str, ModelFactory]
    _kv_state_factories: Dict[str, KvStateFactory] = field(default_factory=dict)

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
        return factory(model_path, device, **model_kwargs)

    def register_kv_state(self, model_type: str, factory: KvStateFactory) -> None:
        key = model_type.strip().lower()
        if not key:
            raise ValueError("model_type must be non-empty")
        self._kv_state_factories[key] = factory

    def create_kv_state(self, model_type: str, model_path: Path | str, device: DeviceType, **kv_kwargs) -> tuple[object | None, dict]:
        key = model_type.strip().lower()
        factory = self._kv_state_factories.get(key)
        if factory is None:
            return None, {}
        return factory(model_path, device, **kv_kwargs)


def _create_qwen2(
    model_path: Path | str,
    device: DeviceType,
    max_model_len: int | None = None,
    tensor_parallel_device_ids: tuple[int, ...] | None = None,
    tensor_parallel_size: int = 1,
    tp_rank: int = 0,
):
    from ..models.qwen2 import Qwen2

    return Qwen2(
        model_path=model_path,
        device=device,
        max_model_len=max_model_len,
        device_ids=tensor_parallel_device_ids,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tp_rank,
    )


def _create_qwen2_kv_state(
    model_path: Path | str,
    device: DeviceType,
    kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
    kv_cache_block_size: int = 16,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    kv_cache_memory_utilization: float = 0.9,
    # tp setting
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    distributed_executor_backend: str = "uni",
    distributed_backend: str = "nccl",
    tensor_parallel_device_ids: tuple[int, ...] | None = None,
    tp_rank: int = 0,
    tp_local_rank: int = 0,
) -> tuple[object, dict]:
    from .runtime_factory import create_kv_state, plan_qwen2_kv_cache

    plan = plan_qwen2_kv_cache(
        model_path=model_path,
        device=device,
        kv_cache_block_size=kv_cache_block_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        kv_cache_memory_utilization=kv_cache_memory_utilization,
    )
    kv_state = create_kv_state(
        device=device,
        kv_cache_layout=kv_cache_layout,
        kv_cache_block_size=kv_cache_block_size,
        plan=plan,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        distributed_backend=distributed_backend,
        tensor_parallel_device_ids=tensor_parallel_device_ids,
        tp_rank=tp_rank,
        tp_local_rank=tp_local_rank,
    )
    return kv_state, {
        "max_model_len": int(plan.max_model_len),
        "kv_cache_capacity_tokens": int(plan.kv_cache_capacity_tokens),
    }


def create_default_registry() -> ModelRegistry:
    registry = ModelRegistry(_factories={})
    registry.register("qwen2", _create_qwen2)
    registry.register_kv_state("qwen2", _create_qwen2_kv_state)
    return registry
