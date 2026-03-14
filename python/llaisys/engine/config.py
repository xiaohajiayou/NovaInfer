from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..libllaisys import DeviceType


@dataclass
class EngineConfig:
    model_type: str = "qwen2"
    model_path: Path | str | None = None

    max_model_len: int | None = None
    end_token_id: int | None = None
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 4096

    kv_cache_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0
    kv_cache_block_size: int = 256

    tensor_parallel_size: int = 1
    distributed_backend: str = "nccl"
    tensor_parallel_device_ids: Sequence[int] | None = None
    tp_rank: int = 0
    tp_local_rank: int = 0

    device: DeviceType = DeviceType.CPU
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        self.kv_cache_block_size = max(1, int(self.kv_cache_block_size))
        self.max_num_seqs = max(1, int(self.max_num_seqs))
        if self.max_model_len is not None:
            self.max_model_len = max(1, int(self.max_model_len))
        if self.end_token_id is not None:
            self.end_token_id = int(self.end_token_id)
        self.max_num_batched_tokens = max(1, int(self.max_num_batched_tokens or 4096))
        self.num_kvcache_blocks = max(0, int(self.num_kvcache_blocks or 0))
        self.enable_prefix_caching = bool(self.enable_prefix_caching)
        self.kv_cache_memory_utilization = float(min(0.98, max(0.01, self.kv_cache_memory_utilization)))

        self.tensor_parallel_size = max(1, int(self.tensor_parallel_size))
        self.tp_rank = max(0, int(self.tp_rank))
        self.tp_local_rank = max(0, int(self.tp_local_rank))
        self.distributed_backend = str(self.distributed_backend or "nccl").strip().lower()
        if self.tensor_parallel_device_ids is not None:
            self.tensor_parallel_device_ids = tuple(int(v) for v in self.tensor_parallel_device_ids)
