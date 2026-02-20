from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout


@dataclass
class EngineConfig:
    model_type: str = "qwen2"
    model_path: Path | str | None = None
    device: DeviceType = DeviceType.CPU
    kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK
    kv_cache_block_size: int = 16
    max_model_len: int | None = None
    kv_cache_capacity_tokens: int | None = None
    max_num_seqs: int = 8
    max_num_batched_tokens: int | None = None
    kv_cache_auto_capacity: bool = False
    kv_cache_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    # Effective values resolved after model runner initialization.
    effective_max_model_len: int | None = None
    effective_kv_cache_capacity_tokens: int | None = None
    num_kvcache_blocks: int | None = None
    effective_enable_prefix_caching: bool | None = None

    def normalized(self) -> "EngineConfig":
        self.kv_cache_block_size = max(1, int(self.kv_cache_block_size))
        self.max_num_seqs = max(1, int(self.max_num_seqs))
        if self.max_model_len is not None:
            self.max_model_len = max(1, int(self.max_model_len))
        if self.kv_cache_capacity_tokens is not None:
            self.kv_cache_capacity_tokens = max(1, int(self.kv_cache_capacity_tokens))
        if self.max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max(1, int(self.max_num_batched_tokens))
        if self.effective_max_model_len is not None:
            self.effective_max_model_len = max(1, int(self.effective_max_model_len))
        if self.effective_kv_cache_capacity_tokens is not None:
            self.effective_kv_cache_capacity_tokens = max(1, int(self.effective_kv_cache_capacity_tokens))
        if self.num_kvcache_blocks is not None:
            self.num_kvcache_blocks = max(0, int(self.num_kvcache_blocks))
        if self.effective_enable_prefix_caching is not None:
            self.effective_enable_prefix_caching = bool(self.effective_enable_prefix_caching)
        self.enable_prefix_caching = bool(self.enable_prefix_caching)
        self.kv_cache_memory_utilization = float(min(0.98, max(0.01, self.kv_cache_memory_utilization)))
        return self
