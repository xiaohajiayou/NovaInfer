from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout


@dataclass
class EngineConfig:
    model_type: str = "qwen2"
    model_path: Path | str | None = None

    max_model_len: int | None = None
    end_token_id: int | None = None
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 4096
    
    kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK
    kv_cache_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0
    kv_cache_block_size: int = 256
    
    device: DeviceType = DeviceType.CPU
    enable_prefix_caching: bool = True
    
    def __post_init__(self) -> None:
        self.kv_cache_block_size = max(1, int(self.kv_cache_block_size))
        self.max_num_seqs = max(1, int(self.max_num_seqs))
        if self.max_model_len is not None:
            self.max_model_len = max(1, int(self.max_model_len))
        if self.end_token_id is not None:
            self.end_token_id = int(self.end_token_id)
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = 4096
        self.max_num_batched_tokens = max(1, int(self.max_num_batched_tokens))
        if self.num_kvcache_blocks is not None:
            self.num_kvcache_blocks = max(0, int(self.num_kvcache_blocks))
        self.enable_prefix_caching = bool(self.enable_prefix_caching)
        self.kv_cache_memory_utilization = float(min(0.98, max(0.01, self.kv_cache_memory_utilization)))
        if self.kv_cache_layout != KvCacheLayout.BLOCK:
            assert self.num_kvcache_blocks == 0
