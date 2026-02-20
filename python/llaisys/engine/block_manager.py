from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Dict

from .sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = int(block_id)
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def update(self, h: int, token_ids: list[int]) -> None:
        self.hash = int(h)
        self.token_ids = [int(t) for t in token_ids]


@dataclass(frozen=True)
class BlockManagerStats:
    block_size: int
    num_blocks: int
    used_blocks: int
    peak_used_blocks: int
    free_blocks: int
    usage: float
    prefix_hits: int
    prefix_misses: int
    prefix_saved_tokens: int


class BlockManager:
    """nano-vllm style block allocator with hash-based prefix reuse."""

    def __init__(self, block_size: int = 16, num_blocks: int = 0, enable_prefix_caching: bool = True):
        self.block_size = max(1, int(block_size))
        self.num_blocks = max(0, int(num_blocks))
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.blocks: list[Block] = [Block(i) for i in range(self.num_blocks)]
        self.free_block_ids: deque[int] = deque(range(self.num_blocks))
        self.used_block_ids: set[int] = set()
        self.peak_used_blocks = 0
        self.hash_to_block_id: Dict[int, int] = {}
        self.prefix_hits = 0
        self.prefix_misses = 0
        self.prefix_saved_tokens = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = hashlib.blake2b(digest_size=8)
        if prefix != -1:
            h.update(int(prefix).to_bytes(8, "little", signed=False))
        buf = array("q", [int(t) for t in token_ids])
        h.update(buf.tobytes())
        return int.from_bytes(h.digest(), "little", signed=False)

    def prepare_sequence(self, seq: Sequence) -> None:
        # nano-vllm style: prefix hit accounting is computed in allocate().
        seq.num_cached_tokens = 0

    def stats(self) -> BlockManagerStats:
        used_blocks = len(self.used_block_ids)
        free_blocks = len(self.free_block_ids)
        usage = (float(used_blocks) / float(self.num_blocks)) if self.num_blocks > 0 else 0.0
        return BlockManagerStats(
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            used_blocks=used_blocks,
            peak_used_blocks=int(self.peak_used_blocks),
            free_blocks=free_blocks,
            usage=usage,
            prefix_hits=int(self.prefix_hits),
            prefix_misses=int(self.prefix_misses),
            prefix_saved_tokens=int(self.prefix_saved_tokens),
        )

    def reset_prefix_cache(self) -> None:
        if not self.enable_prefix_caching:
            return
        self.hash_to_block_id.clear()
        for b in self.blocks:
            b.hash = -1
            b.token_ids = []
        self.prefix_hits = 0
        self.prefix_misses = 0
        self.prefix_saved_tokens = 0

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError("allocate non-free block")
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        if len(self.used_block_ids) > self.peak_used_blocks:
            self.peak_used_blocks = len(self.used_block_ids)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError("deallocate in-use block")
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        if self.num_blocks <= 0:
            return True
        # Match nano-vllm: conservative admission by total blocks.
        return len(self.free_block_ids) >= int(seq.num_blocks)

    def allocate(self, seq: Sequence) -> None:
        if seq.block_table:
            raise RuntimeError("allocate called on non-empty block table")
        if self.num_blocks <= 0:
            return
        if not self.can_allocate(seq):
            raise RuntimeError("cannot allocate for sequence")
        seq.num_cached_tokens = 0
        prefix_enabled = self.enable_prefix_caching
        h = -1
        cache_miss = False
        for i in range(int(seq.num_blocks)):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if (prefix_enabled and len(token_ids) == self.block_size) else -1
            block_id = self.hash_to_block_id.get(h, -1) if prefix_enabled else -1

            valid_hit = (
                (not cache_miss)
                and (block_id >= 0)
                and (self.blocks[block_id].token_ids == token_ids)
                and (h != -1)
            )

            if not valid_hit:
                cache_miss = True
                block_id = int(self.free_block_ids[0])
                block = self._allocate_block(block_id)
            else:
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
                seq.num_cached_tokens += self.block_size

            if prefix_enabled and h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = int(block_id)
            seq.block_table.append(int(block_id))

        if not prefix_enabled:
            seq.num_cached_tokens = 0
            return
        if seq.num_cached_tokens > 0:
            self.prefix_hits += 1
            self.prefix_saved_tokens += int(seq.num_cached_tokens)
        else:
            seq.num_cached_tokens = 0
            self.prefix_misses += 1

    def deallocate(self, seq: Sequence) -> None:
        if self.num_blocks <= 0:
            seq.num_cached_tokens = 0
            seq.block_table.clear()
            return
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        if self.num_blocks <= 0:
            return True
        need_new_block = (len(seq) % self.block_size == 1)
        return len(self.free_block_ids) >= int(need_new_block)

    def may_append(self, seq: Sequence) -> None:
        if self.num_blocks <= 0:
            return
        if not seq.block_table:
            return
        if len(seq) % self.block_size == 1:
            if not self.free_block_ids:
                raise RuntimeError("no free block for append")
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(int(block_id))
            return
        if len(seq) % self.block_size == 0:
            if not self.enable_prefix_caching:
                return
            last_bid = int(seq.block_table[-1])
            token_ids = seq.block(seq.num_blocks - 1)
            if len(token_ids) != self.block_size:
                return
            prefix = -1
            if len(seq.block_table) > 1:
                prefix = int(self.blocks[int(seq.block_table[-2])].hash)
            h = self.compute_hash(token_ids, prefix)
            self.blocks[last_bid].update(h, token_ids)
            self.hash_to_block_id[h] = last_bid
