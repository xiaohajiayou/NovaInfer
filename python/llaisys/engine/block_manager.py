from __future__ import annotations

from collections import deque

from .sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = int(block_id)
        self.ref_count = 0

    def reset(self) -> None:
        self.ref_count = 1


class BlockManager:
    """vLLM-style block allocator (without prefix-cache hashing yet)."""

    def __init__(self, block_size: int = 16, num_blocks: int = 0):
        self.block_size = max(1, int(block_size))
        self.num_blocks = max(0, int(num_blocks))
        self.blocks: list[Block] = [Block(i) for i in range(self.num_blocks)]
        self.free_block_ids: deque[int] = deque(range(self.num_blocks))
        self.used_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError("allocate non-free block")
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError("deallocate in-use block")
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        if self.num_blocks <= 0:
            return True
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        if seq.block_table:
            raise RuntimeError("allocate called on non-empty block table")
        if self.num_blocks <= 0:
            return
        if not self.can_allocate(seq):
            raise RuntimeError("cannot allocate for sequence")
        for _ in range(seq.num_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(int(block_id))

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
        need_new_block = (len(seq) % self.block_size == 0)
        return len(self.free_block_ids) >= int(need_new_block)

    def may_append(self, seq: Sequence) -> None:
        if self.num_blocks <= 0:
            return
        if len(seq) % self.block_size == 0:
            if not self.free_block_ids:
                raise RuntimeError("no free block for append")
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(int(block_id))
