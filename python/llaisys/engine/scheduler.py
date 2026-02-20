from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .block_manager import BlockManager, BlockManagerStats
from .sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerOutputs:
    scheduled_seqs: list[Sequence]
    is_prefill: bool


class RequestScheduler:
    """vLLM-style waiting/running scheduler."""

    def __init__(
        self,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 4096,
        block_size: int = 16,
        num_kvcache_blocks: int = 0,
        enable_prefix_cache: bool = True,
    ):
        self.max_num_seqs = max(1, int(max_num_seqs))
        self.max_num_batched_tokens = max(1, int(max_num_batched_tokens))
        self.enable_prefix_cache = bool(enable_prefix_cache)
        self.block_manager = BlockManager(
            num_blocks=num_kvcache_blocks,
            block_size=block_size,
            enable_prefix_caching=self.enable_prefix_cache,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._seq_by_request: dict[str, Sequence] = {}

    def add(self, seq: Sequence) -> None:
        if self.enable_prefix_cache:
            self.block_manager.prepare_sequence(seq)
        self.waiting.append(seq)
        self._seq_by_request[seq.request_id] = seq

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def block_stats(self) -> BlockManagerStats:
        return self.block_manager.stats()

    def schedule(self, max_num_seqs: int | None = None) -> SchedulerOutputs | None:
        cap = self.max_num_seqs if max_num_seqs is None else max(1, int(max_num_seqs))

        # 1) Prefill admission first (same order as nano-vllm).
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled_seqs) < cap:
            seq = self.waiting[0]
            if (
                num_batched_tokens + len(seq) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            self.waiting.popleft()
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return SchedulerOutputs(scheduled_seqs=scheduled_seqs, is_prefill=True)

        # 2) Decode fallback.
        while self.running and len(scheduled_seqs) < cap:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            return SchedulerOutputs(scheduled_seqs=scheduled_seqs, is_prefill=False)

        return None

    def finish(self, request_id: str) -> None:
        seq = self._seq_by_request.pop(request_id, None)
        if seq is None:
            return
        try:
            self.waiting.remove(seq)
        except ValueError:
            pass
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        if seq.block_table:
            self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
