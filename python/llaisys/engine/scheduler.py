from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..utils.nvtx import nvtx_range
from .block_manager import BlockManager, BlockManagerStats
from .config import EngineConfig
from .sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerOutputs:
    scheduled_seqs: list[Sequence]
    is_prefill: bool
    # vLLM-style scheduler signals (phase-1 subset).
    finished_seq_ids: list[int] = field(default_factory=list)
    free_encoder_mm_hashes: list[int] = field(default_factory=list)
    # seq_id -> num scheduled tokens in this step
    num_scheduled_tokens: dict[int, int] | None = None
    scheduled_new_reqs: list[Sequence] = field(default_factory=list)
    scheduled_cached_reqs: list[Sequence] = field(default_factory=list)
    scheduled_spec_decode_tokens: dict[int, list[int]] | None = None


class RequestScheduler:
    """vLLM-style waiting/running scheduler."""

    def __init__(self, config: EngineConfig):
        self.max_num_seqs = int(config.max_num_seqs)
        self.max_model_len = (
            int(config.max_model_len)
            if config.max_model_len is not None
            else 1 << 30
        )
        self.max_num_batched_tokens = int(config.max_num_batched_tokens)
        self.enable_prefix_cache = bool(config.enable_prefix_caching)
        block_size = int(config.kv_cache_block_size)
        num_kvcache_blocks = int(config.num_kvcache_blocks or 0)
        self.block_manager = BlockManager(
            num_blocks=num_kvcache_blocks,
            block_size=block_size,
            enable_prefix_caching=self.enable_prefix_cache,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._seq_by_id: dict[int, Sequence] = {}
        self._finished_seq_ids_pending: list[int] = []

    def add(self, seq: Sequence) -> None:
        seq.block_size = int(self.block_manager.block_size)
        self._seq_by_id[int(seq.seq_id)] = seq
        prompt_len = len(seq.prompt_token_ids)
        if prompt_len == 0:
            seq.finish_reason = "empty"
            self.finish(seq.seq_id)
            return
        if prompt_len > self.max_model_len or prompt_len > self.max_num_batched_tokens:
            seq.finish_reason = "aborted"
            self.finish(seq.seq_id)
            return
        if int(seq.max_tokens) <= 0:
            seq.finish_reason = "length"
            self.finish(seq.seq_id)
            return
        if self.block_manager.num_blocks > 0 and int(seq.num_blocks) > int(self.block_manager.num_blocks):
            seq.finish_reason = "aborted"
            self.finish(seq.seq_id)
            return
        if self.enable_prefix_cache:
            self.block_manager.prepare_sequence(seq)
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def block_stats(self) -> BlockManagerStats:
        return self.block_manager.stats()

    def schedule(self, max_num_seqs: int | None = None) -> SchedulerOutputs | None:
        with nvtx_range("py/scheduler/schedule"):
            cap = self.max_num_seqs if max_num_seqs is None else max(1, int(max_num_seqs))
            finished_seq_ids = list(self._finished_seq_ids_pending)
            self._finished_seq_ids_pending.clear()

            # 1) Prefill admission first (same order as nano-vllm).
            scheduled_seqs: list[Sequence] = []
            num_seqs = 0
            num_batched_tokens = 0
            with nvtx_range("py/scheduler/schedule/prefill"):
                while self.waiting and num_seqs < cap:
                    seq = self.waiting[0]
                    uncached = max(0, int(len(seq) - int(seq.num_cached_tokens)))
                    if (
                        num_batched_tokens + uncached > self.max_num_batched_tokens
                        or not self.block_manager.can_allocate(seq)
                    ):
                        break
                    num_seqs += 1
                    self.waiting.popleft()
                    self.block_manager.allocate(seq)
                    uncached = max(0, int(len(seq) - int(seq.num_cached_tokens)))
                    num_batched_tokens += uncached
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                    # Keep nano-vllm invariant for runner: every scheduled seq yields one sampled token.
                    if uncached > 0:
                        scheduled_seqs.append(seq)
            if scheduled_seqs:
                num_scheduled_tokens = {
                    int(seq.seq_id): int(len(seq) - int(seq.num_cached_tokens))
                    for seq in scheduled_seqs
                }
                return SchedulerOutputs(
                    scheduled_seqs=scheduled_seqs,
                    is_prefill=True,
                    finished_seq_ids=finished_seq_ids,
                    num_scheduled_tokens=num_scheduled_tokens,
                    scheduled_new_reqs=scheduled_seqs,
                    scheduled_cached_reqs=[],
                    scheduled_spec_decode_tokens={},
                )

            # 2) Decode fallback.
            with nvtx_range("py/scheduler/schedule/decode"):
                while self.running and num_seqs < cap:
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
                num_scheduled_tokens = {int(seq.seq_id): 1 for seq in scheduled_seqs}
                return SchedulerOutputs(
                    scheduled_seqs=scheduled_seqs,
                    is_prefill=False,
                    finished_seq_ids=finished_seq_ids,
                    num_scheduled_tokens=num_scheduled_tokens,
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=scheduled_seqs,
                    scheduled_spec_decode_tokens={},
                )

            if finished_seq_ids:
                return SchedulerOutputs(
                    scheduled_seqs=[],
                    is_prefill=False,
                    finished_seq_ids=finished_seq_ids,
                    num_scheduled_tokens={},
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=[],
                    scheduled_spec_decode_tokens={},
                )
            return None

    def finish(self, seq_id: int) -> None:
        seq = self._seq_by_id.pop(int(seq_id), None)
        if seq is None:
            return
        self._finished_seq_ids_pending.append(int(seq_id))
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

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], eos_token_id: int) -> None:
        with nvtx_range("py/scheduler/postprocess"):
            if len(seqs) != len(token_ids):
                raise RuntimeError("postprocess expects one token per scheduled sequence")
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(int(token_id))
                if (not bool(seq.ignore_eos) and int(token_id) == int(eos_token_id)):
                    seq.finish_reason = "eos_token"
                    self.finish(seq.seq_id)
                    continue
                if seq.stop_token_id_set and int(token_id) in seq.stop_token_id_set:
                    seq.finish_reason = "stop_token"
                    self.finish(seq.seq_id)
                    continue
                if int(seq.max_tokens) >= 0 and int(seq.num_completion_tokens) >= int(seq.max_tokens):
                    seq.finish_reason = "length"
                    self.finish(seq.seq_id)
