from __future__ import annotations

from copy import copy
from enum import Enum, auto
from itertools import count

from .types import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        seq_id: int | None = None,
        block_size: int | None = None,
    ) -> None:
        self.seq_id = int(seq_id) if seq_id is not None else next(Sequence.counter)
        self.sampling_params = sampling_params
        self.block_size = int(Sequence.block_size if block_size is None else block_size)
        self.token_ids = copy([int(t) for t in token_ids])
        self.status = SequenceStatus.WAITING
        self.last_token = int(self.token_ids[-1]) if self.token_ids else -1
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        raw_max_new = self.sampling_params.max_new_tokens
        self.max_tokens = int(raw_max_new) if raw_max_new is not None else -1
        self.ignore_eos = bool(self.sampling_params.ignore_eos)
        self.stop_token_id_set = {int(t) for t in self.sampling_params.stop_token_ids}
        self.finish_reason: str | None = None

    def __len__(self) -> int:
        return int(self.num_tokens)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return int(self.num_tokens - self.num_prompt_tokens)

    @property
    def prompt_token_ids(self) -> list[int]:
        return list(self.token_ids[: self.num_prompt_tokens])

    @property
    def completion_token_ids(self) -> list[int]:
        return list(self.token_ids[self.num_prompt_tokens :])

    @property
    def num_cached_blocks(self) -> int:
        return int(self.num_cached_tokens // self.block_size)

    @property
    def num_blocks(self) -> int:
        if self.num_tokens <= 0:
            return 0
        return int((self.num_tokens + self.block_size - 1) // self.block_size)

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_blocks <= 0:
            return 0
        return int(self.num_tokens - (self.num_blocks - 1) * self.block_size)

    def block(self, i: int) -> list[int]:
        if i < 0 or i >= self.num_blocks:
            raise IndexError("block index out of range")
        start = i * self.block_size
        end = (i + 1) * self.block_size
        return list(self.token_ids[start:end])

    def append_token(self, token_id: int) -> None:
        token_id = int(token_id)
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
