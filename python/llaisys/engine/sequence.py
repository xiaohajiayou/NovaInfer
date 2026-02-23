from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from enum import Enum, auto

from .types import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    request_id: str
    seq_id: int
    token_ids: list[int]
    sampling_params: SamplingParams
    block_size: int = 16

    def __post_init__(self) -> None:
        self.token_ids = copy([int(t) for t in self.token_ids])
        self.status = SequenceStatus.WAITING
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []

    def __len__(self) -> int:
        return int(self.num_tokens)

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
    def num_blocks(self) -> int:
        if self.num_tokens <= 0:
            return 0
        return int((self.num_tokens + self.block_size - 1) // self.block_size)

    @property
    def last_token(self) -> int:
        return int(self.token_ids[-1])

    def block(self, i: int) -> list[int]:
        if i < 0 or i >= self.num_blocks:
            raise IndexError("block index out of range")
        start = i * self.block_size
        end = (i + 1) * self.block_size
        return list(self.token_ids[start:end])

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(int(token_id))
        self.num_tokens += 1
