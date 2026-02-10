from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


class RequestStatus(str, Enum):
    # Aligned with vLLM-style request lifecycle naming.
    WAITING = "waiting"
    WAITING_FOR_REMOTE_KVS = "waiting_for_remote_kvs"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"
    FINISHED_IGNORED = "finished_ignored"


TERMINAL_REQUEST_STATUSES = {
    RequestStatus.FINISHED_STOPPED,
    RequestStatus.FINISHED_LENGTH_CAPPED,
    RequestStatus.FINISHED_ABORTED,
    RequestStatus.FINISHED_IGNORED,
}


@dataclass
class RequestState:
    request_id: str
    status: RequestStatus
    prompt_tokens: List[int]
    output_tokens: List[int] = field(default_factory=list)
    history: List[RequestStatus] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: Optional[int] = None
    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 1.0
    stop_token_ids: Sequence[int] = field(default_factory=tuple)
    stop: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class BatchPlan:
    token_ids: List[int]
    logits_mask: List[int]
    pos_ids: Optional[List[int]] = None
    seq_ids: Optional[List[int]] = None


@dataclass(frozen=True)
class StepResult:
    sampled_token_ids: List[int]
    output_ids: List[int]


@dataclass(frozen=True)
class GenerationOutput:
    request_id: str
    token_ids: List[int]
    finish_reason: str
    status: RequestStatus
    text: Optional[str] = None
    usage: Optional[dict] = None


@dataclass(frozen=True)
class StreamChunk:
    request_id: str
    token_id: Optional[int]
    text_delta: Optional[str]
    status: RequestStatus
    is_finished: bool
    finish_reason: Optional[str] = None
