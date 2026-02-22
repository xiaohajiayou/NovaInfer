from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    model: str
    messages: Sequence[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    temperature: float | None = None
    include_reasoning: bool = True
    stop: Sequence[str] = field(default_factory=tuple)
    stop_token_ids: Sequence[int] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)
