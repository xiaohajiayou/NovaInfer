from __future__ import annotations

from collections import deque

from .types import BatchPlan


class RequestScheduler:
    """Offline scheduler with minimal queue semantics."""

    def __init__(self):
        self._waiting: deque[str] = deque()
        self._running: deque[str] = deque()

    def submit(self, request_id: str) -> None:
        self._waiting.append(request_id)

    def pick_next(self) -> str | None:
        # Prefill-first: new requests enter running queue before decode round-robin.
        if self._waiting:
            req_id = self._waiting.popleft()
            self._running.append(req_id)
            return req_id
        if not self._running:
            return None
        req_id = self._running[0]
        self._running.rotate(-1)
        return req_id

    def pick_next_batch(self, max_items: int) -> list[str]:
        if max_items <= 0:
            return []

        picked: list[str] = []

        # 1) Prefill first: promote new requests into running and schedule them.
        while self._waiting and len(picked) < max_items:
            req_id = self._waiting.popleft()
            self._running.append(req_id)
            picked.append(req_id)

        # 2) Fill remaining slots by decode round-robin from running queue.
        if len(picked) >= max_items or not self._running:
            return picked

        scanned = 0
        running_len = len(self._running)
        while scanned < running_len and len(picked) < max_items and self._running:
            req_id = self._running[0]
            self._running.rotate(-1)
            scanned += 1
            if req_id in picked:
                continue
            picked.append(req_id)

        return picked

    def finish(self, request_id: str) -> None:
        self._remove(self._waiting, request_id)
        self._remove(self._running, request_id)

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    @staticmethod
    def _remove(queue: deque[str], request_id: str) -> None:
        try:
            queue.remove(request_id)
        except ValueError:
            pass

    def build_prefill_plan(self, prompt_tokens: list[int], seq_id: int) -> BatchPlan:
        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")
        logits_mask = [0] * len(prompt_tokens)
        logits_mask[-1] = 1
        seq_ids = [int(seq_id)] * len(prompt_tokens)
        return BatchPlan(
            token_ids=[int(t) for t in prompt_tokens],
            logits_mask=logits_mask,
            seq_ids=seq_ids,
        )

    def build_decode_plan(self, token_id: int, seq_id: int) -> BatchPlan:
        return BatchPlan(token_ids=[int(token_id)], logits_mask=[1], seq_ids=[int(seq_id)])
