from __future__ import annotations

from .types import GenerationOutput


class OutputProcessor:
    """Normalizes engine output for offline callers."""

    def finalize(
        self,
        request_id: str,
        prompt_len: int,
        token_ids: list[int],
        finish_reason: str,
        status,
        text: str | None = None,
    ) -> GenerationOutput:
        total = len(token_ids)
        completion = max(0, total - int(prompt_len))
        return GenerationOutput(
            request_id=request_id,
            token_ids=[int(t) for t in token_ids],
            finish_reason=finish_reason,
            status=status,
            text=text,
            usage={
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(completion),
                "total_tokens": int(total),
            },
        )
