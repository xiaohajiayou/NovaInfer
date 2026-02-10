from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import SamplingParams


class Sampler:
    """Stage-1 sampler: argmax only. Other knobs are accepted for API compatibility."""

    def sample(self, logits_rows: Sequence[np.ndarray], params: SamplingParams) -> list[int]:
        _ = (params.top_k, params.top_p, params.temperature)
        sampled: list[int] = []
        for row in logits_rows:
            sampled.append(int(np.argmax(row)))
        return sampled
