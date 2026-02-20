from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import SamplingParams


class Sampler:
    """Composable sampler supporting argmax/top-k/top-p/temperature."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def sample(self, logits_rows: Sequence[np.ndarray], params: SamplingParams) -> list[int]:
        sampled: list[int] = []
        for row_idx, row in enumerate(logits_rows):
            sampled.append(self._sample_one(np.asarray(row, dtype=np.float64), params, row_idx))
        return sampled

    def sample_per_row(self, logits_rows: Sequence[np.ndarray], params_rows: Sequence[SamplingParams]) -> list[int]:
        if len(logits_rows) != len(params_rows):
            raise ValueError("logits_rows and params_rows size mismatch")
        sampled: list[int] = []
        for row_idx, (row, params) in enumerate(zip(logits_rows, params_rows)):
            sampled.append(self._sample_one(np.asarray(row, dtype=np.float64), params, row_idx))
        return sampled

    def sample_per_row(self, logits_rows: Sequence[np.ndarray], params_rows: Sequence[SamplingParams]) -> list[int]:
        if len(logits_rows) != len(params_rows):
            raise ValueError("logits_rows and params_rows size mismatch")
        sampled: list[int] = []
        for row_idx, (row, params) in enumerate(zip(logits_rows, params_rows)):
            sampled.append(self._sample_one(np.asarray(row, dtype=np.float64), params, row_idx))
        return sampled

    def _sample_one(self, logits: np.ndarray, params: SamplingParams, row_idx: int) -> int:
        if logits.ndim != 1:
            raise ValueError("logits row must be 1D")
        if logits.size == 0:
            raise ValueError("empty logits row")

        # temperature <= 0 collapses to greedy decode.
        if params.temperature <= 0.0:
            return int(np.argmax(logits))

        # Apply temperature first.
        scaled = logits / float(params.temperature)

        # Convert to probabilities in a numerically stable way.
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs_sum = float(np.sum(probs))
        if not np.isfinite(probs_sum) or probs_sum <= 0:
            return int(np.argmax(logits))
        probs = probs / probs_sum

        indices = np.arange(probs.size, dtype=np.int64)

        # top-k filtering.
        top_k = int(params.top_k)
        if top_k > 0 and top_k < probs.size:
            keep = np.argpartition(probs, -top_k)[-top_k:]
            probs = probs[keep]
            indices = indices[keep]

        # top-p filtering.
        top_p = float(params.top_p)
        if 0.0 < top_p < 1.0 and probs.size > 1:
            order = np.argsort(probs)[::-1]
            sorted_probs = probs[order]
            sorted_indices = indices[order]
            cdf = np.cumsum(sorted_probs)
            cutoff = int(np.searchsorted(cdf, top_p, side="left"))
            keep_n = min(sorted_probs.size, max(1, cutoff + 1))
            probs = sorted_probs[:keep_n]
            indices = sorted_indices[:keep_n]

        probs_sum = float(np.sum(probs))
        if probs_sum <= 0:
            return int(np.argmax(logits))
        probs = probs / probs_sum

        if params.seed is not None:
            rng = np.random.default_rng(int(params.seed) + int(row_idx))
            return int(rng.choice(indices, p=probs))
        return int(self._rng.choice(indices, p=probs))
