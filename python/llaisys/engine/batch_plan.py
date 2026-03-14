from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatchPlan:
    is_prefill: bool
    input_ids: np.ndarray
    positions: np.ndarray
    scheduled_token_counts: np.ndarray
    cu_seqlens_q: np.ndarray
    cu_seqlens_k: np.ndarray
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: np.ndarray
    block_table_rows: np.ndarray
    block_table_width: int
    incremental_block_table_update: bool
    temperatures: list[float]
    top_ps: list[float]
    top_ks: list[int]
    seeds: list[int]
    has_seeds: list[int]

    @property
    def n_outputs(self) -> int:
        return len(self.temperatures)
