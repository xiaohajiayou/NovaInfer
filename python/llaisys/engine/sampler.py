from __future__ import annotations

from ctypes import byref
from typing import Optional, Sequence
import math

import numpy as np

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.model import SamplerInput, SamplerOutput
from ..bindings.tensor import Tensor
from .config import EngineConfig


class Sampler:
    """Device-local sampler wrapper over llaisysSamplerSample."""

    def __init__(
        self,
        device: DeviceType,
        max_num_seqs: int | None = None,
        config: EngineConfig | None = None,
        device_id: int = 0,
    ):
        cfg = config or EngineConfig(
            device=device,
            max_num_seqs=(max(1, int(max_num_seqs)) if max_num_seqs is not None else 8),
        )
        self._device = cfg.device
        self._device_id = int(device_id)
        self._max_num_seqs = max(1, int(cfg.max_num_seqs))
        self._sample_counter = 0
        self._temperatures = Tensor((self._max_num_seqs,), DataType.F32, self._device, self._device_id)
        self._top_ps = Tensor((self._max_num_seqs,), DataType.F32, self._device, self._device_id)
        self._top_ks = Tensor((self._max_num_seqs,), DataType.I32, self._device, self._device_id)
        self._seeds = Tensor((self._max_num_seqs,), DataType.I64, self._device, self._device_id)
        self._has_seeds = Tensor((self._max_num_seqs,), DataType.I32, self._device, self._device_id)

    @staticmethod
    def _is_greedy_row(
        temperature: float,
        top_p: float,
        top_k: int,
        has_seed: int,
    ) -> bool:
        if not math.isfinite(temperature) or temperature <= 0.0:
            return True
        if int(top_k) == 1:
            return True
        return (
            int(top_k) <= 0
            and float(top_p) >= 1.0
            and abs(float(temperature) - 1.0) <= 1e-6
            and int(has_seed) == 0
        )

    def _controls_are_all_greedy(
        self,
        n_outputs: int,
        temperatures: Optional[Sequence[float]],
        top_ps: Optional[Sequence[float]],
        top_ks: Optional[Sequence[int]],
        has_seeds: Optional[Sequence[int]],
    ) -> bool:
        if temperatures is None or top_ps is None or top_ks is None or has_seeds is None:
            return True
        for i in range(int(n_outputs)):
            if not self._is_greedy_row(
                float(temperatures[i]),
                float(top_ps[i]),
                int(top_ks[i]),
                int(has_seeds[i]),
            ):
                return False
        return True

    @staticmethod
    def _mix64(x: int) -> int:
        x = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
        return x & 0x7FFFFFFFFFFFFFFF

    def _prepare_effective_seeds(
        self,
        n_outputs: int,
        temperatures: Sequence[float],
        top_ps: Sequence[float],
        top_ks: Sequence[int],
        seeds: Sequence[int],
        has_seeds: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        self._sample_counter = (int(self._sample_counter) + 1) & 0xFFFFFFFFFFFFFFFF
        nonce = self._sample_counter
        out_seeds: list[int] = [0] * int(n_outputs)
        out_has_seeds: list[int] = [0] * int(n_outputs)
        for i in range(int(n_outputs)):
            has_seed = int(has_seeds[i])
            if has_seed != 0:
                out_seeds[i] = int(seeds[i])
                out_has_seeds[i] = 1
                continue
            if self._is_greedy_row(float(temperatures[i]), float(top_ps[i]), int(top_ks[i]), 0):
                out_seeds[i] = 0
                out_has_seeds[i] = 0
                continue
            mixed = self._mix64(
                nonce
                ^ ((int(i) + 1) * 0x9E3779B97F4A7C15)
                ^ ((int(self._device_id) + 1) * 0xBF58476D1CE4E5B9)
            )
            out_seeds[i] = int(mixed)
            out_has_seeds[i] = 1
        return out_seeds, out_has_seeds

    @staticmethod
    def _copy_prefix(dst: Tensor, values: Sequence[int | float], dtype: np.dtype, n: int) -> Tensor:
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim != 1 or int(arr.size) != int(n):
            raise RuntimeError("sampler control shape mismatch")
        view = dst if int(n) == int(dst.shape()[0]) else dst.slice(0, 0, int(n))
        view.copy_from_numpy(arr)
        return view

    def sample_tokens(
        self,
        *,
        logits_tensor: Tensor | None,
        out_ids_dev: Tensor,
        temperatures: Optional[Sequence[float]] = None,
        top_ps: Optional[Sequence[float]] = None,
        top_ks: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        has_seeds: Optional[Sequence[int]] = None,
    ) -> Optional[Tensor]:
        if logits_tensor is None:
            return None

        shape = logits_tensor.shape()
        if len(shape) != 2:
            raise RuntimeError("sampler expects logits to be 2D")
        n_outputs = int(shape[0])
        if n_outputs > self._max_num_seqs:
            raise RuntimeError("sampler outputs exceed configured max_num_seqs")
        if out_ids_dev.device_type() != self._device:
            raise RuntimeError("sampler out_ids_dev device mismatch")
        if out_ids_dev.dtype() != DataType.I64:
            raise RuntimeError("sampler out_ids_dev dtype must be I64")
        if int(out_ids_dev.shape()[0]) != n_outputs:
            raise RuntimeError("sampler out_ids_dev shape mismatch")

        sin = SamplerInput()
        sin.logits = logits_tensor.lib_tensor()
        sin.temperatures = None
        sin.top_ps = None
        sin.top_ks = None
        sin.seeds = None
        sin.has_seeds = None
        keepalive: list[Tensor] = []

        if not self._controls_are_all_greedy(n_outputs, temperatures, top_ps, top_ks, has_seeds):
            if temperatures is None or top_ps is None or top_ks is None or seeds is None or has_seeds is None:
                raise RuntimeError("sampler controls must be fully specified for non-greedy sampling")
            eff_seeds, eff_has_seeds = self._prepare_effective_seeds(
                n_outputs,
                temperatures,
                top_ps,
                top_ks,
                seeds,
                has_seeds,
            )
            temp_t = self._copy_prefix(self._temperatures, temperatures, np.float32, n_outputs)
            top_p_t = self._copy_prefix(self._top_ps, top_ps, np.float32, n_outputs)
            top_k_t = self._copy_prefix(self._top_ks, top_ks, np.int32, n_outputs)
            seeds_t = self._copy_prefix(self._seeds, eff_seeds, np.int64, n_outputs)
            has_seeds_t = self._copy_prefix(self._has_seeds, eff_has_seeds, np.int32, n_outputs)
            keepalive.extend([temp_t, top_p_t, top_k_t, seeds_t, has_seeds_t])
            sin.temperatures = temp_t.lib_tensor()
            sin.top_ps = top_p_t.lib_tensor()
            sin.top_ks = top_k_t.lib_tensor()
            sin.seeds = seeds_t.lib_tensor()
            sin.has_seeds = has_seeds_t.lib_tensor()

        sout = SamplerOutput()
        sout.sampled_ids = out_ids_dev.lib_tensor()

        status = int(LIB_LLAISYS.llaisysSamplerSample(byref(sin), byref(sout)))
        if status != 0:
            raise RuntimeError(f"samplerSample failed with status={status}")
        return out_ids_dev
