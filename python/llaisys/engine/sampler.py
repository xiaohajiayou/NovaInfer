from __future__ import annotations

from ctypes import byref
from typing import Optional, Sequence

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.model import SamplerInput, SamplerOutput
from ..tensor import Tensor
from .config import EngineConfig


class Sampler:
    """Device-local sampler wrapper over llaisysSamplerSample."""

    def __init__(
        self,
        device: DeviceType,
        max_num_seqs: int | None = None,
        config: EngineConfig | None = None,
    ):
        cfg = config or EngineConfig(
            device=device,
            max_num_seqs=(max(1, int(max_num_seqs)) if max_num_seqs is not None else 8),
        )
        self._device = cfg.device
        self._max_num_seqs = max(1, int(cfg.max_num_seqs))

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
        # Stage-1 native sampler currently uses argmax path in C++ and does not
        # consume these controls yet.
        del temperatures, top_ps, top_ks, seeds, has_seeds

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

        sout = SamplerOutput()
        sout.sampled_ids = out_ids_dev.lib_tensor()

        status = int(LIB_LLAISYS.llaisysSamplerSample(byref(sin), byref(sout)))
        if status != 0:
            raise RuntimeError(f"samplerSample failed with status={status}")
        return out_ids_dev
