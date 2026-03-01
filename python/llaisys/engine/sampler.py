from __future__ import annotations

from ctypes import byref
from typing import Optional, Sequence

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.model import SamplerInput, SamplerOutput
from ..tensor import Tensor


class Sampler:
    """Device-local sampler wrapper over llaisysSamplerSample."""

    def __init__(self, device: DeviceType):
        self._device = device
        self._sample_capacity = 0
        self._sampled_ids_buf: Optional[Tensor] = None

    def _ensure_sampled_buffer(self, n_outputs: int) -> None:
        if n_outputs <= self._sample_capacity:
            return
        self._sampled_ids_buf = Tensor((n_outputs,), DataType.I64, self._device, 0)
        self._sample_capacity = n_outputs

    def sample_tokens(
        self,
        *,
        logits_tensor: Optional[Tensor],
        output_ids: Optional[Tensor],
        temperatures: Optional[Sequence[float]] = None,
        top_ps: Optional[Sequence[float]] = None,
        top_ks: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        has_seeds: Optional[Sequence[int]] = None,
    ) -> Optional[Tensor]:
        # Stage-1 native sampler currently uses argmax path in C++ and does not
        # consume these controls yet.
        del temperatures, top_ps, top_ks, seeds, has_seeds

        if logits_tensor is None or output_ids is None:
            return None

        shape = output_ids.shape()
        if len(shape) != 1:
            raise RuntimeError("sampler expects output_ids to be 1D")
        n_outputs = int(shape[0])

        self._ensure_sampled_buffer(n_outputs)
        assert self._sampled_ids_buf is not None
        sampled_ids = self._sampled_ids_buf.slice(0, 0, n_outputs)

        sin = SamplerInput()
        sin.logits = logits_tensor.lib_tensor()
        sin.output_ids = output_ids.lib_tensor()
        sin.temperatures = None
        sin.top_ps = None
        sin.top_ks = None
        sin.seeds = None
        sin.has_seeds = None

        sout = SamplerOutput()
        sout.sampled_ids = sampled_ids.lib_tensor()

        status = int(LIB_LLAISYS.llaisysSamplerSample(byref(sin), byref(sout)))
        if status != 0:
            raise RuntimeError(f"samplerSample failed with status={status}")
        return sampled_ids
