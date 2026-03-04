from __future__ import annotations

from ..libllaisys import DataType, DeviceType
from ..tensor import Tensor


class CpuGpuBuffer:
    """Paired CPU/GPU tensors with reusable non-blocking copy helpers."""

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        dtype: DataType,
        device: DeviceType,
        device_id: int = 0,
        pin_memory: bool = True,
    ) -> None:
        if device == DeviceType.CPU:
            raise ValueError("CpuGpuBuffer requires a non-CPU target device")
        self.cpu = Tensor(tuple(shape), dtype, DeviceType.CPU, 0, pin_memory=pin_memory)
        self.gpu = Tensor(tuple(shape), dtype, device, device_id)

    def copy_to_gpu(self, n: int | None = None, non_blocking: bool = True) -> Tensor:
        if n is None:
            self.gpu.copy_(self.cpu, non_blocking=non_blocking)
            return self.gpu
        dst = self.gpu.slice(0, 0, int(n))
        src = self.cpu.slice(0, 0, int(n))
        dst.copy_(src, non_blocking=non_blocking)
        return dst

    def copy_to_cpu(self, n: int | None = None, non_blocking: bool = True) -> Tensor:
        if n is None:
            self.cpu.copy_(self.gpu, non_blocking=non_blocking)
            return self.cpu
        dst = self.cpu.slice(0, 0, int(n))
        src = self.gpu.slice(0, 0, int(n))
        dst.copy_(src, non_blocking=non_blocking)
        return dst
