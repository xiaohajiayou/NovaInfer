from typing import Sequence, Tuple

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
)
from ctypes import (
    POINTER,
    c_double,
    c_float,
    c_int,
    c_int8,
    c_int32,
    c_int64,
    c_size_t,
    c_ssize_t,
    c_void_p,
    cast,
)


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
    ):
        if tensor:
            self._tensor = tensor
        else:
            _ndim = 0 if shape is None else len(shape)
            _shape = None if shape is None else (c_size_t * len(shape))(*shape)
            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype),
                llaisysDeviceType_t(device),
                c_int(device_id),
            )

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_LLAISYS.tensorDestroy(self._tensor)
            self._tensor = None

    def shape(self) -> Tuple[int]:
        buf = (c_size_t * self.ndim())()
        LIB_LLAISYS.tensorGetShape(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def strides(self) -> Tuple[int]:
        buf = (c_ssize_t * self.ndim())()
        LIB_LLAISYS.tensorGetStrides(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def ndim(self) -> int:
        return int(LIB_LLAISYS.tensorGetNdim(self._tensor))

    def dtype(self) -> DataType:
        return DataType(LIB_LLAISYS.tensorGetDataType(self._tensor))

    def device_type(self) -> DeviceType:
        return DeviceType(LIB_LLAISYS.tensorGetDeviceType(self._tensor))

    def device_id(self) -> int:
        return int(LIB_LLAISYS.tensorGetDeviceId(self._tensor))

    def data_ptr(self) -> c_void_p:
        return LIB_LLAISYS.tensorGetData(self._tensor)

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type}:{self.device_id}>"

    def load(self, data: c_void_p):
        LIB_LLAISYS.tensorLoad(self._tensor, data)

    def copy_from_sequence(self, values: Sequence[int | float]) -> None:
        n = len(values)
        if n <= 0:
            return
        dtype = self.dtype()
        if dtype == DataType.I64:
            buf = (c_int64 * n)(*[int(x) for x in values])
        elif dtype == DataType.I32:
            buf = (c_int32 * n)(*[int(x) for x in values])
        elif dtype == DataType.I8:
            buf = (c_int8 * n)(*[int(x) for x in values])
        elif dtype == DataType.F32:
            buf = (c_float * n)(*[float(x) for x in values])
        elif dtype == DataType.F64:
            buf = (c_double * n)(*[float(x) for x in values])
        else:
            raise RuntimeError(f"copy_from_sequence() unsupported dtype: {dtype}")
        self.load(cast(buf, c_void_p))

    def is_contiguous(self) -> bool:
        return bool(LIB_LLAISYS.tensorIsContiguous(self._tensor))

    def view(self, *shape: int) -> llaisysTensor_t:
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )

    def permute(self, *perm: int) -> llaisysTensor_t:
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        return Tensor(tensor=LIB_LLAISYS.tensorPermute(self._tensor, _perm))

    def slice(self, dim: int, start: int, end: int):
        return Tensor(
            tensor=LIB_LLAISYS.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )

    def contiguous(self):
        return Tensor(tensor=LIB_LLAISYS.tensorContiguous(self._tensor))

    def reshape(self, *shape: int):
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorReshape(self._tensor, _shape, c_size_t(len(shape)))
        )

    def to(self, device: DeviceType, device_id: int = -1):
        return Tensor(
            tensor=LIB_LLAISYS.tensorTo(
                self._tensor, llaisysDeviceType_t(device), c_int(device_id)
            )
        )

    def tolist(self):
        if self.device_type() != DeviceType.CPU:
            raise RuntimeError("tolist() requires CPU tensor; call to(DeviceType.CPU) first")
        shape = self.shape()
        if len(shape) != 1:
            raise RuntimeError("tolist() currently supports only 1D tensors")
        n = int(shape[0])
        if n <= 0:
            return []

        dtype = self.dtype()
        if dtype == DataType.I64:
            ptr = cast(self.data_ptr(), POINTER(c_int64))
            return [int(ptr[i]) for i in range(n)]
        if dtype == DataType.I32:
            ptr = cast(self.data_ptr(), POINTER(c_int32))
            return [int(ptr[i]) for i in range(n)]
        if dtype == DataType.I8:
            ptr = cast(self.data_ptr(), POINTER(c_int8))
            return [int(ptr[i]) for i in range(n)]
        if dtype == DataType.F32:
            ptr = cast(self.data_ptr(), POINTER(c_float))
            return [float(ptr[i]) for i in range(n)]
        if dtype == DataType.F64:
            ptr = cast(self.data_ptr(), POINTER(c_double))
            return [float(ptr[i]) for i in range(n)]
        raise RuntimeError(f"tolist() unsupported dtype: {dtype}")
