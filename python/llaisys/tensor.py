from typing import Sequence, Tuple

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    MemcpyKind,
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
    _runtime_api_cache = {}
    _async_stream_cache = {}

    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
    ):
        self._pending_api = None
        self._pending_stream = None
        self._pending_device_id = None
        self._parent = None
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
            try:
                self.wait()
            except Exception:
                pass
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
        self.wait()
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
        out = Tensor(
            tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )
        out._parent = self
        return out

    def permute(self, *perm: int) -> llaisysTensor_t:
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        out = Tensor(tensor=LIB_LLAISYS.tensorPermute(self._tensor, _perm))
        out._parent = self
        return out

    def slice(self, dim: int, start: int, end: int):
        out = Tensor(
            tensor=LIB_LLAISYS.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )
        out._parent = self
        return out

    def contiguous(self):
        out = Tensor(tensor=LIB_LLAISYS.tensorContiguous(self._tensor))
        out._parent = self
        return out

    def reshape(self, *shape: int):
        _shape = (c_size_t * len(shape))(*shape)
        out = Tensor(
            tensor=LIB_LLAISYS.tensorReshape(self._tensor, _shape, c_size_t(len(shape)))
        )
        out._parent = self
        return out

    def numel(self) -> int:
        n = 1
        for d in self.shape():
            n *= int(d)
        return int(n)

    def nbytes(self) -> int:
        dtype = self.dtype()
        if dtype in (DataType.BOOL, DataType.BYTE, DataType.I8, DataType.U8):
            return self.numel()
        if dtype in (DataType.I16, DataType.U16, DataType.F16, DataType.BF16):
            return self.numel() * 2
        if dtype in (DataType.I32, DataType.U32, DataType.F32):
            return self.numel() * 4
        if dtype in (DataType.I64, DataType.U64, DataType.F64):
            return self.numel() * 8
        raise RuntimeError(f"nbytes() unsupported dtype: {dtype}")

    @classmethod
    def _runtime_api(cls, device_type: DeviceType):
        api = cls._runtime_api_cache.get(device_type)
        if api is not None:
            return api
        from .runtime import RuntimeAPI

        api = RuntimeAPI(device_type)
        cls._runtime_api_cache[device_type] = api
        return api

    @classmethod
    def _async_stream(cls, device_id: int):
        key = int(device_id)
        stream = cls._async_stream_cache.get(key)
        if stream is not None:
            return stream
        api = cls._runtime_api(DeviceType.NVIDIA)
        api.set_device(key)
        stream = api.create_stream()
        cls._async_stream_cache[key] = stream
        return stream

    @staticmethod
    def _normalize_device(device, current_device: DeviceType, current_device_id: int) -> tuple[DeviceType, int]:
        if device is None:
            return current_device, current_device_id
        if isinstance(device, DeviceType):
            if device == current_device:
                return device, current_device_id
            return device, 0
        if isinstance(device, str):
            raw = device.strip().lower()
            if raw.startswith("cuda"):
                raw = raw.replace("cuda", "nvidia", 1)
            if ":" in raw:
                name, idx = raw.split(":", 1)
                if name not in ("cpu", "nvidia"):
                    raise RuntimeError(f"unsupported device string: {device}")
                did = int(idx)
                return (DeviceType.CPU if name == "cpu" else DeviceType.NVIDIA), did
            if raw == "cpu":
                return DeviceType.CPU, 0
            if raw == "nvidia":
                if current_device == DeviceType.NVIDIA:
                    return DeviceType.NVIDIA, current_device_id
                return DeviceType.NVIDIA, 0
            raise RuntimeError(f"unsupported device string: {device}")
        raise RuntimeError(f"unsupported device type: {type(device)}")

    @staticmethod
    def _normalize_dtype(dtype, current_dtype: DataType) -> DataType:
        if dtype is None:
            return current_dtype
        if isinstance(dtype, DataType):
            return dtype
        if isinstance(dtype, str):
            raw = dtype.strip().lower()
            mapping = {
                "float32": DataType.F32,
                "f32": DataType.F32,
                "float16": DataType.F16,
                "f16": DataType.F16,
                "bfloat16": DataType.BF16,
                "bf16": DataType.BF16,
                "int64": DataType.I64,
                "i64": DataType.I64,
                "int32": DataType.I32,
                "i32": DataType.I32,
                "int8": DataType.I8,
                "i8": DataType.I8,
                "float64": DataType.F64,
                "f64": DataType.F64,
            }
            if raw in mapping:
                return mapping[raw]
        raise RuntimeError(f"unsupported dtype spec: {dtype}")

    @staticmethod
    def _looks_like_device_string(raw: str) -> bool:
        v = raw.strip().lower()
        if v.startswith("cuda"):
            v = v.replace("cuda", "nvidia", 1)
        if v in ("cpu", "nvidia"):
            return True
        if ":" in v:
            head, tail = v.split(":", 1)
            return head in ("cpu", "nvidia") and tail.isdigit()
        return False

    @staticmethod
    def _infer_memcpy_kind(dst: DeviceType, src: DeviceType) -> MemcpyKind:
        if dst == DeviceType.CPU and src == DeviceType.CPU:
            return MemcpyKind.H2H
        if dst == DeviceType.NVIDIA and src == DeviceType.CPU:
            return MemcpyKind.H2D
        if dst == DeviceType.CPU and src == DeviceType.NVIDIA:
            return MemcpyKind.D2H
        if dst == DeviceType.NVIDIA and src == DeviceType.NVIDIA:
            return MemcpyKind.D2D
        raise RuntimeError(f"unsupported memcpy route: {src} -> {dst}")

    @classmethod
    def _copy_tensor_bytes(cls, dst: "Tensor", src: "Tensor", non_blocking: bool) -> None:
        dst.wait()
        src.wait()
        kind = cls._infer_memcpy_kind(dst.device_type(), src.device_type())
        if dst.device_type() == DeviceType.NVIDIA or src.device_type() == DeviceType.NVIDIA:
            api = cls._runtime_api(DeviceType.NVIDIA)
            dev_id = dst.device_id() if dst.device_type() == DeviceType.NVIDIA else src.device_id()
            api.set_device(dev_id)
            if non_blocking:
                stream = cls._async_stream(dev_id)
                api.memcpy_async(dst.data_ptr(), src.data_ptr(), src.nbytes(), kind, stream)
                dst._set_pending(api, stream, dev_id)
                src._set_pending(api, stream, dev_id)
                return
        else:
            api = cls._runtime_api(DeviceType.CPU)
        api.memcpy_sync(dst.data_ptr(), src.data_ptr(), src.nbytes(), kind)

    def _set_pending(self, api, stream, device_id=None) -> None:
        t = self
        while t is not None:
            t._pending_api = api
            t._pending_stream = stream
            t._pending_device_id = device_id
            t = t._parent

    def wait(self) -> "Tensor":
        if self._parent is not None:
            self._parent.wait()
        if self._pending_api is not None and self._pending_stream is not None:
            if self._pending_device_id is not None:
                self._pending_api.set_device(int(self._pending_device_id))
            self._pending_api.stream_synchronize(self._pending_stream)
            self._pending_api = None
            self._pending_stream = None
            self._pending_device_id = None
        return self

    def clone(self) -> "Tensor":
        out = Tensor(self.shape(), self.dtype(), self.device_type(), self.device_id())
        out.copy_(self)
        return out

    def copy_(self, src: "Tensor", non_blocking: bool = False) -> "Tensor":
        if not isinstance(src, Tensor):
            raise RuntimeError("copy_ expects a Tensor source")
        if self.shape() != src.shape():
            raise RuntimeError("copy_ requires same shape")
        if self.dtype() != src.dtype():
            raise RuntimeError("copy_ requires same dtype")
        self._copy_tensor_bytes(self, src, non_blocking)
        return self

    def to(
        self,
        device: DeviceType | DataType | str | None = None,
        dtype: DataType | str | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> "Tensor":
        if isinstance(dtype, int):
            raise RuntimeError("Tensor.to(device_id) is not supported. Use device='nvidia:<id>' or device=DeviceType.NVIDIA.")

        device_arg = device
        dtype_arg = dtype
        if isinstance(device, DataType):
            if dtype is not None:
                raise RuntimeError("Tensor.to() got both dtype in first arg and dtype keyword")
            device_arg = None
            dtype_arg = device
        elif isinstance(device, str) and (not self._looks_like_device_string(device)):
            if dtype is not None:
                raise RuntimeError("Tensor.to() got ambiguous positional arg and dtype keyword")
            device_arg = None
            dtype_arg = device

        cur_dev = self.device_type()
        cur_dev_id = self.device_id()
        cur_dtype = self.dtype()

        dst_dev, dst_dev_id = self._normalize_device(device_arg, cur_dev, cur_dev_id)
        dst_dtype = self._normalize_dtype(dtype_arg, cur_dtype)

        if dst_dtype != cur_dtype:
            raise RuntimeError("dtype conversion is not implemented yet in llaisys.Tensor.to")

        if dst_dev == cur_dev and dst_dev_id == cur_dev_id:
            return self.clone() if copy else self

        out = Tensor(self.shape(), dst_dtype, dst_dev, dst_dev_id)
        self._copy_tensor_bytes(out, self, non_blocking)
        return out

    def tolist(self):
        if self.device_type() != DeviceType.CPU:
            raise RuntimeError("tolist() requires CPU tensor; call to(device=DeviceType.CPU) first")
        self.wait()
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
