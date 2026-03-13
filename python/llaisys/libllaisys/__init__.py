import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .llaisys_types import llaisysEvent_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops
from .qwen2 import load_qwen2
from .model import load_model


def _preload_cudnn_from_env() -> None:
    if not sys.platform.startswith("linux"):
        return
    cudnn_home = str(os.getenv("CUDNN_HOME", "")).strip()
    if not cudnn_home:
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    base = Path(cudnn_home)
    candidates = (
        base / "lib" / "libcudnn.so.9",
        base / "lib" / "libcudnn.so",
        base / "lib64" / "libcudnn.so.9",
        base / "lib64" / "libcudnn.so",
    )
    for cand in candidates:
        if not cand.is_file():
            continue
        ctypes.CDLL(str(cand), mode=mode)
        return
    raise FileNotFoundError(f"libcudnn.so not found under CUDNN_HOME={cudnn_home}")


def _preload_nccl_from_python_env() -> None:
    if not sys.platform.startswith("linux"):
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for entry in sys.path:
        try:
            base = Path(entry)
        except Exception:
            continue
        if not base.exists():
            continue
        candidates = (
            base / "nvidia" / "nccl" / "lib" / "libnccl.so.2",
            base / "nvidia" / "nccl" / "lib" / "libnccl.so",
        )
        for cand in candidates:
            if not cand.is_file():
                continue
            try:
                ctypes.CDLL(str(cand), mode=mode)
                return
            except OSError:
                continue


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "libllaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    _preload_cudnn_from_env()
    _preload_nccl_from_python_env()
    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2(LIB_LLAISYS)
load_model(LIB_LLAISYS)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysEvent_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
    "llaisysEvent_t",
]
