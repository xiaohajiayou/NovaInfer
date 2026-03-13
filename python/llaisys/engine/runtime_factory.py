from __future__ import annotations

import ctypes
import os
from ctypes import byref
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..libllaisys import LIB_LLAISYS, DeviceType
from ..libllaisys.model import LlaisysKvStateCreateParams
from ..models import qwen2 as qwen2_impl


@dataclass(frozen=True)
class KvCachePlan:
    max_model_len: int
    kv_cache_capacity_tokens: int


def _available_memory_bytes(device: DeviceType) -> int:
    if device == DeviceType.CPU:
        mem_avail_kb = 0
        mem_total_kb = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem_avail_kb = int(line.split()[1])
                    elif line.startswith("MemTotal:"):
                        mem_total_kb = int(line.split()[1])
        except Exception:
            pass
        kb = mem_avail_kb if mem_avail_kb > 0 else mem_total_kb
        if kb > 0:
            return int(kb) * 1024

        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            if int(ok) != 0:
                avail = int(stat.ullAvailPhys)
                total = int(stat.ullTotalPhys)
                return avail if avail > 0 else total
        except Exception:
            pass
        return 0

    if device == DeviceType.NVIDIA:
        cudart = None
        candidates = [
            find_library("cudart"),
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ]
        for name in candidates:
            if not name:
                continue
            try:
                cudart = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if cudart is None:
            print("[error] runtime_factory: failed to load CUDA runtime (libcudart)")
            return 0

        cuda_set_device = cudart.cudaSetDevice
        cuda_set_device.argtypes = [ctypes.c_int]
        cuda_set_device.restype = ctypes.c_int
        cuda_mem_get_info = cudart.cudaMemGetInfo
        cuda_mem_get_info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        cuda_mem_get_info.restype = ctypes.c_int

        rc = int(cuda_set_device(0))
        if rc != 0:
            print(f"[error] runtime_factory: cudaSetDevice(0) failed, rc={rc}")
            return 0
        free_b = ctypes.c_size_t(0)
        total_b = ctypes.c_size_t(0)
        rc = int(cuda_mem_get_info(ctypes.byref(free_b), ctypes.byref(total_b)))
        if rc != 0:
            print(f"[error] runtime_factory: cudaMemGetInfo failed, rc={rc}")
            return 0
        return int(free_b.value)
    return 0


def _kv_token_bytes(meta) -> int:
    dtype_bytes = int(qwen2_impl._datatype_to_numpy_dtype(meta.dtype).itemsize)
    return int(meta.nlayer) * int(meta.nkvh) * int(meta.dh) * 2 * dtype_bytes


def _estimate_cuda_kv_capacity_tokens(
    *,
    available_bytes: int,
    token_bytes: int,
    block_size: int,
    memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
) -> tuple[int, dict]:
    util = min(0.98, max(0.01, float(memory_utilization)))
    bs = max(1, int(block_size))
    block_bytes = max(1, int(token_bytes) * bs)
    reserve_bytes = max(2 * 1024**3, int(int(available_bytes) * 0.2))
    budget_bytes = int(int(available_bytes) * util) - reserve_bytes
    num_blocks = int(budget_bytes // block_bytes) if budget_bytes > 0 else 0
    capacity_tokens_est = int(num_blocks * bs) if num_blocks > 0 else 0
    logical_cap_tokens = max(1, int(max_model_len) * max(1, int(max_num_seqs)))
    capacity_tokens = min(capacity_tokens_est, logical_cap_tokens)
    probe = {
        "free_bytes": int(available_bytes),
        "budget_bytes": int(budget_bytes),
        "reserve_bytes": int(reserve_bytes),
        "block_bytes": int(block_bytes),
        "num_blocks": int(num_blocks),
        "capacity_tokens_est": int(capacity_tokens_est),
        "capacity_tokens": int(capacity_tokens),
        "logical_cap_tokens": int(logical_cap_tokens),
        "util": float(util),
    }
    return capacity_tokens, probe


def create_kv_state(
    *,
    kv_cache_block_size: int,
    plan: KvCachePlan,
):
    params = LlaisysKvStateCreateParams(
        int(kv_cache_block_size),
        int(plan.max_model_len),
        int(plan.kv_cache_capacity_tokens),
    )
    kv_state = LIB_LLAISYS.llaisysKvStateCreate(byref(params))
    if not kv_state:
        raise RuntimeError("Failed to create kv state")
    return kv_state


def plan_qwen2_kv_cache(
    *,
    model_path: Path | str,
    device: DeviceType,
    kv_cache_block_size: int,
    max_model_len: Optional[int],
    kv_cache_memory_utilization: float,
    max_num_seqs: Optional[int],
) -> KvCachePlan:
    model_path = Path(model_path)
    meta = qwen2_impl._parse_meta(model_path, max_model_len=max_model_len)
    resolved_max_model_len = int(meta.maxseq)
    available_memory_bytes = int(_available_memory_bytes(device))
    if available_memory_bytes <= 0:
        raise RuntimeError(
            "invalid available memory bytes (<= 0): "
            "cannot initialize model with current device memory probe"
        )

    token_bytes = _kv_token_bytes(meta)
    util = min(0.98, max(0.01, float(kv_cache_memory_utilization)))
    if device == DeviceType.NVIDIA:
        auto_max_num_seqs = (
            max(1, int(max_num_seqs))
            if max_num_seqs is not None
            else max(1, int(os.getenv("LLAISYS_KV_AUTO_MAX_SEQS", "8")))
        )
        capacity_tokens, probe = _estimate_cuda_kv_capacity_tokens(
            available_bytes=int(available_memory_bytes),
            token_bytes=token_bytes,
            block_size=int(kv_cache_block_size),
            memory_utilization=util,
            max_model_len=resolved_max_model_len,
            max_num_seqs=auto_max_num_seqs,
        )
        print(
            "[kv] probe "
            f"device={int(device)} free_bytes={probe['free_bytes']} budget_bytes={probe['budget_bytes']} "
            f"reserve_bytes={probe['reserve_bytes']} token_bytes={token_bytes} block_size={int(kv_cache_block_size)} "
            f"util={probe['util']:.2f} max_model_len={resolved_max_model_len} auto_max_num_seqs={auto_max_num_seqs}"
        )
        if capacity_tokens <= 0:
            raise RuntimeError("estimated num_kvcache_blocks <= 0")
        resolved_kv_capacity_tokens = int(capacity_tokens)
        print(
            "[kv] capacity auto "
            f"block_bytes={probe['block_bytes']} blocks={probe['num_blocks']} "
            f"capacity_tokens_est={probe['capacity_tokens_est']} "
            f"logical_cap_tokens={probe['logical_cap_tokens']} "
            f"capacity_tokens={resolved_kv_capacity_tokens}"
        )
    else:
        bs = max(1, int(kv_cache_block_size))
        block_bytes = max(1, int(token_bytes) * bs)
        num_blocks = int((int(available_memory_bytes) * util) // block_bytes)
        capacity_tokens = max(1, int(num_blocks) * bs)
        print(
            "[kv] probe "
            f"device={int(device)} available_bytes={available_memory_bytes} "
            f"token_bytes={token_bytes} block_size={int(kv_cache_block_size)} util={util:.2f}"
        )
        if num_blocks <= 0:
            raise RuntimeError("estimated num_kvcache_blocks <= 0")
        resolved_kv_capacity_tokens = int(capacity_tokens)
        print(
            "[kv] capacity auto "
            f"block_bytes={block_bytes} blocks={num_blocks} "
            f"capacity_tokens={resolved_kv_capacity_tokens}"
        )

    return KvCachePlan(
        max_model_len=int(resolved_max_model_len),
        kv_cache_capacity_tokens=int(resolved_kv_capacity_tokens),
    )
