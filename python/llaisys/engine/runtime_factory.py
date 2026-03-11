from __future__ import annotations

import ctypes
import os
from ctypes import byref, c_char_p, c_int
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from typing import Optional

from ..libllaisys import LIB_LLAISYS, DeviceType
from ..libllaisys.model import KvCacheLayout, LlaisysParallelInitParams, LlaisysRuntimeCreateParams
from ..models import qwen2 as qwen2_impl


@dataclass(frozen=True)
class RuntimePlan:
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
        cudart = _load_cudart()
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


def _load_cudart() -> ctypes.CDLL | None:
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
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _cuda_visible_device_count(cudart: ctypes.CDLL) -> int:
    fn = cudart.cudaGetDeviceCount
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    n = ctypes.c_int(0)
    rc = int(fn(ctypes.byref(n)))
    if rc != 0:
        raise RuntimeError(f"cudaGetDeviceCount failed, rc={rc}")
    return int(n.value)


def _cuda_set_device(cudart: ctypes.CDLL, device_id: int) -> None:
    fn = cudart.cudaSetDevice
    fn.argtypes = [ctypes.c_int]
    fn.restype = ctypes.c_int
    rc = int(fn(int(device_id)))
    if rc != 0:
        raise RuntimeError(f"cudaSetDevice({device_id}) failed, rc={rc}")


def _cuda_mem_free_bytes(cudart: ctypes.CDLL, device_id: int) -> int:
    _cuda_set_device(cudart, int(device_id))
    fn = cudart.cudaMemGetInfo
    fn.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    fn.restype = ctypes.c_int
    free_b = ctypes.c_size_t(0)
    total_b = ctypes.c_size_t(0)
    rc = int(fn(ctypes.byref(free_b), ctypes.byref(total_b)))
    if rc != 0:
        raise RuntimeError(f"cudaMemGetInfo(device={device_id}) failed, rc={rc}")
    return int(free_b.value)


def _cuda_p2p_access(cudart: ctypes.CDLL, dev_a: int, dev_b: int) -> bool:
    fn = cudart.cudaDeviceCanAccessPeer
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    out = ctypes.c_int(0)
    rc = int(fn(ctypes.byref(out), int(dev_a), int(dev_b)))
    if rc != 0:
        return False
    return int(out.value) != 0


def select_tp_device_ids(tp_size: int, explicit_ids: tuple[int, ...] | None) -> tuple[int, ...]:
    tp = max(1, int(tp_size))
    if explicit_ids is not None:
        ids = tuple(int(v) for v in explicit_ids)
        if len(ids) != tp:
            raise RuntimeError(f"tensor_parallel_device_ids length mismatch: tp_size={tp} ndevice={len(ids)}")
        if len(set(ids)) != len(ids):
            raise RuntimeError("tensor_parallel_device_ids contains duplicates")
        if any(v < 0 for v in ids):
            raise RuntimeError("tensor_parallel_device_ids contains negative id")
        return ids

    if tp == 1:
        return (0,)

    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("failed to load CUDA runtime (libcudart) for TP auto device selection")

    ndev = _cuda_visible_device_count(cudart)
    if tp > ndev:
        raise RuntimeError(f"not enough visible GPUs for TP: tp_size={tp} visible={ndev}")

    candidates = tuple(range(ndev))
    free_bytes = {d: _cuda_mem_free_bytes(cudart, d) for d in candidates}

    best: tuple[int, ...] | None = None
    best_score: tuple[int, int] | None = None
    for combo in combinations(candidates, tp):
        p2p_ok = 1
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                a = int(combo[i])
                b = int(combo[j])
                if not (_cuda_p2p_access(cudart, a, b) and _cuda_p2p_access(cudart, b, a)):
                    p2p_ok = 0
                    break
            if p2p_ok == 0:
                break
        score = (p2p_ok, int(sum(free_bytes[int(v)] for v in combo)))
        if best is None or score > best_score or (score == best_score and combo < best):
            best = combo
            best_score = score

    if best is None:
        raise RuntimeError("failed to select TP device ids")

    p2p_matrix = [[int(_cuda_p2p_access(cudart, a, b)) for b in candidates] for a in candidates]
    print(
        "[tp] auto_select "
        f"tp_size={tp} candidates={list(candidates)} selected={list(best)} "
        f"free_bytes={ {k: int(v) for k, v in free_bytes.items()} } p2p_matrix={p2p_matrix}"
    )
    return tuple(int(v) for v in best)


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


def create_runtime(
    *,
    device: DeviceType,
    kv_cache_layout: KvCacheLayout,
    kv_cache_block_size: int,
    plan: RuntimePlan,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    distributed_executor_backend: str = "uni",
    distributed_backend: str = "nccl",
    tensor_parallel_device_ids: tuple[int, ...] | None = None,
    tp_rank: int = 0,
    tp_local_rank: int = 0,
):
    params = LlaisysRuntimeCreateParams(
        int(kv_cache_layout),
        int(kv_cache_block_size),
        int(plan.max_model_len),
        int(plan.kv_cache_capacity_tokens),
    )
    runtime = LIB_LLAISYS.llaisysRuntimeCreate(byref(params))
    if not runtime:
        raise RuntimeError("Failed to create runtime state")

    tp_size = max(1, int(tensor_parallel_size))
    pp_size = max(1, int(pipeline_parallel_size))
    rank = max(0, int(tp_rank))
    local_rank = max(0, int(tp_local_rank))
    if rank >= tp_size:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise RuntimeError(f"tp_rank out of range: rank={rank} tp_size={tp_size}")
    exec_backend = str(distributed_executor_backend or "uni").strip().lower()
    dist_backend = str(distributed_backend or "nccl").strip().lower()

    if tp_size > 1 and device != DeviceType.NVIDIA:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise RuntimeError("tensor parallel currently supports NVIDIA device only")
    if exec_backend != "uni":
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise NotImplementedError(f"distributed_executor_backend={exec_backend} is not supported yet")

    try:
        if device == DeviceType.NVIDIA:
            selected_ids = select_tp_device_ids(tp_size, tensor_parallel_device_ids)
        else:
            if tensor_parallel_device_ids is not None and len(tuple(tensor_parallel_device_ids)) > 0:
                raise RuntimeError("tensor_parallel_device_ids is only valid on NVIDIA device")
            selected_ids = tuple()
    except Exception:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise

    if device == DeviceType.NVIDIA and len(selected_ids) != tp_size:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise RuntimeError(f"parallel device ids mismatch: tp_size={tp_size} selected={len(selected_ids)}")

    ids_arr = (c_int * len(selected_ids))(*[int(v) for v in selected_ids])
    dist_exec_b = str(exec_backend).encode("utf-8")
    dist_b = str(dist_backend).encode("utf-8")
    master_addr_b = str(os.getenv("LLAISYS_TP_MASTER_ADDR", "")).encode("utf-8")
    master_port = int(os.getenv("LLAISYS_TP_MASTER_PORT", "0"))
    init_method = str(os.getenv("LLAISYS_TP_INIT_METHOD", "")).strip()
    if tp_size > 1 and not init_method:
        init_method = "file:///tmp/llaisys_tp_nccl.id"
    init_method_b = init_method.encode("utf-8")
    tp_group_name_b = b"tp"
    single_process_tp = int(os.getenv("LLAISYS_TP_SINGLE_PROCESS", "0"))
    par = LlaisysParallelInitParams(
        int(tp_size),
        int(pp_size),
        int(tp_size * pp_size),
        int(rank),
        int(local_rank),
        c_char_p(dist_exec_b),
        c_char_p(dist_b),
        c_char_p(master_addr_b),
        int(master_port),
        int(0),
        int(1),
        c_char_p(init_method_b),
        c_char_p(tp_group_name_b),
        int(single_process_tp),
        ids_arr,
        int(len(selected_ids)),
    )
    rc = int(LIB_LLAISYS.llaisysRuntimeParallelInit(runtime, byref(par)))
    if rc != 0:
        LIB_LLAISYS.llaisysRuntimeDestroy(runtime)
        raise RuntimeError(
            "llaisysRuntimeParallelInit failed "
            f"(rc={rc}, tp_size={tp_size}, rank={rank}, local_rank={local_rank}, device_ids={list(selected_ids)})"
        )
    return runtime


def plan_qwen2_runtime(
    *,
    model_path: Path | str,
    device: DeviceType,
    kv_cache_block_size: int,
    max_model_len: Optional[int],
    kv_cache_memory_utilization: float,
    max_num_seqs: Optional[int],
) -> RuntimePlan:
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

    return RuntimePlan(
        max_model_len=int(resolved_max_model_len),
        kv_cache_capacity_tokens=int(resolved_kv_capacity_tokens),
    )
