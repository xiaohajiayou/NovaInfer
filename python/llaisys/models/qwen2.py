from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import json
import os
import re
import ctypes
from ctypes.util import find_library

import numpy as np
import safetensors
import torch

from ctypes import POINTER, byref, cast, c_int, c_int32, c_void_p

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.model import (
    KvCacheLayout,
    LlaisysModelCreateParams,
    LlaisysRuntimeCreateParams,
    ModelForwardInput,
    ModelForwardOutput,
    ModelType,
)
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor


_LAYER_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"), "attn_norm_w"),
    # Qwen-style attention.* names.
    (re.compile(r"^model\.layers\.(\d+)\.attention\.wq\.weight$"), "attn_q_w"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.wq\.bias$"), "attn_q_b"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.wk\.weight$"), "attn_k_w"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.wk\.bias$"), "attn_k_b"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.wv\.weight$"), "attn_v_w"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.wv\.bias$"), "attn_v_b"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.wo\.weight$"), "attn_o_w"),
    # HF/Qwen2-style self_attn.*_proj names (observed in DeepSeek-R1-Distill-Qwen-1.5B).
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"), "attn_q_w"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$"), "attn_q_b"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$"), "attn_k_w"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$"), "attn_k_b"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$"), "attn_v_w"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$"), "attn_v_b"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"), "attn_o_w"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"), "mlp_norm_w"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"), "mlp_gate_w"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"), "mlp_up_w"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"), "mlp_down_w"),
)

_GLOBAL_NAMES: Dict[str, str] = {
    "model.embed_tokens.weight": "in_embed",
    "lm_head.weight": "out_embed",
    "model.norm.weight": "out_norm_w",
}


def _maybe_bfloat16_dtype() -> Optional[np.dtype]:
    try:
        import ml_dtypes  # type: ignore

        return np.dtype(ml_dtypes.bfloat16)
    except Exception:
        return None


_BF16_DTYPE = _maybe_bfloat16_dtype()


def _torch_dtype_to_datatype(torch_dtype: Optional[str]) -> DataType:
    if torch_dtype is None:
        return DataType.F32
    torch_dtype = torch_dtype.lower()
    if "bfloat16" in torch_dtype or torch_dtype == "bf16":
        return DataType.BF16 if _BF16_DTYPE is not None else DataType.F32
    if "float16" in torch_dtype or torch_dtype == "fp16" or torch_dtype == "f16":
        return DataType.F16
    if "float32" in torch_dtype or torch_dtype == "fp32" or torch_dtype == "f32":
        return DataType.F32
    return DataType.F32


def _datatype_to_numpy_dtype(dtype: DataType) -> np.dtype:
    if dtype == DataType.F32:
        return np.dtype(np.float32)
    if dtype == DataType.F16:
        return np.dtype(np.float16)
    if dtype == DataType.BF16 and _BF16_DTYPE is not None:
        return _BF16_DTYPE
    # Fallback: use float32 even if meta says BF16 but runtime support is missing.
    return np.dtype(np.float32)


def _as_contiguous(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _detach_tensor_handle(tensor: Tensor) -> c_void_p:
    """Transfer ownership to the backend by detaching the handle from Tensor.__del__."""
    handle = tensor.lib_tensor()
    # The backend takes ownership. Prevent Python-side double free.
    tensor._tensor = None  # type: ignore[attr-defined]
    return handle


@dataclass(frozen=True)
class _MetaInfo:
    dtype: DataType
    nlayer: int
    hs: int
    nh: int
    nkvh: int
    dh: int
    di: int
    maxseq: int
    voc: int
    epsilon: float
    theta: float
    end_token: int


def _read_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_meta(model_path: Path, max_model_len: Optional[int] = None) -> _MetaInfo:
    cfg = _read_config(model_path)

    dtype = _torch_dtype_to_datatype(cfg.get("torch_dtype"))

    hs = int(cfg["hidden_size"])
    nh = int(cfg["num_attention_heads"])
    nkvh = int(cfg.get("num_key_value_heads", nh))
    dh = hs // nh

    eos = cfg.get("eos_token_id", cfg.get("end_token_id", 0))
    if isinstance(eos, Iterable) and not isinstance(eos, (str, bytes)):
        eos_list = list(eos)
        end_token = int(eos_list[0]) if eos_list else 0
    else:
        end_token = int(eos)

    theta = float(cfg.get("rope_theta", 10000.0))

    cfg_maxseq = int(cfg["max_position_embeddings"])
    # KV-cache memory grows linearly with maxseq and can easily reach multiple GB.
    # Cap it by default to keep the stage-1 implementation stable on typical machines.
    cap_maxseq = int(max_model_len) if max_model_len is not None else int(os.getenv("LLAISYS_MAXSEQ", "4096"))
    maxseq = min(cfg_maxseq, cap_maxseq)

    return _MetaInfo(
        dtype=dtype,
        nlayer=int(cfg["num_hidden_layers"]),
        hs=hs,
        nh=nh,
        nkvh=nkvh,
        dh=dh,
        di=int(cfg["intermediate_size"]),
        maxseq=maxseq,
        voc=int(cfg["vocab_size"]),
        epsilon=float(cfg.get("rms_norm_eps", 1e-6)),
        theta=theta,
        end_token=end_token,
    )


def _build_meta_struct(meta: _MetaInfo) -> LlaisysQwen2Meta:
    return LlaisysQwen2Meta(
        meta.dtype,
        meta.nlayer,
        meta.hs,
        meta.nh,
        meta.nkvh,
        meta.dh,
        meta.di,
        meta.maxseq,
        meta.voc,
        meta.epsilon,
        meta.theta,
        meta.end_token,
    )


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

        # Windows fallback: GlobalMemoryStatusEx.
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
            print("[error] qwen2: failed to load CUDA runtime (libcudart)")
            return 0

        cuda_set_device = cudart.cudaSetDevice
        cuda_set_device.argtypes = [ctypes.c_int]
        cuda_set_device.restype = ctypes.c_int

        cuda_mem_get_info = cudart.cudaMemGetInfo
        cuda_mem_get_info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        cuda_mem_get_info.restype = ctypes.c_int

        rc = int(cuda_set_device(0))
        if rc != 0:
            print(f"[error] qwen2: cudaSetDevice(0) failed, rc={rc}")
            return 0

        free_b = ctypes.c_size_t(0)
        total_b = ctypes.c_size_t(0)
        rc = int(cuda_mem_get_info(ctypes.byref(free_b), ctypes.byref(total_b)))
        if rc != 0:
            print(f"[error] qwen2: cudaMemGetInfo failed, rc={rc}")
            return 0
        return int(free_b.value)
    return 0


def _kv_token_bytes(meta: _MetaInfo) -> int:
    dtype = _datatype_to_numpy_dtype(meta.dtype)
    dtype_bytes = int(dtype.itemsize)
    # bytes per token over all layers for K and V
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
    # Conservative reserve to avoid runtime OOM from non-KV allocations.
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


def _device_ids(device_id: int = 0):
    arr = (c_int * 1)(device_id)
    return arr, 1


class Qwen2:
    """Qwen2 model wrapper backed by the LLAISYS C++ runtime."""

    def __init__(
        self,
        model_path: Path | str,
        device: DeviceType = DeviceType.CPU,
        kv_cache_layout: KvCacheLayout = KvCacheLayout.BLOCK,
        kv_cache_block_size: int = 16,
        max_model_len: Optional[int] = None,
        kv_cache_capacity_tokens: Optional[int] = None,
        kv_cache_auto_capacity: bool = False,
        kv_cache_memory_utilization: float = 0.9,
        max_num_seqs: Optional[int] = None,
    ):
        self._model_path = Path(model_path)
        self._device = device
        self._kv_cache_layout = kv_cache_layout
        self._model = None
        self._runtime = None
        self._available_memory_bytes = int(_available_memory_bytes(device))
        if self._available_memory_bytes <= 0:
            raise RuntimeError(
                "invalid available memory bytes (<= 0): "
                "cannot initialize model with current device memory probe"
            )

        meta = _parse_meta(self._model_path, max_model_len=max_model_len)
        self._meta_info = meta
        self._meta_struct = _build_meta_struct(meta)
        self._max_model_len = int(meta.maxseq)
        token_bytes = _kv_token_bytes(meta)
        util = min(0.98, max(0.01, float(kv_cache_memory_utilization)))
        if kv_cache_capacity_tokens is not None:
            self._kv_cache_capacity_tokens = max(1, int(kv_cache_capacity_tokens))
            explicit_blocks = (self._kv_cache_capacity_tokens + int(kv_cache_block_size) - 1) // int(kv_cache_block_size)
            print(
                "[kv] probe "
                f"device={int(device)} available_bytes={self._available_memory_bytes} "
                f"token_bytes={token_bytes} block_size={int(kv_cache_block_size)} util={util:.2f}"
            )
            print(
                "[kv] capacity explicit "
                f"capacity_tokens={self._kv_cache_capacity_tokens} num_blocks={explicit_blocks}"
            )
        elif kv_cache_auto_capacity:
            if device == DeviceType.NVIDIA:
                auto_max_num_seqs = (
                    max(1, int(max_num_seqs))
                    if max_num_seqs is not None
                    else max(1, int(os.getenv("LLAISYS_KV_AUTO_MAX_SEQS", "8")))
                )
                capacity_tokens, probe = _estimate_cuda_kv_capacity_tokens(
                    available_bytes=int(self._available_memory_bytes),
                    token_bytes=token_bytes,
                    block_size=int(kv_cache_block_size),
                    memory_utilization=util,
                    max_model_len=int(self._max_model_len),
                    max_num_seqs=auto_max_num_seqs,
                )
                print(
                    "[kv] probe "
                    f"device={int(device)} free_bytes={probe['free_bytes']} budget_bytes={probe['budget_bytes']} "
                    f"reserve_bytes={probe['reserve_bytes']} token_bytes={token_bytes} block_size={int(kv_cache_block_size)} "
                    f"util={probe['util']:.2f} max_model_len={self._max_model_len} auto_max_num_seqs={auto_max_num_seqs}"
                )
                if capacity_tokens <= 0:
                    raise RuntimeError("estimated num_kvcache_blocks <= 0")
                self._kv_cache_capacity_tokens = int(capacity_tokens)
                print(
                    "[kv] capacity auto "
                    f"block_bytes={probe['block_bytes']} blocks={probe['num_blocks']} "
                    f"capacity_tokens_est={probe['capacity_tokens_est']} "
                    f"logical_cap_tokens={probe['logical_cap_tokens']} "
                    f"capacity_tokens={self._kv_cache_capacity_tokens}"
                )
            else:
                bs = max(1, int(kv_cache_block_size))
                block_bytes = max(1, int(token_bytes) * bs)
                num_blocks = int((int(self._available_memory_bytes) * util) // block_bytes)
                capacity_tokens = max(1, int(num_blocks) * bs)
                print(
                    "[kv] probe "
                    f"device={int(device)} available_bytes={self._available_memory_bytes} "
                    f"token_bytes={token_bytes} block_size={int(kv_cache_block_size)} util={util:.2f}"
                )
                if num_blocks <= 0:
                    raise RuntimeError("estimated num_kvcache_blocks <= 0")
                self._kv_cache_capacity_tokens = int(capacity_tokens)
                print(
                    "[kv] capacity auto "
                    f"block_bytes={block_bytes} blocks={num_blocks} "
                    f"capacity_tokens={self._kv_cache_capacity_tokens}"
                )
        else:
            raise RuntimeError(
                "kv cache capacity is unspecified: set kv_cache_capacity_tokens "
                "or enable kv_cache_auto_capacity"
            )

        dev_ids, ndev = _device_ids(0)
        runtime_params = LlaisysRuntimeCreateParams(
            int(kv_cache_layout),
            int(kv_cache_block_size),
            int(self._max_model_len),
            int(self._kv_cache_capacity_tokens),
        )
        self._runtime = LIB_LLAISYS.llaisysRuntimeCreate(byref(runtime_params))
        if not self._runtime:
            raise RuntimeError("Failed to create runtime state")

        create_params = LlaisysModelCreateParams(
            int(ModelType.QWEN2),
            cast(byref(self._meta_struct), c_void_p),
            device,
            dev_ids,
            ndev,
        )
        self._model = LIB_LLAISYS.llaisysModelCreate(byref(create_params), self._runtime)
        if not self._model:
            LIB_LLAISYS.llaisysRuntimeDestroy(self._runtime)
            self._runtime = None
            raise RuntimeError("Failed to create Qwen2 model instance")
        weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(self._model), POINTER(LlaisysQwen2Weights))
        if not weights_ptr:
            LIB_LLAISYS.llaisysModelDestroy(self._model)
            LIB_LLAISYS.llaisysRuntimeDestroy(self._runtime)
            self._model = None
            self._runtime = None
            raise RuntimeError("Failed to acquire Qwen2 weight slots")

        self._np_dtype = _datatype_to_numpy_dtype(meta.dtype)
        self._closed = False

        self._load_safetensors()

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True

        model = getattr(self, "_model", None)
        runtime = getattr(self, "_runtime", None)
        self._model = None
        self._runtime = None
        if model:
            LIB_LLAISYS.llaisysModelDestroy(model)
        if runtime:
            LIB_LLAISYS.llaisysRuntimeDestroy(runtime)

    def __del__(self):
        # Avoid native destroy in __del__. Finalizer timing during GC can race with
        # other Python/C++ objects and cause hard crashes. Use explicit close() chain.
        try:
            self._model = None
            self._runtime = None
            self._closed = True
        except Exception:
            pass

    # -------------------- Weight Loading --------------------

    def _replace_weight_slot(self, field: str, layer_idx: int, handle: c_void_p) -> None:
        rc = int(
            LIB_LLAISYS.llaisysModelReplaceWeight(
                self._model,
                field.encode("utf-8"),
                c_int32(layer_idx),
                handle,
            )
        )
        if rc != 0:
            # replace failed -> backend did not take ownership, so free the new handle.
            LIB_LLAISYS.tensorDestroy(handle)
            raise RuntimeError(f"Failed to replace weight slot field={field} layer={layer_idx} rc={rc}")

    def _assign_global(self, field: str, array: np.ndarray) -> None:
        tensor = Tensor(
            shape=array.shape,
            dtype=self._meta_info.dtype,
            device=self._device,
            device_id=0,
        )
        tensor.load(array.ctypes.data_as(c_void_p))
        handle = _detach_tensor_handle(tensor)
        self._replace_weight_slot(field, -1, handle)

    def _assign_layer(self, field: str, layer_idx: int, array: np.ndarray) -> None:
        tensor = Tensor(
            shape=array.shape,
            dtype=self._meta_info.dtype,
            device=self._device,
            device_id=0,
        )
        tensor.load(array.ctypes.data_as(c_void_p))
        handle = _detach_tensor_handle(tensor)
        self._replace_weight_slot(field, layer_idx, handle)

    def _map_and_assign(self, name: str, array: np.ndarray) -> bool:
        if name in _GLOBAL_NAMES:
            self._assign_global(_GLOBAL_NAMES[name], array)
            return True

        for pattern, field in _LAYER_PATTERNS:
            m = pattern.match(name)
            if not m:
                continue
            layer_idx = int(m.group(1))
            if layer_idx < 0 or layer_idx >= self._meta_info.nlayer:
                raise ValueError(f"Layer index out of range for {name}: {layer_idx}")
            self._assign_layer(field, layer_idx, array)
            return True

        return False

    def _load_safetensors(self) -> None:
        safetensor_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found under {self._model_path}")

        for file in safetensor_files:
            # Use torch backend for safetensors loading to avoid numpy bfloat16 incompatibility.
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                tensor = data.get_tensor(name)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                array = tensor.detach().cpu().numpy()
                array = _as_contiguous(array, self._np_dtype)
                self._map_and_assign(name, array)

    # -------------------- Inference --------------------

    @property
    def max_seq_len(self) -> int:
        return int(self._max_model_len)

    @property
    def kv_cache_capacity_tokens(self) -> int:
        return int(self._kv_cache_capacity_tokens)

    @property
    def end_token_id(self) -> int:
        return int(self._meta_info.end_token)

    @property
    def kv_cache_layout(self) -> KvCacheLayout:
        return self._kv_cache_layout

    @property
    def runtime_handle(self):
        return self._runtime

    def forward(self, fin: ModelForwardInput, fout: ModelForwardOutput) -> int:
        if not self._model:
            return -1
        return int(LIB_LLAISYS.llaisysModelForward(self._model, byref(fin), byref(fout)))
