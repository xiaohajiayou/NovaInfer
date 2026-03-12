from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import json
import os
import re

import numpy as np

from ctypes import POINTER, byref, cast, c_int, c_int32, c_void_p

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.model import (
    LlaisysModelCreateParams,
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

_TP_SHARD_DIM_BY_FIELD: Dict[str, int] = {
    # Column-parallel (split output dim).
    "attn_q_w": 0,
    "attn_q_b": 0,
    "mlp_gate_w": 0,
    "mlp_up_w": 0,
    # Row-parallel (split input dim).
    "attn_o_w": 1,
    "mlp_down_w": 1,
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
        return DataType.BF16
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
    if dtype == DataType.BF16:
        # Keep BF16 payload as-is even without ml_dtypes; uint16 preserves raw bits.
        if _BF16_DTYPE is not None:
            return _BF16_DTYPE
        return np.dtype(np.uint16)
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


_SAFE_DTYPE_TO_NUMPY: Dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "I16": np.dtype(np.int16),
    "U16": np.dtype(np.uint16),
    "I32": np.dtype(np.int32),
    "U32": np.dtype(np.uint32),
    "I64": np.dtype(np.int64),
    "U64": np.dtype(np.uint64),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
}


_SAFE_DTYPE_NBYTES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _decode_safetensor_array(raw_u8: np.ndarray, dtype_code: str, shape: Tuple[int, ...], target_dtype: np.dtype) -> np.ndarray:
    dtype_code = str(dtype_code).upper()
    nbytes_per_elem = _SAFE_DTYPE_NBYTES.get(dtype_code)
    if nbytes_per_elem is None:
        raise RuntimeError(f"unsupported safetensors dtype: {dtype_code}")

    expected_nbytes = _numel(shape) * int(nbytes_per_elem)
    if int(raw_u8.size) != int(expected_nbytes):
        raise RuntimeError(
            f"safetensors payload size mismatch: dtype={dtype_code} shape={shape} "
            f"payload={int(raw_u8.size)} expected={int(expected_nbytes)}"
        )

    if dtype_code == "BF16":
        # Preserve BF16 payload bits; Tensor(BF16) consumes raw 16-bit lanes.
        base = raw_u8.view(np.uint16)
    else:
        base_dtype = _SAFE_DTYPE_TO_NUMPY.get(dtype_code)
        if base_dtype is None:
            raise RuntimeError(f"unsupported safetensors dtype: {dtype_code}")
        base = raw_u8.view(base_dtype)

    base = base.reshape(shape)
    return _as_contiguous(base, target_dtype)


def _iter_safetensors_arrays(file: Path, target_dtype: np.dtype) -> Iterator[Tuple[str, np.ndarray]]:
    with file.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"invalid safetensors file: {file}")
        header_len = int.from_bytes(prefix, byteorder="little", signed=False)
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise RuntimeError(f"invalid safetensors header: {file}")
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse safetensors header: {file}") from exc

    data_offset = int(8 + header_len)
    mmap_u8 = np.memmap(file, mode="r", dtype=np.uint8, offset=data_offset)
    try:
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(spec, dict):
                continue
            dtype_code = str(spec.get("dtype", ""))
            shape_raw = spec.get("shape", ())
            offsets = spec.get("data_offsets", None)
            if not isinstance(shape_raw, (list, tuple)):
                raise RuntimeError(f"invalid shape in safetensors entry: name={name}")
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                raise RuntimeError(f"invalid data_offsets in safetensors entry: name={name}")
            shape = tuple(int(v) for v in shape_raw)
            start = int(offsets[0])
            end = int(offsets[1])
            if start < 0 or end < start or end > int(mmap_u8.size):
                raise RuntimeError(f"invalid data_offsets range in safetensors entry: name={name}")
            raw = mmap_u8[start:end]
            arr = _decode_safetensor_array(raw, dtype_code, shape, target_dtype=target_dtype)
            yield str(name), arr
    finally:
        del mmap_u8


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


def _device_ids(device_ids: Optional[Sequence[int]] = None):
    if device_ids is None or len(device_ids) == 0:
        arr = (c_int * 1)(0)
        return arr, 1
    ids = [int(v) for v in device_ids]
    arr = (c_int * len(ids))(*ids)
    return arr, len(ids)


class Qwen2:
    """Qwen2 model wrapper backed by the LLAISYS C++ runtime."""

    def __init__(
        self,
        model_path: Path | str,
        device: DeviceType = DeviceType.CPU,
        max_model_len: Optional[int] = None,
        device_ids: Optional[Sequence[int]] = None,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
    ):
        self._model_path = Path(model_path)
        self._device = device
        self._model = None

        meta = _parse_meta(self._model_path, max_model_len=max_model_len)
        self._meta_info = meta
        self._meta_struct = _build_meta_struct(meta)
        self._max_model_len = int(meta.maxseq)
        self._tp_size = max(1, int(tensor_parallel_size))
        self._tp_rank = max(0, int(tensor_parallel_rank))
        if self._tp_rank >= self._tp_size:
            raise ValueError(f"invalid tensor_parallel_rank={self._tp_rank} for tensor_parallel_size={self._tp_size}")
        self._local_device_id = 0
        if device_ids is not None and len(device_ids) > 0:
            local_ids = [int(v) for v in device_ids]
            if self._tp_rank >= len(local_ids):
                raise ValueError(
                    f"tensor_parallel_rank out of range for device_ids: rank={self._tp_rank} ndevice={len(local_ids)}"
                )
            self._local_device_id = int(local_ids[self._tp_rank])

        dev_ids, ndev = _device_ids(device_ids)
        create_params = LlaisysModelCreateParams(
            int(ModelType.QWEN2),
            cast(byref(self._meta_struct), c_void_p),
            device,
            dev_ids,
            ndev,
        )
        self._model = LIB_LLAISYS.llaisysModelCreate(byref(create_params))
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model instance")
        weights_ptr = cast(LIB_LLAISYS.llaisysModelWeights(self._model), POINTER(LlaisysQwen2Weights))
        if not weights_ptr:
            LIB_LLAISYS.llaisysModelDestroy(self._model)
            self._model = None
            raise RuntimeError("Failed to acquire Qwen2 weight slots")

        self._np_dtype = _datatype_to_numpy_dtype(meta.dtype)
        self._closed = False

        self._load_safetensors()

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True

        model = getattr(self, "_model", None)
        self._model = None
        if model:
            LIB_LLAISYS.llaisysModelDestroy(model)

    def __del__(self):
        # Avoid native destroy in __del__. Finalizer timing during GC can race with
        # other Python/C++ objects and cause hard crashes. Use explicit close() chain.
        try:
            self._model = None
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
            device_id=int(self._local_device_id),
        )
        tensor.load(array.ctypes.data_as(c_void_p))
        handle = _detach_tensor_handle(tensor)
        self._replace_weight_slot(field, -1, handle)

    def _assign_layer(self, field: str, layer_idx: int, array: np.ndarray) -> None:
        tensor = Tensor(
            shape=array.shape,
            dtype=self._meta_info.dtype,
            device=self._device,
            device_id=int(self._local_device_id),
        )
        tensor.load(array.ctypes.data_as(c_void_p))
        handle = _detach_tensor_handle(tensor)
        self._replace_weight_slot(field, layer_idx, handle)

    def _map_and_assign(self, name: str, array: np.ndarray) -> bool:
        if name in _GLOBAL_NAMES:
            field = _GLOBAL_NAMES[name]
            self._assign_global(field, self._maybe_shard_array(field, array))
            return True

        for pattern, field in _LAYER_PATTERNS:
            m = pattern.match(name)
            if not m:
                continue
            layer_idx = int(m.group(1))
            if layer_idx < 0 or layer_idx >= self._meta_info.nlayer:
                raise ValueError(f"Layer index out of range for {name}: {layer_idx}")
            self._assign_layer(field, layer_idx, self._maybe_shard_array(field, array))
            return True

        return False

    def _maybe_shard_array(self, field: str, array: np.ndarray) -> np.ndarray:
        if self._tp_size <= 1:
            return array
        shard_dim = _TP_SHARD_DIM_BY_FIELD.get(field)
        if shard_dim is None:
            # Replicated weights in TP mode.
            return array
        if shard_dim < 0 or shard_dim >= array.ndim:
            raise ValueError(f"invalid shard dim for field={field}: dim={shard_dim}, ndim={array.ndim}")
        dim_size = int(array.shape[shard_dim])
        if dim_size % self._tp_size != 0:
            raise ValueError(
                f"TP shard requires divisible dim: field={field} dim_size={dim_size} tp_size={self._tp_size}"
            )
        chunk = dim_size // self._tp_size
        start = int(self._tp_rank) * chunk
        end = start + chunk
        slicer = [slice(None)] * array.ndim
        slicer[shard_dim] = slice(start, end)
        return _as_contiguous(array[tuple(slicer)], self._np_dtype)

    def _load_safetensors(self) -> None:
        safetensor_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found under {self._model_path}")

        for file in safetensor_files:
            for name, array in _iter_safetensors_arrays(file, target_dtype=self._np_dtype):
                self._map_and_assign(name, array)

    # -------------------- Inference --------------------

    @property
    def max_seq_len(self) -> int:
        return int(self._max_model_len)

    @property
    def end_token_id(self) -> int:
        return int(self._meta_info.end_token)

    def forward(self, kv_state, fin: ModelForwardInput, fout: ModelForwardOutput) -> int:
        if not self._model or kv_state is None:
            return -1
        return int(LIB_LLAISYS.llaisysModelForward(self._model, kv_state, byref(fin), byref(fout)))
