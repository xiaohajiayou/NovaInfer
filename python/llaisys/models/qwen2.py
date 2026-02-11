from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import json
import os
import re
import threading

import numpy as np
import safetensors
import torch

from ctypes import POINTER, byref, cast, c_int, c_int8, c_int32, c_int64, c_void_p

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.model import LlaisysBatch, LlaisysModelCreateParams, ModelType
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


def _parse_meta(model_path: Path) -> _MetaInfo:
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
    cap_maxseq = int(os.getenv("LLAISYS_MAXSEQ", "4096"))
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


def _device_ids(device_id: int = 0):
    arr = (c_int * 1)(device_id)
    return arr, 1


class Qwen2:
    """Qwen2 model wrapper backed by the LLAISYS C++ runtime."""

    def __init__(self, model_path: Path | str, device: DeviceType = DeviceType.CPU):
        self._model_path = Path(model_path)
        self._device = device

        meta = _parse_meta(self._model_path)
        self._meta_info = meta
        self._meta_struct = _build_meta_struct(meta)

        dev_ids, ndev = _device_ids(0)
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
            raise RuntimeError("Failed to acquire Qwen2 weight slots")

        self._np_dtype = _datatype_to_numpy_dtype(meta.dtype)
        self._offline_engine = None
        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()
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
        return int(self._meta_info.maxseq)

    @property
    def end_token_id(self) -> int:
        return int(self._meta_info.end_token)

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        # Tokenizer lazy init must be thread-safe: online multi-session can call
        # encode/decode from multiple request threads at once.
        with self._tokenizer_lock:
            if self._tokenizer is None:
                try:
                    from transformers import AutoTokenizer  # type: ignore
                except Exception:
                    from transformers.models.auto.tokenization_auto import AutoTokenizer  # type: ignore
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        return self._tokenizer

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        tokenizer = self._get_tokenizer()
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def encode_chat_messages(self, messages: Sequence[dict]) -> list[int]:
        tokenizer = self._get_tokenizer()
        text = tokenizer.apply_chat_template(
            conversation=[{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages],
            add_generation_prompt=True,
            tokenize=False,
        )
        return [int(t) for t in tokenizer.encode(text)]

    def decode_batch(
        self,
        token_ids: Sequence[int],
        pos_ids: Optional[Sequence[int]] = None,
        seq_ids: Optional[Sequence[int]] = None,
        logits_mask: Optional[Sequence[int]] = None,
    ):
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        n_tokens = len(token_ids)

        token_buf = (c_int64 * n_tokens)(*[int(t) for t in token_ids])
        pos_buf = None
        if pos_ids is not None:
            if len(pos_ids) != n_tokens:
                raise ValueError("pos_ids length mismatch")
            pos_buf = (c_int64 * n_tokens)(*[int(p) for p in pos_ids])

        if logits_mask is None:
            logits_mask = [0] * n_tokens
            logits_mask[-1] = 1
        if len(logits_mask) != n_tokens:
            raise ValueError("logits_mask length mismatch")
        logits_buf = (c_int8 * n_tokens)(*[int(x) for x in logits_mask])

        n_seq_buf = None
        seq_ptr_buf = None
        seq_rows = None
        if seq_ids is not None:
            if len(seq_ids) != n_tokens:
                raise ValueError("seq_ids length mismatch")
            n_seq_buf = (c_int32 * n_tokens)()
            seq_ptr_buf = (POINTER(c_int64) * n_tokens)()
            seq_rows = []
            for i, sid in enumerate(seq_ids):
                row = (c_int64 * 1)(int(sid))
                seq_rows.append(row)
                n_seq_buf[i] = 1
                seq_ptr_buf[i] = row

        batch = LlaisysBatch()
        batch.n_tokens = c_int32(n_tokens)
        batch.token = token_buf
        batch.embd = None
        batch.pos = pos_buf
        batch.n_seq_id = n_seq_buf
        batch.seq_id = seq_ptr_buf
        batch.logits = logits_buf

        status = int(LIB_LLAISYS.llaisysModelDecode(self._model, batch))
        if status != 0:
            raise RuntimeError(f"Decode failed with status={status}")

        n_outputs = int(LIB_LLAISYS.llaisysModelNOutputs(self._model))
        if n_outputs < 0:
            raise RuntimeError("Decode returned invalid output row count")
        if n_outputs == 0:
            return [], []

        output_ids_ptr = LIB_LLAISYS.llaisysModelOutputIds(self._model)
        if not output_ids_ptr:
            raise RuntimeError("Decode returned outputs without output_ids")
        output_ids = np.ctypeslib.as_array(output_ids_ptr, shape=(n_outputs,)).astype(np.int64).tolist()

        logits_rows = []
        for i in range(n_outputs):
            logits_ptr = LIB_LLAISYS.llaisysModelGetLogitsIth(self._model, c_int32(i))
            if not logits_ptr:
                raise RuntimeError(f"Failed to get logits row: {i}")
            row = np.ctypeslib.as_array(logits_ptr, shape=(self._meta_info.voc,))
            logits_rows.append(np.array(row, copy=True))

        return output_ids, logits_rows

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        stop_token_ids: Optional[Sequence[int]] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> Sequence[int]:
        # Stage-1: route offline generation through LLMEngine chain.
        from ..engine.llm_engine import LLMEngine
        from ..engine.types import SamplingParams

        if self._offline_engine is None:
            self._offline_engine = LLMEngine(model_runner=self)

        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_token_ids=tuple(stop_token_ids or ()),
            stop=tuple(stop or ()),
        )
        result = self._offline_engine.generate(inputs=inputs, sampling_params=sampling_params)
        return result.token_ids
