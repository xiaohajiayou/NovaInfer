from ctypes import POINTER, Structure, c_int, c_int32, c_int64, c_char_p, c_void_p
from enum import IntEnum

from .llaisys_types import llaisysDeviceType_t, llaisysStream_t
from .tensor import llaisysTensor_t


llaisysModel_t = c_void_p
llaisysKvState_t = c_void_p


class ModelType(IntEnum):
    UNKNOWN = 0
    QWEN2 = 1
    MOCK = 2

class AttentionPhase(IntEnum):
    PREFILL = 0
    DECODE = 1


class LlaisysModelCreateParams(Structure):
    _fields_ = [
        ("model_type", c_int),
        ("meta", c_void_p),
        ("device", llaisysDeviceType_t),
        ("device_ids", POINTER(c_int)),
        ("ndevice", c_int),
    ]


class LlaisysKvStateCreateParams(Structure):
    _fields_ = [
        ("kv_cache_block_size", c_int32),
        ("max_model_len", c_int32),
        ("kv_cache_capacity_tokens", c_int32),
    ]

class LlaisysKvStats(Structure):
    _fields_ = [
        ("capacity_tokens", c_int64),
        ("used_tokens", c_int64),
        ("free_tokens", c_int64),
        ("peak_used_tokens", c_int64),
    ]

class AttentionMetadata(Structure):
    _fields_ = [
        ("phase", c_int32),
        ("cu_seqlens_q", llaisysTensor_t),
        ("cu_seqlens_k", llaisysTensor_t),
        ("max_seqlen_q", c_int32),
        ("max_seqlen_k", c_int32),
        ("slot_mapping", llaisysTensor_t),
        ("block_tables", llaisysTensor_t),
        ("block_table_width", c_int32),
        ("cudnn_seq_lens_q", llaisysTensor_t),
        ("cudnn_seq_lens_kv", llaisysTensor_t),
        ("cudnn_page_table", llaisysTensor_t),
        ("cudnn_qo_ragged_offset", llaisysTensor_t),
        ("cudnn_b_exec", c_int32),
    ]


class ModelForwardInput(Structure):
    _fields_ = [
        ("input_ids", llaisysTensor_t),
        ("pos_ids", llaisysTensor_t),
        ("logits_indices", llaisysTensor_t),
        ("attention", AttentionMetadata),
    ]


class ModelForwardOutput(Structure):
    _fields_ = [
        ("logits", llaisysTensor_t),
    ]


class SamplerInput(Structure):
    _fields_ = [
        ("logits", llaisysTensor_t),
        ("temperatures", llaisysTensor_t),
        ("top_ps", llaisysTensor_t),
        ("top_ks", llaisysTensor_t),
        ("seeds", llaisysTensor_t),
        ("has_seeds", llaisysTensor_t),
    ]


class SamplerOutput(Structure):
    _fields_ = [
        ("sampled_ids", llaisysTensor_t),
    ]


def load_model(lib):
    lib.llaisysKvStateCreate.argtypes = [POINTER(LlaisysKvStateCreateParams)]
    lib.llaisysKvStateCreate.restype = llaisysKvState_t

    lib.llaisysKvStateDestroy.argtypes = [llaisysKvState_t]
    lib.llaisysKvStateDestroy.restype = None

    lib.llaisysModelCreate.argtypes = [POINTER(LlaisysModelCreateParams)]
    lib.llaisysModelCreate.restype = llaisysModel_t

    lib.llaisysModelDestroy.argtypes = [llaisysModel_t]
    lib.llaisysModelDestroy.restype = None

    lib.llaisysModelType.argtypes = [llaisysModel_t]
    lib.llaisysModelType.restype = c_int

    lib.llaisysModelWeights.argtypes = [llaisysModel_t]
    lib.llaisysModelWeights.restype = c_void_p

    lib.llaisysModelReplaceWeight.argtypes = [llaisysModel_t, c_char_p, c_int32, c_void_p]
    lib.llaisysModelReplaceWeight.restype = c_int

    lib.llaisysModelForward.argtypes = [llaisysModel_t, llaisysKvState_t, POINTER(ModelForwardInput), POINTER(ModelForwardOutput)]
    lib.llaisysModelForward.restype = c_int32
    lib.llaisysSamplerSample.argtypes = [POINTER(SamplerInput), POINTER(SamplerOutput)]
    lib.llaisysSamplerSample.restype = c_int32

    lib.llaisysKvStateRequestFree.argtypes = [llaisysKvState_t, c_int64]
    lib.llaisysKvStateRequestFree.restype = c_int

    lib.llaisysKvStateStats.argtypes = [llaisysKvState_t, POINTER(LlaisysKvStats)]
    lib.llaisysKvStateStats.restype = c_int

    lib.llaisysKvStateResetPrefixCache.argtypes = [llaisysKvState_t]
    lib.llaisysKvStateResetPrefixCache.restype = c_int

__all__ = [
    "llaisysModel_t",
    "llaisysKvState_t",
    "ModelType",
    "AttentionPhase",
    "LlaisysModelCreateParams",
    "LlaisysKvStateCreateParams",
    "LlaisysKvStats",
    "AttentionMetadata",
    "ModelForwardInput",
    "ModelForwardOutput",
    "SamplerInput",
    "SamplerOutput",
    "load_model",
]
