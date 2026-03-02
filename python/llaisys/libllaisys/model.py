from ctypes import POINTER, Structure, c_int, c_int32, c_int64, c_char_p, c_void_p
from enum import IntEnum

from .llaisys_types import llaisysDeviceType_t
from .tensor import llaisysTensor_t


llaisysModel_t = c_void_p
llaisysRuntime_t = c_void_p


class ModelType(IntEnum):
    UNKNOWN = 0
    QWEN2 = 1
    MOCK = 2

class KvCacheLayout(IntEnum):
    SLOT = 0
    BLOCK = 1


class LlaisysModelCreateParams(Structure):
    _fields_ = [
        ("model_type", c_int),
        ("meta", c_void_p),
        ("device", llaisysDeviceType_t),
        ("device_ids", POINTER(c_int)),
        ("ndevice", c_int),
    ]


class LlaisysRuntimeCreateParams(Structure):
    _fields_ = [
        ("kv_cache_layout", c_int32),
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
        ("mode", c_int32),
        ("seq_ids", llaisysTensor_t),
        ("q_seq_rows", llaisysTensor_t),
        ("q_pos", llaisysTensor_t),
        ("slot_mapping", llaisysTensor_t),
        ("context_lens", llaisysTensor_t),
        ("batch_seq_ids", llaisysTensor_t),
        ("block_tables", llaisysTensor_t),
        ("pos_ids_host", llaisysTensor_t),
        ("block_table_width", c_int32),
    ]


class ModelForwardInput(Structure):
    _fields_ = [
        ("input_ids", llaisysTensor_t),
        ("pos_ids", llaisysTensor_t),
        ("logits_mask", llaisysTensor_t),
        ("attention", AttentionMetadata),
    ]


class ModelForwardOutput(Structure):
    _fields_ = [
        ("output_ids", llaisysTensor_t),
        ("logits", llaisysTensor_t),
        ("n_outputs", c_int32),
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
    lib.llaisysRuntimeCreate.argtypes = [POINTER(LlaisysRuntimeCreateParams)]
    lib.llaisysRuntimeCreate.restype = llaisysRuntime_t

    lib.llaisysRuntimeDestroy.argtypes = [llaisysRuntime_t]
    lib.llaisysRuntimeDestroy.restype = None

    lib.llaisysModelCreate.argtypes = [POINTER(LlaisysModelCreateParams), llaisysRuntime_t]
    lib.llaisysModelCreate.restype = llaisysModel_t

    lib.llaisysModelDestroy.argtypes = [llaisysModel_t]
    lib.llaisysModelDestroy.restype = None

    lib.llaisysModelType.argtypes = [llaisysModel_t]
    lib.llaisysModelType.restype = c_int

    lib.llaisysModelWeights.argtypes = [llaisysModel_t]
    lib.llaisysModelWeights.restype = c_void_p

    lib.llaisysModelReplaceWeight.argtypes = [llaisysModel_t, c_char_p, c_int32, c_void_p]
    lib.llaisysModelReplaceWeight.restype = c_int

    lib.llaisysModelForward.argtypes = [llaisysModel_t, POINTER(ModelForwardInput), POINTER(ModelForwardOutput)]
    lib.llaisysModelForward.restype = c_int32
    lib.llaisysSamplerSample.argtypes = [POINTER(SamplerInput), POINTER(SamplerOutput)]
    lib.llaisysSamplerSample.restype = c_int32

    lib.llaisysRuntimeKvSeqCp.argtypes = [llaisysRuntime_t, c_int64, c_int64, c_int64, c_int64]
    lib.llaisysRuntimeKvSeqCp.restype = c_int

    lib.llaisysRuntimeKvSeqRm.argtypes = [llaisysRuntime_t, c_int64, c_int64, c_int64]
    lib.llaisysRuntimeKvSeqRm.restype = c_int

    lib.llaisysRuntimeKvSeqAdd.argtypes = [llaisysRuntime_t, c_int64, c_int64, c_int64, c_int64]
    lib.llaisysRuntimeKvSeqAdd.restype = c_int

    lib.llaisysRuntimeKvSeqKeep.argtypes = [llaisysRuntime_t, c_int64]
    lib.llaisysRuntimeKvSeqKeep.restype = c_int

    lib.llaisysRuntimeKvSeqPosMax.argtypes = [llaisysRuntime_t, c_int64]
    lib.llaisysRuntimeKvSeqPosMax.restype = c_int64

    lib.llaisysRuntimeRequestFree.argtypes = [llaisysRuntime_t, c_int64]
    lib.llaisysRuntimeRequestFree.restype = c_int

    lib.llaisysRuntimeKvStats.argtypes = [llaisysRuntime_t, POINTER(LlaisysKvStats)]
    lib.llaisysRuntimeKvStats.restype = c_int

    lib.llaisysRuntimeKvResetPrefixCache.argtypes = [llaisysRuntime_t]
    lib.llaisysRuntimeKvResetPrefixCache.restype = c_int

__all__ = [
    "llaisysModel_t",
    "llaisysRuntime_t",
    "ModelType",
    "KvCacheLayout",
    "LlaisysModelCreateParams",
    "LlaisysRuntimeCreateParams",
    "LlaisysKvStats",
    "AttentionMetadata",
    "ModelForwardInput",
    "ModelForwardOutput",
    "SamplerInput",
    "SamplerOutput",
    "load_model",
]
