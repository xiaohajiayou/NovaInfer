from ctypes import (
    POINTER,
    Structure,
    c_float,
    c_int,
    c_int8,
    c_int32,
    c_int64,
    c_char_p,
    c_void_p,
)
from enum import IntEnum

from .llaisys_types import llaisysDeviceType_t


llaisysModel_t = c_void_p


class ModelType(IntEnum):
    UNKNOWN = 0
    QWEN2 = 1
    MOCK = 2


class LlaisysBatch(Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(c_int64)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(c_int64)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(c_int64))),
        ("logits", POINTER(c_int8)),
    ]


class LlaisysModelCreateParams(Structure):
    _fields_ = [
        ("model_type", c_int),
        ("meta", c_void_p),
        ("device", llaisysDeviceType_t),
        ("device_ids", POINTER(c_int)),
        ("ndevice", c_int),
    ]


def load_model(lib):
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

    lib.llaisysModelDecode.argtypes = [llaisysModel_t, LlaisysBatch]
    lib.llaisysModelDecode.restype = c_int32

    lib.llaisysModelGetLogits.argtypes = [llaisysModel_t]
    lib.llaisysModelGetLogits.restype = POINTER(c_float)

    lib.llaisysModelGetLogitsIth.argtypes = [llaisysModel_t, c_int32]
    lib.llaisysModelGetLogitsIth.restype = POINTER(c_float)

    lib.llaisysModelNOutputs.argtypes = [llaisysModel_t]
    lib.llaisysModelNOutputs.restype = c_int32

    lib.llaisysModelOutputIds.argtypes = [llaisysModel_t]
    lib.llaisysModelOutputIds.restype = POINTER(c_int32)

    lib.llaisysModelKvSeqCp.argtypes = [llaisysModel_t, c_int64, c_int64, c_int64, c_int64]
    lib.llaisysModelKvSeqCp.restype = c_int

    lib.llaisysModelKvSeqRm.argtypes = [llaisysModel_t, c_int64, c_int64, c_int64]
    lib.llaisysModelKvSeqRm.restype = c_int

    lib.llaisysModelKvSeqAdd.argtypes = [llaisysModel_t, c_int64, c_int64, c_int64, c_int64]
    lib.llaisysModelKvSeqAdd.restype = c_int

    lib.llaisysModelKvSeqKeep.argtypes = [llaisysModel_t, c_int64]
    lib.llaisysModelKvSeqKeep.restype = c_int

    lib.llaisysModelKvSeqPosMax.argtypes = [llaisysModel_t, c_int64]
    lib.llaisysModelKvSeqPosMax.restype = c_int64

    lib.llaisysBatchInit.argtypes = [c_int32, c_int32, c_int32]
    lib.llaisysBatchInit.restype = LlaisysBatch

    lib.llaisysBatchGetOne.argtypes = [POINTER(c_int64), c_int32]
    lib.llaisysBatchGetOne.restype = LlaisysBatch

    lib.llaisysBatchFree.argtypes = [LlaisysBatch]
    lib.llaisysBatchFree.restype = None


__all__ = [
    "llaisysModel_t",
    "ModelType",
    "LlaisysBatch",
    "LlaisysModelCreateParams",
    "load_model",
]
