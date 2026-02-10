from importlib import import_module

from .runtime import RuntimeAPI
from .libllaisys import DeviceType
from .libllaisys import DataType
from .libllaisys import MemcpyKind
from .libllaisys import llaisysStream_t as Stream
from .tensor import Tensor
from .ops import Ops


def __getattr__(name):
    if name == "models":
        mod = import_module(".models", __name__)
        globals()[name] = mod
        return mod
    if name == "engine":
        mod = import_module(".engine", __name__)
        globals()[name] = mod
        return mod
    if name == "LLM":
        llm_cls = import_module(".entrypoints", __name__).LLM
        globals()[name] = llm_cls
        return llm_cls
    if name == "Qwen2":
        qwen2_cls = import_module(".models", __name__).Qwen2
        globals()[name] = qwen2_cls
        return qwen2_cls
    if name in (
        "LLMEngine",
        "SamplingParams",
        "GenerationOutput",
        "StreamChunk",
        "RequestStatus",
        "RequestState",
        "EngineClient",
        "ModelRegistry",
        "create_default_registry",
    ):
        mod = import_module(".engine", __name__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RuntimeAPI",
    "DeviceType",
    "DataType",
    "MemcpyKind",
    "Stream",
    "Tensor",
    "Ops",
    "models",
    "engine",
    "LLM",
    "Qwen2",
    "LLMEngine",
    "EngineClient",
    "ModelRegistry",
    "create_default_registry",
    "SamplingParams",
    "GenerationOutput",
    "StreamChunk",
    "RequestStatus",
    "RequestState",
]
