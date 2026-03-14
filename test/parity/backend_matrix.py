from __future__ import annotations

import os

import torch

import llaisys


def has_llaisys_nvidia_runtime() -> bool:
    try:
        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        return api.get_device_count() > 0
    except Exception:
        return False


def has_torch_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def has_nvidia_runtime() -> bool:
    return has_llaisys_nvidia_runtime()


def parity_device_backend_cases() -> list[tuple[str, str | None]]:
    """
    Auto matrix for parity tests.

    LLAISYS_PARITY_DEVICE_MODE:
      - auto (default): GPU machine -> nvidia(native,cudnn), otherwise cpu.
      - all: cpu + nvidia(native,cudnn) if GPU exists, else cpu.
      - cpu: cpu only.
      - nvidia: nvidia(native,cudnn) if GPU exists, else cpu.
    """
    mode = str(os.environ.get("LLAISYS_PARITY_DEVICE_MODE", "auto")).strip().lower()
    has_gpu = has_nvidia_runtime()

    if mode in {"cpu", "cpu-only"}:
        return [("cpu", None)]

    if mode in {"nvidia", "gpu", "cuda"}:
        if has_gpu:
            return [("nvidia", "native"), ("nvidia", "cudnn")]
        return [("cpu", None)]

    if mode == "all":
        if has_gpu:
            return [("cpu", None), ("nvidia", "native"), ("nvidia", "cudnn")]
        return [("cpu", None)]

    # auto
    if has_gpu:
        return [("nvidia", "native"), ("nvidia", "cudnn")]
    return [("cpu", None)]


def parity_device_backend_layout_cases() -> list[tuple[str, str | None, str]]:
    """
    Parity matrix with KV layout dimension.

    - CPU: run block layout only.
    - NVIDIA: run block layout only.
    """
    out: list[tuple[str, str | None, str]] = []
    for device, backend in parity_device_backend_cases():
        out.append((device, backend, "block"))
    return out
