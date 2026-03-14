from __future__ import annotations

from contextlib import contextmanager

from ..libllaisys import LIB_LLAISYS


def _push(name: str) -> None:
    try:
        LIB_LLAISYS.llaisysNvtxRangePush(name.encode("utf-8"))
    except Exception:
        pass


def _pop() -> None:
    try:
        LIB_LLAISYS.llaisysNvtxRangePop()
    except Exception:
        pass


@contextmanager
def nvtx_range(name: str):
    _push(name)
    try:
        yield
    finally:
        _pop()
