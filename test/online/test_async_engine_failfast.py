from __future__ import annotations

import time

import pytest

from llaisys.server.async_engine import AsyncLLMEngine


class _CrashEngine:
    def step(self):
        raise RuntimeError("intentional step crash")

    def close(self) -> None:
        return None


@pytest.mark.online
def test_async_engine_collect_fails_fast_after_loop_crash():
    async_engine = AsyncLLMEngine(engine=_CrashEngine())
    try:
        deadline = time.time() + 2.0
        while async_engine._loop_exc is None and time.time() < deadline:
            time.sleep(0.01)
        assert async_engine._loop_exc is not None
        with pytest.raises(RuntimeError, match="loop thread crashed"):
            _ = async_engine.collect("req-1")
    finally:
        async_engine.close()
