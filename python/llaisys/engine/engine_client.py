from __future__ import annotations

from .llm_engine import LLMEngine
from .types import GenerationOutput, SamplingParams


class EngineClient:
    """In-process client wrapper around LLMEngine.

    This is the stage-1 form. Stage-2 can swap this with IPC/RPC transport.
    """

    def __init__(self, engine: LLMEngine):
        self._engine = engine

    def submit(self, inputs, sampling_params: SamplingParams) -> str:
        return self._engine.submit(inputs=inputs, sampling_params=sampling_params)

    def step(self) -> list[GenerationOutput]:
        return self._engine.step()

    def collect(self, request_id: str) -> GenerationOutput | None:
        return self._engine.collect(request_id)

    def stream(self, inputs, sampling_params: SamplingParams):
        return self._engine.stream(inputs=inputs, sampling_params=sampling_params)

    def cancel(self, request_id: str) -> bool:
        return self._engine.cancel(request_id)

    def generate(self, inputs, sampling_params: SamplingParams) -> GenerationOutput:
        return self._engine.generate(inputs=inputs, sampling_params=sampling_params)

    def close(self) -> None:
        close_fn = getattr(self._engine, "close", None)
        if callable(close_fn):
            close_fn()
