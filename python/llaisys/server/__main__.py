from __future__ import annotations

import argparse
import signal
import threading

from ..libllaisys import DeviceType
from .async_engine import AsyncLLMEngine
from .http_server import LlaisysHTTPServer
from .openai_server import OpenAIServer


def _parse_device(name: str) -> DeviceType:
    lowered = name.strip().lower()
    if lowered == "cpu":
        return DeviceType.CPU
    if lowered in ("nvidia", "cuda", "gpu"):
        return DeviceType.NVIDIA
    raise ValueError(f"unsupported device: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NovaInfer HTTP server")
    parser.add_argument("--model-path", required=True, help="Local model path")
    parser.add_argument("--model-type", default="qwen2", help="Model type name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--verbose", action="store_true", help="Print HTTP request logs")
    args = parser.parse_args()

    async_engine = AsyncLLMEngine(
        model_type=args.model_type,
        model_path=args.model_path,
        device=_parse_device(args.device),
    )
    openai_server = OpenAIServer(async_engine)
    http = LlaisysHTTPServer(openai_server, host=args.host, port=args.port, verbose=args.verbose)
    http.start()

    stop_event = threading.Event()

    def _handle_signal(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"NovaInfer server started at http://{http.host}:{http.port}")
    print("Endpoints: GET /health, POST /v1/chat/completions, POST /v1/requests/{id}/cancel")
    print("Press Ctrl+C to stop.")

    stop_event.wait()
    http.stop()
    print("NovaInfer server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
