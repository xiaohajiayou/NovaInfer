from __future__ import annotations

import argparse
import signal
import threading

from ..libllaisys import DeviceType
from ..libllaisys.model import KvCacheLayout
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
    parser.add_argument("--kv-cache-layout", default="block", choices=["slot", "block"])
    parser.add_argument("--kv-cache-block-size", default=16, type=int)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--kv-cache-capacity-tokens", default=0, type=int)
    parser.add_argument(
        "--kv-cache-capacity-mode",
        default="auto",
        choices=["explicit", "auto"],
        help="KV cache capacity mode. 'auto' follows available GPU memory budget.",
    )
    parser.add_argument(
        "--kv-cache-memory-utilization",
        default=0.9,
        type=float,
        help="Memory utilization ratio used when kv-cache-capacity-mode=auto.",
    )
    parser.add_argument("--max-num-seqs", default=8, type=int)
    parser.add_argument("--max-num-batched-tokens", default=0, type=int)
    parser.add_argument("--verbose", action="store_true", help="Print HTTP request logs")
    args = parser.parse_args()

    kv_cache_auto_capacity = str(args.kv_cache_capacity_mode) == "auto"
    kv_cache_capacity_tokens = int(args.kv_cache_capacity_tokens) if int(args.kv_cache_capacity_tokens) > 0 else None
    max_num_batched_tokens = int(args.max_num_batched_tokens) if int(args.max_num_batched_tokens) > 0 else None

    async_engine = AsyncLLMEngine(
        model_type=args.model_type,
        model_path=args.model_path,
        device=_parse_device(args.device),
        kv_cache_layout=KvCacheLayout.BLOCK if args.kv_cache_layout == "block" else KvCacheLayout.SLOT,
        kv_cache_block_size=int(args.kv_cache_block_size),
        max_model_len=int(args.max_model_len),
        kv_cache_capacity_tokens=kv_cache_capacity_tokens,
        max_num_seqs=max(1, int(args.max_num_seqs)),
        max_num_batched_tokens=max_num_batched_tokens,
        kv_cache_auto_capacity=kv_cache_auto_capacity,
        kv_cache_memory_utilization=float(args.kv_cache_memory_utilization),
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
    print(
        "Server config: "
        f"device={args.device} kv_cache_layout={args.kv_cache_layout} "
        f"max_model_len={int(args.max_model_len)} "
        f"kv_cache_capacity_mode={args.kv_cache_capacity_mode} "
        f"kv_cache_capacity_tokens={kv_cache_capacity_tokens if kv_cache_capacity_tokens is not None else 'auto'} "
        f"kv_cache_memory_utilization={float(args.kv_cache_memory_utilization):.2f} "
        f"max_num_seqs={max(1, int(args.max_num_seqs))} "
        f"max_num_batched_tokens={max_num_batched_tokens if max_num_batched_tokens is not None else 'auto'}"
    )
    print("Press Ctrl+C to stop.")

    # Use short polling so SIGINT/SIGTERM handlers are observed promptly even
    # when underlying runtime threads are busy.
    while not stop_event.wait(timeout=0.2):
        pass
    http.stop()
    print("NovaInfer server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
