from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .openai_server import OpenAIServer
from .schemas import ChatCompletionRequest, ChatMessage


class LlaisysHTTPServer:
    """Minimal HTTP server exposing OpenAI-compatible endpoints."""

    def __init__(
        self,
        openai_server: OpenAIServer,
        host: str = "127.0.0.1",
        port: int = 8000,
        verbose: bool = False,
    ):
        self._openai_server = openai_server
        self._host = host
        self._port = int(port)
        self._verbose = bool(verbose)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        if self._httpd is not None:
            return

        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "NovaInferHTTP/0.1"

            def log_message(self, format, *args):
                if outer._verbose:
                    msg = format % args
                    print(f"[http] {self.address_string()} {msg}")

            def _write_json(self, status: int, payload: dict):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                return json.loads(raw.decode("utf-8"))

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if outer._verbose:
                    print(f"[http] GET {path}")
                if path == "/health":
                    self._write_json(200, {"status": "ok"})
                    return
                self._write_json(404, {"error": "not_found"})

            def do_POST(self):
                path = urlparse(self.path).path
                if outer._verbose:
                    print(f"[http] POST {path}")
                try:
                    if path == "/v1/chat/completions":
                        payload = self._read_json()
                        req = _parse_chat_request(payload)
                        if req.stream:
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.send_header("Cache-Control", "no-cache")
                            self.send_header("Connection", "keep-alive")
                            self.send_header("Access-Control-Allow-Origin", "*")
                            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                            self.send_header("Access-Control-Allow-Headers", "Content-Type")
                            self.end_headers()
                            for chunk in outer._openai_server.handle_chat_stream(req):
                                if outer._verbose:
                                    token_id = chunk.get("token_id")
                                    req_id = chunk.get("request_id")
                                    done = chunk.get("is_finished")
                                    print(f"[http] stream chunk req={req_id} token_id={token_id} done={done}")
                                line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                                self.wfile.write(line)
                                self.wfile.flush()
                            if outer._verbose:
                                print("[http] stream completed")
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                            return
                        self._write_json(200, outer._openai_server.handle_chat(req))
                        return

                    if path.startswith("/v1/requests/") and path.endswith("/cancel"):
                        # /v1/requests/{request_id}/cancel
                        parts = path.split("/")
                        if len(parts) < 5:
                            self._write_json(400, {"error": "invalid_request_id"})
                            return
                        request_id = parts[3]
                        ok = bool(outer._openai_server.cancel(request_id))
                        self._write_json(200, {"ok": ok, "request_id": request_id})
                        return

                    self._write_json(404, {"error": "not_found"})
                except ValueError as exc:
                    self._write_json(400, {"error": "bad_request", "message": str(exc)})
                except Exception as exc:
                    self._write_json(500, {"error": "internal_error", "message": str(exc)})

        self._httpd = ThreadingHTTPServer((self._host, self._port), Handler)
        self._port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        close_fn = getattr(self._openai_server, "close", None)
        if callable(close_fn):
            close_fn()


def _parse_chat_request(payload: dict) -> ChatCompletionRequest:
    model = str(payload.get("model", "qwen2"))
    messages_raw = payload.get("messages", [])
    if not isinstance(messages_raw, list):
        raise ValueError("messages must be a list")
    messages: list[ChatMessage] = []
    for m in messages_raw:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        messages.append(ChatMessage(role=role, content=content))
    if not messages:
        raise ValueError("messages must be non-empty")

    return ChatCompletionRequest(
        model=model,
        messages=messages,
        stream=bool(payload.get("stream", False)),
        max_tokens=payload.get("max_tokens"),
        top_k=int(payload.get("top_k", 1)),
        top_p=float(payload.get("top_p", 1.0)),
        temperature=float(payload.get("temperature", 1.0)),
        stop=tuple(payload.get("stop", []) or []),
        stop_token_ids=tuple(payload.get("stop_token_ids", []) or []),
        extra={k: v for k, v in payload.items() if k not in {"model", "messages", "stream", "max_tokens", "top_k", "top_p", "temperature", "stop", "stop_token_ids"}},
    )
