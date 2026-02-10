# NovaInfer

## Quick Start

### 1. Install Python package

```bash
pip install -e ./python[test]
```

### 2. Run tests (pytest)

Run all default tests:

```bash
pytest
```

Run parity tests with local model path:

```bash
pytest -vv --model-path /home/hacode/NovaInfer/models/DeepSeek-R1-Distill-Qwen-1.5B -m parity
```

Run real-model multi-session stream regression (reproduces WebUI concurrent chat path):

```bash
pytest -vv test/test_online_real_model_multisession.py \
  --model-path /home/hacode/NovaInfer/models/DeepSeek-R1-Distill-Qwen-1.5B
```

Run stage suites:

```bash
python scripts/run_tests.py --suite stage0 --run-parity never --run-hf never
python scripts/run_tests.py --suite stage1 --run-parity never
python scripts/run_tests.py --suite stage2
```

## Run Inference Services

### 1. Start API server

```bash
PYTHONPATH=python python -m llaisys.server \
  --model-path /home/hacode/NovaInfer/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --model-type qwen2 \
  --device cpu \
  --host 127.0.0.1 \
  --port 8000 \
  --verbose
```

Available endpoints:

- `GET /health`
- `POST /v1/chat/completions`
- `POST /v1/requests/{request_id}/cancel`

### 2. Start Web UI static server

```bash
python -m http.server 8081 -d webui
```

Open `http://127.0.0.1:8081` and set server URL to `http://127.0.0.1:8000`.

## Manual API Debug

Non-stream:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2","messages":[{"role":"user","content":"hello"}],"stream":false,"max_tokens":32}'
```

Stream (SSE):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":32}'
```

Cancel request:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/requests/<request_id>/cancel
```

## Notes

- If Web UI loads but cannot call API, check server URL and CORS-enabled API process.
- `favicon.ico 404` from `python -m http.server` is harmless.
- In restricted sandbox environments, HTTP bind tests may be skipped automatically.
