# NovaInfer


## Build

### 1. Build native library
Install compile tool - xmake
```bash
curl -fsSL https://xmake.io/shget.text | bash
source ~/.xmake/profile
```
CPU (Linux):

```bash
xmake f --mode=release --nv-gpu=n
xmake -j1
xmake install
```

NVIDIA CUDA:

```bash
xmake f --mode=release --nv-gpu=y
xmake -j1
xmake install
```

NVIDIA CUDA + cuDNN frontend (for `LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn`):

```bash
# Ubuntu: install/update cuDNN runtime+dev for CUDA 12 (requires sudo)
sudo apt-get update
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Verify cuDNN >= 9.18
python - <<'PY'
import ctypes, sys
lib = ctypes.cdll.LoadLibrary("libcudnn.so.9")
lib.cudnnGetVersion.restype = ctypes.c_size_t
v = int(lib.cudnnGetVersion())
major = v // 10000
minor = (v % 10000) // 100
patch = v % 100
print(f"detected cuDNN: {major}.{minor}.{patch} ({v})")
if not ((major > 9) or (major == 9 and minor >= 18)):
    raise SystemExit("ERROR: NovaInfer requires cuDNN >= 9.18")
PY

xmake f --mode=release --nv-gpu=y --nv-cudnn=y
xmake -j8
xmake install
```

For repeatable large-vs-small benchmark experiment design and plotting scripts, see:
`doc/novainfer_vs_vllm_perf_experiment_2026-03-12.md`
Notes:

- `xmake f ...` configures build mode/options; rerun it when switching CPU/GPU/cudnn.
- `xmake install` is required so Python loads the latest library from `python/llaisys/libllaisys/`.
- cuDNN frontend headers are vendored under `third_party/cudnn_frontend/include`.

TP=2 smoke (BLOCK + cuDNN, two processes):

```bash
export LD_LIBRARY_PATH=/home/xiaohajiayou/opt/cudnn-linux-x86_64-9.18.1.3_cuda12-archive/lib:$LD_LIBRARY_PATH
export LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn
export LLAISYS_TP_SINGLE_PROCESS=0
export LLAISYS_TP_INIT_METHOD=file:///tmp/llaisys_tp_nccl_smoke.id

CUDA_VISIBLE_DEVICES=4,5 \
python scripts/tp2_smoke.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

Notes:

- TP+cuDNN path currently requires `cudnnGetVersion() >= 91800` (cuDNN 9.18+).
- `--tensor-parallel-device-ids` in scripts uses logical IDs under `CUDA_VISIBLE_DEVICES`.

TP=2 benchmark (two-process launcher):

```bash
export LD_LIBRARY_PATH=/home/xiaohajiayou/opt/cudnn-linux-x86_64-9.18.1.3_cuda12-archive/lib:$LD_LIBRARY_PATH
export LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn

python scripts/bench_tp2_novainfer.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --cuda-visible-devices 4,5 \
  --num-seqs 256 \
  --min-input-len 100 --max-input-len 1024 \
  --min-output-len 100 --max-output-len 1024 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384
```

TP=2 benchmark (built-in `mp` executor):

```bash
export CUDNN_HOME=/home/xiaohajiayou/opt/cudnn-linux-x86_64-9.18.1.3_cuda12-archive
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:/home/xiaohajiayou/NovaInfer/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn

python scripts/bench_tp_novainfer.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2 \
  --cuda-visible-devices 5,6 \
  --tensor-parallel-device-ids 0,1 \
  --num-seqs 256 \
  --min-input-len 100 --max-input-len 1024 \
  --min-output-len 100 --max-output-len 1024 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384
```

TP=2 HF parity (built-in `mp` executor):

```bash
export CUDNN_HOME=/home/xiaohajiayou/opt/cudnn-linux-x86_64-9.18.1.3_cuda12-archive
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:/home/xiaohajiayou/NovaInfer/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn

CUDA_VISIBLE_DEVICES=5,6 \
python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2 \
  --tensor-parallel-device-ids 0,1 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

### 2. Install Python package

```bash
pip install -e ./python[test]
```

`[test]` extra installs pytest and related deps.

## Run Inference Services
```
ssh -N \
    -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:18082:127.0.0.1:8081 \
    -L 127.0.0.1:18003:127.0.0.1:8675 \
    aliyun-2222
```
### 1. Start API server

CPU backend:

```bash
PYTHONPATH=python python -m llaisys.server \
  --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
  --model-type qwen2 \
  --device cpu \
  --kv-cache-capacity-mode auto \
  --kv-cache-memory-utilization 0.9 \
  --host 127.0.0.1 \
  --port 8000 \
  --verbose
```

NVIDIA backend (native paged attention):

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=python python -m llaisys.server \
  --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
  --model-type qwen2 \
  --device nvidia \
  --kv-cache-capacity-mode auto \
  --kv-cache-memory-utilization 0.9 \
  --host 127.0.0.1 \
  --port 8000 \
  --verbose
```

NVIDIA backend (cuDNN paged attention):

```bash
CUDA_VISIBLE_DEVICES=5 \
LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn \
PYTHONPATH=python python -m llaisys.server \
  --model-path /home/xiaohajiayou/NovaInfer/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --model-type qwen2 \
  --device nvidia \
  --kv-cache-memory-utilization 0.9 \
  --host 127.0.0.1 \
  --port 8000 \
  --verbose
```

NVIDIA backend, TP server on one host (rank0 owns HTTP, internal `mp` executor spawns other ranks):

```bash
export CUDNN_HOME=/home/xiaohajiayou/opt/cudnn-linux-x86_64-9.18.1.3_cuda12-archive
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:/home/xiaohajiayou/NovaInfer/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=5,6 \
LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn \
PYTHONPATH=python python -m llaisys.server \
  --model-path /home/xiaohajiayou/NovaInfer/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --model-type qwen2 \
  --device nvidia \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --tensor-parallel-device-ids 0,1 \
  --kv-cache-memory-utilization 0.9 \
  --host 127.0.0.1 \
  --port 8675 \
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

Backend selection summary:

- `--device cpu`: CPU runner (no CUDA backend switch).
- `--device nvidia` + no env `LLAISYS_CUDA_PAGED_ATTN_BACKEND`: default native paged attention backend.
- `--device nvidia` + `LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn`: cuDNN paged attention backend.

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

## Run Tests

### 1. Test axis filters

`pytest` supports these global filters:

- `--device {all,cpu,nvidia}`
- `--layout {all,slot,block}`
- `--backend {all,native,cudnn}`
- `--model-path <local_model_dir>` for tests marked `requires_model`

Examples:

Run all CPU + native + block tests:

```bash
PYTHONPATH=python python -m pytest -q --device cpu --layout block --backend native
```

Run only NVIDIA + cuDNN tests:

```bash
PYTHONPATH=python python -m pytest -q \
  --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia --layout block --backend cudnn
```

### 2. Recommended smoke commands

Core/offline/online (no real model):

```bash
PYTHONPATH=python python -m pytest -q \
  test/core \
  test/engine \
  test/offline \
  test/online/test_online.py \
  test/online/test_online_http.py \
  test/online/test_online_stream_isolation.py \
  --device cpu --layout block --backend native
```

Parity (real model):

```bash
PYTHONPATH=python python -m pytest -q \
  test/parity/test_core_parity.py \
  test/parity/test_offline_parity.py \
  test/parity/test_infer.py \
  --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device all --layout all --backend all
```

Online real-model regression:

```bash
PYTHONPATH=python python -m pytest -q \
  test/online/test_online_real_model_multisession.py \
  --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia --layout block --backend all
```

### 3. CI helper script (optional)

```bash
python scripts/run_tests.py --suite all --run-parity never --run-hf never
python scripts/run_tests.py --suite stage2 --model-path models/DeepSeek-R1-Distill-Qwen-1.5B
```

## Notes

- If Web UI loads but cannot call API, check server URL and CORS-enabled API process.
- `favicon.ico 404` from `python -m http.server` is harmless.
- In restricted sandbox environments, HTTP bind tests may be skipped automatically.
- If `test/test_online_http.py` fails with `http.client.RemoteDisconnected`, check proxy environment variables (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`).
  The HTTP tests now disable proxies explicitly (via `urllib.request.ProxyHandler({})`) to force localhost direct connection.
