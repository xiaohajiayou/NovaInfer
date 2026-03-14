# NovaInfer TP 复现指南

更新时间：2026-03-14
适用分支：`tp`
目的：提供当前 TP 实现的统一复现口径，包括构建、环境、HF parity、吞吐验收与最小 smoke。

---

## 1. 前提

当前 TP 验收依赖以下前提：

1. 后端固定为 `BLOCK + CUDNN`
2. 设备固定为 `NVIDIA + NCCL`
3. cuDNN 必须使用：
   `/home/xiaohajiayou/opt/cudnn-linux-x86_64-9.18.1.3_cuda12-archive`
4. 构建必须启用：
   `--nv-gpu=y --nv-cudnn=y --nv-nccl=y`

---

## 2. 统一环境

先进入仓库并设置统一环境变量：

```bash
cd /home/xiaohajiayou/NovaInfer

export CUDNN_HOME=/home/xiaohajiayou/opt/cudnn-linux-x86_64-9.18.1.3_cuda12-archive
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:/home/xiaohajiayou/NovaInfer/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn
```

说明：

1. `python/llaisys/libllaisys/__init__.py` 已实现优先从 `CUDNN_HOME` preload cuDNN
2. 但仍建议始终显式设置上述环境，避免运行口径漂移

---

## 3. 统一构建

必须按下面的口径重新编译：

```bash
xmake f -c --mode=release --nv-gpu=y --nv-cudnn=y --nv-nccl=y
xmake -j1
xmake install
```

若缺少 `--nv-nccl=y`，`tp_size > 1` 会在运行期失败，错误类似：

```bash
Qwen2 TP requires ENABLE_NCCL_API
```

---

## 4. 检查实际加载的库

确认当前 Python 进程确实加载的是目标 cuDNN / NCCL / libllaisys：

```bash
.venv/bin/python - <<'PY'
import ctypes
import llaisys.libllaisys

with open('/proc/self/maps', 'r', encoding='utf-8', errors='ignore') as f:
    for ln in f:
        if 'libcudnn.so' in ln or 'libnccl.so' in ln or 'libllaisys.so' in ln:
            print(ln.strip())

h = ctypes.CDLL('libcudnn.so.9')
h.cudnnGetVersion.restype = ctypes.c_size_t
print('cudnn_version =', h.cudnnGetVersion())
PY
```

预期：

1. `libcudnn.so` 来自 `CUDNN_HOME`
2. `cudnn_version = 91801`

---

## 5. 最小 Smoke

先验证 TP=2 主链路能否正常初始化并跑通：

```bash
CUDA_VISIBLE_DEVICES=5,6 \
.venv/bin/python scripts/tp2_smoke.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

预期：

1. rank0/rank1 都返回成功
2. 最后打印：
   `exitcodes rank0=0 rank1=0`

---

## 6. HF Parity 复现

口径：

1. `rank0` 必须与 HF 逐 token 完全一致
2. 各 rank 之间输出也必须完全一致

### 6.1 1.5B TP=2

```bash
CUDA_VISIBLE_DEVICES=5,6 \
.venv/bin/python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

### 6.2 7B TP=2

```bash
CUDA_VISIBLE_DEVICES=5,6 \
.venv/bin/python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tp-size 2 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

### 6.3 7B TP=4

```bash
CUDA_VISIBLE_DEVICES=1,2,5,6 \
.venv/bin/python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tp-size 4 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

预期：

1. 打印各 rank `ok`
2. 最后输出：
   `[tp_hf_parity] PASS`

说明：

1. 脚本已改为每次自动生成唯一 `init_method`
2. 不再依赖固定的 `/tmp/llaisys_tp_nccl_*.id`

### 6.4 内建 `mp` executor 口径

当前 TP 只保留一种正式启动口径：

1. `mp`
   - 内建 multiprocess executor 口径
   - 由 `LLMEngine` / `server` 内部拉起子 rank

`uni` 现在只保留给 `tp_size=1` 的单进程路径；`tp_size>1` 一律要求 `mp`。

`1.5B TP=2, mp executor`：

```bash
CUDA_VISIBLE_DEVICES=5,6 \
.venv/bin/python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2 \
  --tensor-parallel-device-ids 0,1 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

当前实测：

1. `HF parity: PASS`
2. `rank0` 与 HF 逐 token 对齐
3. 多 rank 输出一致

---

## 7. 吞吐复现

注意：

1. TP 吞吐不能把各 rank 吞吐相加
2. 当前脚本已经输出标准全局口径：
   `global_throughput = total_tokens / max(rank_run_seconds)`
3. 各 rank 的 `throughput_actual` 仍然会打印，但它们只是本地观测值，不是最终 TP 吞吐

### 7.1 1.5B 单卡基线

```bash
CUDA_VISIBLE_DEVICES=5 \
.venv/bin/python -u scripts/bench_compare_vllm.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --backend novainfer \
  --num-seqs 256 \
  --min-input-len 100 \
  --max-input-len 1024 \
  --min-output-len 100 \
  --max-output-len 1024 \
  --max-model-len 4096 \
  --seed 0 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384
```

### 7.2 1.5B TP=2

```bash
.venv/bin/python -u scripts/bench_tp_novainfer.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2 \
  --cuda-visible-devices 5,6 \
  --num-seqs 256 \
  --min-input-len 100 \
  --max-input-len 1024 \
  --min-output-len 100 \
  --max-output-len 1024 \
  --max-model-len 4096 \
  --seed 0 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384
```

### 7.2.1 1.5B TP=2，内建 `mp` executor

```bash
.venv/bin/python -u scripts/bench_tp_novainfer.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2 \
  --cuda-visible-devices 5,6 \
  --tensor-parallel-device-ids 0,1 \
  --num-seqs 256 \
  --min-input-len 100 \
  --max-input-len 1024 \
  --min-output-len 100 \
  --max-output-len 1024 \
  --max-model-len 4096 \
  --seed 0 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384
```

当前同口径实测：

1. `mp executor`：
   - `global_throughput = 9380.88 tok/s`
2. 历史外部 launcher `uni` 对照：
   - `global_throughput = 9644.08 tok/s`
3. 比值：
   - `mp / uni = 97.3%`

结论：

1. `BatchPlan + shared memory payload` 之后，内建 `mp` 路径已接近外部 launcher
2. server/offline 主链复现统一使用 `mp`
3. `uni` 的 TP 外部 launcher 口径已退役，只保留历史对照意义

### 7.3 7B 单卡基线

```bash
CUDA_VISIBLE_DEVICES=6 \
.venv/bin/python -u scripts/bench_compare_vllm.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --backend novainfer \
  --num-seqs 128 \
  --min-input-len 100 \
  --max-input-len 1024 \
  --min-output-len 100 \
  --max-output-len 1024 \
  --max-model-len 4096 \
  --seed 0 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192
```

### 7.4 7B TP=2

```bash
.venv/bin/python -u scripts/bench_tp_novainfer.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tp-size 2 \
  --cuda-visible-devices 5,6 \
  --num-seqs 128 \
  --min-input-len 100 \
  --max-input-len 1024 \
  --min-output-len 100 \
  --max-output-len 1024 \
  --max-model-len 4096 \
  --seed 0 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192
```

### 7.5 7B TP=4

```bash
.venv/bin/python -u scripts/bench_tp_novainfer.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tp-size 4 \
  --cuda-visible-devices 1,2,5,6 \
  --num-seqs 128 \
  --min-input-len 100 \
  --max-input-len 1024 \
  --min-output-len 100 \
  --max-output-len 1024 \
  --max-model-len 4096 \
  --seed 0 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192
```

---

## 8. 当前参考结果

以下为最近一次在正确构建与环境口径下的参考值：

### 8.1 HF parity

1. `1.5B, TP=2`：`PASS`
2. `7B, TP=2`：`PASS`
3. `7B, TP=4`：`PASS`

### 8.2 吞吐

`1.5B`

1. `TP=1`：`8474.88 tok/s`
2. `TP=2`：以脚本 `summary` 中的 `global_throughput` 为准
3. `TP=2, mp executor`：`9380.88 tok/s`
4. `TP=2, retired uni baseline`：`9644.08 tok/s`

`7B`

1. `TP=1`：`2862.61 tok/s`
2. `TP=2`：以脚本 `summary` 中的 `global_throughput` 为准
3. `TP=4`：以脚本 `summary` 中的 `global_throughput` 为准

---

## 9. 常见错误

### 9.1 忘记设置 `LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn`

现象：

1. 吞吐明显偏低
2. 与之前的 CUDNN 结果完全不可比

### 9.2 忘记带 `--nv-nccl=y`

现象：

1. 单卡可跑
2. 多卡 TP 在 warmup 或首次 forward 直接失败
3. 报错：
   `Qwen2 TP requires ENABLE_NCCL_API`

### 9.3 进程实际吃到了 `.venv` 自带 cuDNN

现象：

1. `ctypes.CDLL("libcudnn.so.9").cudnnGetVersion()` 不是 `91801`
2. `/proc/self/maps` 里 `libcudnn.so` 路径不在 `CUDNN_HOME`

### 9.4 GPU 被其他作业占用

现象：

1. 单卡吞吐和历史结果差很多
2. 多卡不同轮次波动很大
3. `nvidia-smi` 显示目标卡已有大显存占用或高 util

建议：

1. 复验前先看：
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```
2. 尽量使用空闲卡，并连续跑 3 次取中位数

### 9.5 `mp` 与 `uni` 比较口径不一致

现象：

1. `mp` 与 `uni` 差距异常大
2. 不同人复现结果互相对不上

检查：

1. 模型、workload、`CUDA_VISIBLE_DEVICES`、`tensor_parallel_device_ids` 必须完全一致
2. 运行时 `cudnn/nccl` 加载路径必须一致
3. `tp_size > 1` 现在强制要求 `distributed_executor_backend=mp`

---

## 10. 推荐验收顺序

建议按这个顺序执行：

1. `统一环境`
2. `统一构建`
3. `检查实际加载的库`
4. `tp2_smoke`
5. `HF parity`
6. `单卡吞吐`
7. `TP 吞吐`

这样可以最快区分：

1. 是构建问题
2. 是环境问题
3. 还是 TP 逻辑本身问题
