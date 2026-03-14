# MXMACA / 沐曦平台使用说明

## 1. 适用范围

本文档说明 NovaInfer 在 MXMACA 环境上的当前使用方式。

当前口径：

1. 现有 CUDA 风格算子源码继续复用，未复制一套 `maca` 算子目录。
2. MACA 构建链使用 `mxcc -x maca`，不再依赖 fake CUDA SDK 或额外 shell 脚本。
3. Python 对外设备口径暂时仍使用 `DeviceType.NVIDIA` / `--device nvidia`，含义是“CUDA-compatible GPU backend”。

## 2. 环境前提

要求本机已安装 MXMACA，且默认安装在：

```bash
/opt/maca
```

要求模型已存在，例如：

```bash
/root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

## 3. 构建

在仓库根目录执行：

```bash
cd /root/NovaInfer
xmake f --root -c --maca-gpu=y --maca-cudnn=n
xmake --root -r -j4
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/libllaisys.so
```

说明：

1. `--maca-gpu=y` 打开 MXMACA 构建路径。
2. `--maca-cudnn=n` 表示当前先走 `native` attention 路径。
3. `cp ... libllaisys.so ...` 是为了让 Python 包加载到最新构建产物。

## 4. 最小验证

### 4.1 验证 Python 侧能加载动态库

```bash
cd /root/NovaInfer
python - <<'PY'
import sys
sys.path.insert(0, 'python')
import llaisys
print('import ok')
PY
```

### 4.2 验证 runtime 能识别设备

```bash
cd /root/NovaInfer
python - <<'PY'
import sys
sys.path.insert(0, 'python')
import llaisys, torch
api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
print('device_count =', api.get_device_count())
print('torch.cuda.is_available =', torch.cuda.is_available())
PY
```

在 MXMACA 环境下，当前可能出现：

```text
device_count = 1
torch.cuda.is_available = False
```

这不代表 NovaInfer 没走 GPU。它只说明当前 PyTorch 没把该环境识别成标准 CUDA。

### 4.3 验证真实模型离线推理

```bash
cd /root/NovaInfer
PYTHONPATH=python python - <<'PY'
from llaisys.entrypoints.llm import LLM
from llaisys import DeviceType

llm = LLM(
    '/root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    model_type='qwen2',
    device=DeviceType.NVIDIA,
    max_model_len=512,
    max_num_seqs=1,
    kv_cache_memory_utilization=0.5,
)

out = llm.generate('请用一句中文介绍你自己。', max_new_tokens=8)
print(out[0]['text'])
llm.close()
PY
```

## 5. 测试

### 5.1 主测试

```bash
cd /root/NovaInfer
PYTHONPATH=python pytest test --model-path /root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --backend native -q
```

### 5.2 parity 测试

```bash
cd /root/NovaInfer
PYTHONPATH=python pytest test/parity/test_core_parity.py -q --model-path /root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --backend native
```

```bash
cd /root/NovaInfer
PYTHONPATH=python pytest test/parity/test_infer.py -q --model-path /root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --backend native
```

```bash
cd /root/NovaInfer
PYTHONPATH=python pytest test/parity/test_offline_parity.py -q --model-path /root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --backend native
```

## 6. 实现说明

当前 MACA 构建相关文件：

1. `xmake/maca.lua`
2. `xmake/maca_sources/*`

设计要点：

1. `xmake` 不再直接处理原始 `.cu` 文件，否则会触发内建 CUDA SDK 探测。
2. `xmake/maca_sources/*.maca` 是很薄的 wrapper，每个文件只 `#include` 原始 `.cu`。
3. 真正的 GPU 编译器是：

```bash
/opt/maca/mxgpu_llvm/bin/mxcc
```

## 7. 当前限制

1. 对外设备名仍沿用 `nvidia`，尚未引入独立 `maca` 设备类型。
2. 当前验证口径以 `native` attention 为主，未把 `maca-cudnn` 作为默认路径。
3. 部分测试原本依赖 `torch.cuda.is_available()`，已在 parity 路径做兼容；其他强依赖 Torch CUDA 张量路径的测试仍需按实际情况区分。
