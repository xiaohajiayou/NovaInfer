# NovaInfer LLM 推理栈测试计划（2026-03-14）

本文定义当前代码口径下的测试策略、测试矩阵、执行方式、回归规则和通过标准。

关联文档：

1. 需求：`doc/novainfer_requirements_2026-03-14.md`
2. 设计：`doc/novainfer_overview_2026-03-14.md`
3. 性能验证：后续 A100 文档单独记录

## 1. 测试目标

1. 保证单卡主链路正确。
2. 保证在线/离线行为一致。
3. 保证 `BLOCK + cuDNN` 主路径可回归。
4. 保证 TP 正确性与性能不回退。
5. 保证重构时能快速定位影响范围。

## 2. 当前测试分层

### 2.1 Core

目录：`test/core/*`

覆盖：

1. 通用 model API
2. decode batch 语义
3. output/logits API
4. KV cache API
5. model registry
6. qwen2 adapter

### 2.2 Engine

目录：`test/engine/*`

覆盖：

1. state machine
2. scheduler
3. executor
4. block manager
5. model registry / runner split

### 2.3 Offline

目录：`test/offline/*`

覆盖：

1. `LLM.generate`
2. 入口参数规范
3. 离线输出对象
4. offline parity

### 2.4 Online

目录：`test/online/*`

覆盖：

1. OpenAI 风格在线接口
2. HTTP 行为
3. SSE 流式
4. cancel
5. stream isolation
6. 真实模型多会话

### 2.5 Parity

目录：`test/parity/*`

覆盖：

1. 单卡 C API / ModelRunner 与 HF 对拍
2. offline 路径与 HF 对拍
3. backend/device/layout 过滤矩阵

### 2.6 TP 专项

脚本：

1. `scripts/tp_hf_parity.py`
2. `scripts/tp2_smoke.py`
3. `scripts/bench_tp_novainfer.py`

覆盖：

1. TP2/TP4 正确性
2. TP 吞吐与扩展效率
3. 运行时环境打印

## 3. 用例设计原则

### 3.1 API 正确性

1. 非法参数返回明确错误。
2. 句柄 create/destroy 行为可重复、无泄漏语义。
3. 输出行与 `output_ids` 对齐。
4. attention metadata 缺失必须 fail-fast。

### 3.2 状态机正确性

1. 请求必须经历合法状态转移。
2. 终态必须带 `finish_reason`。
3. cancel/abort 不能遗留脏状态。

### 3.3 KV 正确性

1. 不同请求 KV 隔离。
2. prefix cache 命中与失配统计可信。
3. request free 后 block 能回收。
4. capacity/used/free/peak 统计正确。

### 3.4 在线正确性

1. SSE chunk 顺序正确。
2. 多流并发不串 request_id。
3. 终止 chunk 必达。
4. cancel 后客户端能收到合理终止语义。

### 3.5 TP 正确性

1. rank 间输出一致。
2. rank0 输出与 HF 一致。
3. communicator/shape 错误必须 fail-fast。
4. `tp=1` 与非 TP 单卡行为一致。

## 4. 当前推荐执行方式

### 4.1 基础回归

```bash
python scripts/run_tests.py --suite all --run-parity never --run-hf never
```

### 4.2 单卡 HF 对拍

```bash
pytest -q test/parity/test_core_parity.py \
  --model-path /path/to/model \
  --device nvidia \
  --backend cudnn

pytest -q test/parity/test_infer.py \
  --model-path /path/to/model \
  --device nvidia \
  --backend cudnn
```

### 4.3 TP 正确性

```bash
CUDA_VISIBLE_DEVICES=5,6 \
python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2
```

```bash
CUDA_VISIBLE_DEVICES=1,2,5,6 \
python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tp-size 4
```

### 4.4 TP 冒烟

```bash
CUDA_VISIBLE_DEVICES=5,6 python scripts/tp2_smoke.py --model-path /path/to/model
```

## 5. 测试矩阵

### 5.1 主矩阵

| 维度 | 当前主值 | 说明 |
|---|---|---|
| device | `cpu`, `nvidia` | NVIDIA 为主线 |
| backend | `native`, `cudnn` | `cudnn` 为性能主线 |
| layout | `block` | 当前主语义 |
| model | `qwen2` | 当前唯一完整模型 |

### 5.2 TP 矩阵

| 模型 | tp_size | 目标 |
|---|---:|---|
| 1.5B | 1 | 单卡基线 |
| 1.5B | 2 | correctness + throughput |
| 7B | 1 | 单卡基线 |
| 7B | 2 | correctness + throughput |
| 7B | 4 | correctness + throughput |

## 6. 回归触发规则

### 6.1 改动 C API / Core

若改动：

- `include/llaisys/models/model.h`
- `src/llaisys/model.cc`
- `src/llaisys/qwen2/qwen2_model.*`
- `src/ops/*`

至少重跑：

1. `test/core/*`
2. `test/parity/test_core_parity.py`
3. `test/parity/test_infer.py`
4. `tp_hf_parity.py`（涉及 TP 时）

### 6.2 改动 Engine

若改动：

- `engine/llm_engine.py`
- `engine/scheduler.py`
- `engine/executor.py`
- `engine/worker.py`
- `engine/gpu_model_runner.py`

至少重跑：

1. `test/engine/*`
2. `test/offline/*`
3. `test/online/*`
4. `test/parity/test_offline_parity.py`

### 6.3 改动 Model Adapter

若改动：

- `python/llaisys/models/qwen2.py`
- `engine/model_registry.py`
- `engine/runtime_factory.py`

至少重跑：

1. `test/core/test_qwen2_adapter.py`
2. `test/core/test_model_registry.py`
3. `test/parity/test_core_parity.py`
4. TP 用例

## 7. 通过标准

### 7.1 功能通过

1. `scripts/run_tests.py --suite all --run-parity never --run-hf never` 通过。
2. 指定模型路径后，单卡 parity 通过。
3. 在线流式与 cancel 主用例通过。

### 7.2 TP 通过

1. `tp=2` HF parity 通过。
2. `tp=4` HF parity 通过（适用模型）。
3. `tp=1` 单卡 benchmark 不低于当前基线。
4. `tp>1` benchmark 输出标准全局吞吐。

### 7.3 质量通过

1. 所有 bench/test 脚本输出的运行环境信息完整。
2. 文档中的命令可直接复现。
3. 新增模型或重构不破坏现有测试编排入口。

## 8. 性能验证口径

1. 单卡 NovaInfer vs vLLM 采用统一脚本：`scripts/run_perf_experiments.py`。
2. TP 性能采用统一脚本：`scripts/bench_tp_novainfer.py`。
3. TP 全局吞吐按：
   - `global_tokens / max(rank_run_seconds)`
4. 报告中需同时记录：
   - throughput
   - run_seconds
   - speedup
   - scaling efficiency

## 9. 当前已验证结论（代码现状）

1. Core/Engine/Offline/Online 主测试可通过。
2. `BLOCK + cuDNN` 单卡 parity 可通过。
3. TP2/TP4 HF parity 已打通。
4. `ParallelContext` 重构后，`tp=1` 性能未回退。
