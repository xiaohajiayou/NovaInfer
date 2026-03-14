# TP Server MP Executor Design

更新时间：2026-03-14
状态：实现中，已完成正确性打通，性能已接近外部 launcher

## 1. 目标

在当前 TP 已具备底层执行能力的前提下，补齐正式的服务与离线入口，使其对齐 vLLM 的使用方式：

1. `python -m llaisys.server` 可直接启动单机多卡 TP 服务。
2. `LLM` / `LLMEngine` 可直接通过配置启用 TP，而不是依赖 bench 脚本充当 launcher。
3. `bench_tp_novainfer.py` 退回“验收工具”角色，不再是唯一 TP 启动方式。
4. 维持当前 `ParallelContext` 设计，不把 TP 状态重新塞回 `kv_state`。

当前范围只覆盖：

1. 单机
2. NVIDIA
3. `distributed_executor_backend="mp"`
4. `BLOCK + CUDNN`

不覆盖：

1. 多机
2. `ray`
3. 外部 launcher
4. CPU TP

## 2. 当前状态

当前仓库的 TP 分成两层：

1. rank 内能力：已完成
   - `EngineConfig` 已支持 `tensor_parallel_size/tp_rank/tp_local_rank/tensor_parallel_device_ids`
   - `GPUModelRunner` 会创建并绑定 `ParallelContext`
   - `Qwen2Model` 已支持 TP 本地 shape 与 NCCL allreduce

2. 多进程拉起：未产品化
   - 当前只有 [bench_tp_novainfer.py](/home/xiaohajiayou/NovaInfer/scripts/bench_tp_novainfer.py) 在外层 `subprocess.Popen(...)` 拉起多 rank
   - [python/llaisys/server/__main__.py](/home/xiaohajiayou/NovaInfer/python/llaisys/server/__main__.py) 未暴露 TP 参数
   - [LLMEngine](/home/xiaohajiayou/NovaInfer/python/llaisys/engine/llm_engine.py) 未内建 distributed executor

因此当前属于：

- TP 算法链路已完成
- TP 产品入口未完成

## 3. 对齐目标

对齐 vLLM 的点：

1. TP 是 `LLM/Engine/Server` 的正式配置，而不是 bench 特例
2. 使用 `distributed_executor_backend` 选择执行后端
3. 单机多卡场景下，由系统内部管理多进程 worker 生命周期

不对齐 nano-vllm 的点：

1. 不把所有多进程控制逻辑硬塞进一个极简 `ModelRunner`
2. 不手写共享内存 + event 协议作为长期架构

原因：

1. 当前代码已经有 `EngineConfig / LLMEngine / AsyncLLMEngine / Worker / GPUModelRunner / ParallelContext`
2. 分层已经更接近 vLLM，不适合回退成原型式结构

## 4. 目标架构

### 4.1 配置层

`EngineConfig` 扩展为正式的分布式执行配置：

1. `tensor_parallel_size: int`
2. `tensor_parallel_device_ids: Sequence[int] | None`
3. `distributed_executor_backend: str = "uni"`
4. `tp_rank: int`
5. `tp_local_rank: int`
6. `tp_init_method: str | None`

约束：

1. `uni`：单进程单 rank
2. `mp`：单机多进程 TP
3. 其余后端先 fail-fast

### 4.2 Executor 层

新增显式执行器分层：

1. `UniProcExecutor`
   - 当前已有逻辑
   - 单进程调用本地 `Worker`

2. `MPExecutor`
   - 新增
   - 负责拉起 rank1..N 子进程
   - rank0 在父进程
   - 对上暴露与 `UniProcExecutor` 相同的接口：
     - `execute_model(...)`
     - `sample_tokens(...)`
     - `check_health()`
     - `close()`

### 4.3 Worker 层

每个 rank 进程都创建自己的：

1. `Worker`
2. `GPUModelRunner`
3. `ParallelContext`
4. `KvState`

其中：

1. `ParallelContext` 是 rank-local 的
2. `tp_rank/tp_local_rank/device_ids/init_method` 由 executor 注入

### 4.4 Server 层

`python -m llaisys.server` 支持：

1. `--tensor-parallel-size`
2. `--tensor-parallel-device-ids`
3. `--distributed-executor-backend`

在 `mp + tp_size>1` 时：

1. 主进程创建 `AsyncLLMEngine(backend="mp")`
2. 只有主进程绑定 HTTP 端口
3. 子进程不启动 HTTP，仅参与 TP 执行

## 5. 进程模型

### 5.1 单进程

适用：

1. `distributed_executor_backend="uni"`
2. `tensor_parallel_size=1`

结构：

- HTTP / LLM
- AsyncLLMEngine
- LLMEngine
- Executor
- Worker

### 5.2 MP TP

适用：

1. `distributed_executor_backend="mp"`
2. `tensor_parallel_size>1`

结构：

- 主进程：
  - HTTP server
  - AsyncLLMEngine
  - LLMEngine
  - MPExecutor(rank0)
  - Worker(rank0)

- 子进程 rank1..N：
  - MP worker loop
  - Worker(rankN)

### 5.3 启动方式

采用：

1. Python `multiprocessing`
2. start method 固定 `spawn`

原因：

1. CUDA/NCCL 进程安全要求更严格
2. 避免 `fork` 继承父进程 CUDA 状态导致不稳定

## 6. 进程间控制协议

第一版不做通用 RPC 框架，采用最小控制协议。

rank0 负责广播以下控制命令：

1. `execute_model`
2. `sample_tokens`
3. `close`
4. `health_check`

但注意：

1. 对 TP 推理而言，真正需要同步执行的核心是 `execute_model`
2. `sample_tokens` 语义上只在 rank0 输出结果
3. 非 rank0 进程在 `sample_tokens` 阶段不需要重复导出 token list

因此第一版更推荐：

1. `execute_model` 广播到所有 rank
2. `sample_tokens` 只由 rank0 本地完成
3. 子 rank 只参与 forward 和 collective

## 7. 数据流

### 7.1 在线请求

1. HTTP 请求进入主进程
2. `AsyncLLMEngine.submit(...)`
3. `LLMEngine` 调度出 batch
4. `MPExecutor.execute_model(...)`
5. rank0 将同一批 scheduler outputs 广播给所有 rank
6. 所有 rank 各自调用本地 `Worker.execute_model(...)`
7. Qwen2 TP forward 内部完成 NCCL allreduce
8. rank0 拿到 logits 并执行 `sample_tokens(...)`
9. `scheduler.postprocess(...)`
10. 结果经 stream / collect 返回 HTTP

### 7.2 时序图

```text
Client
  -> HTTP Server(rank0)
  -> AsyncLLMEngine(rank0)
  -> LLMEngine(rank0)
  -> MPExecutor(rank0)
  -> Worker(rank0) ----------------------+
  -> Worker(rank1) ------------------+   |
  -> Worker(rank2) --------------+   |   |
                                 |   |   |
                 TP forward + NCCL allreduce
                                 |   |   |
  <- logits(rank0) <-------------+---+---+
  <- sampler(rank0)
  <- scheduler postprocess(rank0)
  <- stream / response(rank0)
```

## 8. API 设计

### 8.1 EngineConfig

新增/冻结：

1. `distributed_executor_backend: str = "uni"`
2. `tp_init_method: str | None = None`

保留：

1. `tensor_parallel_size`
2. `tensor_parallel_device_ids`
3. `tp_rank`
4. `tp_local_rank`
5. `distributed_backend`

### 8.2 LLM / LLMEngine

要求：

1. `LLM(...)` 可直接传 `tensor_parallel_size`
2. `LLMEngine(...)` 可直接传 `distributed_executor_backend="mp"`

## 9. 当前实现进展

已完成：

1. `EngineConfig` 新增 `distributed_executor_backend/tp_init_method`
2. 新增 [mp_executor.py](/home/xiaohajiayou/NovaInfer/python/llaisys/engine/mp_executor.py)
3. [LLMEngine](/home/xiaohajiayou/NovaInfer/python/llaisys/engine/llm_engine.py) 已按 `uni/mp` 选择 executor
4. [server/__main__.py](/home/xiaohajiayou/NovaInfer/python/llaisys/server/__main__.py) 已支持 TP server CLI 参数
5. `runtime_factory/model_registry/gpu_model_runner` 已把 `tp_init_method` 接到 `ParallelContext`

## 10. 当前验证结果

### 10.1 正确性

`1.5B, tp=2, mp executor`：

```bash
CUDA_VISIBLE_DEVICES=5,6 \
python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --distributed-executor-backend mp \
  --tp-size 2 \
  --tensor-parallel-device-ids 0,1 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

结果：`PASS`

### 10.2 启动链

已验证：

1. `LLM.generate(..., distributed_executor_backend="mp", tensor_parallel_size=2)` 可成功返回
2. `python -m llaisys.server --tensor-parallel-size 2 ...`
   - `/health` 返回 `ok`
   - `/v1/chat/completions` 可成功生成

### 10.3 吞吐

同口径 `1.5B, tp=2, 256 seq, [100,1024] x [100,1024], max_num_batched_tokens=16384`：

1. 旧版 `mp executor`（直接 `Pipe` 广播 `SchedulerOutputs`）
   - `global_throughput = 4627.11 tok/s`
2. 当前 `mp executor`（`BatchPlan` 扁平化 + shared memory payload）
   - `global_throughput = 9380.88 tok/s`
3. 历史外部 launcher `uni` 基线
   - `global_throughput = 9644.08 tok/s`

结论：

1. `mp executor` 的主要瓶颈确实来自每步 Python 对象广播
2. 扁平 `BatchPlan` 已经解决大头
3. shared memory payload 进一步把 `mp` 拉到 `uni` 的约 `97.3%`
4. 当前版本已基本达到“server 内建 TP 启动不明显掉性能”的目标

## 11. 性能差距原因判断

当前结论：

1. 问题根因不是 NCCL 或 TP 算法错误，而是 Python 进程间控制开销
2. `rank0 Pipe 广播 SchedulerOutputs` 会触发明显的 pickle / unpickle 和对象复制
3. 改为 `BatchPlan` 后，IPC 载荷被压平
4. 再把 plan payload 放进 shared memory 后，只剩极小控制消息走 `Pipe`
5. 这条路径已经足够把 `mp executor` 拉回到接近历史外部 launcher 基线的性能水平
3. 对用户而言，不再需要 bench 那套 rank 参数显式循环

### 8.3 Server CLI

新增参数：

1. `--tensor-parallel-size`
2. `--tensor-parallel-device-ids`
3. `--distributed-executor-backend`
4. `--tp-init-method`（可选）

默认：

1. `tensor_parallel_size=1`
2. `distributed_executor_backend="uni"`

## 9. 实现路线

### Step 1. 配置打通

1. 扩 `EngineConfig`
2. 扩 `LLM` / `LLMEngine` 参数解析
3. 扩 server CLI

### Step 2. 引入 Executor 选择

1. 当前 [Executor](/home/xiaohajiayou/NovaInfer/python/llaisys/engine/executor.py) 只有单实现
2. 需要改成工厂式：
   - `UniProcExecutor`
   - `MPExecutor`

### Step 3. 实现 MPExecutor

1. `spawn` rank1..N
2. 构造 rank-local `EngineConfig`
3. 每个子进程创建本地 `Worker`
4. 实现最小控制协议

### Step 4. 主进程接管 HTTP

1. server 只在 rank0 进程启动
2. 子进程仅作为 TP worker

### Step 5. 验收与回归

1. 单卡 server 不回归
2. TP server 可启动
3. TP server 可流式返回
4. TP correctness 与 bench 一致

## 10. 失败语义

必须 fail-fast：

1. `tensor_parallel_size>1` 但 `device!=nvidia`
2. `distributed_executor_backend!="mp"` 且 `tp_size>1`
3. `tensor_parallel_device_ids` 长度不匹配
4. 子进程任一 rank 初始化失败
5. 子进程任一 rank 健康检查失败
6. rank0/child 之间控制协议断裂

日志必须打印：

1. `tp_size`
2. `rank`
3. `local_rank`
4. `device_ids`
5. `backend`
6. `init_method`

## 11. 验收标准

### 11.1 功能

1. `python -m llaisys.server --tensor-parallel-size 2 --distributed-executor-backend mp ...` 可启动
2. `curl /v1/chat/completions` 正常返回
3. stream 模式正常返回
4. rank0 端口绑定，非 rank0 不绑定端口

### 11.2 正确性

1. server TP 输出与现有 [tp_hf_parity.py](/home/xiaohajiayou/NovaInfer/scripts/tp_hf_parity.py) 口径一致
2. `tp=1` 与当前单卡服务输出一致

### 11.3 性能

1. `tp=1` server 性能不下降
2. `tp=2/4` server 性能趋势与 bench 一致
3. 不允许因引入 `mp executor` 让单卡路径平白增加明显开销

## 12. 与当前 bench 的关系

保留：

1. [bench_tp_novainfer.py](/home/xiaohajiayou/NovaInfer/scripts/bench_tp_novainfer.py)

但其角色变更为：

1. TP 性能验收工具
2. TP 回归脚本

不再承担：

1. 唯一的 TP 进程拉起入口

## 13. 当前待确认点

### 13.1 rank0 是否唯一承担采样与 HTTP

建议：是。

原因：

1. 当前 sampler 输出语义本来就只需要一份 token ids
2. 非 rank0 再重复导出没有价值

### 13.2 MPExecutor 第一版是否只支持 server + offline generate

建议：是。

原因：

1. 先把主入口做通
2. bench 脚本仍可保留，降低切换风险

### 13.3 是否保留 `uni` 作为默认后端

建议：是。

原因：

1. 单卡仍是主路径
2. `tp=1` 不应强制走 `mp`

## 14. 建议结论

建议按 vLLM 方式推进：

1. TP 作为 `Engine/Server` 正式能力接入
2. 单机多卡使用 `mp executor`
3. `ParallelContext` 继续作为 rank-local TP runtime state
4. 不再让 bench 脚本承担唯一 launcher 角色

这条路线和当前代码结构兼容，也最符合后续产品化服务入口。
