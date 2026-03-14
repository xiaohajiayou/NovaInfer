# NovaInfer LLM 推理栈需求文档（2026-03-14）

本文件描述当前 NovaInfer LLM 推理栈的目标、范围、需求、约束与验收标准。本文不展开实现细节，设计方案见 `doc/novainfer_overview_2026-03-14.md`，测试方案见 `doc/novainfer_test_plan_2026-03-14.md`。

## 1. 目标与范围

### 1.1 目标

1. 提供可复用的 LLM 推理栈，而不是只做单模型脚本。
2. 支持离线推理与在线 OpenAI 兼容服务。
3. 支持连续批处理、多请求并发、KV Cache 复用。
4. 支持 NVIDIA GPU 主路径，并以 `BLOCK + cuDNN` 作为性能主线。
5. 支持多 GPU Tensor Parallel（当前实现为多进程 TP + NCCL）。
6. 保持模型适配成本可控，新增模型时不需要改 Engine 主流程。

### 1.2 当前范围

1. 当前首个完整落地模型为 `Qwen2` 系列兼容模型。
2. 当前主设备路径为：
   - CPU：功能稳定、回归可用。
   - NVIDIA：性能主线。
3. 当前主 KV 形态为 `BLOCK`。
4. 当前对外服务主接口为 `OpenAI chat.completions`。
5. 当前 TP 主线为 `Tensor Parallel`，不包含 `Pipeline Parallel`。

### 1.3 不在本次范围

1. 多模态输入。
2. 分布式服务调度、远程 KV 迁移、跨节点推理。
3. Pipeline Parallel、Expert Parallel。
4. Embedding/Completions 独立产品化接口。

## 2. 分层需求与职责边界

### 2.1 总体分层

1. `Server`：协议适配、HTTP/SSE、取消、错误响应。
2. `Engine`：请求状态机、调度、执行编排、输出组织。
3. `Model Adapter`：模型配置解析、权重加载、Tokenizer、C API 绑定。
4. `Core/C++`：模型 forward、KV Cache、Tensor、算子、采样、并行上下文。
5. `Device Runtime API`：设备内存、拷贝、流、事件、同步。

### 2.2 边界约束

1. Server 不直接做调度或模型前向。
2. Engine 不持有模型私有权重语义。
3. Model Adapter 不实现调度策略。
4. Core 不实现请求公平性和服务协议。
5. 设备能力与模型运行态解耦；KV 状态与并行上下文解耦。

## 3. Core 层需求（C++）

### 3.1 通用模型接口

1. 提供统一模型句柄 create/destroy/forward 接口。
2. 提供统一 weight replace 能力，避免重复赋值泄漏。
3. 提供统一 sampler 接口，执行侧基于 logits 产出 token。
4. 模型 forward 输入必须显式携带 attention metadata。

### 3.2 Tensor 与设备运行时

1. Tensor 支持 CPU/NVIDIA 设备、dtype、shape、slice/view/load。
2. Device Runtime API 提供：
   - malloc/free
   - memcpy sync/async
   - stream create/destroy/sync
   - event create/destroy/record/wait
3. 运行时 API 必须是模型无关的。

### 3.3 KV Cache

1. KV Cache 作为独立运行态对象存在，不与模型权重混合管理。
2. 支持 `BLOCK` 布局。
3. 支持请求级分配、释放、统计、前缀缓存。
4. 资源不足时必须显式失败，不允许 silent fallback。
5. KV 统计必须可观测：capacity/used/free/peak。

### 3.4 Parallel Context

1. 并行上下文必须独立于 KV 状态。
2. 必须提供独立 create/destroy/bind 接口。
3. 一个模型实例只能绑定一个并行上下文。
4. 并行上下文负责：
   - TP size
   - rank/local_rank
   - device_ids
   - backend/init_method
   - communicator 生命周期
5. 当前只支持多进程 TP + NCCL。

### 3.5 模型执行

1. 模型执行必须覆盖：embedding -> transformer blocks -> final norm -> lm head。
2. attention 路径必须支持 `BLOCK + cuDNN`。
3. prefill/decode 均通过统一 forward 协议执行。
4. Tensor Parallel 下必须支持：
   - 本地权重分片
   - 本地 KV head 分片
   - attention/MLP 必要 collective
5. TP 实现必须保证：
   - 正确性对齐 HF
   - `tp=1` 性能不回退

## 4. Engine 层需求（Python）

### 4.1 入口层

1. 提供离线 `LLM.generate/stream`。
2. 提供在线异步入口供 Server 调用。
3. 接收统一 `SamplingParams`。
4. 返回统一输出对象，而不是裸 logits。

### 4.2 请求状态机

1. 维护 `waiting/running/finished_*` 等请求状态。
2. 支持提交、step、collect、cancel。
3. 请求终态必须带 `finish_reason/status/usage`。
4. Engine 层必须可独立测试，不依赖 HTTP Server。

### 4.3 Scheduler

1. 支持 waiting/running 队列。
2. 支持 prefill 与 decode 编排。
3. 受 `max_num_seqs` 与 `max_num_batched_tokens` 约束。
4. 与 BlockManager 协同完成 block 分配、append、free。
5. 调度结果仅描述执行计划，不直接触发计算。

### 4.4 Executor / Worker

1. Executor 负责一次 step 的执行协调。
2. Worker 负责持有 model、kv_state、parallel_context 与 sampler。
3. Worker 必须支持 encode/decode 辅助能力供在线服务复用。
4. Worker 的 GPU 主路径必须支持 TP 参数传入。

### 4.5 模型注册与适配

1. 模型注册必须按 `model_type` 解耦。
2. 新模型接入至少需要：
   - model wrapper
   - kv_state factory
   - parallel_context factory
3. 不允许为了新增模型修改 Engine 主流程。

## 5. Server 层需求

### 5.1 HTTP / OpenAI 兼容

1. 提供 `/health`。
2. 提供 `/v1/chat/completions`。
3. 支持 SSE 流式响应。
4. 支持 `/v1/requests/{id}/cancel`。
5. 错误必须返回结构化 JSON。

### 5.2 输出语义

1. 非流式输出必须包含：
   - id/object/model
   - choices
   - usage
   - request_id/status
2. 流式输出必须包含逐 chunk `delta` 与终止 chunk。
3. reasoning 解析与普通内容输出必须语义明确。

## 6. 模型适配需求

1. 模型配置读取自 HuggingFace 目录。
2. 权重加载支持 `safetensors` 分片。
3. 权重加载链路保持零 torch 运行时依赖。
4. BF16 权重必须按真实位模式解码，不能错误数值转换。
5. TP 下权重分片规则必须模型私有、显式、可验证。

## 7. 可观测性与性能需求

1. 关键执行路径支持 NVTX 标注。
2. bench 脚本必须输出：
   - backend
   - CUDA_VISIBLE_DEVICES
   - cuDNN/NCCL 加载信息
3. TP 吞吐口径必须按：
   - `global_tokens / max(rank_run_seconds)`
4. 必须能区分 `init/warmup/run/total` 时间。

## 8. 验收标准

### 8.1 功能验收

1. Core/Engine/Offline/Online 主测试通过。
2. `Qwen2` 单卡离线与在线主链路可用。
3. `BLOCK + cuDNN` 主路径可用。
4. TP2/TP4 主链路可运行。

### 8.2 正确性验收

1. 单卡 parity 对齐 HF。
2. `tp=2` parity 对齐 HF。
3. `tp=4` parity 对齐 HF（适用模型）。
4. 在线流式输出不串请求、不丢终止块。

### 8.3 性能验收

1. 单卡 `tp=1` 不因 TP 引入而掉性能。
2. TP 吞吐随卡数增加有可解释提升。
3. 单卡 A100 基准需可与 3090 历史口径横向比较。

### 8.4 代码质量验收

1. 需求、设计、测试、性能文档与当前代码一致。
2. 不保留已废弃主路径的误导性描述。
3. 无多余 fallback 语义掩盖真实失败。
