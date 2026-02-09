# 需求文档设计

本需求文档面向“离线推理 + 在线服务”两种形态，描述交付要求与验收标准，不包含实现细节；实现允许分阶段落地。

## 1. 目标与范围

设计目标：
1. 支持 Qwen2 系列模型在本仓库推理栈上完成离线与在线推理。
2. 提供清晰的分层边界与责任划分，确保可扩展到多模型与多设备。
3. 支持多序列并发推理，具备连续批处理与 KV-Cache 复用能力。
4. 为后续 LLM 与多模态模型扩展预留接口。
5. 设计需考虑后续分布式推理场景（如张量并行）。

业务范围：
1. 离线单条/批量推理。
2. 在线服务（OpenAI 接口 + 流式）。
   1. 单用户场景：每个用户仅需一个会话，会话内多轮对话。
   2. 多用户场景：支持多个用户并发会话，每个用户有独立的上下文（尽量复用）。

## 2. 分层与责任边界（需求）

1. Server 层：负责对外 API、会话管理、限流、流式输出与在线请求生命周期管理。
2. Engine 层：负责请求调度与执行计划（含 Scheduler/Executor/Worker 的责任拆分），并驱动推理流程。
3. ModelRunner 层：负责模型适配与执行组织（权重映射、Tokenizer、输入输出组织与单次推理调用）。
4. Core 层（C++）：负责 batch/ubatch 执行、KV-Cache 管理与算子计算。

## 3. Core 层需求（C++）

### 3.1 内存与资源管理
1. 模型权重加载与生命周期管理。
2. 模型计算中间 buffer 设计与复用。
3. KV-Cache 设计与多序列支持。
4. 输出 buffer 管理。

### 3.2 KV-Cache 行为要求
1. 基于 slot（cell）的物理管理，slot 记录逻辑 pos 与 seq_id。
2. 支持多序列混排，但 attention 必须按 seq_id 严格隔离。
3. 资源不足默认失败返回，不自动回收或截断。
4. （可选，阶段三）滑窗注意力不改变逻辑 pos，仅通过 mask 屏蔽窗口外 token。
5. 需提供前缀复用、释放/截断、位置平移、保留与查询等能力接口（不限定 API 名称）；由 Engine 层负责调用与编排。
6. Core 层默认不强制每 `seq_id` 的 slot/cell 硬配额；请求公平性与限额由 Engine/Server 层负责。

### 3.3 计算图与算子
1. 计算图覆盖 embedding → N×block → final norm → lm head。
2. block 需包含 attention（Q/K/V 投影 + RoPE + KV-Cache 读写 + 输出投影）、MLP（gate/up/down + SwiGLU）、残差连接与 RMSNorm。
3. RoPE 位置与 KV-Cache 对齐，新增 token 使用其逻辑 pos。
4. 支持 batch/ubatch 执行路径，prefill/decode 仅体现为输入形态差异。
5. 常用算子：embedding、linear、rms_norm、rope、self_attention、swiglu、argmax。
6. 算子支持常见 dtype 并具备一致性检查。



## 4. Engine 层需求（Python）

### 4.1 Scheduler
1. 维护请求队列与调度策略（prefill/decode 混排）。
2. 组 batch/ubatch，并控制每轮执行的 token 集合。
3. 支持连续批处理与多序列并发。
4. 自回归驱动：支持 prompt prefill 与 decode 循环；支持 max_new_tokens、stop token、stop string；支持流式输出回调与 UTF-8 拼接。
5. 负责请求级公平调度与上下文配额（例如轮转策略、每请求最大上下文/增量 token 限制）。

### 4.2 Executor
1. 接收调度计划并组织执行。
2. 负责与 Worker 协作，驱动模型执行并收集结果。

### 4.3 Worker
1. 负责实例化 ModelRunner 并执行推理。
2. 支持多设备扩展的接口预留（单机多卡 TP 为后续阶段目标），Core 提供多设备内存接口支持。

### 4.4 ModelRunner
1. 模型文件解析：支持 HuggingFace 目录结构，读取 config/tokenizer。
2. 权重加载：支持单文件或分片 safetensors；负责 HF 权重名到 Core 权重槽位的映射与校验。
3. Tokenizer：支持 AutoTokenizer、Qwen2 chat template 与特殊 token（bos/eos/pad/system/user/assistant）。
4. 采样参数透传至采样模块（采样策略由 Engine 层控制）。

### 4.4 模型选择与加载
1. Engine 初始化时必须明确模型类型与模型路径。
2. 首期仅要求支持 Qwen2，但需预留扩展到多模型的接口。
3. 模型实例化与权重加载在 Worker 层完成，Engine 负责配置与路由。

## 5. Server 层需求（Python，按 vLLM 分层）

### 5.1 API Server

1. API 协议：提供 HTTP 服务接口与 OpenAI 兼容 API，覆盖 completions/chat/completions/embeddings 等基础路由。
2. 流式输出：支持 SSE/流式返回，保证 token 级增量输出与正确的结束语义。
3. 请求取消：支持主动取消与超时取消，确保释放对应请求与 KV-Cache 资源。
4. 会话管理：单用户会话、多用户并发；会话内多轮对话上下文可复用。
5. 隔离与限流：请求级隔离、并发限制、排队上限与背压策略。
6. 可选增加用户/会话级 token budget，防止单用户长期占用资源。
7. 日志与错误：提供请求级日志与错误码，关键失败信息可追踪。

### 5.2 AsyncLLMEngine

1. 异步入口：接收 API Server 请求并异步提交到 LLMEngine。
2. 流式转发：将增量 token 结果以流式方式回传给 API Server。
3. 请求生命周期：维护请求状态（排队/执行/结束/取消）与结果聚合。

### 5.3 LLMEngine

1. 调度协作：驱动 Scheduler 进行 prefill/decode 混排与 batch/ubatch 组装。
2. 执行协作：协调 Executor/Worker 进行模型执行并回收结果。
3. 多模型路由：基于 model 字段选择模型类型与实例（首期可仅启用 Qwen2，但需保留接口）。
4. 监控指标：提供吞吐、延迟、KV-Cache 使用率、CPU/GPU 资源与内存统计。

## 6. Web UI（模块需求）

### 6.1 UI 客户端

1. 提供交互式聊天界面（Web UI）。
2. 支持发送请求与展示回复（含流式输出）。
3. 支持连续对话与本地历史记录（单用户场景即可）。

### 6.2 UI 服务端适配

1. 对接 Server 层 OpenAI `chat-completions` API。
2. 支持 SSE 流式响应解析与渲染。
3. 支持请求取消与错误提示。

### 6.3（可选）多会话与 KV 复用

1. 支持多会话创建/切换。
2. 支持修改历史问题并重新生成回答。
3. 支持前缀匹配的 KV-Cache 复用（与 Engine 能力对齐）。

## 7. 设计约束与阶段落地

1. 阶段 1：优先保证离线推理闭环与 argmax 验证。
2. 阶段 2：加入 sampling 与在线推理能力。
3. 阶段 3：连续批处理、前缀缓存与投机解码。
4. 任何阶段必须保持接口稳定，不影响已有推理流程。

## 8. 验收标准

1. 参考 test_infer.py，增加 test_offline.py，验证离线推理结果可复现且与参考实现一致。
2. 增加 test_online.py 测试，验证能支持多用户并发与流式输出。
3. 增加 test_kv_cache.py 测试，验证 KV-Cache 管理与 slot 映射工作稳定，无泄漏。
4. 增加 test_sampling.py 测试，验证采样策略可配置且行为一致。
5. 增加连续批处理的测试，验证连续批处理可显著提升吞吐。
