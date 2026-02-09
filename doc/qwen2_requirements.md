# 需求文档设计

本需求文档面向“离线推理 + 在线服务”两种形态，描述交付要求与验收标准，不包含实现细节；实现允许分阶段落地。

## 1. 目标与范围

设计目标：

1. 支持 Qwen2 在本仓库推理栈完成离线与在线推理。
2. 阶段0即建立通用多模型接口（Core/Engine/Server），而不是仅做 Qwen2 专用实现。
3. 支持多序列并发推理，具备连续批处理与 KV-Cache 复用能力。
4. 为后续 LLM 与多模态模型扩展预留接口。
5. 设计需考虑后续分布式推理场景（如张量并行）。

业务范围：

1. 离线单条/批量推理。
2. 在线服务（OpenAI 接口 + 流式）。
3. 多模型路由（至少支持按 `model` 字段选择实例）。

## 2. 分层与责任边界（需求）

1. Server 层：负责对外 API、会话管理、限流、流式输出与在线请求生命周期管理。
2. Engine 层：负责请求调度与执行计划（含 Scheduler/Executor/Worker 的责任拆分），并驱动推理流程。
3. 模型适配层（`python/llaisys/models/`）：负责模型适配与执行组织（权重映射、Tokenizer、输入输出组织与单次推理调用）。
4. Core 层（C++）：负责 batch/ubatch 执行、KV-Cache 管理与算子计算。
5. 多模型抽象层：提供模型无关的创建/执行/输出/KV 接口，模型私有 API 仅用于内部扩展，不作为主线对外协议。

## 3. Core 层需求（C++）

### 3.1 通用模型接口（阶段0必须）

1. 提供通用模型句柄与 API（如 `LlaisysModel*`）：
   - create/destroy；
   - decode(batch)；
   - logits 查询；
   - KV 序列管理接口。
2. 支持模型类型枚举（至少包含 `qwen2`），并通过统一入口创建模型实例。
3. 对外主线仅保留通用模型 C API；`qwen2` 专用包装若存在仅作为迁移期可选项，不纳入主线验收。

### 3.2 内存与资源管理

1. 模型权重加载与生命周期管理。
2. 模型计算中间 buffer 设计与复用。
3. KV-Cache 设计与多序列支持。
4. 输出 buffer 管理。

### 3.3 KV-Cache 行为要求

1. 基于 slot（cell）的物理管理，slot 记录逻辑 pos 与 seq_id。
2. 支持多序列混排，但 attention 必须按 seq_id 严格隔离。
3. 资源不足默认失败返回，不自动回收或截断。
4. （可选，阶段三）滑窗注意力不改变逻辑 pos，仅通过 mask 屏蔽窗口外 token。
5. 需提供前缀复用、释放/截断、位置平移、保留与查询等能力接口（不限定 API 名称）；由 Engine 层负责调用与编排。
6. Core 层默认不强制每 `seq_id` 的 slot/cell 硬配额；请求公平性与限额由 Engine/Server 层负责。

### 3.4 计算图与算子

1. 计算图覆盖 embedding -> N x block -> final norm -> lm head。
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

1. 负责实例化模型适配器（`python/llaisys/models/*.py`）并执行推理。
2. 支持多设备扩展的接口预留（单机多卡 TP 为后续阶段目标），Core 提供多设备内存接口支持。
3. 支持多模型实例池（至少可按 `model_type` 选择对应适配器）。

### 4.4 模型适配层（models）

1. 模型文件解析：支持 HuggingFace 目录结构，读取 config/tokenizer。
2. 权重加载：支持单文件或分片 safetensors；负责 HF 权重名到 Core 权重槽位的映射与校验。
3. Tokenizer：支持 AutoTokenizer、Qwen2 chat template 与特殊 token（bos/eos/pad/system/user/assistant）。
4. 采样参数透传至采样模块（采样策略由 Engine 层控制）。

### 4.5 模型选择与加载（阶段0必须）

1. Engine 初始化时必须明确模型类型与模型路径。
2. 阶段0要求：在架构上支持多模型路由；实现上至少提供 `qwen2`，并允许注册新增模型而不修改 Engine 主流程。
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
3. 多模型路由：基于 `model` 字段选择模型类型与实例（阶段0必须可扩展，不得写死在 Engine 主流程）。
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

1. 阶段 0：完成 Core 对齐 llama.cpp 的重构，并完成通用多模型接口抽象（统一 `LlaisysModel` 主线接口）。
2. 阶段 1：优先保证离线推理闭环与 argmax 验证。
3. 阶段 2：加入 sampling 与在线推理能力。
4. 阶段 3：连续批处理、前缀缓存与投机解码。
5. 任何阶段必须保持接口稳定，不影响已有推理流程。

## 8. 验收标准

1. 阶段0必须通过既有基线测试（允许把历史测试迁移到通用 `llaisysModel*` 接口）。
2. 增加 `test_core_model_api.py`：验证通用模型接口 create/decode/logits/kv 行为。
3. 增加 `test_qwen2_adapter.py`：验证 `python/llaisys/models/qwen2.py` 基于通用接口可完成权重映射与推理调用。
4. 增加 `test_kv_cache.py`：验证 KV-Cache 管理与 slot 映射工作稳定，无泄漏。
5. 增加 `test_offline.py`：验证离线推理结果可复现且与参考实现一致。
6. 增加 `test_online.py`：验证支持多用户并发与流式输出。
7. 增加 `test_sampling.py`：验证采样策略可配置且行为一致。
8. 增加连续批处理的测试，验证连续批处理可显著提升吞吐。
