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

对齐范围（统一口径）：

1. Core 层按 llama.cpp 语义对齐（SoA batch、`logits/output_ids`、`kv_seq_*`、资源不足失败返回）。
2. Engine 离线链路按 vLLM/`nano-vllm` 的 `LLM.generate -> engine step` 思路对齐（阶段1落地）。
3. Online 服务按 vLLM API Server 分层对齐（阶段2落地）；阶段0不要求 online 全链路实现。

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
4. 需提供通用权重槽位安全替换能力（`llaisysModelReplaceWeight`），用于避免重复赋值时句柄泄漏。

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

## 4. Engine 层需求（Python，严格对齐 vLLM）

### 4.1 LLM / AsyncLLM（入口层）

1. 提供统一入口：离线 `LLM.generate` 与在线异步入口（`AsyncLLM`）。
2. 接收 `SamplingParams`（top-k/top-p/temperature/max_new_tokens/stop）。
3. 将请求标准化后提交到 EngineClient，不直接执行模型前向与采样算法。
4. 返回值语义为“已采样完成”的输出（token/text/finish_reason），而非裸 logits。

### 4.2 EngineClient（客户端层）

1. 负责入口层与 EngineCore（`LLMEngine`）之间的调用封装。
2. 支持同进程直连与可切换 IPC/RPC 形态（后续多进程部署）。
3. 负责请求提交、结果拉取、流式回传、取消信号透传。

### 4.3 LLMEngine / EngineCore（核心编排层）

1. 维护请求生命周期状态机（queued/running/finished/cancelled/failed）。
2. 驱动 step 循环：每轮 `Scheduler -> Executor`，直到请求结束。
3. 维护每请求/每序列状态（token 进度、停止条件、KV 句柄、统计信息）。
4. 负责多模型路由配置下发（选择模型实例，但不写模型专有逻辑）。

### 4.4 Scheduler

1. 维护请求队列与公平调度策略（prefill/decode 混排）。
2. 组 batch/ubatch，并控制每轮执行 token 集合与并发窗口。
3. 支持连续批处理与多序列并发。
4. 负责请求级上下文配额与增量 token 配额（防止单请求长期占用）。
5. 仅产出执行计划，不直接做模型前向或采样。

### 4.5 Executor（执行协调层）

1. 接收 Scheduler 计划并组织一次 step 的执行。
2. 协调 Worker 执行模型前向，收集 `logits + output_ids`。
3. 触发执行侧采样链（阶段1 argmax，阶段2 top-k/top-p/temperature）。
4. 触发停止条件判断（max_new_tokens/eos/stop token/stop string）。

### 4.6 Worker（执行单元）

1. 实例化并持有模型适配器（`python/llaisys/models/*.py`）与 Core 句柄。
2. 执行模型前向（decode/prefill），返回当前 step 的 logits 行。
3. 仅执行计算，不负责调度策略、停止条件决策与请求生命周期管理。
4. 采样可在 Worker/Executor 执行侧完成（对齐 vLLM“执行侧采样”口径），但入口层与 ModelRunner 不参与采样决策。
5. 支持多设备扩展预留（单机多卡 TP 为后续目标）。

### 4.7 Sampler + OutputProcessor（执行侧采样 + 引擎侧结果组织）

1. Sampler 基于 logits 与 SamplingParams 产出下一 token；采样决策在执行侧（Worker/Executor），不在入口层。
2. OutputProcessor 在 EngineCore 结果聚合路径执行：将 sampled token 增量转为文本增量并维护 UTF-8 拼接正确性。
3. OutputProcessor 统一组织输出对象（token/text/finish_reason/usage），在线流式与离线非流式复用同一语义。
4. 执行侧与结果组织侧边界固定：执行侧产出 token 增量，EngineCore 负责请求级输出拼装。

### 4.8 模型适配层（models）

1. 模型文件解析：支持 HuggingFace 目录结构，读取 config/tokenizer。
2. 权重加载：支持单文件或分片 safetensors；负责 HF 权重名到 Core 权重槽位映射与校验。
3. Tokenizer：支持 AutoTokenizer、Qwen2 chat template 与特殊 token。
4. 仅负责模型适配与前向调用，不承担调度、采样策略决策与请求状态管理。

### 4.9 模型选择与加载

1. Engine 初始化时必须明确模型类型与模型路径。
2. 架构上支持多模型路由；当前至少提供 `qwen2`，可新增注册项不改 Engine 主流程。
3. 模型实例化与权重加载在 Worker 层完成；Engine 负责配置与路由编排。

## 5. Server 层需求（Python，对齐 vLLM API Server）

### 5.1 API Server

1. 提供 HTTP 服务接口与 OpenAI 兼容 API（completions/chat/completions/embeddings 基础路由）。
2. 支持 SSE 流式输出，保证 token 级增量语义与正确结束信号。
3. 支持请求取消（主动取消与超时取消），并将取消信号透传给 Engine。
4. 支持会话管理（单用户会话、多用户并发、会话上下文复用）。
5. 提供隔离与限流（并发上限、排队上限、背压策略）。
6. 提供请求日志与错误码，关键失败可追踪。

### 5.2 Server 与 Engine 对接边界

1. 调用链固定为：`API Server -> AsyncLLM -> EngineClient -> LLMEngine(EngineCore)`。
2. Server 仅负责协议适配、鉴权/限流、流式转发；不负责调度、模型前向或采样。
3. Engine 返回统一输出对象（token/text/finish_reason/usage），Server 负责序列化为 OpenAI 响应格式。
4. 监控口径分层：Server 暴露 API 级指标；Engine 暴露调度与执行级指标（吞吐/延迟/KV/资源占用）。

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
2. 阶段 0 兼容例外：为保持 `test/test_infer.py --test`，允许 `ModelRunner.generate()` 临时使用内部 argmax，但该路径仅用于离线兼容，不作为长期主路径。
3. 阶段 0 退出条件：进入阶段1前，离线主流程必须切换为 `LLM -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
4. 阶段 1：优先保证离线推理闭环与 Engine 内 argmax 验证（Core 仅返回 logits）。
5. 阶段 2：在 Engine 层扩展 sampling（top-k/top-p/temperature）与在线推理能力。
6. 阶段 3：连续批处理、前缀缓存与投机解码。
7. 任何阶段必须保持接口稳定，不影响已有推理流程。

### 7.1 阶段0当前完成度口径（As-Built）

1. 已完成（功能闭环）：通用 `LlaisysModel` 主线接口、多序列 SoA decode、`kv_seq_*`、`GetLogits*` 输出接口、`qwen2 + mock` 路由、阶段0核心测试脚本。
2. 已完成（Core 目录重构）：`workspace/kv_cache/output/weights` 已拆分到 `src/llaisys/runtime/`，`qwen2_model.cpp` 仅保留模型专有执行逻辑与算子编排。
3. 已完成（权重槽位安全替换）：提供 `llaisysModelReplaceWeight`，模型适配层通过统一 API 写入权重槽位，避免重复赋值泄漏。
4. 未完成：阶段1要求的 Engine 离线主链路 `LLM -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。

## 8. 验收标准

1. 阶段0必须通过既有基线测试（允许把历史测试迁移到通用 `llaisysModel*` 接口）。
2. 增加 `test_core_model_api.py`：验证通用模型接口 create/decode/logits/kv 行为。
3. 增加 `test_qwen2_adapter.py`：验证 `python/llaisys/models/qwen2.py` 基于通用接口可完成权重映射与推理调用。
4. 增加 `test_kv_cache.py`：验证 KV-Cache 管理与 slot 映射工作稳定，无泄漏。
5. 增加 `test_offline.py`：验证离线推理结果可复现且与参考实现一致。
6. 增加 `test_online.py`：验证支持多用户并发与流式输出。
7. 增加 `test_sampling.py`：验证采样策略可配置且行为一致。
8. 增加连续批处理的测试，验证连续批处理可显著提升吞吐。

### 8.1 当前验收状态（更新于阶段0收敛后）

1. 阶段0验收通过（含 `run_stage0_tests.sh` 全量与 parity 对拍）。
2. 下一里程碑为阶段1（Engine 离线闭环），不再新增阶段0范围内的结构性改动。
