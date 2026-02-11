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

### 2.1 当前实现状态（As-Built，2026-02-11）

1. Core 主线接口已统一到 `llaisysModel*`，并覆盖 `create/decode/logits/kv_seq_*`。
2. `Qwen2Model::decode` 已改为真正 batch 执行（单轮图执行处理整批 token），不再逐 token 调 `infer(..., 1)`。
3. KV 已升级为 unified cell 语义：一个 slot 可关联多个 `seq_id`，并支持 `n_seq_id > 1` token。
4. 多序列隔离在同一轮前向中通过 mask 生效（`seq_id` 集合交集 + `pos` 因果约束）。
5. Engine/Server 阶段2主链路已落地（offline + online + SSE + cancel + WebUI 基础多会话）。
6. KV 元数据流程已引入 llama.cpp 风格 `prepare(ubatches) -> slot_info_vec`、`apply_ubatch`、`rollback_ubatch`、`update(stream_copy_info)`。
7. 未实现项：多 stream 端到端执行路径（含跨 stream K/V tensor 流程）、滑窗策略、阶段3性能优化（高阶连续批处理/前缀缓存收益/投机）。

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
7. KV 元数据实现需对齐 llama.cpp 关键语义：`prepare/find_slot/apply_ubatch/update` 四段式流程、cell 内 `seq_set + pos` 持久化、`seq_pos_min/max` 统计可查询。
8. 阶段2允许“单 stream 执行 + 多 stream 元数据预留”形态，但文档必须明确未打通项，避免误判为全量对齐。

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

1. 维护请求生命周期状态机（对齐 vLLM 风格）：`waiting / waiting_for_remote_kvs / running / preempted / finished_stopped / finished_length_capped / finished_aborted / finished_ignored`。
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
2. 阶段 0 对拍口径：`pytest -q test/test_infer.py --model-path /path/to/local/model` 为 ModelRunner 级 `decode_batch + argmax` 对拍，不经过 Engine。
3. 阶段 0 退出条件：进入阶段1前，离线主流程必须切换为 `LLM -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
4. 阶段 1：优先保证离线推理闭环与 Engine 内 argmax 验证（Core 仅返回 logits）。
   - 阶段1已完成口径：`LLM.generate`（prompt入口）+ Engine 状态机主路径 + stop string + 统一输出对象（含 finish_reason/status/usage）。
   - `waiting_for_remote_kvs/preempted` 可作为预留状态，但需在文档中明确“未激活”。
   - 允许实现瘦身（同进程合并类），但不得移除 `submit/step/cancel` 语义、请求状态机、`Scheduler -> Worker` 边界与统一输出结构。
5. 阶段 2：在 Engine 层扩展 sampling（top-k/top-p/temperature）与在线推理能力。
6. 阶段 3：连续批处理、前缀缓存与投机解码。
7. 任何阶段必须保持接口稳定，不影响已有推理流程。

### 7.1 阶段0当前完成度口径（As-Built）

1. 已完成（功能闭环）：通用 `LlaisysModel` 主线接口、多序列 SoA decode、`kv_seq_*`、`GetLogits*` 输出接口、`qwen2 + mock` 路由、阶段0核心测试脚本。
2. 已完成（Core 目录重构）：`workspace/kv_cache/output/weights` 已拆分到 `src/llaisys/runtime/`，`qwen2_model.cpp` 仅保留模型专有执行逻辑与算子编排。
3. 已完成（权重槽位安全替换）：提供 `llaisysModelReplaceWeight`，模型适配层通过统一 API 写入权重槽位，避免重复赋值泄漏。
4. 已完成（真实 batch decode）：`llaisysModelDecode` 单轮处理整批 token，`output_ids` 与 logits 行索引一致。
5. 已完成（Unified KV）：slot 支持多 `seq_id` 关联；`kv_seq_cp` 语义为附加关联（共享前缀），不再强制复制新 slot。
6. 已完成（KV 流程对齐）：`KvCells`、`prepare/apply_ubatch/rollback_ubatch/update`、`seq_pos_min/max` 统计链路已落地。
7. 已完成（阶段1离线主链路）：`LLM -> LLMEngine -> Scheduler -> Executor -> Worker -> Core` 已落地并通过离线回归测试。

## 8. 验收标准

1. 阶段0必须通过既有基线测试（允许把历史测试迁移到通用 `llaisysModel*` 接口）。
2. 增加 `test_core_model_api.py`：验证通用模型接口 create/decode/logits/kv 行为。
3. 增加 `test_qwen2_adapter.py`：验证 `python/llaisys/models/qwen2.py` 基于通用接口可完成权重映射与推理调用。
4. 增加 `test_kv_cache.py`：验证 KV-Cache 管理与 slot 映射工作稳定，无泄漏。
5. 增加 `test_offline.py`：验证 Engine 级离线行为（状态机、stop、stream、usage）可复现。
6. 增加 `test_offline_parity.py`：验证 Engine 离线链路与 `transformers` 对拍一致。
7. 增加 `test_online.py`：验证支持多用户并发与流式输出。
8. 增加 `test_sampling.py`：验证采样策略可配置且行为一致。
9. 增加连续批处理的测试，验证连续批处理可显著提升吞吐。

### 8.1 当前验收状态（更新于阶段2收敛后）

1. 阶段0验收通过（含 `scripts/run_tests.py --suite stage0` 全量与 parity 对拍）。
2. 阶段1验收通过（`scripts/run_tests.py --suite stage1`；有模型时追加 `test_offline_parity`）。
3. 阶段2核心功能闭环已完成（sampling + online + WebUI 多会话基础能力）。
4. 阶段2后 Core 架构修正已完成：真实 batch decode + unified KV + 单轮 mask 隔离。
5. 阶段3进入前提：阶段2并发稳定性回归长期通过，线上接口语义冻结。

### 8.2 阶段2验收标准（As-Built）

必须满足：

1. `test/test_sampling.py` 通过：`argmax/top-k/top-p/temperature` 行为正确。
2. `test/test_online.py` 通过：non-stream/stream/cancel/concurrent 行为正确。
3. `test/test_online_http.py` 通过或在受限环境下被合理 skip（socket bind 限制）。
4. `test/test_online_stream_isolation.py` 通过：双流并发不串 `request_id`。
5. `test/test_online_real_model_multisession.py` 在本地模型下通过：真实模型并发流式不串线。
6. WebUI 可执行最小闭环：多会话创建、切换、流式渲染、取消请求。

## 9. 实施问题与解决方案（阶段2）

### 9.1 流式终止包丢失

问题：

1. 某些请求流在 `stream timeout + collect(done)` 路径直接返回，未输出 `is_finished=true` 终止块。

方案：

1. 在 `AsyncLLMEngine.stream` 超时分支补发终止 chunk。
2. 增加并发流式隔离与终止语义回归测试。

### 9.2 多会话并发时崩溃（Segmentation fault）

问题：

1. Python `__del__` 中直接 `llaisysModelDestroy`，GC 时机不稳定导致 native 崩溃。
2. C++ `Context` 生命周期与多线程 `Runtime&` 引用存在悬挂风险。
3. `Context::_current_runtime` 未初始化，`setDevice` 中运行时容器曾出现按值拷贝导致状态不一致。

方案：

1. 引入显式 `close` 链：`AsyncLLMEngine -> LLMEngine -> Worker -> Qwen2`。
2. `Qwen2.__del__` 不再做 native destroy，改为显式关闭路径管理。
3. 修复 C++ `Context` 初始化与 `setDevice` 引用语义。
4. `context()` 调整为线程局部常驻对象，规避线程退出析构导致的 Runtime 悬挂（当前阶段接受该有界泄漏换稳定性）。

### 9.3 并发首请求 Tokenizer 初始化异常

问题：

1. 并发首请求时，Tokenizer 延迟初始化可能触发导入/初始化竞态。

方案：

1. `Qwen2` tokenizer 初始化加锁。
2. `Worker.encode_chat_messages/decode_tokens` 增加异常降级路径，确保请求不中断。

### 9.4 Core 伪批处理问题（已修复）

问题：

1. 历史实现中 `decode` 虽接收 batch，但内部逐 token 调 `infer(..., 1)`，不是真正 batch 执行。
2. 历史 KV 元数据是单 slot-单 seq 绑定，无法表达一个 token 同属多个 `seq_id` 的 unified 语义。

方案：

1. `decode` 改为单次 batch 前向：整批 token 在一轮计算图中执行。
2. KV 升级为 unified cell：`alloc_token(seq_ids, n_seq_id, pos)`。
3. attention 增加 mask 路径，按 `seq_id` 集合交集与 `pos` 做可见性判断。

现状：

1. 相关核心用例已覆盖并通过（`test_core_decode_batch/test_kv_cache/test_core_output_api/test_core_model_api`）。

### 9.5 KV 与 llama.cpp 对齐进展（阶段2后）

问题：

1. 早期 KV 设计依赖简化状态（slot 单绑定、线性分配），难以稳定支持多序列混排与覆盖写回滚。
2. 文档口径与源码逐步偏离，出现“文档写了 free_slots_，源码已改为 KvCells”的理解断层。

方案：

1. 结构对齐：引入 `KvCells`，cell 保存 `pos/shift/seq_set`，并维护 `used_` 与 `seq_pos_`。
2. 流程对齐：引入 `prepare -> apply_ubatch -> rollback_ubatch -> update`，把 slot 规划、提交、回滚、状态同步拆开。
3. 回滚对齐：`decode` 异常路径统一调用 `rollback_ubatch`，避免“元数据提交了但前向失败”的脏状态。

当前差距（已明确）：

1. Qwen2 decode 仍限制单 stream 执行（`slot_info.n_stream() == 1`）。
2. `used_slots/slot_visible_for` 当前仅使用 stream0 视图，跨 stream 可见性尚未打通。
3. `update(stream_copy_info)` 目前是元数据级复制，尚未接入统一 K/V tensor 拷贝调度。

## 10. 下一阶段优化路线（阶段3前）

1. 把当前“同批共用采样参数”升级为“每请求独立采样参数”。
2. 落地连续批处理吞吐优化与 benchmark 体系。
3. 补齐前缀缓存命中与回退路径测试。
4. 再评估 `context()` 常驻策略，收敛为可验证、可回收的跨线程 Runtime 生命周期管理。
