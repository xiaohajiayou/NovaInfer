# Qwen2 模型概要设计

## 1. 设计目标

- 基于现在 LLAISYS 完成 DeepSeek-R1-Distill-Qwen-1.5B 的纯 C++ 推理代码基础，保证现在所有测试的正确性前提下，进一步实现core、engine、server、client模块开发。
- 要求覆盖需求文档中所有功能点，确保能够正常交付

## 2. 源码组织

```
llaisys/
├─ include/
│  └─ llaisys/
│     └─ models/
│        └─ qwen2.h          # C API、配置结构、句柄定义
├─ src/
│  └─ llaisys/
│     ├─ runtime/
│     │  ├─ kv_cache/
│     │  │  ├─ kv_cache.hpp  # 通用 KV-Cache 管理
│     │  │  └─ kv_cache.cpp
│     │  ├─ workspace/
│     │  │  ├─ workspace.hpp # 通用 Workspace/buffer 管理
│     │  │  └─ workspace.cpp
│     │  ├─ graph/
│     │  │  ├─ graph.hpp     # 通用计算图编排接口
│     │  │  └─ graph.cpp
│     │  └─ weights/
│     │     ├─ weights.hpp   # 通用权重槽位/校验基类
│     │     └─ weights.cpp
│     └─ qwen2/
│        ├─ qwen2_api.cc     # C API 与 Qwen2Model 绑定
│        ├─ qwen2_model.hpp  # 驱动 + block + weights 声明（合并版）
│        └─ qwen2_model.cpp  # 构造、infer、调度 + weights 校验/映射（合并版）
├─ python/
│  └─ llaisys/
│     ├─ entrypoints/
│     │  └─ llm.py           # LLM (offline entrypoint, align with vLLM)
│     ├─ engine/
│     │  ├─ llm_engine.py    # LLMEngine, Request/SequenceState
│     │  ├─ scheduler.py     # RequestScheduler, BatchPlan/BatchItem
│     │  ├─ executor.py      # Executor
│     │  └─ worker.py        # Worker
│     ├─ server/
│     │  ├─ openai_server.py # OpenAIServer/HttpServer
│     │  ├─ async_engine.py  # AsyncLLMEngine
│     │  └─ schemas.py       # ApiRequest/StreamHandle/响应结构
│     ├─ libllaisys/
│     │  └─ qwen2.py         # ctypes 声明与 Qwen2Handle
│     └─ models/
│        └─ qwen2.py         # 暂时作为ModelRunner: HF→weights 映射、Tokenizer、单步推理
├─ webui/
│  ├─ index.html             # Web UI 入口
│  ├─ app.js                 # 向服务器请求/流式接收
│  └─ styles.css             # UI 样式
└─ doc/
   └─ qwen2_*.md             # 设计/需求/接口文档
```

- **注释约定**：
  - `include/`：关键结构与 API 使用 `//` 说明参数含义和生命周期。
  - `src/llaisys/qwen2/*.cpp`：文件头注明模块职责，复杂函数前加简述；内部逻辑依赖 `ASSERT`/`CHECK_*`。
  - `python/llaisys/libllaisys/qwen2.py`：每个 ctypes 函数注明 `argtypes/restype` 对应的实际意义。
  - `python/llaisys/models/qwen2.py`：类 docstring 描述整体用途，私有方法附注 HF 权重映射与采样策略。
  - `doc/`：保持当前 Markdown 结构，记录设计演进。

以上目录中 `src/llaisys/qwen2/` 及对应 Python/文档文件为本次实现新增内容，提交时需一并创建并按注释约定维护。所有新增文件遵循仓库既有风格：`snake_case` 文件名、命名空间 `llaisys::models::qwen2`，注释使用英语或简明中文，避免冗长段落。

## 3. 架构设计
### 3.1 Online/Offline 总体架构（流程 + 进程）

```
Offline（单机、同步，对齐 vLLM LLM.generate）
Process: main
  User
    ↓
  LLM.generate() (入口/Facade)
    ↓
  LLMEngine
    ↓
  RequestScheduler
    ↓
  Executor
    ↓
  Worker
    ↓
  Qwen2 (ModelRunner)
    ↓
  C++ Core (batch/ubatch + KV + ops)
  ↓
  tokens / logits

注：offline 默认同进程执行，但可复用 llm_engine/worker 的进程化配置以对齐在线架构。

Online（服务化、多用户、连续批处理，对齐 vLLM Async）
Process: api_server
  Client (HTTP/OpenAI)
    ↓
  OpenAIServer / HttpServer (鉴权/限流/流式)
    ↓
  AsyncLLMEngine
    |
    | IPC/RPC
    v
Process: llm_engine
  LLMEngine
    ↓
  RequestScheduler (组 batch/ubatch，prefill/decode 混排)
    ↓
  Executor
    |
    | IPC/RPC (optional)
    v
Process: worker
  Worker
    ↓
  Qwen2 (ModelRunner)
    ↓
  C++ Core (batch/ubatch + KV + ops)
    ↓
  streaming responses

注：对齐 vLLM 结构，api_server 与 llm_engine 建议分进程；worker 可先同进程，后续做 TP 时按每卡/每 rank 进程化。
```

### 3.2 Server 侧分层（对齐 vLLM 架构）

- API Server：`OpenAIServer/HttpServer`，负责 API、会话、限流、流式、取消。
- Async 引擎入口：`AsyncLLMEngine`，负责异步请求提交与流式回传。
- 引擎核心：`LLMEngine`，负责请求生命周期与调度驱动。
- Scheduler：`RequestScheduler`，负责队列、prefill/decode 混排、batch/ubatch 组装。
- Executor：执行计划与资源协调，管理 Worker 调度。
- Worker：模型执行单元，负责调用 ModelRunner 并触发 Core 执行。
- ModelRunner：模型适配与执行组织（权重映射、Tokenizer、输入输出组织、单次推理调用）。
- Core：C++ 推理核（KV-Cache/算子/batch 执行）。


## 4. 模块设计

### 4.1 infer Core

模块架构如下：
```
C++ Core
  ├─ Runtime/Device
  ├─ KV-Cache (slot/seq_id/pos)
  ├─ Compute Graph (embed → block × N → norm → lm head)
  ├─ Ops (linear/rope/attn/mlp/rms_norm)
  └─ Output (logits/token)
```

- 架构设计：推理执行内核，负责 KV-Cache、算子与 batch/ubatch 执行。
- 实现思路：统一 Runtime API 管理 device，KV-Cache 使用 slot/seq_id/pos 映射；计算图支持 prefill/decode 输入形态。
- 4.1.1 需求覆盖与现状
  - 现有实现位于 `src/llaisys/qwen2/qwen2_model.cpp`，单次 infer 支持 ntoken>1 的 prefill 与 ntoken=1 的 decode。
  - KV-Cache 目前是按 [maxseq, nkvh, dh] 连续内存，且只有单序列 `cur_len_` 语义。
  - CPU-only 运行路径由 `src/device/cpu/` 与 `src/ops/*/cpu/` 提供。
- 4.1.2 内存与资源管理（对齐需求 3.1）
  - 权重加载在 Core 内统一管理生命周期，采用句柄写入即转移所有权的模式。
  - Workspace 采用 grow-only 复用策略，按本轮最大 seq_len 扩容。
  - 输出 buffer 提供单步 logits 与采样结果的输出通道。
  - 多用户资源隔离：在 KV-Cache 层统计每个 seq_id 的占用并支持配额/拒绝策略，防止单用户占满资源。
  - 多用户 buffer 使用策略：
    - 权重与常量 buffer 全局共享（只读）。
    - Workspace/中间 buffer 在 Engine/Worker 实例内独占复用，避免并发写冲突。
    - 输出/logits buffer 按请求或 seq_id 维度切片，防止跨用户混用。
  - 申请/使用/释放规则：
    - 申请时机：Engine 调度生成 ubatch 时，按本轮 ntoken 触发 Workspace 扩容；KV-Cache 在新序列创建或新增 token 时分配。
    - 申请规模：Workspace 按本轮最大 ntoken 扩到刚好够用；KV-Cache 按 seq_id 追加对应 pos 区间的 slot。
    - 多用户使用：不同 seq_id 的 KV 互不共享；Workspace 由 Worker 单实例复用；输出 buffer 仅切片给当前 seq_id。
    - 释放时机：Engine 在请求结束/取消时调用 KV-Cache 释放接口；Workspace 不释放，仅随进程生命周期复用。
- 4.1.3 KV-Cache 行为（对齐需求 3.2）
  - 使用 slot（cell）作为物理单元，slot 记录 seq_id 与逻辑 pos。
  - 多序列混排时 attention 按 seq_id 严格隔离，不同序列互不可见。
  - 资源不足默认失败返回，不自动回收或截断。
  - 滑窗注意力仅通过 mask 屏蔽窗口外 token，不修改逻辑 pos。
  - 提供前缀复用、释放/截断、位置平移、保留与查询等能力接口（由 Engine 触发）。
- 4.1.4 计算图与算子（对齐需求 3.3）
  - 计算图覆盖 embedding → block × N → final norm → lm head。
  - block 包含 attention（Q/K/V 投影 + RoPE + KV-Cache 读写 + 输出投影）、MLP（gate/up/down + SwiGLU）、残差与 RMSNorm。
  - RoPE 位置与 KV-Cache 对齐，新增 token 使用逻辑 pos。
  - 支持 batch/ubatch 执行路径，prefill/decode 仅体现为输入形态差异。
  - 常用算子：embedding、linear、rms_norm、rope、self_attention、swiglu、argmax；支持常见 dtype 并做一致性检查。
- 4.1.5 新增数据结构（C++）
  - `struct SeqState { int64_t seq_id; size_t pos_max; bool active; };`
  - `struct KvSlot { int64_t seq_id; size_t pos; };`
  - `struct KvCacheMeta { std::vector<KvSlot> slots; std::unordered_map<int64_t, SeqState> seqs; };`
  - `struct InferenceInput { const int64_t *token_ids; size_t ntoken; int64_t seq_id; size_t pos_start; };`
  - `struct InferenceOutput { tensor_t logits; int64_t next_token; size_t ntoken; };`
- 4.1.6 需要改动的函数（C++）
  - `Qwen2Model::infer(...)`：扩展为接受 `seq_id/pos_start`；将 `cur_len_` 替换为 per-seq `pos_max`。
  - `Qwen2Model::init_kv_cache_()`：保留每层 cache tensor，同时初始化 `KvCacheMeta`。
  - 新增 `Qwen2Model::kv_seq_cp/kv_seq_rm/kv_seq_add/kv_seq_keep/kv_seq_pos_max` 接口。
  - `Qwen2Model::fill_pos_ids_()`：改为基于 `pos_start` 填充。
  - `Qwen2Model::copy_into_cache_()`：增加按 slot/pos 映射写入逻辑。
- 4.1.7 数据结构调整与文件落点
  - 用 `std::unordered_map<int64_t, size_t>` 维护 seq_id -> pos_max，替代单一 `cur_len_`。
  - `LayerCache` 保留 `k_cache/v_cache`，并新增 `std::vector<KvSlot> slot_meta` 或共享 `KvCacheMeta`。
  - `src/llaisys/qwen2/qwen2_model.hpp`：新增 `SeqState/KvSlot/KvCacheMeta/InferenceInput/InferenceOutput` 声明（合并版）。
  - `src/llaisys/qwen2/qwen2_model.cpp`：扩展 `infer()` 签名与逻辑；新增 kv_* 接口实现；替换 `cur_len_` 逻辑（合并版）。
  - `include/llaisys/models/qwen2.h`：暴露新的 infer/kv_* C API。
  - `src/llaisys/runtime/kv_cache/`：通用 KV-Cache 管理与 slot/seq_id/pos 映射。
  - `src/llaisys/runtime/workspace/`：通用 workspace/buffer 管理。
  - `src/llaisys/runtime/graph/`：通用计算图编排接口与执行骨架。
  - `src/llaisys/runtime/weights/`：通用权重槽位定义与校验基类。

Core 推理执行流程（当前实现）：
```
input_ids -> embedding -> blocks(attn+mlp) -> final_norm -> logits -> argmax -> next_token
               |                |                         |
               +-- KV-Cache ---- +                         +-- output buffer
```

### 4.2 infer Engine

模块架构如下：
```
LLMEngine
  ├─ RequestScheduler
  │    └─ batch/ubatch planning
  ├─ Executor
  │    └─ dispatch to Worker
  └─ Worker
       └─ ModelRunner -> Core
```

4.2.1 需求覆盖与现状
  - 当前仓库尚无 Engine 实现，需新增 `LLMEngine/RequestScheduler/Executor/Worker`。
  - ModelRunner 可复用 `python/llaisys/models/qwen2.py` 的权重映射与单次 infer。
4.2.2 核心职责（对齐需求 5.x）
  - 请求生命周期管理与调度驱动，维护状态机与资源协调。
  - prefill/decode 混排与 ubatch 组装，按 seq_id 产出“最后 token logits”。
  - 统一处理 stop/stop string/UTF-8 拼接与流式回传。
4.2.3 新增数据结构（Python）
  - `Request`: `request_id`, `seq_id`, `prompt_tokens`, `max_new_tokens`, `stop`, `status`, `outputs`.
  - `SequenceState`: `seq_id`, `pos`, `finished`, `tokens`, `last_logits`.
  - `BatchPlan`: `items: List[BatchItem]`, `mode: prefill|decode`, `ntoken`.
  - `BatchItem`: `seq_id`, `token_ids`, `pos_start`.
4.2.4 需要改动的函数（Python）
  - `LLMEngine.submit(request)`：入队并返回 request_id。
  - `LLMEngine.step()`：驱动 Scheduler 生成 `BatchPlan`，调用 Executor 执行。
  - `RequestScheduler.build_plan(active_seqs)`：生成 ubatch 计划，区分 prefill/decode。
  - `Executor.run(plan)`：调用 Worker 执行，并把 logits/token 回写到 `SequenceState`。
  - `Worker.infer(batch_items)`：调用 ModelRunner 的单步/批量接口。
4.2.5 结构调整与文件落点
  - 将 `python/llaisys/models/qwen2.py:generate()` 拆成 `prepare_prefill()` + `infer_step()`，供 Engine 复用。
  - `python/llaisys/entrypoints/llm.py`：离线入口 `LLM.generate()`（对齐 vLLM 的 entrypoints/llm.py）。
  - `python/llaisys/engine/llm_engine.py`：`LLMEngine`, `Request`, `SequenceState`, `submit()`, `step()`。
  - `python/llaisys/engine/scheduler.py`：`RequestScheduler`, `BatchPlan`, `BatchItem`, `build_plan()`。
  - `python/llaisys/engine/executor.py`：`Executor`, `run()`。
  - `python/llaisys/engine/worker.py`：`Worker`, `infer()`。
4.2.6 子模块职责
  - Scheduler：队列管理、prefill/decode 混排、batch/ubatch 组装。
  - Executor：执行计划下发、资源协调、Worker 调度。
  - Worker：执行模型推理并回传 logits/token。

Engine 调度流程（预期）：
```
queue -> scheduler -> ubatch plan -> executor -> worker -> logits/tokens -> output
```

### 4.3 infer Server

模块架构如下：
```
API Server
  ├─ Routing/OpenAI API
  ├─ Auth/Rate Limit
  ├─ Streaming (SSE)
  └─ AsyncLLMEngine
       └─ LLMEngine
```

4.3.1 需求覆盖与现状
  - 当前仓库尚无 Server，实现需新增 API 路由与 AsyncLLMEngine 适配层。
  - API 返回格式需与 OpenAI 兼容，包括 stream 与非 stream 响应结构。
4.3.2 核心职责（对齐需求 6.1/6.2）
  - OpenAI 兼容 HTTP 层，负责路由、鉴权、限流、流式输出与取消请求。
  - 处理请求取消、超时回收、请求级日志与错误码。
4.3.3 新增数据结构（Python）
  - `ApiRequest`: `model`, `prompt/messages`, `stream`, `params`, `request_id`.
  - `StreamHandle`: 维护 SSE 通道与取消标记。
4.3.4 需要改动的函数（Python）
  - `OpenAIServer.handle_chat()` / `handle_completion()`：解析请求并调用 AsyncLLMEngine。
  - `AsyncLLMEngine.submit(request)`：异步入队，返回 stream 生成器。
  - `AsyncLLMEngine.cancel(request_id)`：请求取消与资源回收。
4.3.5 文件落点
  - `python/llaisys/server/openai_server.py`：路由与 OpenAI API 解析。
  - `python/llaisys/server/async_engine.py`：`AsyncLLMEngine.submit/cancel`。
  - `python/llaisys/server/schemas.py`：`ApiRequest/StreamHandle` 与响应 schema。

Server 请求路径（预期）：
```
HTTP request -> auth/limit -> parse -> async engine submit -> stream response
```

### 4.4 infer client (Web UI)

模块架构如下：
```
Web UI
  ├─ Prompt/Chat UI
  └─ HTTP/OpenAI Client
```

4.4.1 需求覆盖与现状
  - 当前仓库尚无 Web UI，实现需新增 `webui/index.html`、`webui/app.js`、`webui/styles.css`。
4.4.2 核心职责（对齐需求 6.1/6.2）
  - 提供浏览器端 UI，向 Server 发送请求并接收流式响应。
  - 支持连续对话与本地历史记录（单用户场景）。
4.4.3 新增数据结构（Web）
  - `ChatState`: `messages`, `stream_buffer`, `request_id`。
4.4.4 需要改动的函数（Web）
  - `webui/app.js`: `submitPrompt()`, `handleStreamChunk()`, `cancelRequest()`。

## 5. 主要时序图

Offline（当前实现）：
```
User -> Qwen2.generate -> Qwen2._infer -> C++ Core::infer -> ops -> next_token
```

Online（预期实现）：
```
Client -> API Server -> AsyncLLMEngine -> LLMEngine -> Scheduler -> Executor -> Worker -> Core
   ^                                                                            |
   +----------------------------- streaming tokens -----------------------------+
```



## 6. 未来扩展
