# Qwen2 模型概要设计

## 1. 设计目标

1. 基于现有 LLAISYS 的 Qwen2 推理能力，在不破坏当前测试正确性的前提下，完成 Core、Engine、Server、Client 的分层扩展。
2. 对齐 `doc/qwen2_requirements.md` 中的每条需求，形成可执行的 RFC 级实现指南。
3. 支持离线与在线统一执行链路，支持多序列并发、连续批处理、流式输出，并为多模型/多设备/分布式预留接口。

对齐范围说明：

1. Core 对齐 llama.cpp 语义与接口风格。
2. 离线编排链路对齐 vLLM/`nano-vllm` 的 `LLM.generate -> engine step` 思路。
3. Online 仅对齐 vLLM 架构边界与接口语义，不要求阶段0完成全量服务能力。
## 2. 源码组织与状态

### 2.1 Stage0 当前落地（As-Built）

说明：以下是当前仓库已经落地并通过阶段0测试的实际目录与模块位置。

```
llaisys/
├─ include/
│  └─ llaisys/
│     ├─ runtime/
│     │  └─ infer_types.h     # 通用 SoA batch 结构（已落地）
│     └─ models/
│        ├─ model.h           # 通用模型 C API（已落地）
│        └─ qwen2.h           # Qwen2 专有元数据/权重结构（已落地）
├─ src/
│  ├─ core/
│  │  ├─ runtime/             # 现有运行时实现（历史目录）
│  │  ├─ context/
│  │  └─ storage/
│  └─ llaisys/
│     ├─ model.cc             # 通用模型工厂/路由（含 qwen2 + mock）
│     ├─ runtime.cc
│     ├─ tensor.cc
│     ├─ ops.cc
│     ├─ runtime/
│     │  ├─ workspace/        # 已落地：大 buffer + view（Qwen2Workspace）
│     │  ├─ kv_cache/         # 已落地：KV 元数据管理（KvCache）
│     │  ├─ output/           # 已落地：logits/output_ids 缓冲（OutputBuffer）
│     │  └─ weights/          # 已落地：权重槽位生命周期（replace/destroy）
│     └─ qwen2/
│        ├─ qwen2_api.cc
│        ├─ qwen2_model.hpp
│        └─ qwen2_model.cpp   # 模型专有图执行与算子编排
├─ python/
│  └─ llaisys/
│     ├─ libllaisys/
│     │  ├─ model.py          # 通用 ctypes 绑定（含 ReplaceWeight）
│     │  └─ qwen2.py          # Qwen2 专有 ctypes 结构
│     └─ models/
│        └─ qwen2.py          # Qwen2 适配器（权重写入走 ReplaceWeight）
└─ test/
   └─ test_core_*.py / test_kv_cache.py / test_model_registry.py / test_qwen2_adapter.py
```

阶段边界（As-Built）：

1. Python 侧已落地离线主链路：`LLM -> EngineClient -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
2. 已落地状态机与统一输出结构（含 `finish_reason/status/usage`），并支持离线 `stream()`。
3. 在线专属组件已落地：`AsyncLLMEngine + OpenAIServer + HTTPServer + WebUI`，并已接入阶段2回归测试。

### 2.1.1 Stage2 当前落地（As-Built）

说明：以下是阶段2新增且已在源码落地的关键模块。

1. Server：
   - `python/llaisys/server/async_engine.py`
   - `python/llaisys/server/openai_server.py`
   - `python/llaisys/server/http_server.py`
   - `python/llaisys/server/schemas.py`
   - `python/llaisys/server/__main__.py`
2. Sampling：
   - `python/llaisys/engine/sampling.py`（argmax/top-k/top-p/temperature）
3. 并发与流式隔离回归：
   - `test/test_online_stream_isolation.py`
   - `test/test_online_real_model_multisession.py`
4. WebUI：
   - `webui/index.html`
   - `webui/app.js`
   - `webui/styles.css`

### 2.1.2 Core 架构修正（As-Built，2026-02-11）

本轮针对阶段0/2之间的核心偏差已完成修正，关键点如下：

1. 真正 batch decode：
   - `src/llaisys/qwen2/qwen2_model.cpp` 中 `decode()` 已改为单次 batch 前向执行；
   - 删除“每个 token 调一次 `infer(...,1)`”的伪批处理主路径。
2. Unified KV：
   - `src/llaisys/runtime/kv_cache/kv_cells.hpp/.cpp` 引入 `KvCells`（cell 元信息 `pos/shift/seq_set`）；
   - `src/llaisys/runtime/kv_cache/kv_cache.hpp/.cpp` 引入 `prepare/apply_ubatch/rollback_ubatch/update` 流程；
   - slot 支持多 `seq_id` 关联，并维护 `seq_pos_min/max` 统计。
3. 多序列混排 mask：
   - 新增 `ops::self_attention_masked`，并接入 Qwen2 decode 主路径；
   - 可见性规则为“seq 集合交集 + 因果 pos 约束”。
4. 行为兼容性：
   - 对外 C API 不变（仍是 `llaisysModel*`）；
   - Python Engine/ModelRunner 调用链无需改签名。
5. 当前遗留项：
   - 多 stream 执行路径尚未打通到 Qwen2 decode（当前仍单 stream）；
   - 滑窗策略与阶段3性能优化尚未开始。

### 2.2 目标目录（Target Refactor）

说明：以下为目标态目录。当前阶段0已完成 `runtime/{workspace,kv_cache,output,weights}` 拆分；`runtime/graph` 与 `src/llaisys/models/{model.hpp,model.cpp}` 仍为后续演进项。

```
llaisys/
├─ include/
│  └─ llaisys/
│     ├─ runtime/
│     │  ├─ infer_types.h     # 通用 Batch/Output C 结构
│     │  └─ kv_cache.h        # 通用 KV 管理 C API（seq_*）
│     └─ models/
│        ├─ model.h          # 通用模型 C API（create/decode/logits/kv）
│        └─ qwen2.h          # Qwen2 专有元数据/权重结构定义
├─ src/
│  └─ llaisys/
│     ├─ runtime/
│     │  ├─ kv_cache/
│     │  │  ├─ kv_cache.hpp  # 通用 KV-Cache 管理
│     │  │  └─ kv_cache.cpp
│     │  ├─ workspace/
│     │  │  ├─ workspace.hpp # 通用 Workspace/buffer 管理
│     │  │  └─ workspace.cpp
│     │  ├─ output/
│     │  │  ├─ output.hpp    # 通用 logits/output_ids 缓冲
│     │  │  └─ output.cpp
│     │  ├─ graph/
│     │  │  ├─ graph.hpp     # 通用计算图编排接口
│     │  │  └─ graph.cpp
│     │  └─ weights/
│     │     ├─ weights.hpp   # 通用权重槽位/校验基类
│     │     └─ weights.cpp
│     ├─ models/
│     │  ├─ model.hpp        # 通用模型接口（weights/decode/kv_*）
│     │  └─ model.cpp
│     └─ qwen2/
│        ├─ qwen2_api.cc     # Qwen2 实现注册到通用模型工厂
│        ├─ qwen2_model.hpp  # 驱动 + block + weights 声明（合并版）
│        └─ qwen2_model.cpp  # 构造、infer、调度 + weights 校验/映射（合并版）
├─ python/
│  └─ llaisys/
│     ├─ entrypoints/
│     │  └─ llm.py           # LLM (offline entrypoint, align with vLLM)
│     ├─ common/
│     │  ├─ sampling.py      # 通用采样链与 SamplingParams
│     │  ├─ model_registry.py# 模型注册与路由
│     │  └─ types.py         # Request/Batch/Output 公共类型
│     ├─ engine/
│     │  ├─ llm_engine.py    # LLMEngine, Request/SequenceState
│     │  ├─ scheduler.py     # RequestScheduler, BatchPlan/BatchItem
│     │  ├─ executor.py      # Executor
│     │  └─ worker.py        # Worker
│     ├─ server/
│     │  ├─ openai_server.py # OpenAIServer/HttpServer
│     │  ├─ async_engine.py  # AsyncLLMEngine
│     │  ├─ session_store.py # 会话状态存储
│     │  ├─ metrics.py       # 监控指标输出
│     │  └─ schemas.py       # ApiRequest/StreamHandle/响应结构
│     ├─ libllaisys/
│     │  ├─ model.py         # 通用模型 ctypes 声明（LlaisysModel）
│     │  └─ qwen2.py         # Qwen2 专有 ctypes 结构定义（meta/weights）
│     └─ models/
│        ├─ base.py          # 模型适配抽象接口
│        └─ qwen2.py         # Qwen2 模型适配实现：HF→weights 映射、Tokenizer、单步推理
├─ webui/
│  ├─ index.html             # Web UI 入口
│  ├─ app.js                 # 向服务器请求/流式接收
│  └─ styles.css             # UI 样式
└─ doc/
   └─ qwen2_*.md             # 设计/需求/接口文档
```

解耦拆分规则：

1. `src/llaisys/runtime/*` 与 `python/llaisys/common/*` 只放公共组件，不出现模型专有权重名或层字段。
2. `src/llaisys/qwen2/*` 与 `python/llaisys/models/qwen2.py` 只放 Qwen2 专有逻辑。
3. Engine/Server 仅依赖 `python/llaisys/models/base.py` 抽象接口和 `common/model_registry.py`，不直接依赖具体模型实现。
4. 新增模型时只新增 `src/llaisys/<model>/` 与 `python/llaisys/models/<model>.py`，公共模块不改或少改。

- **注释约定**：
  - `include/`：关键结构与 API 使用 `//` 说明参数含义和生命周期。
  - `src/llaisys/models/*.cpp`：通用模型分发与类型路由逻辑需注明模型类型分支与错误码语义。
  - `src/llaisys/qwen2/*.cpp`：文件头注明模块职责，复杂函数前加简述；内部逻辑依赖 `ASSERT`/`CHECK_*`。
  - `python/llaisys/libllaisys/model.py`：每个通用模型 ctypes 函数注明 `argtypes/restype` 与返回码语义。
  - `python/llaisys/libllaisys/qwen2.py`：Qwen2 专有 ctypes 结构与辅助转换函数注明字段语义。
  - `python/llaisys/models/qwen2.py`：类 docstring 描述整体用途，私有方法附注权重映射、Tokenizer 与单步执行策略。
  - `doc/`：保持当前 Markdown 结构，记录设计演进。

以上目录中 `src/llaisys/models/`、`src/llaisys/qwen2/` 及对应 Python/文档文件为本次实现新增内容，提交时需一并创建并按注释约定维护。所有新增文件遵循仓库既有风格：`snake_case` 文件名；通用层使用 `llaisys::models` 命名空间，模型实现使用 `llaisys::models::<model>` 命名空间，注释使用英语或简明中文，避免冗长段落。

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
  models/<model>.py
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
  models/<model>.py
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
- Worker：模型执行单元，负责调用 `python/llaisys/models/*.py` 适配器并触发 Core 执行。
- models 层：模型适配与执行组织（权重映射、Tokenizer、输入输出组织、单次推理调用）。
- Core：C++ 推理核（KV-Cache/算子/batch 执行）。



## 4. 模块设计

### 4.1 infer Core（C++）

模块架构如下：

```text
C++ Core
  ├─ Runtime/Device
  ├─ KV-Cache (slot/seq_id/pos)
  ├─ Compute Graph (embed -> block x N -> norm -> lm head)
  ├─ Ops (linear/rope/attn/mlp/rms_norm)
  └─ Output (logits/token)
```

Core 公共实现约束（所有 4.1.x 默认遵循）：

1. 代码落点（双态口径）：
   - Stage0 As-Built：`include/llaisys/runtime/infer_types.h`、`include/llaisys/models/model.h`、`include/llaisys/models/qwen2.h`、`src/llaisys/model.cc`、`src/llaisys/runtime/{weights,workspace,kv_cache,output}/`、`src/llaisys/qwen2/qwen2_model.hpp`、`src/llaisys/qwen2/qwen2_model.cpp`、`src/llaisys/qwen2/qwen2_api.cc`、`python/llaisys/libllaisys/model.py`、`python/llaisys/models/qwen2.py`。
   - Target Refactor：`include/llaisys/runtime/kv_cache.h`、`src/llaisys/runtime/graph/`、`src/llaisys/models/{model.hpp,model.cpp}`。
2. C API 使用纯 C 结构体，不暴露 STL 容器。
3. Core 统一输入语义为 SoA batch（`token/pos/n_seq_id/seq_id/logits`）。
4. 阶段0主入口为通用模型 API（`llaisysModel*`），Engine/Server 不依赖模型专用 C API。
5. 权重槽位访问统一走 `llaisysModelWeights`（按 `model_type` 转型）。

#### 4.1.1 需求 3.1.1：模型权重加载与生命周期

实现设计（As-Built）：

1. 在 `src/llaisys/runtime/weights/weights.hpp` 提供机制函数：
   - `replace_slot(llaisysTensor_t *slot, llaisysTensor_t new_handle)`；
   - `destroy_unique(std::vector<llaisysTensor_t *>)`。
2. `Qwen2Model::destroy_weights_()` 统一走 `destroy_unique`，避免共享权重 double free。
3. 通用 C API 增加 `llaisysModelReplaceWeight(model, field_name, layer_idx, new_weight)`，由 `src/llaisys/model.cc` 将字段名路由到 `LlaisysQwen2Weights` 对应槽位，再调用 `replace_slot`。
4. Python `python/llaisys/models/qwen2.py` 在 `_assign_global/_assign_layer` 中调用 `llaisysModelReplaceWeight`，失败时立即 `tensorDestroy(new_handle)`，避免异常路径泄漏。

目标演进（Target）：

1. 若后续新增第二个模型并出现重复代码，再引入 `IWeightStore` 级别抽象；阶段0不强制。

验收：

1. 权重共享（`in_embed == out_embed`）不会 double free。
2. 重复赋值同一槽位不泄漏。

#### 4.1.2 需求 3.1.2：中间 buffer 设计与复用

实现设计（As-Built）：

1. 在 `src/llaisys/runtime/workspace/workspace.hpp` 实现 `Qwen2Workspace`：
   - 一块 `main_arena_`（模型 dtype）；
   - 一块 `i64_arena_`（pos ids）；
   - 通过 slice/view 暴露 `hidden/qkv/attn/mlp/logits/k_ctx/v_ctx`。
2. `Qwen2Model` 只持有 `qwen2_workspace_t` 句柄；每轮 `reserve(ntoken)` grow-only，容量足够时只复用，不重复分配。
3. 当前未引入 `graph_reserve/can_reuse` 抽象接口；这部分属于 `runtime/graph` 目标演进。

目标演进（Target）：

1. 当引入第二个模型时，再抽象为通用 `IWorkspace + GraphMemPlan`。

验收：

1. 连续请求时 `reserve()` 在容量未超出时不触发重新分配（地址稳定）。
2. 算子拿到的输入/输出指针均来自 `view(offset)`，内存连续可直接下发到底层 kernel。
3. 没有每轮重复 new tensor。
4. 本期验收要求“已预留形态不触发热路径临时扩容”。

#### 4.1.3 需求 3.1.3：KV-Cache 设计与多序列支持

实现设计（As-Built）：

1. 在 `src/llaisys/runtime/kv_cache/kv_cache.hpp` 实现 `KvCache`：
   - 核心状态是 `v_cells_ + v_heads_`，按 stream 管理 cell 数组与分配游标；
   - `v_cells_` 的元素类型是 `KvCells`，cell 元信息为 `{pos, shift, seq_set}`；
   - 维护 `seq_to_stream_`（序列归属 stream）和 `seq_slots_cache_`（按 pos 排序的序列 slot 视图缓存）；
   - 对外提供两套接口：llama 风格 `prepare/find_slot/apply_ubatch/rollback_ubatch/update`，以及兼容包装 `alloc_token/alloc_tokens`。
2. 在 `src/llaisys/runtime/kv_cache/kv_cells.hpp` 实现 `KvCells`：
   - `used_` 追踪当前已占用 cell；
   - `seq_pos_` 维护 `seq_id -> (pos -> ref_count)`，用于 `seq_pos_min/max` 常数级查询；
   - `seq_add/seq_rm/seq_keep/pos_add` 保持 cell 集合和统计一致性。
3. `Qwen2Model` 只持有 `std::unique_ptr<KvCache>`，模型层不再持有 slot/cell 元数据。
4. `decode` 路径支持 `n_seq_id > 1` token，输入先转为 `ubatch{seq_sets, pos_values}`，再走 `prepare + apply_ubatch`。
5. `kv_seq_cp` 语义：
   - 同 stream：对命中 cell 直接附加 `dst_seq` 关联（共享 slot）；
   - 跨 stream：复制 cell 元数据到目标 stream，对应 K/V tensor 拷贝由模型层按 `src_slots/dst_slots` 执行。

目标演进（Target）：

1. 当引入第二模型并出现接口分歧时，再评估抽象 `IKvCache` 接口；当前 `KvCache` 已满足阶段2功能口径。
2. 多 stream 主路径目前只完成 KV 元数据层；Qwen2 decode 的 K/V gather/mask 仍走单 stream 口径（见 4.1.16 未对齐项）。

验收：

1. 能打印 `cell[idx] = {seq_ids, pos, used}` 诊断信息。
2. 可同时维护多个 `seq_id` 的 `pos_max`，并在共享 slot 场景保持一致。
3. 模型层不可直接改写 cell 元数据。

#### 4.1.4 需求 3.1.4：输出 buffer 管理

实现设计（As-Built）：

1. SoA `LlaisysBatch` 与通用输出查询接口（`GetLogits/GetLogitsIth/NOutputs/OutputIds`）已全部落地。
2. 在 `src/llaisys/runtime/output/output.hpp` 实现 `OutputBuffer`，统一维护 `logits_f32_` 与 `output_ids_`，并提供 `clear/reserve_rows/append_row`。
3. `Qwen2Model::decode()` 每轮先 `output_->clear()`，按 `batch.logits` 决定是否 `append_row`，输出契约与接口文档一致。
4. 采样仍与 Core 解耦：Core 仅产 logits 行，采样在 Engine/上层执行。

目标演进（Target）：

1. 若需要进一步对齐 llama.cpp，可补充显式 `output_reserve(n_outputs)` C++ API 并把容量策略开放给 Engine。

验收：

1. decode 阶段每活跃序列返回 1 行 logits。
2. prefill 阶段可按策略只返回末 token logits。
3. 当 prefill 请求多行 logits 时，输出行数与 `logits != 0` 的 token 数严格一致。
4. `GetLogits/GetLogitsIth/OutputIds` 返回的指针在同一轮查询内稳定可读。
5. decode 前调用 `output_reserve` 后，`GetLogits*` 指针在本轮推理结束前不变。

#### 4.1.5 需求 3.2.1：slot/cell + 逻辑 pos

实现设计（As-Built）：

1. KV 写入已按显式 slot 执行（而非隐式线性位移）：

```cpp
void copy_into_cache_slot_(tensor_t &cache, size_t slot_idx, const tensor_t &src_row, size_t row_idx);
```

2. `decode` 单轮流程：
   - 先根据 batch 构造 `ubatch{seq_sets, pos_values}` 并校验逻辑 pos；
   - 调用 `prepare({ubatch})` 做 slot 规划（内部会临时 `apply_ubatch` 后回滚）；
   - 调用 `apply_ubatch(slot_info, ubatch)` 正式提交本轮 cell 元数据；
   - 按 `slot_info.idxs` 把新 token 的 K/V 写入对应物理 cell；
   - 用 `used_slots()` 取可见 cell，构建 attention mask 并执行前向；
   - 失败路径统一 `rollback_ubatch(slot_info, ubatch)`，保证元数据回滚。
3. attention 读路径已按 cell 可见性过滤：
   - `slot_visible_for(slot, seq_ids, n_seq_id, qpos)`。
4. 当前实现是“先规划再提交”路径，失败即整轮 `decode` 返回错误码，且 rollback 生效。

验收：

1. 可打印 `token[i] -> cell[idx]` 映射，便于问题定位。
2. 逻辑 pos 与物理 idx 解耦，混排输入下仍正确。
3. 任一步骤失败不会留下“部分 token 已写入、元数据未提交”的脏状态。

#### 4.1.6 需求 3.2.2：多序列混排隔离

实现设计（As-Built）：

1. 已采用统一 mask 路径，不再按 `seq_id` 分组执行。
2. mask 判定条件与 llama.cpp 语义对齐：
   - 集合隔离：`intersect(q.seq_ids, kv.seq_ids) != empty`；
   - causal：`kv.pos <= q.pos`。
3. `n_seq_id > 1` query token 已支持，任一命中即可见。
4. `self_attention` 已新增 masked 接口并在 Qwen2 decode 主路径接入：
   - `ops::self_attention_masked(...)`。

验收：

1. 构造两序列 A/B，A 的输出不受 B token 影响。
2. 构造 `n_seq_id > 1` 的 token，mask 可见性符合集合交集规则。

#### 4.1.7 需求 3.2.3：资源不足默认失败

实现设计：

1. 所有分配函数返回状态码：

```cpp
enum class KvStatus {
    OK,
    OOM_SLOT,
    INVALID_SEQ,
    INVALID_POS,
    EMPTY_RANGE,
    INTERNAL_ERROR
};
```

2. 默认策略：无空闲 cell 时直接 `OOM_SLOT`，不自动踢出其他序列，不自动 `seq_rm`。
3. `decode` 若 `prepare/find_slot` 无法为本轮 ubatch 分配足够 slot，整批失败返回（本轮不 commit）。
4. C API 对外返回 `int`，0=成功，非0=错误码，错误码与 `KvStatus` 一一映射。
5. 不在 Core 中做 `seq_id` 级限额检查；公平性与配额控制由 Engine/Server 层实现。

验收：

1. 压满 cell 后返回可预测错误码（稳定复现）。
2. 不发生隐式踢出或静默回收。
3. 失败后 cache 元数据与 tensor 内容保持提交前状态（可通过 debug 校验）。
4. 在无 Engine 限额时，单 `seq_id` 可占用全部 cell（符合全局池语义）。

#### 4.1.8 需求 3.2.4：滑窗语义（本期不实现）

实现设计：

1. 本期不实现窗口掩码与基于窗口的 cell 复用策略。
2. 资源不足行为统一为 `OOM_SLOT`，不增加隐式覆盖写路径。
3. 降占用由显式接口完成：`kv_seq_rm`、`kv_seq_keep`，必要时由 Engine 执行请求级重建。

验收：

1. 代码路径不存在窗口参数分支。
2. 行为可预测：有空闲 cell 则成功；无空闲 cell 则 `OOM_SLOT`。

#### 4.1.9 需求 3.2.5：KV 管理接口

实现设计：

1. KV 管理函数定义在 `runtime/kv_cache`（公共组件），`Qwen2Model` 仅做转发：

```cpp
KvStatus kv_seq_cp(int64_t dst_seq, int64_t src_seq, int64_t p0, int64_t p1);
KvStatus kv_seq_rm(int64_t seq_id, int64_t p0, int64_t p1);
KvStatus kv_seq_add(int64_t seq_id, int64_t p0, int64_t p1, int64_t delta);
KvStatus kv_seq_keep(int64_t seq_id);
int64_t kv_seq_pos_max(int64_t seq_id) const;
```

2. 区间语义统一为半开区间 `[p0, p1)`，并写入接口注释：
   - `kv_seq_cp(dst, src, p0, p1)`：把 `src` 在区间内命中的 cell 附加 `dst` 关联（前缀复用）；
   - `kv_seq_rm(seq, p0, p1)`：移除 `seq` 与区间内 cell 的关联；若 cell 无任何 `seq_id`，该 cell 释放；
   - `kv_seq_add(seq, p0, p1, delta)`：对命中 cell 的逻辑 `pos += delta`（显式位置平移）；
   - `kv_seq_keep(seq)`：仅保留 `seq`，其他 `seq_id` 全部移除；
   - `kv_seq_pos_max(seq)`：返回该序列当前最大逻辑位置（不存在返回 -1）。
3. 通用模型 C API 在 `include/llaisys/models/model.h` 定义（如 `llaisysModelKvSeq*`）；内部实现路径为 `ModelHandle -> <ModelImpl> -> IKvCache`。
4. ctypes 在 `python/llaisys/libllaisys/model.py` 统一绑定，并保留错误码到异常的映射表。

验收：

1. 每个 API 都有单测覆盖边界区间、空区间、无命中区间。
2. `kv_seq_rm` 后空 cell 可再次被 `find_slot/prepare` 分配。
3. `kv_seq_add` 后 attention 按新 `pos` 生效。

#### 4.1.10 需求 3.3.1：计算图覆盖范围

实现设计：

1. 保持现有图顺序：embedding -> block*N -> final_norm -> lm_head。
2. 在 `src/llaisys/runtime/graph/graph.hpp` 定义通用执行接口（`IGraphRunner`），统一管理“预留、复用判定、重建分配、执行”。
3. `Qwen2Model` 只提供图描述与输入绑定，不直接管理图内存分配细节（由 `IGraphRunner` 持有调度器/allocator）。
4. 初始化阶段采用单 profile 预留策略（与 4.1.2 一致）：执行 `PP reserve`。
5. 运行时采用“复用优先，必要时重建”策略：
   - 先比较 `graph_params`，`can_reuse` 命中则直接执行；
   - 不命中则 `build_graph + alloc_graph` 后执行；
   - 禁止每轮无条件重建图。
6. Qwen2 图描述接口建议：

```cpp
void build_qwen2_graph(GraphPlan *plan, const UBatchView &ubatch);
void bind_qwen2_graph_io(GraphPlan *plan, const WorkspaceView &ws);
```

7. GraphRunner 接口建议（RFC）：

```cpp
struct GraphReserveParams {
    int32_t n_tokens;
};

struct GraphExecParams {
    int32_t n_tokens;
    llaisysDataType_t dtype;
    int32_t n_outputs;
};

class IGraphRunner {
public:
    virtual bool graph_reserve(const GraphReserveParams &params) = 0;
    virtual bool can_reuse(const GraphExecParams &params) const = 0;
    virtual bool alloc_graph(const GraphExecParams &params) = 0;
    virtual bool run_graph(const GraphExecParams &params) = 0;
    virtual ~IGraphRunner() = default;
};
```

验收：

1. 与旧 `infer` 在单序列路径输出一致。
2. 更换模型时无需改 `runtime/graph` 执行框架，只新增模型图描述。
3. 连续 decode 多轮时，复用命中场景不发生 `alloc_graph`。

#### 4.1.11 需求 3.3.2：block 组成完整性

实现设计：

1. 每层固定调用顺序写入注释和断言。
2. 对 Q/K/V shape、RoPE shape、MLP shape 统一检查函数：

```cpp
void check_block_shapes_(...) const;
```

验收：

1. 形状不匹配时 fail-fast，错误信息包含层号。

#### 4.1.12 需求 3.3.3：RoPE 与 KV 对齐

实现设计（As-Built）：

1. 当前位置填充函数已改为按 batch 明确位置值写入：

```cpp
void fill_pos_ids_from_values_(const tensor_t &pos_ids, const std::vector<int64_t> &pos_values);
```

2. 禁止再用 `cur_len_ + i` 推导 pos；当 `batch.pos == NULL` 时，按 `seq_pos_max(seq)+1` 推导并校验。

验收：

1. 混排序列时每个 token 的 pos 都可回溯到输入。

#### 4.1.13 需求 3.3.4：batch/ubatch 路径

实现设计（As-Built）：

1. 新增对齐 llama.cpp 的主入口（通用模型 API）：

```cpp
int32_t llaisysModelDecode(struct LlaisysModel * model, struct LlaisysBatch batch);
struct LlaisysBatch llaisysBatchInit(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
struct LlaisysBatch llaisysBatchGetOne(int64_t * token, int32_t n_tokens);
void llaisysBatchFree(struct LlaisysBatch batch);
```

2. 模型执行统一走 `llaisysModelDecode`，不定义模型专用 `Decode` 公共入口。
3. 每次 `decode` 处理一个 batch，当前实现为“单轮前向执行整个 batch”（不再逐 token `infer` 循环）。
4. 状态码语义对齐 llama.cpp：`0=success`、`1=no KV slot`、`2=aborted`、`-1=invalid input`、`<-1=fatal`。
5. 历史单序列调用由 Python 适配层在本地转换为 `llaisysBatchGetOne`/`llaisysModelDecode`，不再新增 C 层专用入口。

验收：

1. prefill/decode 仅体现在 `batch` 组装，不再有两套图代码。
2. 输出读取统一通过 `GetLogits/GetLogitsIth`，行数与 `batch.logits` 标记一致。

#### 4.1.14 需求 3.3.5：算子覆盖

实现设计：

1. CPU 路径作为 mandatory；GPU 路径标注 staged。
2. 保持 `ops::xxx` 统一入口，设备分派在 op 层。

验收：

1. CPU 端通过当前全部推理测试。

#### 4.1.15 需求 3.3.6：dtype 检查

实现设计：

1. 所有 op 入口统一检查 dtype/device/contiguous。
2. 缺失的检查补到对应 `src/ops/*/op.cpp`。

验收：

1. 非法 dtype 输入稳定报错，不出现 silent wrong result。

#### 4.1.16 Core 对齐 llama.cpp 的关键口径（As-Built + Target）

当前已落地（As-Built）：

1. 输入是 token 列表，不是 `[B,T]`。
2. batch 输入采用 SoA 连续数组（`token/pos/n_seq_id/seq_id/logits`）。
3. 物理 cell/slot 和逻辑 pos 分离。
4. 多序列隔离靠 `seq_id + pos` mask（`self_attention_masked` 已接入）。
5. KV 元数据结构对齐到 `KvCells`（cell 内 `seq_set + pos + shift`）并维护 `seq_pos_min/max` 统计。
6. `prepare(ubatches) -> slot_info_vec`、`apply_ubatch()`、`rollback_ubatch()`、`update(do_shift, stream_copy_info)` 已落地。
7. 覆盖写路径会先回收旧 cell 的序列关联，再按新 token 写入并修复序列位置区间。
8. 资源不足默认失败，回收依赖显式 `seq_*`；Core 不做 per-seq 硬配额。
9. Workspace 采用 runtime 组件 `reserve + view`；输出缓冲已抽到 `runtime/output`。
10. 对外主线为 `LlaisysModel` 通用接口；`n_seq_id > 1` token 已在 decode 主路径支持。

当前未完全对齐项（Target）：

1. 多 stream 仍未打通到 Qwen2 decode 主路径：当前 `decode` 仍要求 `slot_info.n_stream() == 1`。
2. `used_slots()` 与 `slot_visible_for()` 当前仅读取 stream0，尚未扩展为跨 stream 视图。
3. `update(stream_copy_info)` 目前只更新 KV 元数据；跨 stream 的 K/V tensor 批量拷贝与调度尚未并入统一流程。
4. `find_slot(cont=true)` 的连续块策略、SWA 相关复用分支尚未接入主执行路径（本期禁用滑窗）。

### 4.2 infer Engine（Python）

模块架构如下：

```text
LLMEngine
  ├─ RequestScheduler
  ├─ Executor
  └─ Worker
       └─ models/<model>.py -> Core
```

Engine 公共实现约束：

1. 代码落点：`python/llaisys/entrypoints/llm.py`、`python/llaisys/engine/llm_engine.py`、`python/llaisys/engine/scheduler.py`、`python/llaisys/engine/executor.py`、`python/llaisys/engine/worker.py`。
2. Engine 只处理请求状态和计划，不写模型细节。
3. Scheduler 输出 `BatchPlan`，Executor 负责调 Worker，Worker 只负责执行。

#### 4.2.1 需求 4.1.1：请求队列与调度策略

实现设计：

1. 在 `llm_engine.py` 定义：

```python
class RequestStatus(Enum):
    WAITING = "waiting"
    WAITING_FOR_REMOTE_KVS = "waiting_for_remote_kvs"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"
    FINISHED_IGNORED = "finished_ignored"
```

2. `LLMEngine` 维护 `requests: dict[str, Request]` 和 `active_seq_ids: set[int]`。
3. `submit()` 只入队，`step()` 负责状态迁移。
4. 请求级公平性在该层实现（例如 `max_context_tokens_per_request`、轮转 decode），不下沉到 Core。

验收：

1. 任一请求状态迁移都可追踪。

#### 4.2.2 需求 4.1.2：batch/ubatch 组装

实现设计：

1. `scheduler.py` 定义：

```python
@dataclass
class BatchItem:
    seq_id: int
    token_ids: list[int]
    pos_start: int
    logits: bool

@dataclass
class BatchPlan:
    mode: Literal["prefill", "decode"]
    items: list[BatchItem]
```

2. `build_plan()` 返回单个 ubatch；后续可扩展返回多个 ubatch。
3. Executor 在下发前把 `BatchPlan` 转成 SoA `LlaisysBatch` 缓冲（连续 `token/pos/n_seq_id/seq_id/logits` 数组）。

验收：

1. Worker 仅消费 `BatchPlan`。

#### 4.2.3 需求 4.1.3：连续批处理与多序列并发

实现设计：

1. Scheduler 每轮从 active 序列挑选 decode token。
2. 有新请求时允许 prefill + decode 混排（先按 simple 策略：prefill 优先）。
3. 需实现最小公平策略：避免单长序列长期独占 step（例如 round-robin 或 age-based）。

验收：

1. 两个并发请求可在同一循环中交替推进。

#### 4.2.4 需求 4.1.4：自回归驱动与停止条件

实现设计：

1. `SequenceState` 增加字段：`tokens`, `pending_text`, `finished`, `stop_reason`。
2. `LLMEngine._apply_outputs()` 统一处理 stop token/string。
3. 流式场景下每次 decode 后推送 delta 文本。

验收：

1. stop token 与 stop string 行为一致且可复现。

#### 4.2.5 需求 4.2.1：Executor 执行编排

实现设计：

1. `executor.py`：

```python
class Executor:
    def __init__(self, worker: Worker): ...
    def run(self, plan: BatchPlan) -> dict[int, "StepOutput"]: ...
```

2. Executor 不关心调度策略，只做执行和结果转换。

验收：

1. 可替换不同 Worker 实现而不改 Scheduler。

#### 4.2.6 需求 4.2.2：Executor-Worker 协作

实现设计：

1. 定义统一输出：

```python
@dataclass
class StepOutput:
    seq_id: int
    logits: Any | None
    next_token: int | None
```

2. Worker 异常由 Executor 捕获并转换为 request 失败状态。

验收：

1. 单个 Worker 失败不影响其他请求状态一致性。

#### 4.2.7 需求 4.3.1：Worker 实例化与执行

实现设计：

1. `worker.py` 中 lazy init `python/llaisys/models/*.py` 模型适配器。
2. 暴露：

```python
class Worker:
    def warmup(self) -> None: ...
    def run(self, plan: BatchPlan) -> dict[int, StepOutput]: ...
```

3. `run()` 内部把 BatchPlan 转成 ctypes `LlaisysBatch`（SoA）并调用通用 `llaisysModelDecode`。

验收：

1. 首次请求后不再重复加载模型。

#### 4.2.8 需求 4.3.2：多设备扩展预留

实现设计：

1. Worker 配置：`device`, `device_id`, `rank`, `world_size`。
2. 首期只实现 `world_size=1`，但保留字段。

验收：

1. 配置可序列化，后续可扩展。

#### 4.2.9 需求 4.4.1：模型文件解析

实现设计：

1. `python/llaisys/models/qwen2.py` 提供：

```python
def load_config(model_path: str) -> dict: ...
def load_tokenizer(model_path: str): ...
```

2. Worker 通过这些函数创建模型适配器实例。

验收：

1. Engine 层不直接读取 `config.json`。

#### 4.2.10 需求 4.4.2：权重加载与映射校验

实现设计：

1. 在 `python/llaisys/models/qwen2.py` 增加：

```python
def validate_weight_coverage(self) -> None: ...
```

2. 在 `_load_safetensors()` 后调用，缺失即抛错。

验收：

1. 缺少关键权重时启动失败并给出字段名。

#### 4.2.11 需求 4.4.3：Tokenizer 能力

实现设计：

1. 模型适配器提供统一接口：

```python
def encode_chat(self, messages: list[dict]) -> list[int]: ...
def decode_tokens(self, tokens: list[int]) -> str: ...
```

2. 内部封装 qwen2 chat template 与 special tokens。

验收：

1. offline/online 两入口编码结果一致。

#### 4.2.12 需求 4.4.4：采样参数透传

实现设计：

1. 新增 `python/llaisys/common/sampling.py`：

```python
@dataclass
class SamplingParams:
    top_k: int
    top_p: float
    temperature: float
```

2. `LLMEngine` 在 decode 时用 logits + SamplingParams 采样。
3. 采样目标行由 Engine 显式选择（通过 `GetLogitsIth` 对应输出行）；允许“返回 logits 但不采样”。

验收：

1. 参数改变可导致输出分布变化。

#### 4.2.13 需求 4.5.1：模型类型与路径显式

实现设计：

1. 新增配置：

```python
@dataclass
class LLMConfig:
    model_type: str
    model_path: str
    device: str = "cpu"
```

2. `LLMEngine.__init__()` 校验必填。

验收：

1. 缺失字段时报配置错误。

#### 4.2.14 需求 4.5.2：阶段0多模型注册与路由

实现设计：

1. 新增 `python/llaisys/common/model_registry.py`：

```python
MODEL_REGISTRY = {
    "qwen2": Qwen2Model,
    # "llama": LlamaModel,
    # "mistral": MistralModel,
}
```

2. Worker 通过 registry 创建模型适配器；Engine 主流程不依赖具体模型类名。

验收：

1. 新增模型只需要新增 `python/llaisys/models/<model>.py` 并注册，不改 Engine 主流程。

#### 4.2.15 需求 4.5.3：Worker 加载，Engine 路由

实现设计：

1. Engine 根据请求的 `model` 字段选择 Worker/模型适配器。
2. Worker 在首次任务时按 `model_type` 懒加载模型实例。

验收：

1. Engine 不直接实例化具体模型（如 `Qwen2`）。

### 4.3 infer Server（Python）

模块架构如下：

```text
API Server
  ├─ Routing/OpenAI API
  ├─ Auth/Rate Limit
  ├─ Streaming (SSE)
  └─ AsyncLLMEngine
       └─ LLMEngine
```

Server 公共实现约束：

1. 代码落点：`python/llaisys/server/openai_server.py`、`python/llaisys/server/async_engine.py`、`python/llaisys/server/session_store.py`、`python/llaisys/server/metrics.py`、`python/llaisys/server/schemas.py`。
2. API 层不直接调用 Core。
3. 全链路携带 `request_id`。

#### 4.3.1 需求 5.1.1：OpenAI 兼容 API

实现设计：

1. `openai_server.py` 路由：`/v1/chat/completions`、`/v1/completions`。
2. `schemas.py` 定义请求/响应 dataclass，与 OpenAI 关键字段对齐。

验收：

1. 兼容基础 SDK 调用。

#### 4.3.2 需求 5.1.2：SSE 流式输出

实现设计：

1. `handle_chat(stream=True)` 返回 SSE 响应。
2. 每个 chunk 序列化为 OpenAI delta 结构，结束发 `[DONE]`。

验收：

1. 客户端能逐 token 渲染。

#### 4.3.3 需求 5.1.3：请求取消

实现设计：

1. `async_engine.cancel(request_id)`。
2. Server 暴露取消通道（HTTP 或连接断开触发）。

验收：

1. 取消后请求状态转为 `finished_aborted` 且停止推送。

#### 4.3.4 需求 5.1.4：会话管理

实现设计：

1. 引入 `python/llaisys/server/session_store.py` 中的 `SessionStore`：`session_id -> seq_id list`。
2. chat 请求可带 `session_id` 复用上下文。

验收：

1. 同 session 多轮具备上下文连续性。

#### 4.3.5 需求 5.1.5：隔离与限流

实现设计：

1. middleware 实现并发/队列限额。
2. 超限返回 `429`。
3. 可选增加用户/会话级 token budget（防止单用户长期占用 KV 资源）。

验收：

1. 压测下无无限排队。

#### 4.3.6 需求 5.1.6：日志与错误可追踪

实现设计：

1. 统一日志字段：`request_id/model/latency/status`。
2. 错误响应带 `request_id`。

验收：

1. 单请求可以跨层追踪。

#### 4.3.7 需求 5.2.1：AsyncLLMEngine 异步入口

实现设计：

1. `async_engine.py` 提供：

```python
class AsyncLLMEngine:
    async def submit(self, req) -> str: ...
    async def stream(self, request_id: str): ...
    async def cancel(self, request_id: str): ...
```

2. 内部维护每请求 `asyncio.Queue`。

验收：

1. API 与 Engine 异步解耦。

#### 4.3.8 需求 5.2.2：流式转发

实现设计：

1. Async 层从 Engine 接收 token 事件写入 queue。
2. Server 逐条消费 queue 输出 SSE。

验收：

1. 高并发下每请求输出顺序正确。

#### 4.3.9 需求 5.2.3：请求生命周期管理

实现设计：

1. 定义生命周期状态机。
2. `submit/stream/cancel` 均检查合法状态迁移。

验收：

1. 非法状态迁移会被拒绝并记录。

#### 4.3.10 需求 5.3.1：LLMEngine 调度协作

实现设计：

1. `LLMEngine` 提供 `step()` 循环入口。
2. Async 层通过后台任务驱动 step。

验收：

1. prefill/decode 混排可持续推进。

#### 4.3.11 需求 5.3.2：LLMEngine 执行协作

实现设计：

1. Engine 调用 Executor 获取 step 结果。
2. 结果统一写回 Request/SequenceState。

验收：

1. 异常请求与正常请求可并存处理。

#### 4.3.12 需求 5.3.3：多模型路由

实现设计：

1. Engine 通过 `model` 字段选择 WorkerPool。
2. 阶段0要求 WorkerPool 支持多模型注册机制；默认至少注册 qwen2，可按配置扩展其他模型。

验收：

1. 未注册模型返回明确错误。

#### 4.3.13 需求 5.3.4：监控指标

实现设计：

1. 指标最小集：QPS、TPOT、P50/P99、KV 占用、队列长度。
2. Engine 与 Server 暴露统一 metrics 接口。

验收：

1. 可导出文本指标并被采集。

### 4.4 infer client (Web UI)

模块架构如下：

```text
Web UI
  ├─ Prompt/Chat UI
  └─ HTTP/OpenAI Client
```

WebUI 公共实现约束：

1. 文件落点：`webui/index.html`、`webui/app.js`、`webui/styles.css`。
2. 协议只依赖 OpenAI chat-completions。

#### 4.4.1 需求 6.1.1：交互式聊天界面

实现设计：

1. `index.html` 包含：输入区、消息区、发送按钮、取消按钮。
2. `styles.css` 提供桌面和移动端布局。

验收：

1. 能在浏览器完成基础交互。

#### 4.4.2 需求 6.1.2：发送请求与展示回复

实现设计：

1. `app.js` 实现 `submitPrompt()`，调用 `/v1/chat/completions`。
2. 非流式响应一次性渲染 assistant 消息。

验收：

1. 非流式场景可稳定显示完整回复。

#### 4.4.3 需求 6.1.3：连续对话与本地历史

实现设计：

1. `ChatState = {messages, request_id, stream_buffer}`。
2. 在 `localStorage` 序列化存储并在加载时恢复。

验收：

1. 刷新后会话内容仍在。

#### 4.4.4 需求 6.2.1：对接 chat-completions

实现设计：

1. 请求体固定字段：`model/messages/stream/max_tokens/temperature`。
2. 与 Server schema 对齐，避免自定义字段。

验收：

1. 同一请求可切换 stream true/false。

#### 4.4.5 需求 6.2.2：SSE 解析与渲染

实现设计：

1. `handleStreamChunk(line)` 解析 `data: {json}`。
2. 增量拼接到最后一条 assistant 消息。

验收：

1. token 级增量渲染无乱序。

#### 4.4.6 需求 6.2.3：取消与错误提示

实现设计：

1. `cancelRequest()` 调用取消接口或中断连接。
2. UI 显示错误提示并恢复按钮状态。

验收：

1. 取消后不会继续追加 token。

#### 4.4.7 需求 6.3.1（可选）：多会话切换

实现设计：

1. `sessions: ChatState[]` + `activeSessionId`。
2. 切换时重新渲染消息区。

验收：

1. 多会话切换不串内容。

#### 4.4.8 需求 6.3.2（可选）：修改历史重生成

实现设计：

1. 允许编辑某条 user 消息。
2. 截断其后的消息并重发请求。

验收：

1. 可从中间节点重生后续对话。

#### 4.4.9 需求 6.3.3（可选）：前缀 KV 复用

实现设计：

1. UI 请求参数增加 `reuse_prefix`。
2. 命中信息显示在调试栏。

验收：

1. 命中复用时首 token 延迟明显下降。
## 5. 主要时序图

Offline（目标）：

```text
User -> LLM.generate -> LLMEngine -> Scheduler -> Executor -> Worker -> Core -> tokens
```

Online（目标）：

```text
Client -> API Server -> AsyncLLMEngine -> LLMEngine -> Scheduler -> Executor -> Worker -> Core
   ^                                                                              |
   +------------------------------- SSE streaming --------------------------------+
```

## 6. 交付验收
### 6.1 阶段 0：
完成 Core 对齐 llama.cpp 的重构，落地通用多模型 API（`LlaisysModel`），实现完整的runtime能力、多序列推理能力。
- `pytest -q test/test_infer.py --model-path /path/to/local/model` 定义为 ModelRunner 级对拍（`decode_batch + argmax` 对齐 `transformers`），不经过 Engine 层。
- 退出条件：进入阶段1前，离线主流程必须切换到 `LLM -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
- 必须通过此前已有全部测试（包含 `pytest -q test/test_infer.py --model-path /path/to/local/model`）。
- `test/test_core_decode_batch.py`：验证 SoA batch/decode 路径在单序列场景与多序列场景下的正确性。
-  `test/test_core_output_api.py`：验证 `GetLogits/GetLogitsIth/NOutputs/OutputIds` 行为一致、行数与 `logits` 标记一致。
-  `test/test_kv_cache.py`：KV slot/cell 语义、`seq_id + pos` 隔离、`kv_seq_*` 接口行为。
-  `test/test_core_model_api.py`：通用 `LlaisysModel` 接口（create/decode/logits/kv）行为验证。
-  `test/test_model_registry.py`：多模型注册与按 `model` 字段路由验证（至少覆盖 qwen2 + 一个 mock 模型）。
-  `test/test_qwen2_adapter.py`：验证 `python/llaisys/models/qwen2.py` 基于通用 C API 的适配行为。
-  `test/test_core_parity.py`：与 `transformers` 在 batch+argmax 路径做逐步 next-token 对拍（有本地模型时必跑）。
- 里程碑补充（已完成）：`decode` 真 batch 执行、Unified KV（slot 可多 `seq_id`）、`self_attention_masked` 隔离路径。
### 6.2 阶段 1：
Engine 内 argmax + offline完整实现。
- `test/test_offline.py`：离线一致性与流式/非流式行为。
- `test/test_llm_entrypoint.py`：`LLM` 入口契约（token兼容、prompt/prompts、batch params、stream）。
- `test/test_offline_parity.py`：Engine 离线链路与 `transformers` 对拍（single + multi-seq，有本地模型时执行）。

阶段1实现原则（已落地，作为约束保留）：

1. 允许把 `EngineClient/OutputProcessor` 等类在同进程场景合并到 `LLMEngine`，减少代码层级与样板。
2. 不允许删除以下在线前置契约：`submit/step/cancel` 语义、请求状态机、`Scheduler -> Worker` 边界、统一输出结构。
3. 阶段1以离线可交付为目标，online 专属能力可延后到阶段2，但接口语义必须提前保留。

阶段1完成判定（已满足）：

1. 离线主入口统一为 `LLM -> EngineClient -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
2. Core 仍只返回 `logits/output_ids`，采样不下沉到 Core。
3. 状态机与输出对象在离线场景可稳定复现并有测试覆盖。
4. 阶段1回归闭环已稳定：`test_offline + test_llm_entrypoint + test_engine_state_machine + test_engine_model_registry`；有模型时追加 `test_offline_parity`。
### 6.3 阶段 2（整体功能闭环）：
当前状态：阶段2核心功能已落地并通过回归（以当前源码为准）。
- Engine 采样链：`top-k/top-p/temperature` 已实现。
- online 完整最小闭环：HTTP/OpenAI 兼容入口、SSE 流式、取消请求已实现。
- Web UI 最小可用：多会话、流式展示、取消、调试面板已实现。
- 端到端路径已打通：`WebUI -> Server -> AsyncLLMEngine -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
- Core 架构修正已纳入阶段2稳定基线：多序列混排在单轮前向执行并通过 mask 隔离。
- 阶段2验收测试（As-Built）：
  - `test/test_sampling.py`
  - `test/test_online.py`
  - `test/test_online_http.py`
  - `test/test_online_stream_isolation.py`
  - `test/test_online_real_model_multisession.py`（需本地模型）
### 6.4 阶段 3（性能优化）：
在阶段2整体功能稳定后，再做吞吐/时延优化。
- 连续批处理 + 前缀缓存 + 投机（提升吞吐与时延）。
- 连续批处理 benchmark：吞吐收益验证。
- 前缀缓存命中率与收益验证。
- 投机解码回退正确性与收益验证。
### 6.5 全阶段要求：
接口稳定，主线接口保持 `llaisysModel*` 统一命名与语义。
全阶段回归要求：新增能力不得破坏通用接口与既有测试基线。


## 7. 未来扩展 暂不考虑（后续再议）

1. `memory_update -> re-reserve` 自动链路（含内存形态变化后的图重保留策略）。
2. 完整 `PP -> TG(split_only) -> PP` 三段预留细节。

## 8. 阶段2复盘（问题与修复）

### 8.1 问题：流式终止包偶发缺失

现象：

1. 某些请求流最后没有 `is_finished=true`，前端只能等 `[DONE]` 或超时。

修复：

1. 在 `AsyncLLMEngine.stream` 超时分支发现请求已完成时，主动补发终止 chunk。
2. 增加在线流式隔离与终止语义回归测试。

### 8.2 问题：多线程并发场景下 Segmentation fault

现象：

1. 真实模型并发流式测试中，创建或释放模型阶段出现段错误。

修复：

1. Python 侧引入显式 `close()` 链路，不再依赖 `__del__` 进行 native destroy。
2. C++ `Context` 修复：
   - `_current_runtime` 显式初始化；
   - `setDevice` 使用引用语义；
   - `context()` 改为线程局部常驻对象，避免线程退出析构导致 Runtime 悬挂引用。

说明：

1. 当前策略优先稳定性，代价是每线程一个有界常驻 Context；后续可在阶段3前做可回收重构。

### 8.3 问题：并发首请求 Tokenizer 初始化异常

现象：

1. 并发首请求时，`AutoTokenizer` 导入/初始化偶发失败。

修复：

1. Qwen2 tokenizer 延迟初始化加锁。
2. Worker 增加编码/解码异常降级路径，避免请求直接失败。

## 9. 后续优化路线（阶段3）

1. 采样参数从“同 ubatch 统一”升级为“每请求独立采样”。
2. 在不改接口前提下加强连续批处理策略与吞吐优化。
3. 引入前缀缓存命中与回退机制，并补齐命中率/收益指标。
4. 抽象统一 metrics 导出接口（QPS、TTFT、TPOT、队列长度、KV占用）。
