# Qwen2 模型概要设计

## 1. 设计目标

1. 基于现有 LLAISYS 的 Qwen2 推理能力，在不破坏当前测试正确性的前提下，完成 Core、Engine、Server、Client 的分层扩展。
2. 对齐 `doc/qwen2_requirements.md` 中的每条需求，形成可执行的 RFC 级实现指南。
3. 支持离线与在线统一执行链路，支持多序列并发、连续批处理、流式输出，并为多模型/多设备/分布式预留接口。
## 2. 源码组织

```
llaisys/
├─ include/
│  └─ llaisys/
│     ├─ runtime/
│     │  ├─ infer_types.h     # 通用 Batch/Output C 结构
│     │  └─ kv_cache.h        # 通用 KV 管理 C API（seq_*）
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
│     ├─ models/
│     │  ├─ model.hpp        # 通用模型接口（weights/decode/kv_*）
│     │  └─ model.cpp
│     └─ qwen2/
│        ├─ qwen2_api.cc     # C API 与 Qwen2Model 绑定
│        ├─ qwen2_model.hpp  # 驱动 + block + weights 声明（合并版）
│        └─ qwen2_model.cpp  # 构造、infer、调度 + weights 校验/映射（合并版）
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
│     ├─ model_runners/
│     │  ├─ base.py          # ModelRunner 抽象接口
│     │  └─ qwen2_runner.py  # Qwen2 ModelRunner 实现
│     ├─ server/
│     │  ├─ openai_server.py # OpenAIServer/HttpServer
│     │  ├─ async_engine.py  # AsyncLLMEngine
│     │  ├─ session_store.py # 会话状态存储
│     │  ├─ metrics.py       # 监控指标输出
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

解耦拆分规则：

1. `src/llaisys/runtime/*` 与 `python/llaisys/common/*` 只放公共组件，不出现模型专有权重名或层字段。
2. `src/llaisys/qwen2/*` 与 `python/llaisys/model_runners/qwen2_runner.py` 只放 Qwen2 专有逻辑。
3. Engine/Server 仅依赖 `model_runners/base.py` 抽象接口和 `common/model_registry.py`，不直接依赖具体模型实现。
4. 新增模型时只新增 `src/llaisys/<model>/` 与 `python/llaisys/model_runners/<model>_runner.py`，公共模块不改或少改。

- **注释约定**：
  - `include/`：关键结构与 API 使用 `//` 说明参数含义和生命周期。
  - `src/llaisys/qwen2/*.cpp`：文件头注明模块职责，复杂函数前加简述；内部逻辑依赖 `ASSERT`/`CHECK_*`。
  - `python/llaisys/libllaisys/qwen2.py`：每个 ctypes 函数注明 `argtypes/restype` 对应的实际意义。
  - `python/llaisys/model_runners/qwen2_runner.py`：类 docstring 描述整体用途，私有方法附注权重映射、Tokenizer 与单步执行策略；必要时可保留对 `python/llaisys/models/qwen2.py` 的兼容调用。
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

1. 代码落点：`include/llaisys/runtime/infer_types.h`、`include/llaisys/runtime/kv_cache.h`、`include/llaisys/models/qwen2.h`、`src/llaisys/runtime/{weights,workspace,kv_cache,graph}/`、`src/llaisys/qwen2/qwen2_model.hpp`、`src/llaisys/qwen2/qwen2_model.cpp`、`src/llaisys/qwen2/qwen2_api.cc`、`python/llaisys/libllaisys/qwen2.py`。
2. C API 使用纯 C 结构体，不暴露 STL 容器。
3. Core 统一输入语义为 SoA batch（`token/pos/n_seq_id/seq_id/logits`）。
4. 旧接口 `llaisysQwen2ModelInfer(model, token_ids, ntoken)` 保留，用于兼容。

#### 4.1.1 需求 3.1.1：模型权重加载与生命周期

实现设计：

1. 在 `src/llaisys/runtime/weights/weights.hpp` 定义公共权重容器接口（如 `IWeightStore`），统一管理句柄生命周期与去重释放。
2. `Qwen2Model` 仅持有 `IWeightStore` 句柄，并通过键名读取权重，不直接管理每个槽位的销毁逻辑。
3. 覆盖写入保护函数下沉到 `runtime/weights`：

```cpp
void replace_weight_slot(const char *name, llaisysTensor_t new_handle);
```

4. `replace_weight_slot()` 规则：
   - 同名槽位旧句柄与新句柄相同则直接返回；
   - 不同句柄则先安全释放旧句柄，再写入新句柄；
   - 支持共享句柄引用计数或别名去重。
5. 在 `qwen2_api.cc` 的 create/destroy 错误路径保持 fail-fast，避免跨 C 边界异常泄漏。
6. Python `python/llaisys/model_runners/qwen2_runner.py` 继续使用 `_detach_tensor_handle()` 转移所有权（可复用兼容模块实现）。

验收：

1. 权重共享（`in_embed == out_embed`）不会 double free。
2. 重复赋值同一槽位不泄漏。

#### 4.1.2 需求 3.1.2：中间 buffer 设计与复用

实现设计：

1. 在 `src/llaisys/runtime/workspace/workspace.hpp` 定义公共 Workspace（arena）接口，采用“一块连续大内存 + offset/view”的模式（对齐 llama.cpp 思路）。
2. `Qwen2Model` 仅持有 `IWorkspace` 句柄，不直接持有中间 buffer 成员或 `workspace_token_cap_` 这类容量状态。
3. Workspace 最小接口定义：

```cpp
struct WorkspaceView {
    void *ptr;
    size_t bytes;
};

class IWorkspace {
public:
    virtual bool reserve(size_t bytes, size_t alignment) = 0;  // grow-only
    virtual WorkspaceView view(size_t offset, size_t bytes) = 0;
    virtual size_t capacity() const = 0;
    virtual ~IWorkspace() = default;
};
```

4. 图执行前先做一次内存规划（按本轮 `n_tokens` 计算每个中间张量所需字节和 offset），得到 `GraphMemPlan`：

```cpp
struct BufferSlot {
    const char *name;
    size_t offset;
    size_t bytes;
};

struct GraphMemPlan {
    size_t total_bytes;
    std::vector<BufferSlot> slots;
};
```

5. 执行流程固定为：`plan -> reserve(total_bytes) -> view(offset)`。若容量足够则不重新分配，只复用已有 arena。
6. Worker 单实例复用 Workspace，不做请求级释放；进程退出时统一回收。
7. 初始化阶段图预留流程（本期）：
   - 至少执行一次 `graph_reserve(profile=PP)`，完成 prefill 常见形态预留。
8. `graph_reserve` 的职责是 allocator 语义，不是单纯 build graph：
   - 重置调度器状态；
   - 用保守 ubatch 构图；
   - 否则执行 `reserve/alloc`，把 compute buffers 预留到位。
9. 对外行为约束：初始化完成后，常见 prefill/decode 形态不应在首轮触发临时扩容。

验收：

1. 连续请求时 `reserve()` 在容量未超出时不触发重新分配（地址稳定）。
2. 算子拿到的输入/输出指针均来自 `view(offset)`，内存连续可直接下发到底层 kernel。
3. 没有每轮重复 new tensor。
4. 本期验收要求“已预留形态不触发热路径临时扩容”。

#### 4.1.3 需求 3.1.3：KV-Cache 设计与多序列支持

实现设计：

1. 在 `src/llaisys/runtime/kv_cache/kv_cache.hpp` 新增（公共组件，cell 为一等对象）：

```cpp
using SeqId = int64_t;
using Pos = int64_t;

struct KvCell {
    size_t idx;             // physical slot index
    int64_t pos;
    uint32_t generation;    // optional: debug/reuse tracking
    bool used;
    std::vector<SeqId> seq_ids;  // persistent mapping: cell -> seq set
};

struct SeqPosStat {
    SeqId seq_id;
    Pos pos_min;
    Pos pos_max;
    bool active;
    size_t n_cells;
};

struct SlotInfo {
    // token[i] -> cell idxs[i], transient for current ubatch
    std::vector<size_t> idxs;
};

struct UBatchView {
    int32_t n_tokens;
    const int64_t * token;      // SoA: contiguous token ids
    const int64_t * pos;        // SoA: contiguous positions (nullable => auto)
    const int32_t * n_seq_id;   // SoA: per-token number of seq ids
    int64_t * const * seq_id;   // SoA: per-token seq-id pointer
    const int8_t * logits;      // SoA: output-row request mask
};

class IKvCache {
public:
    virtual KvStatus find_slots(const UBatchView &ubatch, SlotInfo *out) = 0;
    virtual KvStatus commit_slots(const UBatchView &ubatch, const SlotInfo &slots) = 0;
    virtual bool cell_has_seq(size_t idx, SeqId seq_id) const = 0;
    virtual bool cell_has_any_seq(size_t idx, const SeqId *seq_ids, int32_t n_seq_id) const = 0;
    virtual Pos cell_pos(size_t idx) const = 0;
    virtual ~IKvCache() = default;
};
```

2. `Qwen2Model` 仅新增 KV 组件句柄（不直接持有 slot 元数据）：

```cpp
std::shared_ptr<llaisys::runtime::IKvCache> kv_cache_;
```

3. `init_kv_cache_()` 负责构建 `IKvCache` 实例并注入配置（`maxseq/nlayer/nkvh/dh`），并分配每层 `k_cache/v_cache` 张量（cell 维度等于 `maxseq`）。
4. slot 分配与序列绑定逻辑放到 `runtime/kv_cache` 内部，不放在 `Qwen2Model`：

```cpp
KvStatus alloc_or_attach(int64_t seq_id, int64_t pos, size_t *slot_idx);
KvStatus attach_seq(size_t slot_idx, int64_t seq_id);
```

5. 明确 KV 元数据不变量（实现里要有断言）：
   - `idx` 是物理位置，`pos` 是逻辑位置，二者不可混用；
   - `used == false` 时 `seq_ids` 必须为空；
   - `seq_pos_stat` 只是统计缓存，必要时可由 `cells` 重建；
   - 持久映射方向是 `cell -> (seq_ids, pos)`，`seq -> cells` 使用统计或临时扫描，不维护强一致全局反向表；
   - Core 不实现每个 `seq_id` 的硬性 cell 配额（对齐 llama.cpp），仅管理全局 cell 池。

验收：

1. 能打印 `cell[idx] = {seq_ids, pos, used}` 诊断信息。
2. `IKvCache` 可同时维护多个 `seq_id` 的 `pos_max`，并在重建统计后保持一致。
3. 模型层不可直接改写 cell 元数据。

#### 4.1.4 需求 3.1.4：输出 buffer 管理

实现设计：

1. 在 `include/llaisys/runtime/infer_types.h` 新增与 llama.cpp 对齐的 SoA batch 结构（`qwen2.h` 仅 include 并复用）：

```c
typedef struct LlaisysBatch {
    int32_t n_tokens;
    int64_t * token;      // used when embd == NULL
    float   * embd;       // optional, reserved for embedding/multimodal path
    int64_t * pos;        // nullable: auto-tracked when NULL
    int32_t * n_seq_id;   // nullable: default 1 when NULL
    int64_t ** seq_id;    // nullable: default seq_id=0 when NULL
    int8_t  * logits;     // output-row mask, 0 means no logits row for this token
} LlaisysBatch;
```

2. 输出读取改为上下文查询式接口（对齐 llama.cpp）：

```c
float * llaisysQwen2ModelGetLogits(struct LlaisysQwen2Model * model);
float * llaisysQwen2ModelGetLogitsIth(struct LlaisysQwen2Model * model, int32_t i);
int32_t llaisysQwen2ModelNOutputs(struct LlaisysQwen2Model * model);
const int32_t * llaisysQwen2ModelOutputIds(struct LlaisysQwen2Model * model);
```

3. 输出契约：logits 行只为 `batch.logits[i] != 0` 的 token 生成，且按 batch 出现顺序紧凑存放。
4. `batch.logits` 语义是“是否输出该行 logits”，不是“是否必须采样”。
5. `output_ids[j]` 表示第 `j` 条输出行对应 batch 内哪个 token 索引，便于 Engine 精确回填到序列状态。
6. 采样与 logits 输出解耦：采样默认在 Engine 执行；Core 仅提供 logits 缓冲查询。
7. logits/output_ids 存放在 runtime 的连续输出缓冲区（由 workspace/output 组件管理），生命周期至少覆盖到下一次 `Decode()` 调用前。
8. 增加 `output_reserve(n_outputs)` 语义（对齐 llama.cpp）：

```cpp
bool output_reserve(int32_t n_outputs);
```

9. 在每次 `decode()`（以及可选 `encode()`）前先计算本轮 `n_outputs` 并调用 `output_reserve(n_outputs)`，保证输出缓冲区稳定，避免热路径临时扩容。

验收：

1. decode 阶段每活跃序列返回 1 行 logits。
2. prefill 阶段可按策略只返回末 token logits。
3. 当 prefill 请求多行 logits 时，输出行数与 `logits != 0` 的 token 数严格一致。
4. `GetLogits/GetLogitsIth/OutputIds` 返回的指针在同一轮查询内稳定可读。
5. decode 前调用 `output_reserve` 后，`GetLogits*` 指针在本轮推理结束前不变。

#### 4.1.5 需求 3.2.1：slot/cell + 逻辑 pos

实现设计：

1. 替换线性写入接口，按 `SlotInfo.idxs` 写入：

```cpp
void copy_into_cache_slot_(tensor_t &cache, size_t slot_idx, const tensor_t &src_row, size_t row_idx);
```

2. 执行采用两阶段，避免半写入状态：
   - 阶段 A：`find_slots(ubatch) -> SlotInfo`（只做规划，不改元数据）；
   - 阶段 B：按 `idxs` 写 K/V 后 `commit_slots(ubatch, slots)`（原子更新元数据）。
3. `seq_pos_stat`（`pos_min/pos_max/n_cells`）由 KV 组件维护，`Qwen2Model` 只通过查询接口读取。
4. attention 读路径以 cell 为准：
   - 先拿候选 cell 集（used cells）；
   - 再过滤 `cell_has_any_seq(idx, q.seq_id_set)`（query 的 seq 集与 cell 的 seq 集有交集）；
   - 最后做 causal 判断 `cell_pos(idx) <= q.pos`。
5. `prefill/decode` 在 Core 内不分两套写 cache 代码，统一走 “batch(SoA) + SlotInfo”。

验收：

1. 可打印 `token[i] -> cell[idx]` 映射，便于问题定位。
2. 逻辑 pos 与物理 idx 解耦，混排输入下仍正确。
3. 任一步骤失败不会留下“部分 token 已写入、元数据未提交”的脏状态。

#### 4.1.6 需求 3.2.2：多序列混排隔离

实现设计：

1. 仅保留统一 mask 路径，不采用按 `seq_id` 分组执行的过渡实现。
2. mask 判定条件与 llama.cpp 语义对齐：
   - 集合隔离：`intersect(q.seq_ids, kv.seq_ids) != empty`；
   - causal：`kv.pos <= q.pos`。
3. 对 `n_seq_id > 1` 的 query token，任一命中即视为可见（多 `seq_id` 视作等价集合）。
4. `self_attention` 接口应接收 mask 或可推导 mask 的元信息，避免在调度层做分组绕行。

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
3. `decode` 若任一 token `find_slots` 失败，整批失败返回（本轮不 commit）。
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
3. 公共 C API 在 `include/llaisys/runtime/kv_cache.h` 定义（如 `llaisysKvSeq*`）；`qwen2.h` 保留 `llaisysQwen2KvSeq*` 兼容包装，内部实现路径为 `Qwen2Model -> IKvCache`。
4. ctypes 在 `python/llaisys/libllaisys/qwen2.py` 同步绑定，并保留错误码到异常的映射表。

验收：

1. 每个 API 都有单测覆盖边界区间、空区间、无命中区间。
2. `kv_seq_rm` 后空 cell 可再次被 `find_slots` 分配。
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

实现设计：

1. `fill_pos_ids_()` 改为按 batch 的 `pos` 列填充：

```cpp
void fill_pos_ids_from_batch_(const tensor_t &pos_ids, const UBatchView &ubatch);
```

2. 禁止再用 `cur_len_ + i` 生成 pos（仅在 `batch.pos == NULL` 的兼容路径允许自动位置）。

验收：

1. 混排序列时每个 token 的 pos 都可回溯到输入。

#### 4.1.13 需求 3.3.4：batch/ubatch 路径

实现设计：

1. 新增对齐 llama.cpp 的主入口：

```cpp
int32_t llaisysQwen2ModelDecode(struct LlaisysQwen2Model * model, struct LlaisysBatch batch);
struct LlaisysBatch llaisysBatchInit(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
struct LlaisysBatch llaisysBatchGetOne(int64_t * token, int32_t n_tokens);
void llaisysBatchFree(struct LlaisysBatch batch);
```

2. 每次 `decode` 处理一个 batch；内部可拆多个 ubatch 执行，外部只关心单次返回状态码。
3. 状态码语义对齐 llama.cpp：`0=success`、`1=no KV slot`、`2=aborted`、`-1=invalid input`、`<-1=fatal`。
4. 旧入口 `infer(token_ids, ntoken)` 仅作为兼容 helper，内部走 `llaisysBatchGetOne`/`llaisysQwen2ModelDecode` 路径。

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

#### 4.1.16 Core 对齐 llama.cpp 的关键口径

具体落地：

1. 输入是 token 列表，不是 `[B,T]`。
2. batch 输入采用 SoA 连续数组（`token/pos/n_seq_id/seq_id/logits`）。
3. 物理 cell/slot 和逻辑 pos 分离。
4. 多序列隔离靠 `seq_id + pos` mask。
5. 资源不足默认失败，回收依赖显式 `seq_*`。
6. Core 不做 per-seq 硬配额；请求公平性在 Engine/Server 处理。
7. Workspace 采用 `graph_reserve + reserve/view`，不是模型内逐 tensor 手工扩容。
8. 运行期图执行采用 `can_reuse` 优先，必要时 `build+alloc` 重建。
9. 输出缓冲采用 `output_reserve(n_outputs)`，与采样解耦。

### 4.2 infer Engine（Python）

模块架构如下：

```text
LLMEngine
  ├─ RequestScheduler
  ├─ Executor
  └─ Worker
       └─ ModelRunner -> Core
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
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
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

1. `worker.py` 中 lazy init `ModelRunner`。
2. 暴露：

```python
class Worker:
    def warmup(self) -> None: ...
    def run(self, plan: BatchPlan) -> dict[int, StepOutput]: ...
```

3. `run()` 内部把 BatchPlan 转成 ctypes `LlaisysBatch`（SoA）并调用 `llaisysQwen2ModelDecode`。

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

1. `python/llaisys/model_runners/qwen2_runner.py` 提供：

```python
def load_config(model_path: str) -> dict: ...
def load_tokenizer(model_path: str): ...
```

2. Worker 通过这些函数创建 ModelRunner。

验收：

1. Engine 层不直接读取 `config.json`。

#### 4.2.10 需求 4.4.2：权重加载与映射校验

实现设计：

1. 在 ModelRunner 增加：

```python
def validate_weight_coverage(self) -> None: ...
```

2. 在 `_load_safetensors()` 后调用，缺失即抛错。

验收：

1. 缺少关键权重时启动失败并给出字段名。

#### 4.2.11 需求 4.4.3：Tokenizer 能力

实现设计：

1. ModelRunner 提供统一接口：

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

#### 4.2.14 需求 4.5.2：首期 Qwen2，预留多模型

实现设计：

1. 新增 `python/llaisys/common/model_registry.py`：

```python
MODEL_REGISTRY = {
    "qwen2": Qwen2ModelRunner,
}
```

2. Worker 通过 registry 创建 runner。

验收：

1. 新增模型只需要注册，不改 Engine 主流程。

#### 4.2.15 需求 4.5.3：Worker 加载，Engine 路由

实现设计：

1. Engine 根据请求的 `model` 字段选择 Worker。
2. Worker 在首次任务时加载模型实例。

验收：

1. Engine 不直接实例化 `Qwen2`。

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

1. 取消后请求状态转为 `cancelled` 且停止推送。

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
2. 首期 WorkerPool 仅注册 qwen2。

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

## 6. 未来扩展

### 6.1 对齐需求 7.x 的分阶段落地

1. 阶段 0：完成 Core 对齐 llama.cpp 的重构（含 batch/ubatch SoA、slot/cell KV、输出接口与图复用基础），并通过此前已有全部测试（含 `test/test_infer.py --test`）。
2. 阶段 1：离线闭环 + argmax（先保证正确性与兼容性）。
3. 阶段 2：sampling + online（补齐可用性）。
4. 阶段 3：连续批处理 + 前缀缓存 + 投机（提升吞吐与时延）。
5. 全阶段要求：接口稳定，旧接口可兼容迁移。

### 6.2 对齐需求 8.x 的测试与验收

1. 阶段 0 验收门槛：必须通过此前已有全部测试（至少包含 `test/test_infer.py --test`）。
2. `test/test_core_decode_batch.py`：验证 SoA batch/decode 路径在单序列场景与旧 `infer()` 输出一致。
3. `test/test_core_output_api.py`：验证 `GetLogits/GetLogitsIth/NOutputs/OutputIds` 行为一致、行数与 `logits` 标记一致。
4. `test/test_kv_cache.py`：KV slot/cell 语义、`seq_id + pos` 隔离、`kv_seq_*` 接口行为。
5. `test/test_core_compat.py`：旧 C API `llaisysQwen2ModelInfer` 兼容性回归（重构后仍可用）。
6. `test/test_offline.py`：离线一致性与流式/非流式行为。
7. `test/test_online.py`：在线并发、流式、取消。
8. `test/test_sampling.py`：采样链行为。
9. 连续批处理 benchmark：吞吐收益验证。
10. 全阶段回归要求：新增能力不得破坏旧接口与既有测试基线。

### 6.3 暂不考虑（后续再议）

1. `memory_update -> re-reserve` 自动链路（含内存形态变化后的图重保留策略）。
2. 完整 `PP -> TG(split_only) -> PP` 三段预留细节。
