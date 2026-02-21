# Qwen2 接口设计（阶段0~2，多模型统一接口版）

本文档定义阶段0需要落地的 Core 对外接口与结构体。  
阶段0口径：对外只暴露通用多模型 API（工厂模式）；`qwen2` 仅作为一个 `model_type` 实现。

> 文档状态更新（2026-02-18）  
> 本文历史描述默认把 `kv_seq_*` 作为主语义。当前实现已进入 `SLOT/BLOCK` 双布局阶段，且 BLOCK 为主线。  
> 若冲突，以下述“0.1 当前接口口径（2026-02）”为准。详细计划见 `doc/qwen2_next_dev_plan_2026-02.md`。
> 目录口径补充（2026-02-20）：测试目录已重构到 `test/core|engine|offline|online|parity|ops|utils`。

## 0.2 阅读优先级（当前 vs 历史）

1. 当前执行口径优先：`doc/qwen2_next_dev_plan_2026-02.md`。
2. 本文中的 `As-Built` 段为阶段快照，不等同于当前所有行为边界。
3. 若本文历史段与“0.1 当前接口口径”冲突，以“0.1 当前接口口径”为准。

## 0.1 当前接口口径（2026-02）

1. `LlaisysModelCreateParams` 已包含：
   - `kv_cache_layout`（`SLOT=0 / BLOCK=1`，`<0` 走默认 BLOCK）
   - `kv_cache_block_size`（`<=0` 默认 `16`）
2. `llaisysModelDecode` 仍是统一主入口；Python 正常推理路径不依赖 `llaisysModelKvSeq*`。
3. BLOCK 主路径已切到显式 batch 元数据输入（对齐 nano-vllm runner 数据流）：
   - `slot_mapping`
   - `context_lens`
   - `batch_seq_ids`
   - `block_tables`
   - `n_batch_seq`
   - `block_table_width`
4. `llaisysModelKvSeq*` 当前主要用于兼容与测试。BLOCK 下部分语义已与 SLOT 不同：
   - BLOCK 下 `llaisysModelKvSeq*` 统一按未实现处理（`INTERNAL_ERROR(5)`）。
5. 当前推荐口径：BLOCK 作为性能主线，SLOT 作为兼容/回归模式。
6. NVIDIA + auto capacity 口径（当前实现）：
   - `max_num_seqs` 会从 EngineConfig 透传到 Qwen2 runner，
   - 并参与 KV auto capacity 的 `logical_cap_tokens = max_model_len * max_num_seqs` 估算；
   - 若未显式提供 `max_num_seqs`，才回落环境变量 `LLAISYS_KV_AUTO_MAX_SEQS`。

历史状态快照（As-Built）：

1. 主线接口已切到通用 `llaisysModel*`。
2. 为阶段0测试引入了 `LLAISYS_MODEL_TYPE_MOCK`（测试路由模型），不影响 `qwen2` 主线接口语义。
3. `include/llaisys/runtime/kv_cache.h` 属于目标头文件，当前仓库尚未落地该文件；KV 状态码语义目前由 `model.h` 文档约定与实现保持一致。
4. 已提供 `llaisysModelReplaceWeight`，用于安全替换权重槽位（同句柄 no-op，异句柄先释放旧权重）。
5. `llaisysModelDecode` 已是“真 batch 执行”语义（单轮前向处理整批 token）。
6. `SLOT` 路径支持 `n_seq_id > 1` token（一个 token 绑定多个 `seq_id`）。
7. `BLOCK` 主路径要求一 token 对应一个 `seq_id`，并由上层提供显式 block 元数据。
8. KV 已是 unified slot 语义：一个 slot 可绑定多个 `seq_id`（SLOT 路径）。

对齐范围说明：

1. 本文档只定义 Core 通用 C API（阶段0主线），不定义 Engine/Server Python 类接口。
2. 与 vLLM/`nano-vllm` 的对齐主要体现在“Core 返回 logits，采样在上层执行”的职责边界。
3. online 服务协议（OpenAI API、流式、取消）由 Server 文档约束，不在本接口文档中展开。

## 1. 头文件边界

1. `include/llaisys/models/model.h`：通用模型接口（唯一主入口）。
2. `include/llaisys/runtime/infer_types.h`：通用 batch/output 结构。
3. `include/llaisys/runtime/kv_cache.h`：KV 状态码与通用语义（目标态，当前未落地）。
4. `include/llaisys/models/qwen2.h`：Qwen2 专有元数据/权重槽位定义（不包含独立 decode/logits/kv API）。

## 2. 阶段0公开结构体

### 2.1 通用模型类型与创建参数

```c
typedef enum LlaisysModelType {
    LLAISYS_MODEL_TYPE_UNKNOWN = 0,
    LLAISYS_MODEL_TYPE_QWEN2   = 1,
    LLAISYS_MODEL_TYPE_MOCK    = 2, // stage0 test-only route
} LlaisysModelType;

typedef struct LlaisysModelCreateParams {
    LlaisysModelType model_type;
    const void *meta;               // points to model-specific meta struct
    llaisysDeviceType_t device;
    int *device_ids;
    int ndevice;
    int32_t kv_cache_layout;        // LlaisysKvCacheLayout, <0 means default(BLOCK)
    int32_t kv_cache_block_size;    // <=0 means default(16)
} LlaisysModelCreateParams;

struct LlaisysModel;
```

### 2.2 通用 batch（SoA）

```c
typedef struct LlaisysBatch {
    int32_t n_tokens;
    int64_t *token;      // used when embd == NULL
    float   *embd;       // optional, reserved for multimodal path
    int64_t *pos;        // nullable: auto-tracked when NULL
    int32_t *n_seq_id;   // nullable: default 1 when NULL
    int64_t **seq_id;    // nullable: default seq_id=0 when NULL
    int8_t  *logits;     // non-zero => keep logits row
    int32_t *slot_mapping;     // [n_tokens], BLOCK optional
    int32_t *context_lens;     // [n_batch_seq], BLOCK optional
    int64_t *batch_seq_ids;    // [n_batch_seq], BLOCK optional
    int32_t *block_tables;     // [n_batch_seq * block_table_width], BLOCK optional
    int32_t n_batch_seq;       // BLOCK optional
    int32_t block_table_width; // BLOCK optional
} LlaisysBatch;
```

### 2.3 Qwen2 专有元数据与权重槽位（用于 `model_type=qwen2`）

```c
struct LlaisysQwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct LlaisysQwen2Weights {
    llaisysTensor_t in_embed;
    llaisysTensor_t out_embed;
    llaisysTensor_t out_norm_w;
    llaisysTensor_t *attn_norm_w;
    llaisysTensor_t *attn_q_w;
    llaisysTensor_t *attn_q_b;
    llaisysTensor_t *attn_k_w;
    llaisysTensor_t *attn_k_b;
    llaisysTensor_t *attn_v_w;
    llaisysTensor_t *attn_v_b;
    llaisysTensor_t *attn_o_w;
    llaisysTensor_t *mlp_norm_w;
    llaisysTensor_t *mlp_gate_w;
    llaisysTensor_t *mlp_up_w;
    llaisysTensor_t *mlp_down_w;
};
```

## 3. 阶段0公开函数（C API）

### 3.1 通用模型主接口（`model.h`）

```c
__export struct LlaisysModel *llaisysModelCreate(
    const struct LlaisysModelCreateParams *params);

__export void llaisysModelDestroy(struct LlaisysModel *model);

__export LlaisysModelType llaisysModelType(const struct LlaisysModel *model);

__export void *llaisysModelWeights(struct LlaisysModel *model);
__export int llaisysModelReplaceWeight(
    struct LlaisysModel *model,
    const char *field_name,
    int32_t layer_idx,
    llaisysTensor_t new_weight);

__export int32_t llaisysModelDecode(
    struct LlaisysModel *model,
    struct LlaisysBatch batch);

__export float *llaisysModelGetLogits(struct LlaisysModel *model);
__export float *llaisysModelGetLogitsIth(struct LlaisysModel *model, int32_t i);
__export int32_t llaisysModelNOutputs(struct LlaisysModel *model);
__export const int32_t *llaisysModelOutputIds(struct LlaisysModel *model);
```

`llaisysModelWeights` 语义：

1. 返回模型私有权重槽位表指针（`void *`）。
2. 调用方先读 `llaisysModelType(model)`，再按类型转换：
   - `LLAISYS_MODEL_TYPE_QWEN2` -> `struct LlaisysQwen2Weights *`。
   - `LLAISYS_MODEL_TYPE_MOCK` -> `NULL`（无模型专有权重槽位）。
3. 生命周期与模型句柄一致，调用方不得释放该指针本身。

`llaisysModelReplaceWeight` 语义：

1. 用于通过字段名+层号安全替换槽位句柄，避免直接覆写指针导致泄漏。
2. 全局槽位（如 `in_embed/out_embed/out_norm_w`）要求 `layer_idx = -1`。
3. 分层槽位（如 `attn_q_w/mlp_down_w`）要求 `layer_idx` 在 `[0, nlayer)`。
4. 返回码：`0` 成功，`-1` 参数非法，`-2` 模型类型不支持，`-3` 字段名非法，`-4` 层索引非法。

`Decode` 返回码（阶段0约定）：

1. `0`：成功
2. `1`：无可用 KV slot
3. `2`：执行被中止（保留码）
4. `-1`：输入参数非法
5. `< -1`：内部致命错误

`Decode` 与采样职责边界（阶段0约定）：

1. `llaisysModelDecode` 只负责执行计算并产出 logits 行（由 `batch.logits` 与 `output_ids` 描述）。
2. Core 不执行采样决策（不在 decode 主路径内做 argmax/top-k/top-p/temperature）。
3. 采样由上层 Engine/Executor 基于 `GetLogitsIth` 返回结果执行。
4. 阶段0兼容期允许模型适配层临时使用离线路径 argmax（仅用于旧测试兼容），不改变 `decode` 主路径职责。

`Decode` 输入约束（当前口径）：

1. `SLOT`：`n_seq_id[i] >= 1`，支持一个 token 绑定多个 `seq_id`。
2. `BLOCK`：主路径要求 `n_seq_id[i] == 1`，并要求显式 `slot_mapping/context_lens/batch_seq_ids/block_tables` 元数据。
3. 对于显式提供 `pos[i]` 的输入，`pos` 必须与当前可推进位置一致。
4. `batch.logits` 仅控制“是否返回该 token 的 logits 行”，不改变 KV 写入与前向执行。

### 3.2 通用 Batch 辅助接口

```c
__export struct LlaisysBatch llaisysBatchInit(
    int32_t n_tokens,
    int32_t embd,
    int32_t n_seq_max);

__export struct LlaisysBatch llaisysBatchGetOne(
    int64_t *token,
    int32_t n_tokens);

__export void llaisysBatchFree(struct LlaisysBatch batch);
```

### 3.3 通用 KV 序列接口（`model.h`，兼容态）

```c
__export int llaisysModelKvSeqCp(
    struct LlaisysModel *model,
    int64_t dst_seq,
    int64_t src_seq,
    int64_t p0,
    int64_t p1);

__export int llaisysModelKvSeqRm(
    struct LlaisysModel *model,
    int64_t seq_id,
    int64_t p0,
    int64_t p1);

__export int llaisysModelKvSeqAdd(
    struct LlaisysModel *model,
    int64_t seq_id,
    int64_t p0,
    int64_t p1,
    int64_t delta);

__export int llaisysModelKvSeqKeep(
    struct LlaisysModel *model,
    int64_t seq_id);

__export int64_t llaisysModelKvSeqPosMax(
    struct LlaisysModel *model,
    int64_t seq_id);

__export int llaisysModelRequestFree(
    struct LlaisysModel *model,
    int64_t seq_id);

typedef struct LlaisysKvStats {
    int64_t capacity_tokens;
    int64_t used_tokens;
    int64_t free_tokens;
    int64_t peak_used_tokens;
} LlaisysKvStats;

__export int llaisysModelKvStats(
    struct LlaisysModel *model,
    struct LlaisysKvStats *out_stats);

__export int llaisysModelKvResetPrefixCache(
    struct LlaisysModel *model);
```

区间语义统一为半开区间 `[p0, p1)`。  
KV 返回码：

1. `0`：OK
2. `1`：OOM_SLOT
3. `2`：INVALID_SEQ
4. `3`：INVALID_POS
5. `4`：EMPTY_RANGE
6. `5`：INTERNAL_ERROR

KV 语义补充（当前口径）：

1. `llaisysModelKvSeq*` 在 SLOT 下保留 legacy 语义；BLOCK 下统一返回 `INTERNAL_ERROR(5)`。
2. `llaisysModelRequestFree(seq_id)` 已落地：释放该请求绑定的全部 KV（BLOCK/SLOT 通用）。
3. `llaisysModelKvStats` 已落地：返回 capacity/used/free/peak 四项 token 统计。
4. `llaisysModelKvResetPrefixCache` 已落地：
   - BLOCK 路径下，当仍有活跃 KV 占用时返回 `INTERNAL_ERROR(5)`；
   - 当无活跃占用时返回 `OK(0)`，并清理 runtime 请求元数据（`req_to_blocks/seq_to_stream/block_tables`）；
   - 当前 BLOCK 前缀哈希索引主存于 Python `BlockManager`，该接口主要提供 runtime 侧 reset 契约。

## 4. 调用契约

1. 阶段0默认 `LlaisysModel` 非线程安全，同一模型句柄不得并发调用。
2. 权重句柄写入权重槽位后由模型接管生命周期；共享权重必须去重释放。
3. `BatchInit/GetOne` 分配的内存由调用方 `BatchFree` 释放。
4. `GetLogits*` 返回的缓冲属于模型内部，不允许调用方释放。
5. `OutputIds[j]` 对应 batch token 索引，`NOutputs` 与 `batch.logits` 标记行数一致。

## 5. Python 绑定要求（阶段0）

主线绑定（`python/llaisys/libllaisys/model.py`）：

1. `llaisysModelCreate`
2. `llaisysModelDestroy`
3. `llaisysModelType`
4. `llaisysModelWeights`
5. `llaisysModelReplaceWeight`
6. `llaisysModelDecode`
7. `llaisysModelGetLogits`
8. `llaisysModelGetLogitsIth`
9. `llaisysModelNOutputs`
10. `llaisysModelOutputIds`
11. `llaisysModelKvSeqCp/Rm/Add/Keep/PosMax`
12. `llaisysBatchInit/GetOne/Free`

模型适配层（`python/llaisys/models/qwen2.py`）通过上述通用接口完成 Qwen2 的权重映射与推理调用，不引入独立的 `qwen2_*` ctypes 主流程。

## 6. 阶段0验收对齐清单

1. 通用模型接口 `llaisysModel*` 可完整驱动 create/decode/logits/kv 行为。
2. Engine/Server 主流程只依赖 `LlaisysModel`，不出现 `qwen2_*` 分支 API 依赖。
3. `python/llaisys/models/qwen2.py` 可基于通用接口完成离线闭环。
4. 新增模型时仅新增 `model_type + models/<model>.py + src/llaisys/<model>/`，无需改动通用接口签名。

## 7. 迁移期可选项（不纳入主线验收）

1. 若短期需要保持历史调用，可在 Python 层提供本地适配函数，把旧调用转发到 `llaisysModel*`。
2. 若保留 `qwen2_*` C 包装接口，应标记为 deprecated，且不得引入独立语义分叉。

## 8. 本期不纳入接口范围

1. `memory_update -> re-reserve` 自动链路相关公开接口。
2. 完整 `PP -> TG(split_only) -> PP` 三段预留策略开关或参数。

## 9. 阶段2 Python/Server 接口契约（历史快照，As-Built）

说明：

1. 本节为当前源码已实现的 Python 层接口，用于 offline/online 集成。
2. Core 主线仍是 `llaisysModel*`，本节不替代 C API，只补“上层如何调用”。

### 9.1 Engine 入口（Python）

关键类与方法：

1. `python/llaisys/entrypoints/llm.py`
   - `LLM.generate(...)`
   - `LLM.stream(...)`
   - `LLM.submit(...) / LLM.step() / LLM.collect(...) / LLM.cancel(...)`
   - `LLM.close()`
2. `python/llaisys/server/async_engine.py`
   - `AsyncLLMEngine.submit(...) / collect(...) / cancel(...)`
   - `AsyncLLMEngine.stream(...) / generate(...)`
   - `AsyncLLMEngine.is_finished(...) / get_request_status(...)`
   - `AsyncLLMEngine.close()`

约束：

1. `LLMEngine.step()` 当前支持单步多请求合批执行（continuous batching 基础形态）。
2. 同一 step 已支持按请求应用独立采样参数；后续重点转向调度层 request-aware 优化与 prefix cache。

### 9.2 OpenAI 兼容服务接口（Python）

服务入口：

1. `python -m llaisys.server --model-path ... --device cpu --host 127.0.0.1 --port 8000 [--verbose]`

HTTP 路由（历史快照）：

1. `GET /health`
2. `POST /v1/chat/completions`
3. `POST /v1/requests/{request_id}/cancel`

流式协议：

1. SSE `data: {json}\n\n`
2. 结束标记 `data: [DONE]\n\n`
3. chunk 关键字段：`request_id`、`token_id`、`is_finished`、`choices[0].delta.content`

### 9.3 WebUI 对接接口（历史快照）

1. WebUI 作为静态站点运行：`python -m http.server 8081 -d webui`。
2. 前端请求固定调用 `POST /v1/chat/completions`（`stream=true`）。
3. 支持多会话切换、流式渲染、取消请求与调试日志输出。

## 10. 生命周期与并发契约（阶段2补充）

1. 模型与引擎对象应显式 `close()`，避免依赖 `__del__` 触发 native 资源释放。
2. 推荐关闭顺序：`HTTPServer.stop() -> OpenAIServer.close() -> AsyncLLMEngine.close() -> LLMEngine.close() -> Worker.close() -> ModelRunner.close()`。
3. `Qwen2` tokenizer 为延迟初始化并带锁，允许并发请求下安全首用。
4. 若 tokenizer 初始化失败，Worker 会走降级编码路径，保证请求不会因导入异常直接中断。

## 11. 已知差异与后续接口计划

1. 目前 OpenAI 路由仅覆盖 chat-completions，`/v1/completions` 与 embeddings 为后续扩展。
2. `sampling_params` 已支持请求级并在同一 step 按请求独立应用。
3. 监控接口当前以日志为主，标准 metrics 导出接口（Prometheus 等）为下一步。
4. Core 已实现 unified KV + mask 隔离与 `kv_seq_add(delta)` 位置平移；滑窗与性能优化接口仍在后续阶段。
