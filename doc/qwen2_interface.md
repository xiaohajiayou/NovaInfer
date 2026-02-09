# Qwen2 接口设计（阶段0，多模型统一接口版）

本文档定义阶段0需要落地的 Core 对外接口与结构体。  
阶段0口径：对外只暴露通用多模型 API（工厂模式）；`qwen2` 仅作为一个 `model_type` 实现。

## 1. 头文件边界

1. `include/llaisys/models/model.h`：通用模型接口（唯一主入口）。
2. `include/llaisys/runtime/infer_types.h`：通用 batch/output 结构。
3. `include/llaisys/runtime/kv_cache.h`：KV 状态码与通用语义。
4. `include/llaisys/models/qwen2.h`：Qwen2 专有元数据/权重槽位定义（不包含独立 decode/logits/kv API）。

## 2. 阶段0公开结构体

### 2.1 通用模型类型与创建参数

```c
typedef enum LlaisysModelType {
    LLAISYS_MODEL_TYPE_UNKNOWN = 0,
    LLAISYS_MODEL_TYPE_QWEN2   = 1,
} LlaisysModelType;

typedef struct LlaisysModelCreateParams {
    LlaisysModelType model_type;
    const void *meta;               // points to model-specific meta struct
    llaisysDeviceType_t device;
    int *device_ids;
    int ndevice;
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
3. 生命周期与模型句柄一致，调用方不得释放该指针本身。

`Decode` 返回码（阶段0约定）：

1. `0`：成功
2. `1`：无可用 KV slot
3. `2`：执行被中止（保留码）
4. `-1`：输入参数非法
5. `< -1`：内部致命错误

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

### 3.3 通用 KV 序列接口（`model.h`）

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
```

区间语义统一为半开区间 `[p0, p1)`。  
KV 返回码：

1. `0`：OK
2. `1`：OOM_SLOT
3. `2`：INVALID_SEQ
4. `3`：INVALID_POS
5. `4`：EMPTY_RANGE
6. `5`：INTERNAL_ERROR

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
5. `llaisysModelDecode`
6. `llaisysModelGetLogits`
7. `llaisysModelGetLogitsIth`
8. `llaisysModelNOutputs`
9. `llaisysModelOutputIds`
10. `llaisysModelKvSeqCp/Rm/Add/Keep/PosMax`
11. `llaisysBatchInit/GetOne/Free`

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
