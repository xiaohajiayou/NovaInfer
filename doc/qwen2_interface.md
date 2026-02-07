# Qwen2 模型接口设计（对齐当前头文件原型）

本文档严格对齐 `include/llaisys/models/qwen2.h:7` 起定义的原型，采用“后端分配权重槽位 + 前端通过 tensor 句柄写入数据”的加载方式，而不是名称映射式 `load_weight(name, ptr)`。

## 1. C API（真实原型）

头文件：`include/llaisys/models/qwen2.h`

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

struct LlaisysQwen2Model;

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const struct LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice);

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model);

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    struct LlaisysQwen2Model *model);

__export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken);
```

接口语义约定如下（用于消除实现歧义）：

- `llaisysQwen2ModelCreate`
  - `meta` 描述模型结构与数值超参。
  - `device/device_ids/ndevice` 与现有 runtime 体系保持一致；本阶段允许仅支持 CPU（`LLAISYS_DEVICE_CPU`）。
  - 后端应在此阶段完成：
    - 权重槽位（`LlaisysQwen2Weights` 内所有字段与每层数组）分配。
    - KV-Cache 与必要工作区的初始化（至少基于 `meta->maxseq`）。
- `llaisysQwen2ModelWeights`
  - 返回后端持有的权重句柄表（指针稳定，生命周期从 create 到 destroy）。
  - 前端不负责分配该结构体本身，但需要为各字段创建 `llaisysTensor_t` 并写入数据。
- `llaisysQwen2ModelInfer`
  - 输入为“本次新增”的 `ntoken` 个 token id（支持 prompt 与 decode 两阶段）。
  - 返回值为“下一 token”的采样结果（阶段性目标建议使用 argmax，与 `test/test_infer.py --test` 对齐）。
  - KV-Cache 在模型内部维护，并在 infer 中更新。

## 1.1 内存契约（Ownership / Overflow / Workspace）

- 权重所有权：一旦某个 `llaisysTensor_t` 句柄被写入 `LlaisysQwen2Weights` 槽位，即视为由 `model` 接管所有权。
- 权重释放责任：`llaisysQwen2ModelDestroy` 负责释放所有被接管的权重句柄、KV-Cache 与工作区内存。
- 重复赋值策略：如果某个槽位被重复赋值，后端应先释放旧句柄，再接管新句柄，避免泄漏。
- 权重共享策略：允许 `in_embed == out_embed`（weight tying）；销毁时必须基于指针去重，确保同一句柄只释放一次。
- KV-Cache 溢出策略：在 `infer` 开头强制检查 `cur_len + ntoken <= meta.maxseq`；不满足时 fail-fast（当前原型无错误码）。
- Python 侧限流建议：`generate` 在 prompt 之后应计算剩余容量并裁剪 `max_new_tokens`，避免触发后端 fail-fast。
- Workspace 策略：`ensure_workspace_(ntoken)` 只在每次 `infer` 开头调用一次；内部通过 `kvlen = cur_len + ntoken` 推导尺寸。
- Workspace 扩容策略：采用 grow-only + 复用（不 shrink）；在 `destroy` 统一释放。

## 2. Meta 字段定义（必须落地）

`LlaisysQwen2Meta` 来自 `include/llaisys/models/qwen2.h:7`，建议在实现与 Python 侧保持以下解释一致：

- `dtype`：权重与主干计算的期望数据类型（`llaisysDataType_t`，定义见 `include/llaisys.h:28`）。
- `nlayer`：层数。
- `hs`：hidden size。
- `nh`：attention head 数。
- `nkvh`：KV head 数（GQA/MQA）。
- `dh`：head dim（通常满足 `hs == nh * dh`）。
- `di`：MLP intermediate size。
- `maxseq`：KV-Cache 的最大序列长度。
- `voc`：词表大小。
- `epsilon`：RMSNorm epsilon。
- `theta`：RoPE base（用于 `rope_theta`）。
- `end_token`：终止 token id（用于 generate 停止条件）。

## 3. 权重加载协议（句柄式而非名称映射）

当前原型没有 `load_weight(name, ptr)`，因此“权重命名映射”必须放在前端（Python）完成，后端只校验张量形状/类型并消费句柄。

推荐加载流程（与现有 tensor C API 配套）：

1. 调用 `llaisysQwen2ModelCreate(meta, device, device_ids, ndevice)`。
2. 调用 `llaisysQwen2ModelWeights(model)` 获取 `weights`。
3. 对每个权重：
   - 前端根据 `meta` 推导 shape。
   - 使用 `tensorCreate(...)` 创建 `llaisysTensor_t`。
   - 使用 `tensorLoad(tensor, data_ptr)` 写入 safetensors 数据。
   - 将 tensor 句柄赋值到 `weights` 对应字段（或对应层索引）。

### 3.1 权重字段与期望 shape（建议作为强校验）

下面 shape 约定用于前后端一致性校验。这里明确写死当前 `ops::linear` 的真实约定：`in=[M,K]`、`weight=[N,K]`、`out=[M,N]`，也即 **weight 形状为 `[out_dim, in_dim]`（与 PyTorch linear 一致）**。因此本节所有线性层权重 shape 均以 `[out, in]` 描述，默认不需要转置。

- 全局权重
  - `in_embed`：`[voc, hs]`（`model.embed_tokens.weight`）。
  - `out_embed`：`[voc, hs]`（`lm_head.weight`，允许与 `in_embed` 共享）。
  - `out_norm_w`：`[hs]`（`model.norm.weight`）。
- 每层权重（数组长度均为 `nlayer`）
  - `attn_norm_w[i]`：`[hs]`（`input_layernorm.weight`）。
  - `attn_q_w[i]`：`[nh * dh, hs]`（`attention.wq.weight`）。
  - `attn_q_b[i]`：`[nh * dh]`（`attention.wq.bias`，若存在）。
  - `attn_k_w[i]`：`[nkvh * dh, hs]`（`attention.wk.weight`）。
  - `attn_k_b[i]`：`[nkvh * dh]`（`attention.wk.bias`，若存在）。
  - `attn_v_w[i]`：`[nkvh * dh, hs]`（`attention.wv.weight`）。
  - `attn_v_b[i]`：`[nkvh * dh]`（`attention.wv.bias`，若存在）。
  - `attn_o_w[i]`：`[hs, nh * dh]`（`attention.wo.weight`）。
  - `mlp_norm_w[i]`：`[hs]`（`post_attention_layernorm.weight`）。
  - `mlp_gate_w[i]`：`[di, hs]`（`mlp.gate_proj.weight`）。
  - `mlp_up_w[i]`：`[di, hs]`（`mlp.up_proj.weight`）。
  - `mlp_down_w[i]`：`[hs, di]`（`mlp.down_proj.weight`）。

实现建议（后端侧）：

- 对每个权重句柄做以下检查并在失败时快速失败（`ASSERT`/`CHECK_*`）：
  - 非空。
  - dtype 与 `meta->dtype` 兼容（或可被 ops 安全处理）。
  - 维度与 shape 契约一致。
  - device 与模型 device 一致。

### 3.2 RoPE 的位置口径（与 KV-Cache 对齐）

当前 `ops::rope` 要求：

- 输入形状为 `[seqlen, nhead, dim]`。
- `pos_ids` 为 `int64` 的 1-D 张量，长度等于 `seqlen`。

为与 KV-Cache 语义对齐，建议在每次 `infer(ntoken)` 中明确采用以下口径：

- `qlen = ntoken`。
- `kvlen = cur_len + ntoken`（`cur_len` 为进入本次 infer 前的缓存长度）。
- 本次 RoPE 只作用在“新写入缓存”的 K/Q 上，因此：
  - `pos_ids = [cur_len, cur_len + 1, ..., cur_len + ntoken - 1]`。
  - `pos_ids.shape == [ntoken]`，dtype 为 `I64`，device 与模型一致。

该口径与当前 `self_attention` 的 causal 处理方式兼容（其内部通过 `offset = kvlen - seqlen` 将 Q 对齐到 K 的末尾）。

### 3.3 权重映射完整性（强烈建议写成显式清单）

为避免“少映射一个权重导致结果飘或崩溃”，建议将 Python 侧映射至少覆盖以下模式（按层索引 `i`）：

- 全局：
  - `model.embed_tokens.weight -> in_embed`
  - `lm_head.weight -> out_embed`
  - `model.norm.weight -> out_norm_w`
- 每层：
  - `model.layers.{i}.input_layernorm.weight -> attn_norm_w[i]`
  - `model.layers.{i}.attention.wq.weight -> attn_q_w[i]`
  - `model.layers.{i}.attention.wq.bias -> attn_q_b[i]`（若存在）
  - `model.layers.{i}.attention.wk.weight -> attn_k_w[i]`
  - `model.layers.{i}.attention.wk.bias -> attn_k_b[i]`（若存在）
  - `model.layers.{i}.attention.wv.weight -> attn_v_w[i]`
  - `model.layers.{i}.attention.wv.bias -> attn_v_b[i]`（若存在）
  - `model.layers.{i}.attention.wo.weight -> attn_o_w[i]`
  - `model.layers.{i}.post_attention_layernorm.weight -> mlp_norm_w[i]`
  - `model.layers.{i}.mlp.gate_proj.weight -> mlp_gate_w[i]`
  - `model.layers.{i}.mlp.up_proj.weight -> mlp_up_w[i]`
  - `model.layers.{i}.mlp.down_proj.weight -> mlp_down_w[i]`

## 4. C++ 侧建议接口（与原型一致的内部形态）

命名空间建议：`llaisys::models::qwen2`

为贴合当前 C API，内部接口应围绕“权重句柄表 + infer”组织，而不是 name-based loader：

```cpp
class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device,
               int *device_ids,
               int ndevice);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights() noexcept;
    int64_t infer(int64_t *token_ids, size_t ntoken);

private:
    void validate_weights_or_die_();
    void ensure_workspace_(size_t ntoken);
    void run_block_(size_t layer_idx, tensor_t hidden, size_t new_tokens);
};
```

## 5. Python ctypes 封装（按真实原型绑定）

文件目标：`python/llaisys/libllaisys/qwen2.py`

ctypes 侧重点不再是 `load_weight(name, array)`，而是：

1. 定义 `LlaisysQwen2Meta` 与 `LlaisysQwen2Weights` 的 ctypes 结构体。
2. 绑定以下函数：
   - `llaisysQwen2ModelCreate`
   - `llaisysQwen2ModelDestroy`
   - `llaisysQwen2ModelWeights`
   - `llaisysQwen2ModelInfer`
3. 通过 `tensorCreate/tensorLoad` 构造权重 tensor，并写入 `weights` 结构体字段。

建议绑定草图如下（字段需与头文件完全一致）：

```python
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]
```

## 6. Python 模型 API（与测试脚本对齐）

文件目标：`python/llaisys/models/qwen2.py`

对齐 `test/test_infer.py:60` 的需求，推荐行为契约：

- `__init__`
  - 解析 HF 配置并填充 `LlaisysQwen2Meta`。
  - create 模型并拿到 `weights` 句柄。
  - 遍历 safetensors，把每个权重映射到 `weights` 中对应槽位（映射逻辑在 Python）。
- `generate`
  - 先把 prompt 一次性送入 `llaisysQwen2ModelInfer(model, token_ids, ntoken)`。
  - 然后循环单 token decode：
    - 把上一 token 作为输入调用 infer。
    - 将返回 token 追加到输出序列。
    - 遇到 `end_token` 终止。

## 7. 错误处理（基于当前函数签名的现实约束）

当前原型没有状态码返回值，因此错误处理建议明确为“失败即终止/抛出致命错误”，避免产生“ctypes 可以捕获 C++ 异常”的误导：

- 不允许让 C++ 异常跨越 C API 边界传播（未定义行为）。
- C API 内部应自行 `catch (...)` 并转为：
  - `ASSERT(false)` / `std::abort()`，或
  - 统一的致命错误路径（例如打印错误后终止）。
- Python 侧应把此类错误视为“调用约束被破坏”，重点依赖前置校验（shape/dtype/device/maxseq）。
