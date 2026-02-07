# Qwen2 模型概要设计

## 1. 设计目标

- 复用 LLAISYS 现有张量与算子，完成 DeepSeek-R1-Distill-Qwen-1.5B 的纯 C++ 推理。
- 保持 Python/ctypes 层仅负责配置解析、权重映射与推理驱动，所有算子调度、KV-Cache 与内存生命周期在 C++ 层执行。
- 构建清晰的模块边界，后续可扩展多设备、优化器或其它模型。
- 严格对齐当前 C 头文件原型：通过 `llaisysQwen2ModelWeights` 暴露权重槽位，前端使用 tensor 句柄写入数据。
- 支持 offline 与 online 推理一致的“流式输出接口”语义（增量结果）。
- 为连续批处理与多序列混排（`seq_id` + slot）预留结构位置。

## 2. 总体架构
```
Python Qwen2 模型
    └── ctypes 封装 (python/llaisys/libllaisys/qwen2.py)
          └── C API (include/llaisys/models/qwen2.h)
                └── Qwen2Model C++ 实现 (src/llaisys/qwen2/)
                      ├── 权重槽位 (LlaisysQwen2Weights 句柄表)
                      ├── 计算图 (Embedding → Block → Final Norm → LM Head)
                      ├── KV-Cache (slot/seq_id/pos)
                      ├── Workspace/中间 buffer
                      └── Sampler/流式输出接口
    └── Server 层（可选）
          ├── 请求池/连续批处理
          ├── 会话管理
          └── HTTP/Web UI/流式输出
```

- **Python 层**：解析 HuggingFace 配置与 safetensors，将 HF 权重名映射到 `LlaisysQwen2Weights` 各槽位；提供 `generate` 与采样逻辑。
- **ctypes 层**：声明 C API 与 tensor API，负责 Python buffer、`llaisysTensor_t` 与后端权重槽位互通。
- **C API**：屏蔽 C++ 具体实现，提供创建、销毁、获取权重槽位与推理函数（不提供 name-based loader）。
- **C++ 模型**：搭建完整推理管线，复用 ops 中的 embedding、rope、self_attention、linear、rms_norm、swiglu、argmax；提供 sampler 链与流式输出接口。

## 2.1 代码目录与注释规范

```
llaisys/
├─ include/
│  └─ llaisys/
│     └─ models/
│        └─ qwen2.h          # C API、配置结构、句柄定义
├─ src/
│  └─ llaisys/
│     └─ qwen2/
│        ├─ qwen2_api.cc     # C API 与 Qwen2Model 绑定
│        ├─ qwen2_model.hpp  # Qwen2Model / Weights / KvCache / Workspace 声明
│        ├─ qwen2_model.cpp  # 构造、infer、KV-Cache 逻辑
│        ├─ qwen2_block.cpp  # Transformer Block（Attention + MLP）实现
│        └─ qwen2_validate.cpp # 权重 shape/dtype/device 校验（建议）
├─ python/
│  └─ llaisys/
│     ├─ libllaisys/
│     │  └─ qwen2.py         # ctypes 声明与 Qwen2Handle
│     └─ models/
│        └─ qwen2.py         # Python 模型包装、HF→weights 映射、generate
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

## 3. 关键模块（按需求分解的详细设计）

### 3.1 Python 层

- 模型文件解析：读取 `config.json`、支持单文件与分片 `safetensors`，允许 dtype 强制覆盖与降级。
- Tokenizer：支持 encode/decode、chat template、stop token/stop words。
- 自回归推理：支持 prefill + decode 循环、停止条件与采样参数透传。
- Offline 推理：支持批量离线与流式增量输出接口（与 online 统一口径）。

### 3.2 Infer Core：内存设计

- 权重加载：权重句柄写入即转移所有权，支持共享与去重释放，首次 infer 前强校验 shape/dtype/device。
- 中间 buffer：Workspace 采用 grow-only，可按本轮 seq_len 扩容并复用。
- KV-Cache：slot（cell）为基本单元，slot 记录 `seq_id`/逻辑 `pos`；slot->(seq_id,pos) 为持久映射。
- 输出 buffer：提供单步 logits 与采样结果输出；提供流式增量结果接口。

### 3.3 Infer Core：批处理设计

- ubatch 表示本轮要计算的 token 集合（可来自多个序列）。
- 支持 batch 拆分为多个 ubatch 依次执行。
- 不同长度序列可混排进入同一 ubatch，前提是 attention 按 `seq_id` 隔离。
- 提供等价于 batch builder 与 decode 的接口能力（不绑定具体函数名）。

### 3.4 Infer Core：模型计算图

- 计算图覆盖 embedding → N×block → final norm → lm head。
- Block 包含 attention（Q/K/V 投影 + RoPE + KV-Cache 读写 + 输出投影）、MLP（gate/up/down + SwiGLU）、残差与 RMSNorm。
- RoPE 使用逻辑 `pos`；历史 token 视为已编码。
- 支持 prefill 与 decode 两阶段；decode 允许 ubatch 多序列 token。
- 多序列混排时：attention 按 `seq_id` 隔离；采样按 `seq_id` 取最新 token logits。

### 3.5 后端管理与算子实现

- 统一 Runtime API 管理 device，CPU/GPU 走统一调用入口（GPU 为后续目标）。
- 常用算子：embedding、linear、rms_norm、rope、self_attention、swiglu、argmax。
- 算子支持常见 dtype 并进行一致性检查。

### 3.6 采样

- 采样采用可组合的 sampler 链，支持 Argmax/Top-k/Top-p/Temperature。
- 采样参数集中管理并支持配置顺序。
- 采样上下文可复用/可重置，多序列按 `seq_id` reset。
- ubatch/multi-seq 下按 `seq_id` 采样，仅对每序列最新 token logits 采样。

### 3.7 输出策略

- 支持流式增量输出接口（offline 与 online 统一语义）。
- 投机解码接口预留：Ngram/MTP/EAGLE（后续阶段补充）。

### 3.8 KV-Cache 接口设计

- 支持 `seq_cp/seq_rm/seq_add/seq_keep/seq_pos_max` 等能力。
- 默认资源不足失败返回；显式调用 seq_* 或滑窗策略时回收/复用。
- 滑窗注意力不修改逻辑 `pos`，仅通过 mask 屏蔽窗口外 token。

### 3.9 Infer Server

- 会话管理：单用户与多用户并发，支持请求隔离与限流。
- 在线推理：HTTP/OpenAI 兼容 API、流式输出、请求取消。
- 并发调度：请求池/队列 + 连续批处理循环。
- 前缀缓存：相似度匹配与 KV 复用，支持 context shift。

## 4. 主要数据流

1. Python 解析配置并构造 `LlaisysQwen2Meta`。
2. 调用 `llaisysQwen2ModelCreate(meta, device, device_ids, ndevice)`，后端分配权重槽位与 KV-Cache。
3. 调用 `llaisysQwen2ModelWeights(model)` 获取 `weights` 句柄表。
4. Python 遍历 safetensors：
   - 将 HF 权重名映射到 `weights` 的具体字段/层索引。
   - 使用 `tensorCreate + tensorLoad` 创建并填充 `llaisysTensor_t`。
   - 把 tensor 句柄写入 `weights` 对应槽位。
5. `generate`：
   - For prompt: 一次性调用 `infer` 进行 prefill（同序列多 token）。
   - For decode: 循环调用 `infer` 获取当前步 logits → 采样 → 生成下一 token。
   - 多序列/连续批处理：构建 ubatch（含 `seq_id/pos`），在一次 infer 中处理多个序列 token。
   - Python 端负责维护输入 token 序列、停止条件与流式输出。

## 5. 未来扩展

- **设备适配**：接口已包含 `device/device_ids/ndevice`，后续可实现 CUDA 版本 ops，只需在 `self_attention`/`linear` 等算子派发即可。
- **内存池**：初版使用智能指针管理 tensor，后续可扩展 `WorkspaceAllocator`，减少中间张量重复分配。
- **多批次/多请求**：通过 ubatch + slot 化 KV-Cache 实现连续批处理与多用户服务。

以上设计遵循现有仓库命名与代码组织，确保模块解耦、易于调试并满足作业 #3 的所有需求。
