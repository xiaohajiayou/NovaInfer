# Qwen2 推理系统需求设计（新版）

本需求文档面向“离线推理 + 在线服务”两种形态，覆盖 Python 层、Infer Core、Batch 机制、KV-Cache 管理、采样与输出策略、以及推理 Server 的基本能力。本文为需求设计，不是实现细节；实现时允许分阶段落地。

## 1. 目标与范围

1. 支持 Qwen2 系列模型在本仓库推理栈上完成离线与在线推理。
2. 提供清晰的 Python 入口与 Infer Core 能力划分。
3. 支持多序列并发推理，具备连续批处理与 KV-Cache 复用能力。
4. 为后续LLM与多模态模型扩展预留接口。
5. 设计需考虑后续分布式推理场景,比如张量并行。

## 2. Python 层需求

### 2.1 模型文件解析

1. 支持解析 HuggingFace 目录结构。
2. 支持 `config.json` 与 tokenizer 文件读取。
3. 支持单文件或分片 `safetensors` 权重加载。

### 2.2 Tokenizer

1. 使用AutoTokenizer,完成模型输入和输出的tokenize与detokenize。

### 2.3 自回归推理流程
1. 支持 prompt prefill 与 decode 循环。
2. 支持 max_new_tokens、stop token、stop string。
3. 支持采样参数透传至采样模块。
4. 支持流式输出接口或回调。

### 2.4 Offline 推理流程

1. 支持脚本式、单次输入的离线推理。
2. 支持批量离线推理，优先吞吐。
3. 支持保存结果与耗时统计。

## 3. Infer Core 需求

### 3.1 内存设计

1. 模型权重加载。
2. 模型计算中间 buffer 设计。
3. KV-Cache 设计与多序列支持。
4. 输出 buffer 管理。

#### 3.1.1 模型权重加载

1. 权重在 Core 内统一管理生命周期。
2. 支持权重共享与去重释放。
3. 支持权重校验，包含 shape、dtype、device 一致性。

#### 3.1.2 模型计算中间 buffer

1. Workspace 采用 grow-only 策略并可复用。
2. 支持按 seq_len 大小扩容。
3. 支持多设备内存分配接口。

#### 3.1.3 KV-Cache

1. KV-Cache 以 slot（cell）为基本单元，物理位置为 cell 索引（idx）。
2. 每个 slot 记录逻辑 `pos` 与 `seq_id`（或 seq_id 集合）；`pos` 是序列内位置，不是物理偏移。
3. 持久映射以 slot -> (seq_id, pos) 为主，不强制维护 seq_id -> slots 显式表；需至少维护每个序列的 pos 区间统计。
4. 允许多序列混排 token，但 attention 必须按 seq_id 严格隔离；不同序列互不可见。
5. 资源不足默认失败返回；除非显式调用 seq_* 接口或启用滑窗策略，不自动回收或截断。
6. 滑窗注意力不改变逻辑 pos，仅通过 mask 屏蔽窗口外 token；只有显式 seq_add/shift 才会调整 pos。

#### 3.1.4 输出 buffer 管理

1. 支持单步 logits 输出 buffer（仅当前步）。
2. 提供当前步采样结果的输出接口。
3. 支持流式输出的增量结果接口，offline 与 online 统一口径。

### 3.2 批处理设计

1. 设计 batch 与 ubatch 机制：ubatch 表示本轮要计算的 token 集合（可来自多个序列）。
2. 支持将一个 batch 划分为多个 ubatch 依次执行。
3. 支持不同长度序列混合的连续批处理（前提是按 `seq_id` 做注意力隔离）。
4. 提供等价于 batch builder 与 decode 的接口能力（不绑定具体函数名）。


### 3.3 模型计算图

1. 计算图以 Qwen2 结构为准，覆盖 embedding → N×block → final norm → lm head。
2. 每个 block 必须包含：attention（Q/K/V 投影 + RoPE + KV-Cache 读写 + attention 输出投影）、MLP（gate/up/down + SwiGLU）、残差连接与 RMSNorm。
3. RoPE 位置与 KV-Cache 对齐：每个新增 token 使用其逻辑 `pos`，历史 token 视为已编码。
4. 计算图需支持 prompt prefill 与 decode 两阶段：
   - prefill：一次处理多 token（同序列），更新 KV-Cache；
   - decode：每轮处理 1 个 token（或 ubatch 内多序列 token），更新 KV-Cache。
5. 支持按照 batch 或 ubatch 运行计算图；ubatch 为本轮要计算的 token 集合。
6. 当启用多序列混排时：attention 必须按 `seq_id` 隔离；采样需对每个 `seq_id` 的“最后 token logits”分别执行。
7. 计算图不要求支持其他模型或多模态组件（后续重构再考虑）。

### 3.4 后端管理与算子实现

1. 后端需提供统一的 Runtime API 管理 device。
2. CPU 与 GPU 算子需统一调用接口（GPU 实现可作为后续阶段目标）。
3. 常用算子包括 embedding、linear、rms_norm、rope、self_attention、swiglu、argmax。
4. 算子需支持常见 dtype 并具备一致性检查。

### 3.5 采样

1. 采样应设计为“可组合的 sampler 链”，支持按顺序组合多种采样/约束算子。
2. 至少支持：Argmax、Top-k、Top-p、Temperature（后续可扩展 min-p/tfs/typical/penalties）。
3. 采样参数应集中管理（类似 `sampling_params`），并允许配置 sampler 顺序。
4. 采样上下文需可复用与可重置，支持在多序列场景下按序列 reset。
5. 在 ubatch/multi-seq 情况下，采样应按 `seq_id` 分别执行，仅对每个序列“最新 token logits”进行采样。
6. 采样输入为当前步 logits；不要求缓存多步 logits。

### 3.6 输出策略

#### 3.6.1 投机解码

1. Ngram 推测。
2. MTP 推测。
3. EAGLE 推测。

投机解码应支持回退与验证机制，并不破坏标准采样流程。

### 3.7 KV-Cache 接口设计

1. `llama_kv_self_seq_cp`：前缀共享，支持系统 prompt 复用。
2. `llama_kv_self_seq_rm`：释放或截断某序列上下文。
3. `llama_kv_self_seq_add`：滑窗平移或位置平移（K-shift/RoPE shift），调整逻辑 pos。
4. `llama_kv_self_seq_keep`：保留某序列，清除其它序列。
5. `llama_kv_self_seq_pos_max`：查询序列已占用最大 pos。

接口需支持多序列并发场景，并保证 slot 与 seq_id 一致；资源不足时默认失败返回。

## 4. Infer Server 需求

### 4.1 单用户会话管理

1. 提供单会话上下文。
2. 支持会话内多轮对话。

### 4.2 多用户管理

1. 支持多会话并发。
2. 支持请求级别隔离与限流。

### 4.3 在线推理

1. HTTP 服务接口。
2. OpenAI 兼容 API。
3. 流式输出。
4. 取消请求。

### 4.4 Web UI

1. 提供可选的调试与演示界面。
2. 支持流式展示输出。

### 4.5 并发请求管理

1. 请求池或队列。
2. 独立的循环线程或进程处理。
3. 支持动态批处理。

### 4.6 指标与监控

1. 吞吐与延迟统计。
2. KV-Cache 使用率。
3. GPU/CPU 资源占用与内存统计。

### 4.7 连续批处理

1. 每轮从池中取出请求组成 batch。
2. 执行一次推理后，将未完成请求放回池中。
3. 支持按 token 级别连续迭代。
4. 支持与 KV-Cache slot 管理对齐。

### 4.8 前缀缓存

1. 维护跨用户的前缀相似度匹配。
2. 当相似度满足阈值时复用 KV。
3. 支持 context shift 后的连续推理。

## 5. 设计约束与阶段落地

1. 阶段 1：优先保证离线推理闭环与 argmax 验证。
2. 阶段 2：加入 sampling 与在线推理能力。
3. 阶段 3：连续批处理、前缀缓存与投机解码。
4. 任何阶段必须保持接口稳定，不影响已有推理流程。

## 6. 验收标准

1. 参考test_infer.py, 增加test_offline.py,验证离线推理结果可复现且与参考实现一致。
2. 增加test_online.py测试,验证能支持多用户并发与流式输出。
3. 增加test_kv_cache.py测试,验证KV-Cache 管理与 slot 映射工作稳定，无泄漏。
4. 增加test_sampling.py测试,验证采样策略可配置且行为一致。
5. 增加连续批处理的测试,验证连续批处理可显著提升吞吐。
