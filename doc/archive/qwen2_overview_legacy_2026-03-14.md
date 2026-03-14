# Qwen2 概览（当前口径）

## 1. 当前状态（2026-02-21）

1. 主线架构：`BLOCK(PagedKvImpl)` 为性能演进主线；`SLOT(UnifiedKvImpl)` 作为兼容与回归对照。
2. 执行链路：Python `engine/scheduler/executor/worker` 负责调度，C++ `Qwen2Model` 负责 runner 执行。
3. 前缀缓存：主能力在 Python `BlockManager`/Scheduler 路径；C++ 保持运行时接口与统计。
4. CPU 策略：CPU attention 路径冻结为“功能稳定 + 回归可用”，后续性能主线转向 CUDA。
5. 测试组织：测试目录已重构为 `core/engine/offline/online/parity/ops/utils` 分层。
6. CI 状态：GitHub Actions 已同步新测试路径，`push/pull_request` 自动运行。
7. CUDA 现状：`BLOCK + NVIDIA` 主路径已可跑通离线推理与 benchmark；`SLOT + NVIDIA` 仍不支持。
8. 基准对比：已新增 `scripts/bench_compare_vllm.py` 用于 NovaInfer/vLLM 同口径对比。
9. 已修复关键问题：
   - KV auto capacity 估算已接入 `max_num_seqs`（不再固定默认 `8`）。
   - `bench_compare_vllm.py --backend both` 改为子进程隔离运行，避免同进程 CUDA 初始化冲突。
   - vLLM tokenized 输入已改为 `prompt_token_ids` 结构，修复批量 token prompt 报错。

## 2. 代码主路径

1. C++：
   - `src/llaisys/qwen2/qwen2_model.cpp`
   - `src/llaisys/runtime/kv_cache/unified_kv.*`
   - `src/llaisys/runtime/kv_cache/paged_kv.*`
2. Python：
   - `python/llaisys/models/qwen2.py`
   - `python/llaisys/engine/*`
   - `python/llaisys/server/*`
3. 测试：
   - `test/core/*`
   - `test/engine/*`
   - `test/offline/*`
   - `test/online/*`
   - `test/parity/*`
   - `test/utils/batch_builders.py`

## 3. 文档索引

1. 需求：`doc/qwen2_requirements.md`
2. 接口：`doc/qwen2_interface.md`
3. 测试：`doc/qwen2_test_plan.md`
4. 计划：`doc/qwen2_next_dev_plan_2026-02.md`
5. 历史归档：`doc/archive/qwen2_overview_legacy_2026-02-18.md`


## 3. Core 层需求（C++）

### 3.1 外部接口支持



### 3.2 内存与资源管理
1. 模型计算中间buffer、输出buffer、kv-cache buffer均在初始化时申请大buffer，后续在运行时根据需要从大buffer中切分。


### 3.3 KV-Cache 行为要求
1. 支持Slot和Block布局
2. Slot布局需要cell和slot概念（在UnifiedKvImpl中管理）
   - cell：slot的容器，每个cell可以存放多个slot
   - slot：kv-cache的一个slot，每个slot存放一个key-value对
3. Block布局需要block和page概念（在BlockManager中管理）
   - block：page的容器，每个block可以存放多个page
   - page：kv-cache的一个page，每个page存放多个slot


### 3.4 计算图与算子

1. 目前仅实现 Qwen2 模型的计算图

## 4. Engine 层需求（Python，严格对齐 vLLM）

### 4.1 LLM / AsyncLLM（入口层）



### 4.2 EngineClient（客户端层）

1. 负责入口层与 EngineCore（`LLMEngine`）之间的调用封装。
2. 支持同进程直连与可切换 IPC/RPC 形态（目前为多线程，后续多进程部署。可能存在性能阻塞）。
3. 负责请求提交、结果拉取、流式回传、取消信号透传。

### 4.3 LLMEngine / EngineCore（核心编排层）

已实现/可见：

状态枚举与终态集合存在：types.py
状态机转移表存在：llm_engine.py（_ALLOWED_TRANSITIONS）
FINISHED_* 在流程里有实际使用（stop/length/abort/ignored）
未落地/未使用：

WAITING_FOR_REMOTE_KVS 只出现在转移表里，没有实际进入该状态的逻辑
PREEMPTED 同样未在请求状态里使用（scheduler 只 preempt Sequence，不更新 RequestStatus）
结论：状态机框架在，但 waiting_for_remote_kvs / preempted 这些还没接起来。

### 4.4 Scheduler

1. Slot布局为公平调度
2. Block布局目前prefill和decode分开调度
3. 现在没有实现需求里说的“请求级上下文配额与增量 token 配额”，当前只做了全局 token budget 和单请求最大生成长度

### 4.5 Executor（执行协调层）



### 4.6 Worker（执行单元）


### 4.7 Sampler + OutputProcessor（执行侧采样 + 引擎侧结果组织）

Sampler 在执行侧：executor.py 里 Sampler.sample(...) / sample_per_row(...)
OutputProcessor 在 EngineCore 侧：llm_engine.py 里 _complete_request() 调用 OutputProcessor.finalize(...)
输出对象统一：output_processor.py（GenerationOutput 组织）
执行侧/结果组织侧边界：执行侧只回 logits/采样结果，Engine 负责拼装 output（符合描述）
- 目前采样是在cpu侧,存在性能损耗.
### 4.8 模型适配层（models）



### 4.9 模型选择与加载



## 5. Server 层需求（Python，对齐 vLLM API Server）

### 5.1 API Server

OpenAI 兼容 API 路由：只看到 /v1/chat/completions，没有 /v1/completions 或 /v1/embeddings。实现见 http_server.py。
SSE 流式输出：已支持，chat/completions + stream 分支写 text/event-stream，并发送 [DONE]。见 http_server.py、openai_server.py。
请求取消：已支持 /v1/requests/{id}/cancel，并调用 cancel() 透传到 async engine。见 http_server.py、async_engine.py。
会话管理（多会话/上下文复用）：没有看到会话状态或会话存储逻辑；目前是请求级一次性处理。
隔离与限流（并发/排队/背压）：没有看到 rate limit、队列上限或背压策略配置。
请求日志与错误码：有基础错误响应（bad_request/internal_error），但没有系统化的日志/错误码体系；仅在 verbose 模式下打印。见 http_server.py。

### 5.2 Server 与 Engine 对接边界

1. 调用链固定为：`API Server -> AsyncLLM -> EngineClient -> LLMEngine(EngineCore)`。
2. Server 仅负责协议适配、鉴权/限流、流式转发；不负责调度、模型前向或采样。
3. Engine 返回统一输出对象（token/text/finish_reason/usage），Server 负责序列化为 OpenAI 响应格式。
4. 监控口径分层：Server 暴露 API 级指标；Engine 暴露调度与执行级指标（吞吐/延迟/KV/资源占用）。
5. 目前为多线程模型,存在并发瓶颈：
  - encode_chat_messages（apply_chat_template + tokenizer.encode）
    在 Python 线程执行
  - decode_tokens（给 stream 产出 text_delta）在 Python 线程执行
  - AsyncLLMEngine 目前是单 loop 线程驱动 step + chunk 分发

  所以当并发上来时，确实会出现你现象：
  - GPU kernel 不是满载
  - CPU 在做 tokenize/detokenize、拼装 chunk、队列调度
  - 请求进场和 token 下发被串住

  但也有一条现实：

  - 真正 decode 计算在 C/CUDA，通常会释放 GIL，不是“完全假线程”
  - 低并发下这套还够用，问题多在高并发/小请求场景暴露

  可以用这两个指标快速判定是不是这个瓶颈：

  1. GPU utilization 低（比如 20%~40%），但请求延迟高
  2. Python 进程单核或少数核很高，engine loop 忙，chunk/tokenize 路
     径耗时明显

  如果要解，优先做：tokenize 前移/并行化、stream detokenize 降频或
  批量化、控制面与推理面进一步解耦。
## 6. Web UI（模块需求）

### 6.1 UI 客户端

1. 提供交互式聊天界面（Web UI）。
2. 支持发送请求与展示回复（含流式输出）。
3. 支持连续对话与本地历史记录（单用户场景即可）。

### 6.2 UI 服务端适配

1. 对接 Server 层 OpenAI `chat-completions` API。
2. 支持 SSE 流式响应解析与渲染。
3. 支持请求取消与错误提示。

### 6.3 多会话与 KV 复用

1. 支持多会话创建/切换。
2. 支持修改历史问题并重新生成回答。（未实现）
3. 支持前缀匹配的 KV-Cache 复用（与 Engine 能力对齐）。
