# Qwen2 测试计划

本文件定义 Qwen2 推理栈在阶段 0/1/2/3 的测试范围、执行方式、通过标准与回归规则。  
目标：把“需求 -> 设计 -> 测试”闭环落到可执行清单，避免阶段验收口径不一致。

> 文档状态更新（2026-02-18）  
> 当前 KV 为 `SLOT/BLOCK` 双布局，且 BLOCK 为主线演进路径。  
> 若与旧用例口径冲突，以 0.1 节“当前测试策略（2026-02）”为准。详细开发计划见 `doc/qwen2_next_dev_plan_2026-02.md`。
> 计划口径补充：原 M2/M3 已合并为“调度与 KV 语义一次改造阶段”。
> 测试目录口径更新（2026-02-20）：`test/core|engine|offline|online|parity|ops|utils`。

## 0.2 阅读优先级（当前 vs 历史）

1. 本文主体（1~9 节）为当前执行规则。
2. 历史复盘内容已归档到 `doc/archive/qwen2_test_plan_stage2_postmortem_legacy_2026-02.md`。
3. 若历史复盘与当前执行规则冲突，以本文当前规则为准。

## 0.1 当前测试策略（2026-02）

1. 基础正确性：默认在 BLOCK 布局验证主链路。
2. 回归与对照：关键 Core/Engine 用例保留 SLOT 对照运行。
3. `kv_seq_*`：作为兼容接口测试，不再代表 BLOCK 主语义目标。
4. 性能对比：统一只走 `decode` 主路径，不把 `kv_seq_*` 纳入 benchmark 口径。

## 1. 测试范围与分层

1. Core（C++）：`llaisysModel*` 通用接口、SoA batch/decode、logits 输出、KV 接口。
2. ModelRunner（Python）：Qwen2 适配层权重映射、通用 C API 调用、离线兼容路径。
3. Engine（Python）：调度/执行/采样（阶段 1 开始）。
4. Server/UI（Python/Web）：在线 API、流式、取消、Web UI（阶段 2 开始）。

对齐参考：

1. Core 正确性以 `transformers` 对拍与阶段0回归为准。
2. 离线编排语义参考 vLLM/`nano-vllm`（`LLM.generate -> engine step`）。
3. online 行为以 OpenAI API 语义与 vLLM API Server 分层为准（阶段2开始）。

## 2. 阶段测试矩阵

### 2.1 阶段 0（Core 重构 + 通用多模型 API）

必须通过：
1. 既有基线：`pytest -q test/parity/test_infer.py --model-path /path/to/local/model`
2. `test/core/test_core_model_api.py`：`create/decode/get_logits*/kv` 行为。
3. `test/core/test_core_decode_batch.py`：SoA batch 在单序列/多序列场景下正确（含 `n_seq_id > 1` 共享 token 用例）。
4. `test/core/test_core_output_api.py`：`GetLogits/GetLogitsIth/NOutputs/OutputIds` 与 `batch.logits` 对齐。
5. `test/core/test_kv_cache.py`：`seq_id + pos` 隔离、`kv_seq_*` 返回码与行为（按 SLOT/BLOCK 分支断言）。
6. `test/core/test_qwen2_adapter.py`：`python/llaisys/models/qwen2.py` 基于通用 C API 可用。
7. `test/core/test_model_registry.py`：多模型注册/路由（至少 `qwen2 + mock`）。
8. `test/parity/test_core_parity.py`：与 `transformers` 做 batch+argmax 逐步对拍（有本地模型时必跑）。

阶段 0 特殊约束：
1. `test/parity/test_infer.py` 定义为 ModelRunner 级对拍：走 `decode_batch + argmax`，不经过 Engine。
2. 不允许在 Core `decode` 主路径内做采样决策（Core 只产 logits）。

### 2.2 阶段 1（offline + Engine 内 argmax）

必须通过：
1. 阶段 0 全量用例。
2. `test/offline/test_offline.py`：离线生成一致性、流式/非流式行为。
3. `test/offline/test_llm_entrypoint.py`：`LLM.generate/stream` 入口契约（token兼容、prompt/prompts、batch params）。
4. `test/engine/test_engine_state_machine.py`：状态流转（`waiting -> running -> finished_*`）与异常路径。
5. `test/engine/test_engine_model_registry.py`：新增模型注册不修改 Engine 主流程。
6. `test/parity/test_offline_parity.py`：Engine 离线链路与 `transformers` 对拍（single + multi-seq，需本地模型）。

新增约束：
1. 离线主流程必须切到 `LLM -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
2. 采样在 Engine 执行（argmax），Core 继续只返回 logits。
3. `waiting_for_remote_kvs/preempted` 若未激活，需在文档标注为“预留状态”。
4. 即便做同进程瘦身实现，也必须保留 `submit/step/cancel`、状态机、`Scheduler -> Worker` 与统一输出结构的测试可见性。

### 2.3 阶段 2（Engine 采样链 + online）

必须通过：
1. 阶段 1 全量用例。
2. `test/engine/test_sampling.py`：top-k/top-p/temperature 行为。
3. `test/online/test_online.py`：在线并发、流式、取消、错误码。
4. `test/online/test_online_stream_isolation.py`：并发流式下 request_id 隔离（DummyRunner）。
5. `test/online/test_online_real_model_multisession.py`：真实模型并发流式不串线（需本地模型）。

说明：
1. 未提供 `--model-path` 时，`requires_model` 用例会自动 skip，不影响无模型环境的快速回归。

### 2.4 阶段 3（连续批处理 + 前缀缓存 + 投机）

必须通过：
1. 阶段 2 全量用例。
2. 连续批处理吞吐 benchmark（固定输入集，记录 tokens/s 与延迟）。
3. 前缀缓存命中率与正确性测试。
4. 投机解码回退正确性测试（不改变最终输出语义）。

## 3. 用例设计要点

### 3.1 Core API 用例（阶段 0）

1. `ModelCreate` 参数错误返回空句柄。
2. `Decode` 输入非法返回 `-1`，资源不足返回 `1`。
3. `batch.logits` 掩码仅输出选中行，`output_ids` 与 logits 行索引一致。
4. 多序列 decode 时，不同 `seq_id` 互不可见（上下文隔离）。
5. `kv_seq_rm/cp/add/keep/pos_max` 返回码与区间语义一致。
6. 与 `transformers` 对拍：固定模型与参数下，多序列逐步 next token 一致。

### 3.2 ModelRunner 用例（阶段 0）

1. 权重映射完整性（必须权重缺失时报错）。
2. 通用接口调用链可用：`llaisysModelCreate -> llaisysModelDecode -> GetLogitsIth`。
3. `decode_batch + argmax` 在固定参数下与 `transformers` next-token 对拍一致。

### 3.3 Engine/Server 用例（阶段 1/2）

1. 请求状态流转：`waiting -> running -> finished_*`（覆盖 `finished_stopped / finished_length_capped / finished_aborted / finished_ignored`），并预留 `waiting_for_remote_kvs / preempted`。
2. 采样策略切换生效（argmax 与 top-k/top-p 温度）。
3. 流式输出分片可拼接，stop 条件一致。
4. 取消请求后 KV 资源释放。

## 4. 执行方式

统一入口（推荐）：
```bash
python scripts/run_tests.py --suite all
```

本地原生库前置要求（重要）：

```bash
xmake -y
xmake install
```

说明：
1. Python 默认加载 `python/llaisys/libllaisys/libllaisys.so`（或 `.dylib`）。
2. 仅执行 `xmake` 可能只更新 `build/...` 下产物，未同步到 Python 实际加载路径。

底层测试执行器：`pytest`。`scripts/run_tests.py` 仅负责按阶段和策略拼装 pytest 命令。

统一入口参数说明：
1. `--suite {stage0,stage1,stage2,all}`：选择执行阶段。
2. `--model-path /path/to/local/model`：提供本地模型路径（用于 parity/HF 相关测试）。
3. `--run-parity {auto,always,never}`：parity 策略。
4. `--run-hf {auto,always,never}`：阶段0中 HF 依赖测试（`test/parity/test_infer.py`）策略。

推荐命令：
```bash
# 快速回归（不跑 parity，不跑 HF 依赖）
python scripts/run_tests.py --suite all --run-parity never --run-hf never

# 阶段0（带本地模型，自动跑 parity/HF）
python scripts/run_tests.py --suite stage0 --model-path /path/to/local/model --run-parity auto --run-hf auto

# 阶段1（带本地模型，自动跑 offline parity）
python scripts/run_tests.py --suite stage1 --model-path /path/to/local/model --run-parity auto

# 阶段2（sampling + online）
python scripts/run_tests.py --suite stage2
```

测试编排统一入口：
1. `scripts/run_tests.py`（唯一入口）。

在线与采样（阶段 2 起）：
```bash
pytest -q test/engine/test_sampling.py
pytest -q test/online/test_online.py
pytest -q test/online/test_online_stream_isolation.py
pytest -q test/online/test_online_real_model_multisession.py --model-path /path/to/local/model
```

直接全量 pytest（可选）：
```bash
PYTHONPATH=python python -m pytest -q
```

## 5. 环境与复现要求

1. 固定模型版本与 tokenizer 版本（建议锁定模型路径/commit）。
2. 固定随机种子（Python/NumPy/PyTorch）。
3. 明确设备类型（CPU 或 NVIDIA）与运行参数（maxseq、dtype）。
4. 对耗时相关用例，记录机器配置并提供阈值。

## 6. 通过标准

1. 阶段门槛：该阶段“必须通过”测试全部通过。
2. 无 blocker 级缺陷（崩溃、错误返回码、跨序列污染、输出错行）。
3. 阶段新增能力不破坏前一阶段基线测试。

### 6.1 当前状态（阶段2收敛后）

1. 阶段0已通过全量测试与 parity 对拍。
2. 阶段1已通过离线全量测试；有模型路径时通过 `test/parity/test_offline_parity.py` 对拍。
3. 阶段2核心能力已落地：采样链、在线接口、SSE、取消、多会话并发隔离回归。
4. 后续回归默认从 `scripts/run_tests.py` 开始，失败再下钻单测。
5. 下一重点为阶段3吞吐优化（连续批处理、前缀缓存、投机）。
6. Core 架构修正已完成：`decode` 真 batch 执行 + unified KV + 单轮 mask 隔离；对应核心回归已通过。

## 7. 阶段2历史复盘（已归档）

阶段2实施问题与解决过程已归档，不再作为当前执行规则正文维护。  
参见：`doc/archive/qwen2_test_plan_stage2_postmortem_legacy_2026-02.md`

## 8. 回归规则

1. 改动 `src/llaisys/model.cc` 或 `include/llaisys/models/model.h`：重跑阶段 0 全量 Core API 用例。
2. 改动 `src/llaisys/qwen2/qwen2_model.*`：重跑 Core API + KV + infer 基线。
3. 改动 `python/llaisys/models/qwen2.py`：重跑 adapter + infer/offline。
4. 改动 `engine/server`：重跑 offline/online/sampling 对应用例。
5. 改动 `scripts/run_tests.py`：至少重跑 `--suite stage0 --run-parity never --run-hf never` 与 `--suite stage1 --run-parity never`。

## 9. 性能评测口径（slot vs block）

1. 统一入口：仅用 `decode` 主路径，不调用 `kv_seq_*`。
2. 统一 workload：固定请求到达序列、prompt 长度分布、采样参数、max_new_tokens。
3. 统一环境：同硬件、同模型、同 batch 策略、同编译参数。
4. 指标：
   - 吞吐：tokens/s
   - 时延：TTFT、TPOT、P50/P95
   - 内存：KV 占用、峰值 RSS、free block 变化
5. 结果解释：
   - 端到端对比可接受算子不同（方案级对比）；
   - 需额外补“归因实验”（只换 KV 形态或只换算子）。
