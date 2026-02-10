# Qwen2 测试计划

本文件定义 Qwen2 推理栈在阶段 0/1/2/3 的测试范围、执行方式、通过标准与回归规则。  
目标：把“需求 -> 设计 -> 测试”闭环落到可执行清单，避免阶段验收口径不一致。

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
1. 既有基线：`test/test_infer.py --test`
2. `test/test_core_model_api.py`：`create/decode/get_logits*/kv` 行为。
3. `test/test_core_decode_batch.py`：SoA batch 在单序列/多序列场景下正确。
4. `test/test_core_output_api.py`：`GetLogits/GetLogitsIth/NOutputs/OutputIds` 与 `batch.logits` 对齐。
5. `test/test_kv_cache.py`：`seq_id + pos` 隔离、`kv_seq_*` 返回码与行为。
6. `test/test_qwen2_adapter.py`：`python/llaisys/models/qwen2.py` 基于通用 C API 可用。
7. `test/test_model_registry.py`：多模型注册/路由（至少 `qwen2 + mock`）。
8. `test/test_core_parity.py`：与 `transformers` 做 batch+argmax 逐步对拍（有本地模型时必跑）。

阶段 0 特殊约束：
1. 允许 `ModelRunner.generate()` 临时使用内部 argmax，仅用于 `test_infer` 兼容。
2. 不允许在 Core `decode` 主路径内做采样决策（Core 只产 logits）。

### 2.2 阶段 1（offline + Engine 内 argmax）

必须通过：
1. 阶段 0 全量用例。
2. `test/test_offline.py`：离线生成一致性、流式/非流式行为。

新增约束：
1. 离线主流程必须切到 `LLM -> LLMEngine -> Scheduler -> Executor -> Worker -> Core`。
2. 采样在 Engine 执行（argmax），Core 继续只返回 logits。

### 2.3 阶段 2（Engine 采样链 + online）

必须通过：
1. 阶段 1 全量用例。
2. `test/test_sampling.py`：top-k/top-p/temperature 行为。
3. `test/test_online.py`：在线并发、流式、取消、错误码。

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
3. 离线 `generate()` 在固定种子/参数下可复现。

### 3.3 Engine/Server 用例（阶段 1/2）

1. 请求状态流转：`queued -> running -> finished/cancelled/failed`。
2. 采样策略切换生效（argmax 与 top-k/top-p 温度）。
3. 流式输出分片可拼接，stop 条件一致。
4. 取消请求后 KV 资源释放。

## 4. 执行方式

推荐脚本（阶段0）：
```bash
./test/run_stage0_tests.sh python /path/to/local/model
```

说明：
1. `run_stage0_tests.sh` 默认 `RUN_PARITY=auto`，当传入模型路径时会自动跑 `test_core_parity.py`。
2. 若仅做快速回归可设 `RUN_PARITY=0`。

基础命令（按阶段执行）：
```bash
pytest -q test/test_infer.py
pytest -q test/test_core_model_api.py
pytest -q test/test_core_decode_batch.py
pytest -q test/test_core_output_api.py
pytest -q test/test_kv_cache.py
pytest -q test/test_qwen2_adapter.py
pytest -q test/test_model_registry.py
python test/test_core_parity.py --model /path/to/local/model
```

在线与采样（阶段 2 起）：
```bash
pytest -q test/test_sampling.py
pytest -q test/test_online.py
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

### 6.1 当前状态（阶段0收敛后）

1. 阶段0已通过全量测试与 parity 对拍。
2. 后续回归默认从 `run_stage0_tests.sh` 开始，失败再下钻单测。

## 7. 回归规则

1. 改动 `src/llaisys/model.cc` 或 `include/llaisys/models/model.h`：重跑阶段 0 全量 Core API 用例。
2. 改动 `src/llaisys/qwen2/qwen2_model.*`：重跑 Core API + KV + infer 基线。
3. 改动 `python/llaisys/models/qwen2.py`：重跑 adapter + infer/offline。
4. 改动 `engine/server`：重跑 offline/online/sampling 对应用例。
