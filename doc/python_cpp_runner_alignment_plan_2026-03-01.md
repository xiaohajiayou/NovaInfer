# Python -> C++ 全链路重构对齐方案（vLLM Runner 解耦口径，2026-03-01）

## 1. 目标与原则

目标：把当前推理链路对齐到 vLLM 风格的职责边界。

1. Runner（Python Worker）负责编排与采样。
2. Model Adapter 只负责 forward，不承担采样主流程。
3. C++ `decode()` 只做分发；输入准备与 attention metadata 构建显式化。
4. Python -> C++ 仅传结构化 step 输入，减少散乱参数、减少重复拷贝。

设计原则：

1. 先做“行为等价重构”，不先改算法。
2. 先稳定接口，再做性能专项。
3. 以 BLOCK 路径为主线，SLOT 路径仅保留必要功能覆盖。
4. 每阶段都有可回滚开关和验收基线。

## 2. 当前链路与核心问题

当前主链路（简化）：

1. `Scheduler` 产出 `SchedulerOutputs`。
2. `Executor._flatten()` 生成 `BatchPlan`。
3. `Worker.execute(plan)` 调 `model_runner.decode_batch(...)`。
4. `Qwen2.decode_batch`（Python adapter）构造 `LlaisysBatch` 后调 C++ `llaisysModelDecode`。
5. C++ `Qwen2Model::decode` 按 SLOT/BLOCK 分支，执行 layer forward 并返回 sampled ids。

当前问题：

1. Executor 里做了较多输入整形，Runner 编排职责不够集中。
2. Python adapter 既做 forward 输入组装，又承担采样参数透传语义；职责容易混杂。
3. C++ decode 内部参数列表较长（`run_layers_and_collect_`），上下游语义难收敛。
4. attention metadata 的构建与上传逻辑分散在路径代码中，不利于统一 profiling/优化。
5. Python -> C++ 之间存在“metadata 先构造为 host 容器，再按路径上传”的重复工作点，缺乏统一契约。

## 3. 目标职责划分（对齐口径）

### 3.1 Python 层

`Executor`（薄层）

1. 只负责调度步调用和结果回传映射。
2. 不持有复杂 flatten 细节。

`Worker`（Runner 主体）

1. `prepare_step_inputs(outputs, sampling_params...)`。
2. `build_attention_metadata(model_input)`。
3. `run_model_forward(model_input, attn_meta)`。
4. `sample(logits, sampling_metadata)`。
5. `postprocess(...)`。

`ModelAdapter`（模型实现）

1. 只保留 `forward(model_input, attn_meta)`。
2. 可选 `adjust_logits(...)` 钩子（默认 no-op）。
3. 不再承载主采样流程。

### 3.2 C++ 层（严格对齐 vLLM 语义）

新增独立 C++ Runner 层，模型类不再承担 decode 编排：

1. `CppModelRunner`（或 `RuntimeRunner`）负责 step 编排。
2. `InputPreparer` 负责把 Python 传入的 step 输入规整为 C++ 内部输入对象。
3. `AttentionMetadataBuilder` 负责 slot/paged metadata 构建与设备上传。
4. `Sampler` 负责基于 logits 的采样（可先 CPU，再演进 GPU）。
5. `Qwen2Model` 只实现 `forward(...)` 计算图，不处理 slot/block 路径分叉与采样。

关键约束：

1. `slot/block` 差异属于 metadata builder，不属于 `Qwen2Model`。
2. `Qwen2Model` 不直接接收 `LlaisysBatch`，step 输入统一进入 C++ runner。
3. `decode` 概念归 runner；模型仅 `forward`。

## 4. 新接口与数据契约

### 4.1 Python 统一对象

新增（或由现有 `BatchPlan` 演进）：

1. `ModelStepInput`
2. `AttentionMetadata`
3. `SamplingMetadata`
4. `ForwardOutput`

建议字段（最小闭环）：

`ModelStepInput`

1. `token_ids`
2. `pos_ids`
3. `seq_ids`
4. `logits_mask`
5. `slot_mapping`（BLOCK）
6. `context_lens`（BLOCK）
7. `batch_seq_ids`（BLOCK）
8. `block_tables` + `block_table_width`（BLOCK）

`AttentionMetadata`

1. `mode`: `slot` / `paged`
2. `q_seq_rows`（paged）
3. `q_pos`（paged）
4. `seq_lens`（paged）
5. `block_tables`（paged）
6. `attn_mask`（slot）

`SamplingMetadata`

1. `temperatures`
2. `top_ps`
3. `top_ks`
4. `seeds`
5. `has_seeds`

`ForwardOutput`

1. `output_ids`
2. `sampled_ids`
3. 可选 `logits` / `hidden_states`

### 4.2 C++ 统一对象

引入四类对象，职责与 vLLM 对齐：

1. `RunnerStepInput`：runner 入参（来自 Python/ctypes）。
2. `ModelInput`：送给模型 forward 的输入（tokens/positions/hidden 等）。
3. `AttentionMetadata`：注意力访问信息（slot 或 paged）。
4. `ForwardOutput`：模型前向输出（logits/hidden/aux）。

`RunnerStepInput` 建议字段：

1. `token_ids`
2. `pos_ids`
3. `seq_ids`
4. `logits_mask`
5. `sampling_metadata`（温度/topk/topp/seed）
6. `slot_mapping/context_lens/batch_seq_ids/block_tables/block_table_width`（BLOCK）

`AttentionMetadata`（C++）建议字段：

1. `mode`
2. `attn_mask`（slot）
3. `q_seq_rows/q_pos/seq_lens/block_tables/block_table_width`（paged）
4. device buffers / upload cache handle（层间复用）

`ModelInput`（C++）建议字段：

1. `ntoken`
2. `input_ids`
3. `pos_ids`
4. `hidden`（可选，若 embedding 前置于 runner）

`ForwardOutput`（C++）建议字段：

1. `logits`（或 logits handle）
2. `output_rows`
3. `sampled_ids`（由 sampler 产生后回填）

## 5. Python -> C++ 调用链调整

目标调用链：

1. `Executor.execute_scheduler_step(outputs, ...)`
2. `Worker.execute_step(outputs, ...)`
3. `prepare_step_inputs(...) -> ModelStepInput`
4. `build_attention_metadata(...) -> AttentionMetadata`
5. `adapter.forward(model_step_input, attention_metadata)`（模型无采样）
6. `worker.sample(...)`（统一采样）
7. `postprocess(...)`

切换策略（一次性切换）：

1. 不保留旧 `Worker.execute(plan)` 兼容入口。
2. `Qwen2.decode_batch(...)` 直接切换为新 C++ runner step API，移除旧 decode 直连语义。
3. 不引入 feature flag；通过“完整测试门禁 + 性能门禁”一次性上线。

## 6. C++ 侧重构拆分（严格 Runner/Model 解耦）

### 6.1 新组件与职责

1. `src/llaisys/runtime/runner/model_runner.hpp/.cpp`
职责：`execute_step(RunnerStepInput) -> StepOutput`，编排 prepare/meta/forward/sample/postprocess。

2. `src/llaisys/runtime/runner/attention_metadata_builder.hpp/.cpp`
职责：根据布局构建 `AttentionMetadata`，并管理 device 侧 metadata 缓冲复用。

3. `src/llaisys/runtime/runner/sampler.hpp/.cpp`
职责：统一采样策略实现与随机性控制。

4. `src/llaisys/model.hpp`（或等价接口）
职责：定义模型统一接口 `forward(const ModelInput&, const AttentionMetadata&, ForwardOutput*)`。

5. `src/llaisys/qwen2/qwen2_model.hpp/.cpp`
职责：仅实现 Qwen2 forward 图；不承担 step 编排、slot/block metadata 构建、采样。

### 6.2 关键流程伪代码

```cpp
StepOutput CppModelRunner::execute_step(const RunnerStepInput& in) {
  ModelInput model_in = input_preparer_.prepare(in);
  AttentionMetadata attn_meta = attn_builder_.build(in, model_in);
  ForwardOutput fwd_out;
  model_->forward(model_in, attn_meta, &fwd_out);
  std::vector<int32_t> sampled = sampler_.sample(fwd_out.logits, in.sampling_metadata, in.logits_mask);
  return postprocess_(in, sampled);
}
```

```cpp
void Qwen2Model::forward(const ModelInput& in,
                         const AttentionMetadata& attn_meta,
                         ForwardOutput* out) {
  // embedding/rope/layers/lm_head
  // if (attn_meta.mode == paged) -> paged attention
  // else -> masked attention
  // write logits to out
}
```

### 6.3 过渡策略（无兼容层）

1. 直接删除 `Qwen2Model::decode(const LlaisysBatch&)` 编排职责。
2. C-API 入口直接绑定到 C++ runner `execute_step(...)`。
3. 旧调用链不保留 alias，不保留双路径并行。

## 7. 内存与拷贝整改要点

针对你提到的 metadata 拷贝链路，统一口径如下：

1. Python 侧只构建一次 step 输入，不做重复中间 flatten。
2. Python->C 仍有一次 ctypes 边界拷贝（短期接受）；中期切 C-API struct view/zero-copy 优化。
3. C++ runner 内部避免“vector -> 临时容器 -> GPU”多段拷贝。
4. `AttentionMetadataBuilder` 负责 step 级一次上传与层间复用。
5. 模型 forward 只消费 metadata，不再自行拼 metadata。
6. 能用 device indices 的路径不再回落 host gather/scatter。

性能验证关注点：

1. `decode/block/build_attn_metadata`
2. `decode/upload_slot_indices`
3. `memcpy/*`（H2D/D2H）
4. `decode/lm_head_linear`

## 8. 分阶段实施计划

### Phase 0: 文档与测试设计（当前阶段）

1. 固化本文档与命名约定。
2. 先完成完整测试矩阵与门禁脚本（功能/稳定性/性能）。

### Phase 1: Python Runner 解耦

1. 把 `Executor._flatten` 下沉到 Worker。
2. 新增 `Worker.execute_step` 五段式流程。
3. `sample()` 从 adapter 主路径迁移到 Worker。

交付物：

1. 新接口可跑通现有测试。
2. 删除旧接口调用点并完成调用方迁移。

### Phase 2: C++ Runner 落地

1. 引入 `CppModelRunner` / `AttentionMetadataBuilder` / `Sampler`。
2. `Qwen2Model` 增加统一 `forward(...)` 接口。
3. 删除旧 decode 编排路径并统一到 runner 入口。

交付物：

1. 模型代码内不再出现 slot/block metadata 构建逻辑。
2. slot/block 均通过统一 runner 路径回归。
3. Python/C API/C++ 不存在 legacy 分叉入口。

### Phase 3: 拷贝与 metadata 优化

1. 对齐 BLOCK metadata 复用路径（builder 统一管理）。
2. 减少 host 临时容器和重复上传。
3. 结合 NVTX/NSYS 定位剩余 memcpy 热点。

交付物：

1. 关键场景 tokens/s 无回退。
2. memcpy 时间占比下降或持平。

## 9. 验收标准（必须可量化）

### 9.1 功能一致性

1. greedy 结果逐 token 一致（固定 seed）。
2. top-k/top-p/temperature 行为一致。
3. BLOCK 路径和 SLOT 路径在各自支持场景无行为回归。
4. 多请求并发下 request_id -> output_id 映射不乱序。

### 9.2 稳定性

1. `python scripts/run_tests.py --suite all --run-parity never --run-hf never` 通过。
2. 至少补充 3 类新单测：
   - Worker 新链路（prepare/build_meta/sample）
   - C++ StepPlan 组装正确性
   - C++ runner 编排正确性（prepare/meta/forward/sample/postprocess）
3. 无新增 crash、double free、非法内存访问。

### 9.3 性能

固定基准配置下（与现有对比脚本同口径）：

1. `tokens/s` 不低于基线的 98%。
2. `TTFT` 不高于基线的 105%。
3. `decode` 阶段 `memcpy` 总时长不劣化超过 5%。
4. runner 分层后，`attn_meta/build` 在总 step 时间中占比可观测。
5. 与重构前基线对比报告齐备（无兼容路径掩护）。

### 9.4 可观测性

1. 保持 NVTX 标签分层：`prepare/*`, `attn_meta/*`, `forward/*`, `sample/*`, `memcpy/*`。
2. `nsys stats` 能直接定位 step 内 metadata 构建与 memcpy 开销。

## 10. 风险与回滚

主要风险：

1. 旧接口调用方依赖隐式行为。
2. sample 迁移后，停止词/ignore_eos 边界行为差异。
3. BLOCK metadata 契约变更导致调度侧字段不一致。
4. 一次性切换失败时影响面大（无兼容回退路径）。

回滚策略：

1. 通过 Git 提交粒度回滚（按 Phase 回滚），不依赖运行时开关。
2. 上线前冻结基线 commit，失败时直接回退到冻结点。
3. 所有 schema 变更提供一键回退脚本（数据结构与接口签名同步回滚）。

## 11. 文件级改造清单（执行参考）

Python：

1. `python/llaisys/engine/executor.py`
2. `python/llaisys/engine/worker.py`
3. `python/llaisys/engine/types.py`
4. `python/llaisys/models/qwen2.py`

C++：

1. `src/llaisys/qwen2/qwen2_model.hpp`
2. `src/llaisys/qwen2/qwen2_model.cpp`
3. `src/llaisys/runtime/runner/model_runner.hpp`（新增）
4. `src/llaisys/runtime/runner/model_runner.cpp`（新增）
5. `src/llaisys/runtime/runner/attention_metadata_builder.hpp`（新增）
6. `src/llaisys/runtime/runner/attention_metadata_builder.cpp`（新增）
7. `src/llaisys/runtime/runner/sampler.hpp`（新增）
8. `src/llaisys/runtime/runner/sampler.cpp`（新增）
9. `src/ops/self_attention/cuda/self_attention_cuda.hpp`
10. `src/ops/self_attention/cuda/self_attention_cuda.cu`
11. `src/utils/nvtx.hpp`
12. `src/utils/nvtx.cpp`

文档与测试：

1. `doc/qwen2_next_dev_plan_2026-02.md`（同步里程碑）
2. `doc/qwen2_test_plan.md`（补新增测试项）
3. `scripts/bench_compare_vllm.py`（固定口径复测）

## 12. 里程碑完成定义（DoD）

M1（Python 解耦完成）：

1. Runner 具备 prepare/build_meta/forward/sample/postprocess 五段式。
2. adapter 不再包含主采样路径。

M2（C++ StepPlan 完成）：

1. C++ runner 成为唯一 step 编排入口。
2. slot/block metadata 仅在 builder 中构建。
3. `Qwen2Model` 仅保留 forward 图语义。

M3（性能收敛完成）：

1. 达到 9.3 指标。
2. NSYS 报告无新增显著 memcpy 热点。
