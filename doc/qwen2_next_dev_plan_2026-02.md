# Qwen2 下一步开发计划（当前执行版，2026-02-20）

## 1. 阶段状态

1. M1 基线固化：进行中  
2. M2 连续批处理收敛：已完成  
3. M3 前缀缓存（BLOCK 主线）：已完成  
4. M4 KV 接口与可观测性：已完成  
5. M5 服务面与回归闭环：进行中（功能闭环已通过，online 性能收官未完成）

当前统一结论：
1. `BLOCK` 是主线，`SLOT` 仅作为兼容与对照。
2. CPU 路径冻结在“功能稳定 + 回归可用”，不再做性能型重构。
3. 下一性能主线转向 CUDA（page-attention / block-sparse）。

## 2. 当前实现口径

1. 调度仍在 Python 层（`engine/scheduler/executor/worker`），C++ 作为 runner 执行层。
2. BLOCK 主路径使用显式 batch 元数据（`slot_mapping/context_lens/batch_seq_ids/block_tables`）。
3. `kv_seq_*` 保留兼容接口，不作为 BLOCK 主语义。
4. 前缀缓存主能力在 Python `BlockManager` + Scheduler；C++ 保持运行时接口与统计。

## 3. 基线记录（持续追加）

固定口径建议：
1. 固定 `seed`、OMP 参数、prompt 数、输入输出长度。
2. 同场景至少跑 3 轮，记录波动（不要只看单轮）。
3. 主要指标：`completion_tokens / seconds / tokens_per_sec / avg_req_latency_ms / prefix_hits/misses/saved_tokens`。

已记录样本（保留当前有效实测）：

| Date | Model | Device | Layout | num_prompts | rounds | max_input_len | max_output_len | max_model_len | kv_capacity_tokens | OMP_NUM_THREADS | completion_tokens | seconds | tokens/s | avg_req_latency_ms | prefix_hits | prefix_misses | prefix_saved_tokens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-19 | DeepSeek-R1-Distill-Qwen-1.5B | cpu | slot | 8 | 3 | 256 | 256 | 2048 | 8192 | 8 | 4050 | 1087.2130 | 3.7251 | 45300.54 | 0 | 0 | 0 |
| 2026-02-19 | DeepSeek-R1-Distill-Qwen-1.5B | cpu | block | 8 | 3 | 256 | 256 | 2048 | 8192 | 8 | 4050 | 801.0294 | 5.0560 | 33376.23 | 16 | 9 | 2656 |
| 2026-02-19 | DeepSeek-R1-Distill-Qwen-1.5B | cpu | slot | 8 | 3 | 256 | 256 | 2048 | 8192 | 8 | 4050 | 1708.4516 | 2.3706 | 71185.48 | 0 | 0 | 0 |
| 2026-02-19 | DeepSeek-R1-Distill-Qwen-1.5B | cpu | block | 8 | 3 | 256 | 256 | 2048 | 8192 | 8 | 4050 | 895.6154 | 4.5220 | 37317.31 | 16 | 9 | 2656 |

说明：同参数存在抖动，后续统一以多轮统计结论为准。

## 4. 未完成清单（面向下一阶段）

1. M1：继续补足多轮基线，并收敛测试波动解释口径。
2. M5：online 端到端性能收官对比（最终与 vLLM 对齐评估）。
3. 指标导出：统一在线/离线指标 schema（当前以 bench/log 为主）。
4. CUDA 主线：启动 page-attention 性能专项（不在本阶段展开）。

## 5. 回归与验收口径

1. 功能回归主命令：`python scripts/run_tests.py --suite all --run-parity never --run-hf never`
2. 真实模型回归：
   - `python scripts/run_tests.py --suite stage0 --model-path /path/to/model --run-parity always --run-hf always`
   - `python scripts/run_tests.py --suite stage1 --model-path /path/to/model --run-parity always`
   - `pytest -q test/online/test_online_real_model_multisession.py --model-path /path/to/model`
3. 目录口径：测试文件已重构到 `test/core|engine|offline|online|parity|ops|utils`。

## 6. CUDA 开发阶段计划（执行清单）

### 阶段 1：CUDA 基础通路打通

范围：
1. `src/device`：CUDA 内存分配/释放/拷贝/同步。
2. `tensor`：CUDA 下 `create/load/view/slice/isContiguous` 语义一致。

交付：
1. CUDA tensor 可创建、写入、读取、切片、视图。
2. 无隐式回 CPU 路径。

验收标准：
1. CUDA tensor 基础单测通过。
2. `device=nvidia` 的最小算子 smoke（伪数据）通过。

### 阶段 2：Linear CUDA 主路径

范围：
1. `src/ops/linear/*` 增加 CUDA 实现。
2. `src/ops/linear/op.cpp` 增加 `LLAISYS_DEVICE_NVIDIA` 分发。

交付：
1. `linear(out, in, weight, bias)` 支持 CUDA `fp16/bf16`。

验收标准：
1. 与 CPU 结果对齐（按 dtype 误差阈值）。
2. Qwen2 单层权重抽样前向一致。
3. 线性层执行不回退到 CPU。

### 阶段 3：Paged Attention CUDA 主路径

范围：
1. `src/ops/self_attention/*` 实现 `self_attention_paged` CUDA 版本。
2. 输入协议保持当前口径：`used_slots/row_ptr/col_idx`。

交付：
1. CUDA paged attention correctness 版本可用。

验收标准：
1. 与 CPU `self_attention_paged` 在固定 batch 上对齐（阈值内）。
2. 支持可变 `nnz`、多 token 行场景。
3. `decode_block_path_` 可稳定调用 CUDA attention 分支。

### 阶段 4：Qwen2 端到端 CUDA decode 跑通

范围：
1. `qwen2_model.cpp` 去除 CPU-only 假设（如 host memcpy 路径）。
2. 权重/workspace/KV cache 全链路 GPU 驻留。

交付：
1. `device=nvidia` 下完成 prefill+decode。

验收标准：
1. `core/engine/offline` 关键用例在 CUDA 跑通。
2. online server 可启动并返回流式结果。
3. 无跨设备 dtype/contiguous 断言错误。

### 阶段 5：辅助算子 CUDA 化（去 fallback）

范围：
1. `rms_norm`、`rope`、`swiglu`、`add`（必要时 embedding/logits 路径）补 CUDA。
2. 对应 `op.cpp` 完成 CUDA dispatch。

交付：
1. decode 热路径中 CPU fallback 显著减少。

验收标准：
1. `bench_kv_layout.py --device nvidia` 稳定跑通。
2. profile 中主热路径为 CUDA kernel。

### 阶段 6：性能收敛与基线固化

范围：
1. kernel 参数与访存优化、launch/拷贝开销清理。
2. 固化 CUDA benchmark 配置与日志模板。

交付：
1. CUDA 基线表（tokens/s、TTFT、avg latency、显存占用）。
2. slot vs block（CUDA）对比结论。

验收标准：
1. 同配置多轮波动可解释。
2. BLOCK 相对 SLOT 收益可复现。
3. 文档基线区持续更新。

### 阶段 7：TP 预留与最小实现

范围：
1. `EngineConfig`/runtime 增加 TP 参数（`tensor_parallel_size/rank/world_size`）。
2. attention/MLP 切分与必要通信接口骨架（先 correctness）。

交付：
1. TP=1/2 配置链路打通。

验收标准：
1. TP=2 最小端到端推理通过。
2. TP=1/TP=2 输出一致性在阈值内。
3. 文档补齐通信点与后续优化计划。
