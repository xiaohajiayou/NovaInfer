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
