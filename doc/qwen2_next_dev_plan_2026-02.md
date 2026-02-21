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
| 2026-02-21 | DeepSeek-R1-Distill-Qwen-1.5B | nvidia | block | 8 | 3 | 256 | 256 | 2048 | 16384 | 8 | 3453 | 35.7193 | 96.6704 | 1488.31 | 16 | 9 | 1696 |

说明：
1. 同参数存在抖动，后续统一以多轮统计结论为准。
2. 当前脚本口径中，表内 `seconds` 按 `run_seconds` 记录（不包含 init/warmup/close）。

### 3.1 NovaInfer vs vLLM 对比基线（`scripts/bench_compare_vllm.py`）

固定配置（本次）：
1. `CUDA_VISIBLE_DEVICES=2`
2. `num_seqs=256`
3. `input_len=[100,1024]`
4. `output_len=[100,1024]`
5. `max_model_len=4096`
6. `max_num_seqs=256`
7. `max_num_batched_tokens=16384`
8. `kv_cache_capacity_mode=auto`
9. `seed=0`

结果记录（持续追加）：

| Date | Model | Device | Backend | num_seqs | max_model_len | expected_total_tokens | init_seconds | warmup_seconds | run_seconds | total_seconds | tokens_per_sec(run) | Status | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-02-21 | DeepSeek-R1-Distill-Qwen-1.5B | nvidia | novainfer | 256 | 4096 | 140084 | 6.6772 | 2.8732 | 664.8953 | 674.4868 | 210.6858 | PASS | `auto_max_num_seqs=256`，KV auto capacity 生效 |
| 2026-02-21 | DeepSeek-R1-Distill-Qwen-1.5B | nvidia | vllm | 256 | 4096 | 140084 | N/A | N/A | N/A | N/A | N/A | FAIL | 首次失败：`AttributeError: 'list' object has no attribute 'get'`（vLLM 输入需 `{"prompt_token_ids":[...]}`） |

修复与口径说明：
1. `scripts/bench_compare_vllm.py` 已修复 vLLM tokenized prompt 输入格式（`prompt_token_ids`）。
2. `--backend both` 已改为子进程隔离执行 `novainfer/vllm`，避免同进程 CUDA 初始化冲突。
3. NovaInfer 吞吐按 `run_seconds` 统计，不包含 `init/warmup/close`。

### 3.2 本轮实现与修复清单（2026-02-21）

功能实现：
1. `scripts/bench_compare_vllm.py` 新增 NovaInfer/vLLM 对比基准主脚本（nano-vllm 风格口径）。
2. NovaInfer bench 输出已补齐 `init/warmup/run/finish` 分段计时与 KV 统计打印，便于在线排障。
3. `backend=both` 支持串行子进程执行并汇总结果，避免多后端同进程 CUDA 状态污染。

性能与稳定性优化：
1. CUDA paged attention 元数据支持按 decode-step 复用，减少层内重复上传。
2. CUDA benchmark 计时口径统一：吞吐按 `run_seconds`（纯推理阶段）计算。
3. KV auto capacity 在 NVIDIA 下加入保守预算与日志字段，便于 OOM 与容量诊断。

关键 bug 修复：
1. 修复 NVIDIA auto capacity 估算未使用 `max_num_seqs` 的问题（已透传并参与估算）。
2. 修复 vLLM 路径 tokenized 输入格式不兼容（`list[list[int]]` -> `prompt_token_ids`）。
3. 修复 `linear_cuda` add-bias 大批量下 grid 配置越界问题（索引/网格改为 `size_t` 口径）。

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

## 7. CUDA 性能瓶颈分析与优化记录（2026-02-21）

### 7.1 分析目标与背景

1. 目标：解释“GPU 已跑通但吞吐偏低/波动大”的原因，给出可执行优化路径。
2. 现象：
   - 小 workload 下可见较高瞬时吞吐（如 >60 tok/s，极小样本可到 90 tok/s+）。
   - 长 workload/混合场景下，端到端吞吐被 init/warmup 与 host-side 开销拉低。

### 7.2 Profiling 方法（当前固定口径）

1. 基准命令（短集，便于快速定位）：
   - `python scripts/bench_kv_layout.py --device nvidia --layout block --num-prompts 2 --rounds 1 --max-input-len 64 --max-output-len 32 --max-model-len 1024 --max-num-seqs 4 --max-num-batched-tokens 512 ...`
2. Nsight Systems（推荐）：
   - `nsys profile --trace=cuda,nvtx,osrt --sample=none ...`
   - `nsys stats --force-export=true --report cuda_api_gpu_sum,cuda_gpu_kern_sum,cuda_api_sum <rep>`
3. NVTX 分段：
   - `block.init`
   - `block.warmup`
   - `block.run`
   - `block.round_1`

### 7.3 关键证据（已记录）

1. `nsys_after_stepmeta`（优化后、短集）：
   - `cudaMemcpy` 约 `549.8 ms`（API）
   - `H2D memcpy` 约 `350.2 ms`
   - `cudaMalloc` 约 `157.0 ms`
   - `cudaFree` 约 `19.8 ms`
   - `paged_attention_warp_kernel` 约 `120.9 ms`
2. `nsys_memcpy_fast_v2`（同口径复测）：
   - `cudaMemcpy` 约 `474.5 ms`（仍是 API 第一大项）
   - `cudaMalloc` 约 `169.1 ms`
   - `cudaFree` 约 `135.4 ms`
   - `paged_attention_warp_kernel` 约 `115.4 ms`
3. NVTX 汇总（同批次）：
   - `block.init` 约 `4.29 s`
   - `block.warmup` 约 `2.08 s`
   - `block.run` 约 `236 ms`
4. 结论：
   - 当前“慢”的主因不是单一 kernel 算力不足，而是大量 API/内存管理开销叠加（`cudaMemcpy/cudaMalloc/cudaFree`）。
   - 端到端观察到的长耗时，常被 `init/warmup` 放大；纯 run 阶段明显更快。

### 7.4 已完成优化（本轮）

1. Paged attention 元数据由“层内重复上传”改为“decode-step 级准备/复用”。
2. benchmark 计时口径收敛：
   - 吞吐统一按 `run_seconds` 计算；
   - `init/warmup/close` 独立统计，避免混入推理吞吐。
3. `bench_compare_vllm.py`：
   - `both` 改为子进程隔离，避免多后端同进程 CUDA 初始化问题。
   - NovaInfer 增加 `init/warmup/run/finish + kv_stats` 打印，便于调试。

### 7.5 主要瓶颈归因（当前判断）

1. 高频 `cudaMemcpy`：
   - token/position/logits 路径与注意力元数据路径存在频繁 H2D/D2H 传输。
   - 小块、频繁调用导致 API 时间占比过高。
2. 高频 `cudaMalloc/cudaFree`：
   - decode 热路径仍有临时 buffer/tensor 分配释放。
   - allocator 抖动直接拖慢短步前向。
3. paged attention kernel 本身已是热点，但不是唯一主瓶颈：
   - kernel 时间占比高于其他算子，但 API/内存管理总开销同样显著。

### 7.6 优化策略与落地顺序（执行版）

P0（优先级最高，先做）：
1. 热路径去动态分配：
   - 将 decode 中临时 tensor/metadata 全量搬入 workspace 复用。
   - 目标：显著降低 `cudaMalloc/cudaFree` 次数。
2. 降低小块 memcpy 次数：
   - 按 step 打包上传 metadata，避免 layer 内重复上传。
   - 输出/logits 路径改为按需回传，减少 D2H 细粒度拷贝。

P1（完成 P0 后）：
1. paged attention 内核继续对齐高性能实现（访存模式、并行映射、共享内存策略）。
2. 进一步减少 launch 开销（融合轻量 kernel / 批量化调用）。

P2（稳定后）：
1. 固化 A/B 基准矩阵（同配置多轮，记录 median/p95）。
2. 与 vLLM 做同口径端到端对比，分离 init/run 指标。

### 7.7 验收指标（本阶段）

1. `cudaMalloc + cudaFree` 总时间占比下降到可接受区间（相对当前显著下降）。
2. `cudaMemcpy` API 占比明显下降，H2D 小包次数下降。
3. `block.run` 吞吐稳定提升（同 seed/同 workload）。
4. 文档与基线表持续追加：每次优化都必须补 profile 截图/统计摘要与结论。

### 7.8 任务看板（已验证 / 待验证）

已验证：
1. 已定位并复现主瓶颈：`cudaMemcpy/cudaMalloc/cudaFree` 在短集 profile 中占比显著。
2. 已确认 `run` 与 `init/warmup` 需分离统计，吞吐必须按 `run_seconds` 计算。
3. 已完成 paged attention 元数据 step 级复用改造（减少层内重复上传）。
4. 已完成 `bench_compare_vllm.py` 调试增强：
   - NovaInfer 输出 `init/warmup/run/finish + kv_stats`；
   - `backend=both` 子进程隔离，避免 CUDA 初始化冲突。
5. 已修复 vLLM tokenized 输入格式（`prompt_token_ids`），可继续对比跑数。

待验证（下一轮执行）：
1. Workspace 全覆盖：
   - decode 热路径临时 tensor/metadata 全量搬入 workspace 复用；
   - 验证 `cudaMalloc/cudaFree` 次数与总时间下降。
2. memcpy 收敛：
   - 梳理 token/pos/logits/attention metadata 拷贝路径；
   - 合并小包 H2D，减少细粒度 `cudaMemcpy` 调用。
3. paged attention 内核收敛：
   - 在当前 warp kernel 基础上继续访存/并行映射优化；
   - 目标是提升 run 阶段 tokens/s，并压低 kernel 时间方差。
4. 对比基准闭环：
   - 跑通 NovaInfer/vLLM 同配置 full run；
   - 产出同表对比（tokens/s、run_seconds、init/warmup）。
5. 稳定性验证：
   - 同配置至少 3 轮，记录 median/p95；
   - 若波动超阈值，补充资源竞争与上下文污染排查结论。
