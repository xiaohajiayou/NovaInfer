# Qwen2 下一步开发计划（2026-02）

## 1. 冲突点与统一结论

1. `SLOT` vs `BLOCK` 定位冲突
   - 统一结论：`BLOCK` 为性能主线，`SLOT` 仅保留基础可用与回归/对照。
2. `kv_seq_*` 接口定位冲突
   - 统一结论：保留兼容，不作为 BLOCK 长期主语义；BLOCK 主线收敛到 request 生命周期接口。
3. 调度归属冲突（Python vs C++）
   - 统一结论：现阶段保留 Python 调度（`engine/scheduler/executor/worker`），C++ 作为 runner 执行层。
4. vLLM 对齐粒度冲突
   - 统一结论：优先对齐“请求级 append-only + block manager + page-native 执行数据流”，不强求一次性对齐所有工程细节。

## 2. 目标架构（目标态）

1. Python 层
   - `LLMEngine`: 请求生命周期与 step 驱动。
   - `RequestScheduler`: prefill/decode 混排与公平调度。
   - `BlockManager`（新增，BLOCK 专用）: 维护 `request -> block_table` 逻辑映射、alloc/append/free/preempt。
   - `Executor/Worker`: 组装 step 输入并调用 C++ decode。
2. C++ 层
   - `Qwen2Model` 继续承担 runner 职责（后续可再拆 runner 类）。
   - `PagedKvImpl`: 物理 KV + block 元数据执行路径。
   - `UnifiedKvImpl`: SLOT 兼容路径。
3. 执行数据流
   - Python 输出 request-step 计划；
   - C++ 消费 `token/pos/seq_id`（过渡期）并逐步过渡到 block-table 驱动输入。

## 3. 里程碑与任务拆分（当前执行版）

> 约束：本阶段**不做 CPU 算子优化，不做 CUDA 算子开发**。  
> 目标：先完成算子外能力（调度、前缀缓存、接口、基准、可观测性），为后续 CUDA 性能阶段铺路。

### M1: 口径与基线固化（立即执行）

1. 文档统一到同一口径：
   - `BLOCK` 主线；
   - `SLOT` 仅作为兼容与回归对照；
   - `kv_seq_*` 为兼容接口，不作为 BLOCK 长期主语义。
2. 固化 benchmark 默认场景与运行参数：
   - 固定 seed、prompt 分布、并发、rounds、OMP 配置；
   - 输出统一指标（`tokens/s`、`TTFT/TPOT`、`P50/P95`、KV 水位）。
3. 基线数据入库（文档化）：
   - `slot vs block` 在同一配置下至少 3 轮结果。

验收：
1. `scripts/bench_kv_layout.py` 参数口径稳定，日志能复现同一实验。
2. 文档中存在可执行命令与基线记录模板。

执行建议（固定基线命令）：
1. 小规模冒烟（先看流程稳定）：
```bash
python scripts/bench_kv_layout.py \
  --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu \
  --layout both \
  --block-size 16 \
  --num-prompts 4 \
  --rounds 1 \
  --seed 0 \
  --max-input-len 256 \
  --max-output-len 256 \
  --max-model-len 1024 \
  --kv-cache-capacity-tokens 8192 \
  --omp-num-threads 8 \
  --omp-proc-bind spread \
  --omp-places cores
```
2. 固化基线（至少 3 轮）：
```bash
python scripts/bench_kv_layout.py \
  --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu \
  --layout both \
  --block-size 16 \
  --num-prompts 8 \
  --rounds 3 \
  --seed 0 \
  --max-input-len 512 \
  --max-output-len 512 \
  --max-model-len 1024 \
  --kv-cache-capacity-tokens 32768 \
  --omp-num-threads 8 \
  --omp-proc-bind spread \
  --omp-places cores
```

基线记录模板（填入同一文档或独立报告）：
| Date | Model | Device | Layout | num_prompts | rounds | max_input_len | max_output_len | max_model_len | kv_capacity_tokens | OMP_NUM_THREADS | completion_tokens | seconds | tokens/s | avg_req_latency_ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-xx | DeepSeek-R1-Distill-Qwen-1.5B | cpu | slot | 8 | 3 | 512 | 512 | 1024 | 32768 | 8 |  |  |  |  |
| 2026-02-xx | DeepSeek-R1-Distill-Qwen-1.5B | cpu | block | 8 | 3 | 512 | 512 | 1024 | 32768 | 8 |  |  |  |  |

### M2: 连续批处理收敛（调度主线）

1. `StepPlan` 彻底 request-aware：
   - 以 request/sequence 为中心，而非 token 列表中心；
   - 保留与现有 batch decode 的 flatten 兼容层。
2. prefill/decode 调度规则收敛：
   - admission、preempt、resume、finish 行为明确；
   - 避免 starvation 与 silent drop。
3. 同批多请求独立采样参数落地：
   - 移除“同一 step 仅取首请求采样参数”的限制。
4. 请求生命周期接口打通：
   - `request_free` 在 engine -> worker -> core 全链路一致。

验收：
1. `engine/offline/online` 现有用例通过。
2. 多请求并发下无串线、无悬挂请求、无未释放请求。

### M3: 前缀缓存（BLOCK 主线能力）

1. 设计并实现 prefix 索引（block 级）：
   - prefix key/hash；
   - request -> block_table 前缀复用映射。
2. 命中路径与失效路径：
   - 命中时跳过重复 prefill；
   - truncate/free 后引用计数与可见性一致。
3. 运维接口与观测：
   - prefix hit/miss；
   - 复用节省 token 数；
   - `reset_prefix_cache`（可选开关）。

验收：
1. 前缀命中场景正确，输出与无前缀缓存一致。
2. 命中率与节省统计可查询、可打印。

### M4: KV 接口与可观测性补齐

1. BLOCK 主线接口补齐：
   - `llaisysModelRequestFree`；
   - `llaisysModelKvStats`（used/free blocks、used tokens、watermark）；
   - `llaisysModelKvResetPrefixCache`（如启用前缀缓存）。
2. `kv_seq_*` 保持兼容但降级为 legacy 路径：
   - 文档明确 BLOCK 下语义边界（如 tail truncate）。
3. 错误码与日志统一：
   - 容量不足、越界、非法状态转换可定位。

验收：
1. Python 层可读取 KV 统计并输出。
2. BLOCK/SLOT 切换时接口行为可预期且文档一致。

### M5: 服务面与回归闭环

1. Online 接口补齐（非算子）：
   - 评估并补充 `/v1/completions`；
   - embeddings 维持“可选后续”，但文档要明确状态。
2. Metrics 导出：
   - 日志指标 + 统一导出格式（后续可接 Prometheus）。
3. 回归矩阵固化：
   - `slot/block x offline/online x core/api`。

验收：
1. `pytest -q` 主回归稳定通过。
2. 基准脚本与服务链路在 BLOCK 下可稳定运行。

## 4. 接口策略（兼容期）

1. 保留：
   - `llaisysModelDecode`
   - `llaisysModelGetLogits*`
2. 兼容保留（后续收敛）：
   - `llaisysModelKvSeqCp/Rm/Add/Keep/PosMax`
3. 计划新增（BLOCK 主线）：
   - `llaisysModelRequestFree`
   - `llaisysModelKvStats`
   - `llaisysModelKvResetPrefixCache`

## 5. 风险与控制

1. 风险：SLOT/BLOCK 代码路径继续耦合导致维护成本上升。
   - 控制：调度策略分层 + C++ 分支内聚 + 测试矩阵拆分。
2. 风险：接口迁移破坏现有 Python 调用。
   - 控制：兼容期保留旧接口，新增接口先灰度接入。
3. 风险：性能优化引入正确性回退。
   - 控制：每步改造后先跑核心回归，再跑全量回归。

## 6. 执行顺序（建议）

1. M1 基线固化。
2. M2 连续批处理收敛。
3. M3 前缀缓存。
4. M4 KV 接口与可观测性。
5. M5 服务面与回归闭环。

> 下一阶段（不在本文执行范围）：CUDA 算子与 page-attention 性能专项。
