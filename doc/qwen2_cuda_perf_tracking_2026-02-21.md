# Qwen2 CUDA 性能追踪记录（2026-02-21）

## 1. 目标与范围

本记录覆盖本轮 Qwen2 BLOCK decode 在 NVIDIA 路径上的性能排查，重点关注：

1. `cudaMemcpy` 占比过高与来源定位。
2. paged attention native/cudnn 两条路径的吞吐对比。
3. 与 vLLM 基准脚本同口径下的差距。
4. 已完成适配项、未完成项、回退原因与后续方向。

## 2. 基准配置口径

### 2.1 快速调试口径（bench_kv_layout）

- 模型：`models/DeepSeek-R1-Distill-Qwen-1.5B`
- 设备：`--device nvidia`
- 布局：`--layout block --block-size 16`
- 压测参数：`--num-prompts 8 --rounds 1 --max-input-len 128 --max-output-len 64 --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 1024 --kv-cache-capacity-mode auto --seed 0`

### 2.2 全量对比口径（bench_compare_vllm）

- 脚本：`scripts/bench_compare_vllm.py`
- 请求规模：`--num-seqs 256`
- 输入长度：`100~1024`
- 输出长度：`100~1024`
- `max_model_len=4096`
- `max_num_seqs=256`
- `max_num_batched_tokens=16384`
- `seed=0`

## 3. 已记录的关键结果

### 3.1 bench_kv_layout（中等规模）结果

1. native（同上配置）
- `run_seconds=1.7602`
- `tokens_per_sec=130.6685`

2. cudnn（早期版本）
- `run_seconds=8.6978`
- `tokens_per_sec=26.4435`

3. cudnn（优化后版本，nsys 对应 after6/after9）
- `block.run` 已恢复到约 `3.82s` 量级（见 4.2 节）。
- 端到端较最差版本显著改善，但仍明显落后 native。

### 3.2 bench_compare_vllm（全量）结果

1. NovaInfer + cuDNN（当前）
- `expected_total_tokens=140084`
- `actual_total_tokens=139064`
- `time=252.2567s`
- `throughput_actual=592.5592 tok/s`

2. vLLM（`--vllm-fair-mode`）
- `expected_total_tokens=140084`
- `actual_total_tokens=140084`
- `time=15.7908s`
- `throughput_actual=8871.2303 tok/s`
- 运行日志显示 attention backend 仍为 `FLASH_ATTN`。

## 4. NSYS 关键演进（cudnn 路径）

### 4.1 主要报告文件

- `nsys_cudnn_after6.log`
- `nsys_cudnn_after7.log`
- `nsys_cudnn_after8.log`
- `nsys_cudnn_after9.log`

### 4.2 `block.run`（NVTX）变化

1. after6：`3.8239s`（`3823931506 ns`）
2. after7：`11.6718s`
3. after8：`14.9569s`
4. after9：`3.8140s`（`3814031204 ns`，已恢复至 after6 水平）

### 4.3 回退期共性症状

1. `cudaGetDeviceProperties_v2_v12000` 明显抬升（次数或单次长尾上升）。
2. `cuLibraryLoadData/Unload` 增多，表明运行期仍触发额外构建/加载路径。
3. `cudaMalloc/cudaFree` 仍是 API 时间大头之一。

## 5. 本轮已完成的适配项

1. decode step 级 metadata 准备
- BLOCK 路径改为每个 decode step 准备一次 paged metadata，不再按层重复准备。

2. slot 索引上传复用
- `slot_idxs` 改为 step 级上传一次，层内复用。

3. KV 写入改为 CUDA scatter kernel
- 替换逐 token 的 host/runtime copy，改为设备侧 scatter。

4. workspace 复用补齐
- `input_ids` 与 `attn_mask` 使用 workspace 复用，减少临时 tensor 创建。

5. cuDNN key/capacity 与 plan 预构建机制
- 引入 plan key 管理与分桶预构建。
- 增加预构建控制环境变量并在 Python 入口自动注入默认 hint。

6. 元数据缓冲改造
- 增加更长生命周期的 metadata 缓冲复用。
- 引入差分更新路径，降低无效整块更新概率。

## 6. 本轮遇到的问题与结论

1. 问题：仅按“当前 bucket”预构建会导致 run 期反复首次建图，`block.run` 大幅回退。
- 体现为 after7/after8 的 `block.run` 暴涨。
- 结论：必须保留稳定的预构建下限与容量稳定策略，避免运行中 shape 抖动触发构建。

2. 问题：`cudaGetDeviceProperties` 出现单次超长尾（百毫秒级）。
- 结论：构建链路仍有内部查询/加载开销，且会对端到端产生显著冲击。

3. 问题：cuDNN kernel 时间不是主要矛盾，但总吞吐仍显著落后。
- 结论：当前瓶颈主要是“适配层与运行时管理成本”，而非 attention 核函数本体算力。

4. 问题：vLLM fair mode 仍使用 `FLASH_ATTN`。
- 结论：当前对比是“NovaInfer(cudnn) vs vLLM(FLASH_ATTN 栈)”，不是同一 attention backend 的纯算子对比。

## 7. 当前适配现状（截至本记录）

1. native 路径
- 稳定，且在当前中等规模参数下明显领先 cudnn。

2. cudnn 路径
- 功能可用，`block.run` 已从回退状态恢复（after9）。
- 仍存在较重 API 开销，吞吐远低于 vLLM 的 FLASH_ATTN 栈。

3. FlashInfer/FlashAttention 路径
- 尚未完成可用接入。
- 若目标是接近 vLLM 吞吐，后续应优先投入该方向。

## 8. 下一步建议（执行优先级）

1. 将“运行期首次 build”降为 0（强约束）。
2. 继续压缩 cudnn 路径的 `cudaMalloc/cudaFree` 热点。
3. 进一步降低 metadata 的 H2D 成本与同步阻塞。
4. 启动 FlashAttention/FlashInfer 接入分支，与 native/cudnn 做 A/B 固定口径回归。

