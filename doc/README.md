# 文档索引（当前版）

## 1. 当前应阅读顺序

1. `doc/qwen2_overview.md`：当前实现总览（简版，持续同步）
2. `doc/qwen2_requirements.md`：需求与验收约束
3. `doc/qwen2_interface.md`：C/Python 接口口径
4. `doc/qwen2_test_plan.md`：测试策略与回归规则
5. `doc/qwen2_next_dev_plan_2026-02.md`：执行计划与未完成清单
6. `doc/qwen2_cuda_perf_tracking_2026-02-21.md`：本轮 CUDA/cudnn 性能数据、适配现状与问题复盘
7. `doc/python_cpp_runner_alignment_plan_2026-03-01.md`：Python->C++ 全链路对齐 vLLM Runner 的重构方案与验收标准

## 2. 归档文档

1. `doc/archive/qwen2_overview_legacy_2026-02-18.md`：历史大版本概览（保留追溯，不再作为当前口径）
2. `doc/archive/qwen2_test_plan_stage2_postmortem_legacy_2026-02.md`：阶段2历史复盘（保留追溯，不作为当前执行规则）

## 3. 目录与路径口径

1. 测试目录已重构：`test/core|engine|offline|online|parity|ops|utils`
2. CI 与 `scripts/run_tests.py` 已同步到上述新路径。
3. 对比基准脚本：`scripts/bench_compare_vllm.py`（NovaInfer/vLLM 同口径基准）。

## 4. NVTX + nsys（性能分析）

1. 工具函数位置：`src/utils/nvtx.hpp`、`src/utils/nvtx.cpp`
2. 用法（RAII）：
   `LLAISYS_NVTX_SCOPE("qwen2.decode.layer")`
3. 用法（手动）：
   `llaisys::utils::nvtx_range_push("qwen2.decode"); ...; llaisys::utils::nvtx_range_pop();`
4. 采集命令示例：
   `nsys profile -t cuda,nvtx,osrt -o novainfer_report --force-overwrite true <your_run_cmd>`
5. 查看报告：
   `nsys stats novainfer_report.nsys-rep`
