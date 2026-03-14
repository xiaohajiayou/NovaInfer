# 文档索引（当前版）

## 1. 当前应阅读顺序

1. `doc/novainfer_overview_2026-03-14.md`：当前 NovaInfer LLM 推理栈设计总览
2. `doc/novainfer_requirements_2026-03-14.md`：需求与验收约束
3. `doc/qwen2_interface.md`：C/Python 接口口径
4. `doc/novainfer_test_plan_2026-03-14.md`：测试策略与回归规则
5. `doc/qwen2_next_dev_plan_2026-02.md`：执行计划与未完成清单
6. `doc/qwen2_cuda_perf_tracking_2026-02-21.md`：CUDA/cudnn 性能跟踪与问题复盘
7. `doc/python_cpp_runner_alignment_plan_2026-03-01.md`：Python/C++ runner 对齐方案
8. `doc/tp_parallel_context_merge_plan_2026-03-14.md`：TP 与 ParallelContext 重构方案与当前状态
9. `doc/tp_repro_guide_2026-03-14.md`：TP 复现指南

## 2. 性能文档

1. `doc/novainfer_vs_vllm_perf_experiment_2026-03-12.md`：3090 单机 NovaInfer vs vLLM 实验
2. `doc/novainfer_a100_perf_validation_2026-03-14.md`：A100 集群单卡与 TP 性能验证

## 3. 归档文档

1. `doc/archive/qwen2_requirements_legacy_2026-03-14.md`
2. `doc/archive/qwen2_overview_legacy_2026-03-14.md`
3. `doc/archive/qwen2_test_plan_legacy_2026-03-14.md`
4. `doc/archive/qwen2_overview_legacy_2026-02-18.md`
5. `doc/archive/qwen2_test_plan_stage2_postmortem_legacy_2026-02.md`

## 4. 目录与路径口径

1. 测试目录：`test/core|engine|offline|online|parity|ops|utils`
2. 对比基准脚本：`scripts/bench_compare_vllm.py`
3. 单卡对比实验：`scripts/run_perf_experiments.py`
4. TP 基准脚本：`scripts/bench_tp_novainfer.py`

## 5. NVTX + nsys

1. 工具：`src/utils/nvtx.hpp`、`src/utils/nvtx.cpp`
2. 采集示例：
   `nsys profile -t cuda,nvtx,osrt -o novainfer_report --force-overwrite true <cmd>`
3. 查看报告：
   `nsys stats novainfer_report.nsys-rep`
