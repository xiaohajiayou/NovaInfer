# 文档索引（当前版）

## 1. 当前应阅读顺序

1. `doc/qwen2_overview.md`：当前实现总览（简版，持续同步）
2. `doc/qwen2_requirements.md`：需求与验收约束
3. `doc/qwen2_interface.md`：C/Python 接口口径
4. `doc/qwen2_test_plan.md`：测试策略与回归规则
5. `doc/qwen2_next_dev_plan_2026-02.md`：执行计划与未完成清单

## 2. 归档文档

1. `doc/archive/qwen2_overview_legacy_2026-02-18.md`：历史大版本概览（保留追溯，不再作为当前口径）
2. `doc/archive/qwen2_test_plan_stage2_postmortem_legacy_2026-02.md`：阶段2历史复盘（保留追溯，不作为当前执行规则）

## 3. 目录与路径口径

1. 测试目录已重构：`test/core|engine|offline|online|parity|ops|utils`
2. CI 与 `scripts/run_tests.py` 已同步到上述新路径。
3. 对比基准脚本：`scripts/bench_compare_vllm.py`（NovaInfer/vLLM 同口径基准）。
