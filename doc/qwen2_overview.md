# Qwen2 概览（当前口径）

## 1. 当前状态（2026-02-20）

1. 主线架构：`BLOCK(PagedKvImpl)` 为性能演进主线；`SLOT(UnifiedKvImpl)` 作为兼容与回归对照。
2. 执行链路：Python `engine/scheduler/executor/worker` 负责调度，C++ `Qwen2Model` 负责 runner 执行。
3. 前缀缓存：主能力在 Python `BlockManager`/Scheduler 路径；C++ 保持运行时接口与统计。
4. CPU 策略：CPU attention 路径冻结为“功能稳定 + 回归可用”，后续性能主线转向 CUDA。
5. 测试组织：测试目录已重构为 `core/engine/offline/online/parity/ops/utils` 分层。
6. CI 状态：GitHub Actions 已同步新测试路径，`push/pull_request` 自动运行。

## 2. 代码主路径

1. C++：
   - `src/llaisys/qwen2/qwen2_model.cpp`
   - `src/llaisys/runtime/kv_cache/unified_kv.*`
   - `src/llaisys/runtime/kv_cache/paged_kv.*`
2. Python：
   - `python/llaisys/models/qwen2.py`
   - `python/llaisys/engine/*`
   - `python/llaisys/server/*`
3. 测试：
   - `test/core/*`
   - `test/engine/*`
   - `test/offline/*`
   - `test/online/*`
   - `test/parity/*`
   - `test/utils/batch_builders.py`

## 3. 文档索引

1. 需求：`doc/qwen2_requirements.md`
2. 接口：`doc/qwen2_interface.md`
3. 测试：`doc/qwen2_test_plan.md`
4. 计划：`doc/qwen2_next_dev_plan_2026-02.md`
5. 历史归档：`doc/archive/qwen2_overview_legacy_2026-02-18.md`
