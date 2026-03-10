# Python -> C++ 全链路重构对齐方案（vLLM Runner 解耦口径，2026-03-01）


  目标

  1. 单一路径：Scheduler -> ModelRunner.prepare_* -> forward -> sample ->
     Scheduler.postprocess。
  2. 配置单一真源：只用 EngineConfig，初始化时一次性回填模型元信息。
  3. 删除重复状态和重复检查：不再在 Worker/Engine/Runner 多层兜底。

  核心不变式

  1. Scheduler.schedule() 返回的 seqs 就是本轮执行集合，后面不再二次重组。
  2. postprocess(seqs, token_ids) 只改 seq 状态/finish_reason，不返回映射。
  3. prepare_prefill/prepare_decode 直接产出 forward 所需全部 tensor。
  4. execute_model 和 sample_tokens 两段形式保留，但内部只做“调用 + 最小组
     包”。

  数据结构对齐

  1. Sequence 对齐 nano-vllm字段：seq_id/status/token_ids/last_token/
     num_tokens/num_prompt_tokens/num_cached_tokens/block_table/
     temperature/max_tokens/ignore_eos/finish_reason。
  2. request_id 不放在 Sequence，放 Engine 映射层：seq_id <-> request_id。
  3. 删除 effective_*、runtime_max_*、kv_cache_capacity_tokens 等中间字段
     （除非被 C++ runtime 必需）。

  Config 设计

  1. EngineConfig 启动后即确定：max_model_len/end_token_id/
     num_kvcache_blocks。
  2. 模型 config.json/meta 在模型加载后一次性读出并回填 config。
  3. 后续 Scheduler/Engine/Runner 全部只读 config，不再回传修正。

  ModelRunner 设计（重点）

  1. 仅保留两个 prepare：prepare_prefill(seqs)、prepare_decode(seqs)。
  2. prepare 内完成：
      - 计算 input_ids/positions
      - 计算 attention metadata（按 nano-vllm语义：cu_seqlens_q/
        cu_seqlens_k/max_seqlen_q/max_seqlen_k/slot_mapping/block_tables）
      - 直接生成 host/device tensor
  3. execute_model 内直接：
      - fin = prepare_prefill(...) if is_prefill else prepare_decode(...)
      - 组 C 接口入参并 forward
      - 不再有 _prepare_inputs/_build_forward_io/_PreparedForward/
        _ExecuteModelState 这类中间层。
  4. sample_tokens 直接返回 list[int]（GPU 同步后 tolist），与 nano-vllm 对
     齐。

  C++/Kernel 设计

  1. attention 元数据接口改成 nano-vllm语义（cu_seqlens_* + slot_mapping +
     block_tables），不再以 query_start_loc/seq_lens 为主。
  2. op.cpp 去掉“metadata 必须 CPU”的硬编码，设备约束与执行设备一致。
  3. CUDA 路径去掉运行时再推导 row 的重复逻辑，prepare 阶段已给足数据。
  4. 删除不用的 metadata builder kernel（只保留被主链路实际使用的）。

  Engine/Scheduler 设计

  1. Engine step 对齐 nano-vllm：
      - seqs = schedule()
      - token_ids = execute + sample
      - postprocess(seqs, token_ids)
      - outputs = finished seqs
  2. finish 判定放 Scheduler（长度/EOS/stop），Engine 只做结果封装。
  3. _complete_request 只做输出组装和资源释放，不再做策略判断。

  落地顺序

  1. 先改数据契约（Sequence + Config + Scheduler不变式）。
  2. 再改 Runner prepare/execute/sample 单路径。
  3. 最后改 C++ AttentionMetadata 与 kernel入口。
  4. 清理冗余字段与旧 helper。
  5. 全量测试 + 真实模型冒烟。



  • 现状扫描结论（基于当前仓库）

  1. 当前主线是可运行的：我刚跑了 pytest -q test/engine test/offline test/
     online，结果 48 passed, 3 skipped。
  2. 主要复杂度集中在 gpu_model_runner.py（744 行），核心问题是“prepare/拷
     贝/metadata/状态”混在一起。
  3. CPUModelRunner 继承 GPUModelRunner，导致大量“反向约束检
     查”（CpuGpuBuffer 判断）和重复 builder。见 cpu_model_runner.py。
  4. Engine 仍维护较多映射状态（_seq_by_id/_request_id_by_seq_id/
     _seq_id_by_request），虽然已经比之前干净，但还能继续收敛。见
     llm_engine.py。
  5. C++ 侧仍是 slot/block 双模式接口，Python 当前几乎都在走 block 主路径；
     接口冗余还在。见 model.h、qwen2_model.cpp。

  基于当前代码的调整设计（对齐 nano-vllm）

  1. 第一步：先把 Python 控制面收敛成“单一路径”
      - SchedulerOutputs 只保留必要字段：scheduled_seqs/is_prefill/
        finished_seq_ids。
      - LLMEngine.step() 固化为：schedule -> execute_model -> sample_tokens
        -> scheduler.postprocess -> complete finished。
      - postprocess 只改 seq（写 finish_reason/status），不返回额外结构。
  2. 第二步：重构 ModelRunner 为 nano 风格 prepare
      - 删除 PreparedTensors dataclass 和 _build_slot_tensors/
        _build_block_tensors 两层。
      - 只保留 prepare_prefill(seqs)、prepare_decode(seqs)、
        prepare_sample(seqs)，每个函数直接产出执行所需 tensor。
      - execute_model 保留函数形态，但内部只做：prepare -> forward -> 缓存
        采样输入。
      - sample_tokens 保留函数形态，但直接返回 list[int]。
  3. 第三步：模式收敛到 BLOCK（对齐你当前主路径）
      - Python 层先去掉 slot 分支。
      - C++ Qwen2Model::forward 只接受 ATTENTION_MODE_BLOCK，删
        prepare_slot_attention_state_ 路径。
      - AttentionMetadata 精简为 block 必需字段集合。
  4. 第四步：配置单一真源
      - max_model_len/end_token_id 在 model wrapper 创建后一次性回填到
        EngineConfig。
      - 后续 Scheduler/Engine/Runner 全部只读 config，不再二次推导和回传改
        写。
  5. 第五步：kernel/ops 接口清理
      - 对齐 block metadata 契约，删无效参数传递。
      - self_attention op 层去掉与实际设备约束冲突的检查（保持单一设备契
        约）。

  建议执行顺序

  1. 先做 Python 侧单路径（低风险，测试覆盖完整）。
  2. 再做 BLOCK-only 接口收敛（会改到 C++/ABI）。
  3. 最后做 kernel 参数精简和命名对齐。