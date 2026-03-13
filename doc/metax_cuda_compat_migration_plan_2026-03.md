# NovaInfer 适配沐曦 GPU 迁移方案（2026-03）

## 1. 结论

当前不建议一开始就单独写一套 `metax` 后端。

更合理的首版路线是：

1. 先按沐曦 `MXMACA/cu-bridge` 的 CUDA 兼容路线迁移现有 NVIDIA/CUDA 代码。
2. 首版只目标支持：
   - CUDA runtime 风格的设备/内存/stream 能力
   - `cuBLAS/cuBLASLt` 对应线性层路径
   - 自研 CUDA kernel 路径
3. 首版明确关闭 `cuDNN frontend` 路径，只保留 `native` paged attention。

原因很直接：

1. 你们项目里大部分热路径已经有 CUDA 版本。
2. 沐曦官方材料明确强调 CUDA 兼容、`cuBLAS/cuBLASLt` 兼容和 `cu-bridge` 迁移工具链。
3. 但没有足够明确的官方依据证明 `cuDNN frontend graph API` 可以无缝复用。

所以第一阶段的判断是：

1. 不要先重写整套后端。
2. 也不要默认 `cudnn` 这条能直接跑。
3. 先用现有 CUDA 路径验证能跑到什么程度，再决定抽象和重构范围。

## 2. 官方依据

本结论基于以下官方或官方维护资料：

1. 沐曦官方新闻：MXMACA 提供 CUDA 兼容能力  
   https://www.metax-tech.com/ndetail/12531.html
2. 沐曦官方新闻：强调 `cuBLAS` / `cuBLASLt` 兼容以及 vLLM、FlashAttention 生态适配  
   https://www.metax-tech.com/ndetail/12518.html
3. 沐曦开发者论坛：CUDA 项目通过 `cu-bridge` / `cucc` 迁移的实际使用方式  
   https://developer.metax-tech.com/forum/t/cuda-samplesce-shi-bao-cuo/319/
4. 沐曦官方软件仓库索引：`cu-bridge` 作为独立开发环境存在，不要求安装原生 CUDA toolkit  
   https://repos.metax-tech.com/
5. 沐曦维护的 vLLM 适配仓库：环境脚本中明确设置 `MACA_PATH`、`CUCC_PATH`、`CUDA_PATH=/opt/maca/tools/cu-bridge`  
   https://gitee.com/metax-maca/vLLM-metax/blob/master/env.sh

从这些资料只能稳妥推出两件事：

1. 普通 CUDA kernel 和 `cuBLAS/cuBLASLt` 路径值得优先尝试复用。
2. `cuDNN frontend` 不能先当成“肯定兼容”。

## 3. 现有代码中的 NVIDIA 耦合点

### 3.1 构建层

1. [`xmake.lua`](/home/liwenxiao/NovaInfer/xmake.lua)
2. [`xmake/nvidia.lua`](/home/liwenxiao/NovaInfer/xmake/nvidia.lua)

当前特征：

1. 构建开关是 `--nv-gpu` 和 `--nv-cudnn`。
2. CUDA 规则直接使用 xmake `cuda` toolchain。
3. `llaisys-ops-cuda` 直接链接：
   - `cublas`
   - `cublasLt`
   - `cudnn`

这意味着首版迁移不应该直接改业务层，应该先让构建系统能切到沐曦 `cu-bridge` 编译链。

### 3.2 设备层

1. [`src/device/nvidia`](/home/liwenxiao/NovaInfer/src/device/nvidia)
2. [`src/tensor/tensor.cpp`](/home/liwenxiao/NovaInfer/src/tensor/tensor.cpp)

当前特征：

1. Tensor 运行时直接通过 `LLAISYS_DEVICE_NVIDIA` 走 NVIDIA API。
2. 设备分配、复制、同步都绑定在 `device/nvidia` 实现。

这部分决定了“能不能在沐曦卡上创建 tensor 并跑 kernel”，是首个阻塞点。

### 3.3 算子层

已有 CUDA 版本的热点算子包括：

1. [`src/ops/linear/cuda/linear_cuda.cu`](/home/liwenxiao/NovaInfer/src/ops/linear/cuda/linear_cuda.cu)
2. [`src/ops/rms_norm/cuda/rms_norm_cuda.cu`](/home/liwenxiao/NovaInfer/src/ops/rms_norm/cuda/rms_norm_cuda.cu)
3. [`src/ops/add/cuda/add_cuda.cu`](/home/liwenxiao/NovaInfer/src/ops/add/cuda/add_cuda.cu)
4. [`src/ops/self_attention/cuda/self_attention_cuda.cu`](/home/liwenxiao/NovaInfer/src/ops/self_attention/cuda/self_attention_cuda.cu)
5. 其余 `src/ops/*/cuda/*.cu`

当前要点：

1. `linear` 依赖 `cublas` / `cublasLt`。
2. `self_attention` 同时包含：
   - native CUDA kernel 路径
   - cuDNN frontend paged attention 路径

### 3.4 Python 运行时层

1. [`python/llaisys/engine/gpu_model_runner.py`](/home/liwenxiao/NovaInfer/python/llaisys/engine/gpu_model_runner.py)
2. [`python/llaisys/entrypoints/llm.py`](/home/liwenxiao/NovaInfer/python/llaisys/entrypoints/llm.py)

当前特征：

1. Python 侧仍以 `nvidia` 为设备名。
2. attention backend 切换靠 `LLAISYS_CUDA_PAGED_ATTN_BACKEND` 环境变量。
3. `cudnn` 相关 metadata 由 GPU runner 显式构造。

首版迁移时这层不需要大改语义，但要保证：

1. `native` 后端可跑。
2. `cudnn` 后端在沐曦路径上被明确禁用或自动回避。

## 4. 首版迁移目标

首版不要追求“全部 NVIDIA 特性原样可用”，目标应收敛为：

1. `device=nvidia` 这条内部实现能在沐曦 `MXMACA/cu-bridge` 环境中编译和运行。
2. Qwen2 可以用 `native` paged attention 完成 `prefill + decode`。
3. `core / infer / offline / online` 主链路能过基础回归。
4. `cudnn` 后端在沐曦环境中默认关闭。

换句话说，首版目标是“复用 CUDA 路径跑通”，不是“复刻 NVIDIA 全功能栈”。

## 5. 建议实施顺序

### 阶段 A：构建链路迁移

目标：

1. 让 `xmake` 能在沐曦 `cu-bridge` 环境下编译 `.cu` 文件。

建议动作：

1. 新增一个独立构建选项，例如：
   - `mx-gpu`
   - `mx-cudnn` 不需要
2. 保持现有 `nvidia` 代码目录不动，先只切 toolchain 和 link/search path。
3. 让 `xmake/nvidia.lua` 的职责拆开：
   - CUDA-like 编译链配置
   - NVIDIA 专有库链接配置

最小改造建议：

1. 把 `cuda` 构建规则保留。
2. 把 `add_links("cublas", "cublasLt", "cudnn")` 改为按平台/后端条件注入。
3. 新增环境探测：
   - `MACA_PATH`
   - `CUCC_PATH`
   - `CUDA_PATH=/opt/maca/tools/cu-bridge`

验收：

1. 能编出 `llaisys` 动态库。
2. 不启用 `nv-cudnn` 时，所有 `.cu` 文件可通过。

### 阶段 B：设备运行时验证

目标：

1. 验证 `device/nvidia` 这一层是否能在沐曦环境中直接工作。

建议动作：

1. 先不改 API 名，直接验证以下操作：
   - `malloc/free`
   - H2D / D2H memcpy
   - stream create/sync
   - event/timing
2. 跑最小 tensor smoke test：
   - 创建 GPU tensor
   - 写入已知值
   - 复制回 CPU
   - 校验内容一致

如果这一步失败，再决定是否单独抽象 `device/metax`。

判断标准：

1. 如果只是个别 API 名或 flag 不兼容，优先做兼容层。
2. 如果 runtime 行为大面积不兼容，再考虑新后端目录。

### 阶段 C：BLAS 路径验证

目标：

1. 先跑通 `linear`，因为这是最可能直接受益于官方兼容承诺的部分。

建议动作：

1. 优先验证 [`src/ops/linear/cuda/linear_cuda.cu`](/home/liwenxiao/NovaInfer/src/ops/linear/cuda/linear_cuda.cu)
2. 分别验证：
   - `cublasSgemm`
   - `cublasGemmEx`
   - `cublasLtMatmul`
3. 先只看 correctness，再看性能。

验收：

1. `test_ops linear` 通过。
2. `bf16/f16/f32` 至少有一条稳定精度口径。

### 阶段 D：native attention 路径跑通

目标：

1. 只让 `self_attention_paged` 的 native 路径跑通。

原因：

1. 这是你们真实推理闭环的关键。
2. 这条不依赖 `cuDNN frontend`。
3. 只要 native paged attention 能跑，Qwen2 就有机会先完整推理。

建议动作：

1. 在 [`src/ops/self_attention/cuda/self_attention_cuda.cu`](/home/liwenxiao/NovaInfer/src/ops/self_attention/cuda/self_attention_cuda.cu) 中：
   - 沐曦环境下直接绕过 `ENABLE_CUDNN_API` / `cudnn` backend 选择
   - 固定走 native kernel 路径
2. 先用最小 case 验证：
   - 单 batch decode
   - block layout
   - 再扩到 prefill

验收：

1. `test_ops self_attention` 通过。
2. `test_core_parity` 的单卡 native block case 能跑。

### 阶段 E：Qwen2 端到端回归

目标：

1. 在沐曦卡上跑通真实模型。

建议动作：

1. 先跑：
   - `test/parity/test_core_parity.py`
   - `test/parity/test_infer.py`
   - `test/parity/test_offline_parity.py`
2. 只开：
   - `--device nvidia`
   - `--layout block`
   - `--backend native`

不要一开始就开：

1. `cudnn`
2. 多会话 online 大并发
3. vLLM 对比 benchmark

### 阶段 F：抽象收敛

只有在前面几阶段验证完后，再决定是否做更大改造。

建议分两种情况：

1. 兼容良好：
   - 保持现有目录
   - 增加 `CUDA-like backend` 配置层
   - 不新增 `src/device/metax`
2. 兼容性一般：
   - 抽 `device/common` + `blas/common`
   - 单独做 `device/metax`
   - `cudnn` 相关仍然只在 NVIDIA 下存在

## 6. 不建议首版做的事情

1. 不要先实现沐曦版 `cudnn frontend` 替代路径。
2. 不要先改 Python scheduler 语义。
3. 不要先改 `block/slot` 协议。
4. 不要先做跨平台统一大重构。
5. 不要先写一整套 `metax` 算子目录。

这些动作投入大，而且在“现有 CUDA 路径到底能复用多少”未验证前，风险太高。

## 7. 最小验证矩阵

建议按下面顺序做，不要一口气跑全量。

### 7.1 构建与设备

1. `llaisys` 能编过
2. tensor create/copy/sync smoke 通过

### 7.2 单算子

1. `linear`
2. `rms_norm`
3. `rope`
4. `swiglu`
5. `argmax`
6. `self_attention native`

### 7.3 模型闭环

1. `core parity native block`
2. `infer parity native block`
3. `offline parity native block`
4. `online single request native block`

### 7.4 性能

1. 单算子 profile
2. Qwen2 小 batch 端到端
3. Qwen2 大 batch benchmark

## 8. 对当前仓库的具体改造建议

### 必做

1. 把 `xmake/nvidia.lua` 中“CUDA-like toolchain 配置”和“NVIDIA 专有库配置”分开。
2. 在 runtime/backend 选择上增加“本平台不支持 `cudnn`”的显式分支。
3. 增加一份沐曦环境的 smoke test 文档和命令模板。

### 建议做

1. 把设备名从“物理厂商名”逐步转成“能力名”：
   - 例如内部抽象为 `gpu_cuda_compatible`
2. 把 `linear` 的 BLAS 调用封成更薄的一层，减少未来迁移面。
3. 把 `self_attention` 的 backend 选择逻辑集中起来，不要散在 Python 和 C++ 多处判断。

### 暂时不要做

1. 不要拆 `python` 调度器。
2. 不要重写 KV cache 协议。
3. 不要为了沐曦先改测试口径。

## 9. 推荐结论

如果现在立项，建议这样定口径：

1. 首版沐曦适配目标：`native block attention + CUDA compatibility path + Qwen2 end-to-end`
2. 首版不承诺：`cudnn backend`
3. 首版优先级：
   - 编译链
   - device/runtime
   - linear
   - native attention
   - Qwen2 parity

这条路线的优点是：

1. 风险最小
2. 最大化复用现有 CUDA 代码
3. 能快速判断“需不需要单独写一套后端”

只有在这条路线验证失败后，才有必要进入“沐曦专用后端”的重构方案。
