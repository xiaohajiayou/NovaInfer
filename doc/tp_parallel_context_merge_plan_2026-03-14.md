# TP ParallelContext Merge Plan

## Goal

Merge `origin/main` structural cleanup into `tp` while:

- keeping `tp=1` performance unchanged
- keeping TP correctness against HF
- simplifying code by removing TP state from `kv_state`
- making the result readable and directly mergeable

## Frozen Decisions

1. One model instance binds exactly one `ParallelContext`.
2. `ParallelContext` is held by the Python runner and bound into the C++ model.
3. `llaisysKvStateParallelInit(...)` is deleted without compatibility.
4. `ModelCreateParams` remains model-only. TP runtime state moves to `ParallelContext`.
5. NCCL communicator is owned by `ParallelContext`.
6. Only the current multi-process TP mode is supported. No unused extra fields are kept.
7. No generic multi-model TP base class in this round. Only common communication plumbing may be shared.

## Target Responsibilities

### Model

- weights
- model metadata
- TP-local shape state
- forward execution
- workspace / attention backend state

### ParallelContext

- `tensor_parallel_size`
- `rank`
- `local_rank`
- `device_ids`
- `distributed_backend`
- `init_method`
- NCCL communicator

### KvState

- paged KV cache
- request/sequence KV lifecycle
- KV stats / prefix cache

## Implementation Route

### Step 1. Accept main structural cleanup

- keep block-only KV path
- keep flattened `src/llaisys/kv_cache`, `src/llaisys/workspace`, `src/llaisys/weights`
- keep Python binding/module cleanup from main

### Step 2. Introduce `ParallelContext`

New public C API:

- `llaisysParallelContextCreate(...)`
- `llaisysParallelContextDestroy(...)`
- `llaisysModelBindParallelContext(...)`

Removed public C API:

- `llaisysKvStateParallelInit(...)`

### Step 3. Rewire Python engine

- `runtime_factory.create_kv_state(...)` becomes KV-only
- new `runtime_factory.create_parallel_context(...)`
- `model_registry` creates both `kv_state` and `parallel_context`
- `GPUModelRunner` owns:
  - `_kv_state`
  - `_parallel_context`
- model wrapper binds the parallel context once during initialization

### Step 4. Rewire C++ model path

- `LlaisysModelImpl` owns the bound `ParallelContext`
- `LlaisysKvStateImpl` becomes KV-only again
- `Qwen2Model::bind_parallel_context(...)` receives the already-created NCCL communicator
- `Qwen2Model` no longer creates or destroys NCCL communicators

## New Model Adaptation Rule

Future TP model adaptation should only require model-local logic for:

1. weight sharding rules
2. local shape derivation
3. collective insertion points

Common distributed plumbing stays out of model code and lives in `ParallelContext`.

## Acceptance Criteria

### Correctness

1. `test/core/test_core_model_api.py` passes
2. `test/parity/test_core_parity.py` passes for `nvidia+cudnn+block`
3. `test/parity/test_infer.py` passes for `nvidia+cudnn+block`
4. `scripts/tp_hf_parity.py` passes for:
   - 1.5B, `tp_size=2`
   - 7B, `tp_size=4`

### Performance

1. `tp=1` single-card benchmark does not regress against current main-level baseline
2. TP throughput is reported using:
   - `global_tokens / max(rank_run_seconds)`
3. TP throughput remains above single-card baseline where expected

### Code Quality

1. No TP state remains in `kv_state`
2. No compatibility shim for removed `llaisysKvStateParallelInit(...)`
3. `git diff --check` passes

## Validation Log

### Build

Validated with:

```bash
xmake f --mode=release --nv-gpu=y --nv-cudnn=y --nv-nccl=y
xmake -j1
xmake install
```

### Tests

Passed:

```bash
pytest -q test/core/test_core_model_api.py test/core/test_model_registry.py test/core/test_kv_cache.py
pytest -q test/parity/test_core_parity.py --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --backend cudnn
pytest -q test/parity/test_infer.py --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --backend cudnn
pytest -q test/core test/engine test/offline test/online
```

### HF Parity

Passed:

```bash
CUDA_VISIBLE_DEVICES=5,6 \
.venv/bin/python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tp-size 2 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

```bash
CUDA_VISIBLE_DEVICES=1,2,5,6 \
.venv/bin/python scripts/tp_hf_parity.py \
  --model-path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tp-size 4 \
  --max-new-tokens 8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096
```

### Throughput

Single card 1.5B:

- `8565.5063 tok/s`

TP2 1.5B:

- `global_throughput=10442.6781 tok/s`

Single card 7B:

- `2850.9992 tok/s`

TP4 7B:

- `global_throughput=5063.4811 tok/s`

## Current Status

- merge conflicts resolved
- build passing
- core/engine/offline/online tests passing
- HF parity passing for TP2 and TP4
- single-card throughput preserved
- TP throughput restored

## Remaining Work Before User Commit

1. user review
2. optional broader TP matrix rerun if desired
3. final merge commit by user
