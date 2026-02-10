# Test Layout And Execution

This project now uses one unified entrypoint:

```bash
python scripts/run_tests.py --suite all
```

## Test Layers

- `stage0` (Core): C API/model/kernel correctness, KV semantics, adapter/model registry, infer baseline.
- `stage1` (Offline Engine): scheduler/executor/state machine, LLM entrypoint behavior, offline contract.
- `stage2` (Online + Sampling): sampling chain behavior, online server streaming/cancel/concurrency.
- `parity` (optional): compare against HF reference model outputs, requires local model path.

## Standard Commands

- Run all fast suites (no parity by default if no model path):
```bash
python scripts/run_tests.py --suite all
```

- Run stage0 only:
```bash
python scripts/run_tests.py --suite stage0
```

- Run stage1 only:
```bash
python scripts/run_tests.py --suite stage1
```

- Run stage2 only:
```bash
python scripts/run_tests.py --suite stage2
```

- Run with parity enabled:
```bash
python scripts/run_tests.py --suite all --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B --run-parity auto
```

- Run HF-dependent infer check in stage0:
```bash
python scripts/run_tests.py --suite stage0 --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B --run-hf always
```

## Unified Entry

Use `scripts/run_tests.py` as the only orchestrator entrypoint.

## Design Principles

- One command path for CI/local runs.
- Fast tests and expensive parity tests separated by policy (`--run-parity`).
- Stage naming follows architecture milestones (`stage0`, `stage1`, ...).
- Existing tests stay intact; organization and execution are standardized first.
