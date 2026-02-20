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

## Directory Layout

- `test/core/`: core C API + model/KV behavior tests
- `test/engine/`: scheduler/executor/state-machine/block-manager tests
- `test/offline/`: offline entrypoint/LLM interface tests
- `test/online/`: online server/API streaming/concurrency tests
- `test/parity/`: HF/reference parity tests
- `test/ops/`: low-level op tests
- `test/utils/`: shared testing helpers (e.g., batch builders)

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

## Naming Convention

- File name: `test_<scope>_<subject>.py`
- Test case name: `test_<given>_<when>_<then>`
- Scopes:
  - `core`: C API/model decode/KV semantics
  - `engine`: scheduler/executor/state machine
  - `offline` / `online`: entrypoints and server contracts
  - `parity`: HF/reference parity checks
  - `ops`: operator-level behavior

## Batch Builder Convention

- For tests that call `llaisysModelDecode` or need BLOCK/SLOT batch assembly:
  - Use `test/utils/batch_builders.py::build_decode_batch`
  - Do not hand-write `LlaisysBatch` construction in new tests.
- Shared state for BLOCK cross-step mapping:
  - Use `BlockBatchState` when one logical request spans multiple decode calls.
- Rationale:
  - Keep SLOT/BLOCK semantics consistent across tests.
  - Avoid duplicated ctypes pointer-lifetime handling in every file.

## Runtime Sequence (Current)

### 1) Offline `LLM.generate(...)` (batch path)

```mermaid
sequenceDiagram
    participant User
    participant LLM as entrypoints.LLM
    participant Client as EngineClient
    participant Engine as LLMEngine
    participant Sched as RequestScheduler
    participant Exec as Executor/Worker

    User->>LLM: generate(inputs, sampling_params)
    LLM->>LLM: normalize inputs -> list[list[int]]
    LLM->>LLM: normalize sampling params per request
    loop each request
        LLM->>Client: submit(token_ids, params)
        Client->>Engine: submit(...)
        Engine->>Engine: create RequestState + Sequence
        Engine->>Sched: add(sequence to waiting queue)
    end

    loop until all requests collected
        LLM->>Client: step()
        Client->>Engine: step()
        Engine->>Sched: schedule()
        Engine->>Exec: execute_scheduler_step(...)
        Exec-->>Engine: sampled tokens
        LLM->>Client: collect(req_id)
        Client->>Engine: collect(req_id)
        Engine-->>LLM: GenerationOutput or None
    end

    LLM->>LLM: pack outputs (completion-only token_ids)
    LLM-->>User: list[dict]
```

Notes:
- `LLM.generate` no longer supports legacy single-token-list input (`list[int]`).
- Accepted forms are `str`, `list[str]`, and `list[list[int]]`.
- Return shape is always structured batch output (`list[dict]`), even for one request.

### 2) `LLMEngine.step()` scheduling branch

```mermaid
flowchart TD
    A["engine step"] --> B["scheduler schedule"]
    B -->|"none"| C["scan waiting queue for impossible requests"]
    C --> D["mark finished aborted if over budget"]
    D --> R["return completions"]

    B -->|"prefill"| E["validate runtime state"]
    B -->|"decode"| E
    E --> F["executor run scheduled step"]
    F --> G["append sampled token"]
    G --> H{"has finish reason"}
    H -->|"yes"| I["complete request and release resources"]
    H -->|"no"| J["keep request running"]
    I --> R
    J --> R
```

Key behavior aligned to current implementation:
- Prefill admission is attempted first; decode is fallback when no new prefill can be admitted.
- Request lifecycle is managed in `RequestState` (`waiting -> running -> finished_*`).
- Scheduler data (`waiting/running`) and block/KV bookkeeping live under `RequestScheduler` + `BlockManager`.

### 3) Online `OpenAIServer` request path

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant Server as OpenAIServer
    participant Async as AsyncLLMEngine
    participant Bg as Async loop thread
    participant Engine as LLMEngine

    Client->>Server: chat.completions request
    Server->>Server: messages to prompt token ids
    Server->>Async: generate or stream
    Async->>Async: enqueue submit command
    Bg->>Engine: submit(inputs, params)
    Bg->>Engine: step repeatedly
    Engine-->>Bg: outputs or inflight tokens

    alt non-streaming
        Async->>Async: collect until final output exists
        Async-->>Server: GenerationOutput
        Server-->>Client: OpenAI-style JSON response
    else streaming
        Bg-->>Async: emit token chunks to stream queue
        Async-->>Server: StreamChunk iterator
        Server-->>Client: OpenAI-style SSE chunks
    end
```

Online notes:
- `AsyncLLMEngine` uses a single background loop thread to mutate `LLMEngine` state.
- API thread only sends commands (`submit/collect/cancel/watch_stream`) through queues.
- This avoids concurrent direct mutation of scheduler/runtime state.
