#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${1:-}"

echo "[stage1] root=$ROOT_DIR"
if [[ -n "$MODEL_PATH" ]]; then
  echo "[stage1] model=$MODEL_PATH"
fi

PYTHONPATH=python python ./test/test_offline.py
PYTHONPATH=python python ./test/test_llm_entrypoint.py
PYTHONPATH=python python ./test/test_engine_model_registry.py
PYTHONPATH=python python ./test/test_engine_state_machine.py

if [[ -n "$MODEL_PATH" ]]; then
  PYTHONPATH=python python ./test/test_offline_parity.py --model "$MODEL_PATH" --case all --test
fi

echo "[stage1] offline tests passed"
