#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-python}"
MODEL_PATH="${2:-${MODEL_PATH:-}}"
PARITY_MODE="${RUN_PARITY:-auto}" # auto|1|0
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

echo "[stage0] root=${ROOT_DIR}"
echo "[stage0] mode=${MODE}"
if [[ -n "${MODEL_PATH}" ]]; then
  echo "[stage0] model=${MODEL_PATH}"
fi
echo "[stage0] parity_mode=${PARITY_MODE}"

run_parity_if_needed() {
  if [[ "${PARITY_MODE}" == "0" ]]; then
    echo "[stage0] parity skipped (RUN_PARITY=0)"
    return 0
  fi

  if [[ "${PARITY_MODE}" == "1" ]]; then
    if [[ -z "${MODEL_PATH}" ]]; then
      echo "[stage0] RUN_PARITY=1 requires model path"
      return 3
    fi
    echo "[stage0] parity case=all"
    PYTHONPATH=python python ./test/test_core_parity.py --model "${MODEL_PATH}"
    return 0
  fi

  # auto mode: run parity only when model path is provided.
  if [[ -n "${MODEL_PATH}" ]]; then
    echo "[stage0] parity case=all (auto)"
    PYTHONPATH=python python ./test/test_core_parity.py --model "${MODEL_PATH}"
  else
    echo "[stage0] parity skipped (auto without model path)"
  fi
}

if [[ "${MODE}" == "python" ]]; then
  PYTHONPATH=python python ./test/test_core_model_api.py
  PYTHONPATH=python python ./test/test_core_output_api.py
  PYTHONPATH=python python ./test/test_core_decode_batch.py
  PYTHONPATH=python python ./test/test_kv_cache.py
  PYTHONPATH=python python ./test/test_model_registry.py
  PYTHONPATH=python python ./test/test_qwen2_adapter.py
  if [[ -n "${MODEL_PATH}" ]]; then
    PYTHONPATH=python python ./test/test_infer.py --model "${MODEL_PATH}" --test
  else
    PYTHONPATH=python python ./test/test_infer.py --test
  fi
  run_parity_if_needed
elif [[ "${MODE}" == "pytest" ]]; then
  if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[stage0] pytest not installed, install first: python -m pip install pytest"
    exit 1
  fi
  PYTHONPATH=python python -m pytest -q \
    test/test_core_model_api.py \
    test/test_core_output_api.py \
    test/test_core_decode_batch.py \
    test/test_kv_cache.py \
    test/test_model_registry.py \
    test/test_qwen2_adapter.py
  if [[ -n "${MODEL_PATH}" ]]; then
    PYTHONPATH=python python ./test/test_infer.py --model "${MODEL_PATH}" --test
  else
    PYTHONPATH=python python ./test/test_infer.py --test
  fi
  run_parity_if_needed
else
  echo "usage: $0 [python|pytest] [model_path]"
  echo "or: MODEL_PATH=/path/to/model $0 [python|pytest]"
  echo "parity: RUN_PARITY=auto|1|0 (default: auto; auto runs parity when model_path is set)"
  echo "single case: PYTHONPATH=python python ./test/test_core_parity.py --model /path/to/model --case single"
  echo "multi case : PYTHONPATH=python python ./test/test_core_parity.py --model /path/to/model --case multi"
  exit 2
fi

echo "[stage0] all tests passed"
