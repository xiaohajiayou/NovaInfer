#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STAGE0_PYTEST_FILES = [
    "test/test_core_model_api.py",
    "test/test_core_output_api.py",
    "test/test_core_decode_batch.py",
    "test/test_kv_cache.py",
    "test/test_model_registry.py",
    "test/test_qwen2_adapter.py",
    "test/test_infer.py",
    "test/test_core_parity.py",
]

STAGE1_PYTEST_FILES = [
    "test/test_offline.py",
    "test/test_llm_entrypoint.py",
    "test/test_engine_model_registry.py",
    "test/test_engine_state_machine.py",
    "test/test_offline_parity.py",
]

STAGE2_PYTEST_FILES = [
    "test/test_sampling.py",
    "test/test_online.py",
    "test/test_online_http.py",
    "test/test_online_stream_isolation.py",
    "test/test_online_real_model_multisession.py",
]


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _run_pytest(files: list[str], env: dict[str, str], model_path: str | None = None) -> None:
    cmd = [PYTHON, "-m", "pytest", "-q", *files]
    if model_path:
        cmd.extend(["--model-path", model_path])
    _run(cmd, env)


def _run_stage0(
    env: dict[str, str],
    model_path: str | None,
    run_parity: str,
    run_hf: str,
) -> None:
    files = list(STAGE0_PYTEST_FILES)
    should_run_hf = (run_hf == "always") or (run_hf == "auto" and bool(model_path))
    should_run_parity = (run_parity == "always") or (run_parity == "auto" and bool(model_path))
    if not should_run_hf:
        files.remove("test/test_infer.py")
        print("[skip] stage0 HF-dependent infer test")
    if not should_run_parity:
        files.remove("test/test_core_parity.py")
        print("[skip] stage0 parity")
    _run_pytest(files, env, model_path=model_path)


def _run_stage1(env: dict[str, str], model_path: str | None, run_parity: str) -> None:
    files = list(STAGE1_PYTEST_FILES)
    should_run_parity = (run_parity == "always") or (run_parity == "auto" and bool(model_path))
    if not should_run_parity:
        files.remove("test/test_offline_parity.py")
        print("[skip] stage1 parity")
    _run_pytest(files, env, model_path=model_path)


def _run_stage2(env: dict[str, str], model_path: str | None) -> None:
    _ = model_path
    _run_pytest(STAGE2_PYTEST_FILES, env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified test runner for NovaInfer")
    parser.add_argument(
        "--suite",
        choices=["stage0", "stage1", "stage2", "all"],
        default="all",
        help="Which suite to run",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", ""),
        help="Local HF model path (required for parity tests when enabled)",
    )
    parser.add_argument(
        "--run-parity",
        choices=["auto", "always", "never"],
        default=os.environ.get("RUN_PARITY", "auto"),
        help="Parity test policy",
    )
    parser.add_argument(
        "--run-hf",
        choices=["auto", "always", "never"],
        default=os.environ.get("RUN_HF", "auto"),
        help="HF-dependent policy for stage0 infer parity-like test",
    )
    args = parser.parse_args()

    model_path = args.model_path.strip() or None

    env = dict(os.environ)
    env["PYTHONPATH"] = f"python{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    print(f"[config] suite={args.suite}")
    print(f"[config] model_path={model_path or '<none>'}")
    print(f"[config] run_parity={args.run_parity}")
    print(f"[config] run_hf={args.run_hf}")

    if args.suite in ("stage0", "all"):
        _run_stage0(
            env=env,
            model_path=model_path,
            run_parity=args.run_parity,
            run_hf=args.run_hf,
        )
    if args.suite in ("stage1", "all"):
        _run_stage1(env=env, model_path=model_path, run_parity=args.run_parity)
    if args.suite in ("stage2", "all"):
        _run_stage2(env=env, model_path=model_path)

    print("[ok] all selected test suites passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
