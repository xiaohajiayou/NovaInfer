#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class BenchCase:
    name: str
    num_seqs: int
    min_input_len: int
    max_input_len: int
    min_output_len: int
    max_output_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_memory_utilization: float
    vllm_fair_mode: bool


CASES: dict[str, BenchCase] = {
    "smallseqs_tightbatch": BenchCase(
        name="smallseqs_tightbatch",
        num_seqs=20,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_num_seqs=20,
        max_num_batched_tokens=4096,
        kv_cache_memory_utilization=0.7,
        vllm_fair_mode=True,
    ),
    "smallseqs_smallbatch": BenchCase(
        name="smallseqs_smallbatch",
        num_seqs=20,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_num_seqs=20,
        max_num_batched_tokens=8192,
        kv_cache_memory_utilization=0.7,
        vllm_fair_mode=True,
    ),
    "largeseqs_smallbatch": BenchCase(
        name="largeseqs_smallbatch",
        num_seqs=256,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        kv_cache_memory_utilization=0.5,
        vllm_fair_mode=True,
    ),
    "largeseqs_largebatch": BenchCase(
        name="largeseqs_largebatch",
        num_seqs=256,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        kv_cache_memory_utilization=0.5,
        vllm_fair_mode=True,
    ),
}


def _run_one_backend(
    *,
    root: Path,
    model_path: Path,
    backend: str,
    case: BenchCase,
    seed: int,
    max_model_len: int,
    env: dict[str, str],
    vllm_gpu_memory_utilization: float,
) -> tuple[dict, str]:
    script = root / "scripts" / "bench_compare_vllm.py"
    with tempfile.NamedTemporaryFile(prefix=f"perf_{backend}_{case.name}_", suffix=".json", delete=False) as tf:
        result_json = Path(tf.name)
    cmd = [
        sys.executable,
        str(script),
        "--model-path",
        str(model_path),
        "--backend",
        backend,
        "--num-seqs",
        str(case.num_seqs),
        "--min-input-len",
        str(case.min_input_len),
        "--max-input-len",
        str(case.max_input_len),
        "--min-output-len",
        str(case.min_output_len),
        "--max-output-len",
        str(case.max_output_len),
        "--max-model-len",
        str(max_model_len),
        "--seed",
        str(seed),
        "--max-num-seqs",
        str(case.max_num_seqs),
        "--max-num-batched-tokens",
        str(case.max_num_batched_tokens),
        "--kv-cache-memory-utilization",
        str(case.kv_cache_memory_utilization),
        "--result-json",
        str(result_json),
    ]
    if backend == "vllm":
        cmd.extend(["--vllm-gpu-memory-utilization", str(vllm_gpu_memory_utilization)])
    if case.vllm_fair_mode:
        cmd.append("--vllm-fair-mode")

    p = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True)
    stdout = p.stdout or ""
    stderr = p.stderr or ""
    if p.returncode != 0:
        raise RuntimeError(
            f"bench failed backend={backend} case={case.name} seed={seed}\n"
            f"cmd={' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    row = json.loads(result_json.read_text(encoding="utf-8"))
    try:
        result_json.unlink(missing_ok=True)
    except Exception:
        pass
    return row, (stdout + ("\n" + stderr if stderr else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible large/small NovaInfer-vs-vLLM perf experiments.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=Path("perf_results.jsonl"))
    parser.add_argument("--output-log-dir", type=Path, default=Path("perf_logs"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "smallseqs_tightbatch",
            "smallseqs_smallbatch",
            "largeseqs_smallbatch",
            "largeseqs_largebatch",
        ],
        choices=sorted(CASES.keys()),
        help="Subset of cases to run",
    )
    parser.add_argument(
        "--backend-order",
        nargs="+",
        default=["novainfer", "vllm"],
        choices=["novainfer", "vllm"],
        help="Order of backend execution in each repeat",
    )
    parser.add_argument(
        "--cudnn-backend",
        default="cudnn",
        choices=["cudnn", "native"],
        help="LLAISYS_CUDA_PAGED_ATTN_BACKEND value for NovaInfer runs",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.output_log_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LLAISYS_CUDA_PAGED_ATTN_BACKEND"] = args.cudnn_backend

    rows: list[dict] = []
    t0 = time.time()
    for case_name in args.cases:
        case = CASES[case_name]
        for rep in range(args.repeats):
            seed = int(args.seed_base) + rep
            for backend in args.backend_order:
                row, raw_log = _run_one_backend(
                    root=root,
                    model_path=args.model_path,
                    backend=backend,
                    case=case,
                    seed=seed,
                    max_model_len=int(args.max_model_len),
                    env=env,
                    vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
                )
                rec = {
                    "ts": int(time.time()),
                    "case": case.name,
                    "repeat": rep,
                    "seed": seed,
                    "backend": backend,
                    "llaisys_paged_attn_backend": args.cudnn_backend,
                    "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
                    "run_config": asdict(case),
                    "result": row,
                }
                rows.append(rec)
                log_name = f"{case.name}_rep{rep}_{backend}.log"
                (args.output_log_dir / log_name).write_text(raw_log, encoding="utf-8")
                print(
                    f"[perf] case={case.name} rep={rep} backend={backend} "
                    f"throughput={float(row.get('actual_tokens_per_sec', 0.0)):.2f} tok/s "
                    f"time={float(row.get('seconds', 0.0)):.3f}s"
                )

    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[perf] done runs={len(rows)} elapsed={time.time() - t0:.1f}s "
        f"jsonl={args.output_jsonl} logs={args.output_log_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
