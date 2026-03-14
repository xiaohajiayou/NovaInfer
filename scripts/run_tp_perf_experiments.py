#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TpCase:
    name: str
    model_alias: str
    model_path: str
    tp_size: int
    cuda_visible_devices: str
    num_seqs: int
    min_input_len: int
    max_input_len: int
    min_output_len: int
    max_output_len: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_memory_utilization: float


CASES: dict[str, TpCase] = {
    "qwen15b_small_tp1": TpCase(
        name="qwen15b_small_tp1",
        model_alias="qwen15b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tp_size=1,
        cuda_visible_devices="5",
        num_seqs=64,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen15b_small_tp2": TpCase(
        name="qwen15b_small_tp2",
        model_alias="qwen15b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tp_size=2,
        cuda_visible_devices="5,6",
        num_seqs=64,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen15b_large_tp1": TpCase(
        name="qwen15b_large_tp1",
        model_alias="qwen15b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tp_size=1,
        cuda_visible_devices="5",
        num_seqs=256,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen15b_large_tp2": TpCase(
        name="qwen15b_large_tp2",
        model_alias="qwen15b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tp_size=2,
        cuda_visible_devices="5,6",
        num_seqs=256,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen7b_small_tp1": TpCase(
        name="qwen7b_small_tp1",
        model_alias="qwen7b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tp_size=1,
        cuda_visible_devices="6",
        num_seqs=64,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen7b_small_tp2": TpCase(
        name="qwen7b_small_tp2",
        model_alias="qwen7b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tp_size=2,
        cuda_visible_devices="5,6",
        num_seqs=64,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen7b_small_tp4": TpCase(
        name="qwen7b_small_tp4",
        model_alias="qwen7b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tp_size=4,
        cuda_visible_devices="1,2,5,6",
        num_seqs=64,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen7b_large_tp1": TpCase(
        name="qwen7b_large_tp1",
        model_alias="qwen7b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tp_size=1,
        cuda_visible_devices="6",
        num_seqs=128,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen7b_large_tp2": TpCase(
        name="qwen7b_large_tp2",
        model_alias="qwen7b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tp_size=2,
        cuda_visible_devices="5,6",
        num_seqs=128,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        kv_cache_memory_utilization=0.9,
    ),
    "qwen7b_large_tp4": TpCase(
        name="qwen7b_large_tp4",
        model_alias="qwen7b",
        model_path="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tp_size=4,
        cuda_visible_devices="1,2,5,6",
        num_seqs=128,
        min_input_len=100,
        max_input_len=1024,
        min_output_len=100,
        max_output_len=1024,
        max_model_len=4096,
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        kv_cache_memory_utilization=0.9,
    ),
}


def _run_case(root: Path, case: TpCase, seed: int) -> tuple[dict, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = case.cuda_visible_devices
    env.setdefault("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "cudnn")

    with tempfile.NamedTemporaryFile(prefix=f"tp_perf_{case.name}_", suffix=".json", delete=False) as tf:
        result_json = Path(tf.name)

    if case.tp_size == 1:
        script = root / "scripts" / "bench_compare_vllm.py"
        cmd = [
            sys.executable,
            str(script),
            "--model-path",
            case.model_path,
            "--backend",
            "novainfer",
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
            str(case.max_model_len),
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
    else:
        script = root / "scripts" / "bench_tp_novainfer.py"
        cmd = [
            sys.executable,
            str(script),
            "--model-path",
            case.model_path,
            "--tp-size",
            str(case.tp_size),
            "--cuda-visible-devices",
            case.cuda_visible_devices,
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
            str(case.max_model_len),
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

    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"tp perf case failed: case={case.name} seed={seed}\ncmd={' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    row = json.loads(result_json.read_text(encoding="utf-8"))
    result_json.unlink(missing_ok=True)
    return row, (stdout + ("\n" + stderr if stderr else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TP performance experiment matrix.")
    parser.add_argument("--output-jsonl", type=Path, default=Path("tp_perf_results.jsonl"))
    parser.add_argument("--output-log-dir", type=Path, default=Path("tp_perf_logs"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--cases", nargs="+", default=list(CASES.keys()), choices=sorted(CASES.keys()))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.output_log_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for case_name in args.cases:
        case = CASES[case_name]
        for rep in range(args.repeats):
            seed = int(args.seed_base) + rep
            row, raw_log = _run_case(root, case, seed)
            rec = {
                "ts": int(time.time()),
                "case": case.name,
                "repeat": rep,
                "seed": seed,
                "model_alias": case.model_alias,
                "tp_size": int(case.tp_size),
                "run_config": asdict(case),
                "result": row,
            }
            rows.append(rec)
            (args.output_log_dir / f"{case.name}_rep{rep}.log").write_text(raw_log, encoding="utf-8")
            throughput = float(row.get("global_throughput", row.get("actual_tokens_per_sec", 0.0)) or 0.0)
            print(f"[tp_perf] case={case.name} rep={rep} throughput={throughput:.2f} tok/s")

    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[tp_perf] done runs={len(rows)} elapsed={time.time() - t0:.1f}s jsonl={args.output_jsonl} logs={args.output_log_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
