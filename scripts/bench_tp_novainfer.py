from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _default_init_method(tag: str) -> str:
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return f"file://{(tmp_dir / f'llaisys_{tag}_{os.getpid()}.id').resolve()}"


def _tail(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def _summarize_paths(raw: str, limit: int = 4) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for item in str(raw or "").split(":"):
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        parts.append(item)
    if not parts:
        return "<unset>"
    head = parts[: max(1, int(limit))]
    suffix = "" if len(parts) <= len(head) else f" ... (+{len(parts) - len(head)} more)"
    return " | ".join(head) + suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NovaInfer TP bench via built-in mp executor.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--tp-size", default=2, type=int)
    parser.add_argument("--num-seqs", default=256, type=int)
    parser.add_argument("--min-input-len", default=100, type=int)
    parser.add_argument("--max-input-len", default=1024, type=int)
    parser.add_argument("--min-output-len", default=100, type=int)
    parser.add_argument("--max-output-len", default=1024, type=int)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-num-seqs", default=256, type=int)
    parser.add_argument("--max-num-batched-tokens", default=16384, type=int)
    parser.add_argument("--kv-cache-memory-utilization", default=0.9, type=float)
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1",
        type=str,
        help="Physical GPU list exported to CUDA_VISIBLE_DEVICES, e.g. 4,5,6,7",
    )
    parser.add_argument(
        "--tensor-parallel-device-ids",
        default="",
        type=str,
        help="Optional logical ids under CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3. Empty means auto-select.",
    )
    parser.add_argument(
        "--init-method",
        default="",
        type=str,
        help="TP init method shared by all ranks.",
    )
    parser.add_argument(
        "--result-json",
        default="",
        type=str,
        help="Optional path to write structured summary JSON.",
    )
    parser.add_argument(
        "--distributed-executor-backend",
        default="mp",
        type=str,
        help="TP bench now requires mp. This flag is kept only for CLI compatibility.",
    )
    args = parser.parse_args()
    if str(args.distributed_executor_backend).strip().lower() != "mp":
        raise ValueError("bench_tp_novainfer.py only supports distributed_executor_backend=mp")

    tp_size = max(1, int(args.tp_size))
    py = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path("python")
    bench = Path(__file__).resolve().with_name("bench_compare_vllm.py")

    common = [
        str(py),
        str(bench),
        "--model-path",
        str(args.model_path),
        "--backend",
        "novainfer",
        "--num-seqs",
        str(args.num_seqs),
        "--min-input-len",
        str(args.min_input_len),
        "--max-input-len",
        str(args.max_input_len),
        "--min-output-len",
        str(args.min_output_len),
        "--max-output-len",
        str(args.max_output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--seed",
        str(args.seed),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--kv-cache-memory-utilization",
        str(args.kv_cache_memory_utilization),
        "--tensor-parallel-size",
        str(tp_size),
        "--tensor-parallel-device-ids",
        str(args.tensor_parallel_device_ids),
        "--distributed-executor-backend",
        str(args.distributed_executor_backend),
    ]

    init_method = str(args.init_method).strip() or _default_init_method(f"tp{tp_size}_nccl_bench")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "cudnn")
    env.setdefault("LLAISYS_TP_SINGLE_PROCESS", "0")
    env["LLAISYS_TP_INIT_METHOD"] = init_method
    print(
        "[tp_bench] runtime_env "
        f"backend={env.get('LLAISYS_CUDA_PAGED_ATTN_BACKEND', '<unset>')} "
        f"cuda_visible_devices={env.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"cudnn_home={env.get('CUDNN_HOME', '<unset>')} "
        f"ld_library_path={_summarize_paths(env.get('LD_LIBRARY_PATH', ''))}"
    )

    root = Path(tempfile.mkdtemp(prefix=f"tp{tp_size}_bench_"))
    rows: list[dict] = []
    run_log = root / "mp.log"
    run_json = root / "mp.json"
    cmd = common + [
        "--result-json",
        str(run_json),
    ]
    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        rc = int(proc.wait())
    print(f"[tp_bench] tp_size={tp_size} backend=mp rcs={[rc]} log_dir={root}")
    if rc != 0:
        print("[tp_bench] rank=0 tail:")
        print(_tail(run_log))
        return 1
    try:
        rows.append(json.loads(run_json.read_text(encoding="utf-8")))
    except Exception:
        print("[tp_bench] failed to parse result json")
        print(_tail(run_log))
        return 1

    tputs = [float(r.get("actual_tokens_per_sec", 0.0)) for r in rows]
    run_secs = [float(r.get("run_seconds", r.get("seconds", 0.0))) for r in rows]
    total_tokens = [int(r.get("actual_total_tokens", r.get("total_tokens", 0))) for r in rows]
    global_tokens = total_tokens[0] if total_tokens else 0
    global_run_seconds = max(run_secs) if run_secs else 0.0
    global_throughput = (float(global_tokens) / float(global_run_seconds)) if global_run_seconds > 0.0 else 0.0
    for rank, row in enumerate(rows):
        print(
            "[tp_bench] "
            f"rank={rank} throughput_actual={float(row.get('actual_tokens_per_sec', 0.0)):.4f} tok/s "
            f"run_seconds={float(row.get('run_seconds', row.get('seconds', 0.0))):.4f}"
        )
    summary = {
        "tp_size": tp_size,
        "global_tokens": int(global_tokens),
        "global_run_seconds": float(global_run_seconds),
        "global_throughput": float(global_throughput),
        "throughput_actual_avg": float(sum(tputs) / max(1, len(tputs))),
        "throughput_actual_min": float(min(tputs)),
        "throughput_actual_max": float(max(tputs)),
        "run_seconds_avg": float(sum(run_secs) / max(1, len(run_secs))),
        "rows": rows,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "backend": env.get("LLAISYS_CUDA_PAGED_ATTN_BACKEND", ""),
    }
    print(
        "[tp_bench] summary "
        f"global_tokens={summary['global_tokens']} "
        f"global_run_seconds={summary['global_run_seconds']:.4f} "
        f"global_throughput={summary['global_throughput']:.4f} "
        f"throughput_actual_avg={summary['throughput_actual_avg']:.4f} "
        f"throughput_actual_min={summary['throughput_actual_min']:.4f} "
        f"throughput_actual_max={summary['throughput_actual_max']:.4f} "
        f"run_seconds_avg={summary['run_seconds_avg']:.4f}"
    )
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
