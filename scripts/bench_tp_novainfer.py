from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _default_init_method(tag: str) -> str:
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    return f"file://{(tmp_dir / f'llaisys_{tag}_{os.getpid()}.id').resolve()}"


def _tail(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NovaInfer TP bench in N processes.")
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
    args = parser.parse_args()

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
    ]

    init_method = str(args.init_method).strip() or _default_init_method(f"tp{tp_size}_nccl_bench")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "cudnn")
    env.setdefault("LLAISYS_TP_SINGLE_PROCESS", "0")
    env["LLAISYS_TP_INIT_METHOD"] = init_method

    root = Path(tempfile.mkdtemp(prefix=f"tp{tp_size}_bench_"))
    procs: list[subprocess.Popen] = []
    logs: list[Path] = []
    jsons: list[Path] = []
    for rank in range(tp_size):
        r_log = root / f"rank{rank}.log"
        r_json = root / f"rank{rank}.json"
        cmd = common + [
            "--tp-rank",
            str(rank),
            "--tp-local-rank",
            str(rank),
            "--result-json",
            str(r_json),
        ]
        fh = r_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        proc._llaisys_log_fh = fh  # type: ignore[attr-defined]
        procs.append(proc)
        logs.append(r_log)
        jsons.append(r_json)

    rcs: list[int] = []
    for proc in procs:
        rc = int(proc.wait())
        rcs.append(rc)
        try:
            proc._llaisys_log_fh.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"[tp_bench] tp_size={tp_size} rcs={rcs} log_dir={root}")
    if any(rc != 0 for rc in rcs):
        for rank, r_log in enumerate(logs):
            print(f"[tp_bench] rank={rank} tail:")
            print(_tail(r_log))
        return 1

    rows: list[dict] = []
    for rank, r_json in enumerate(jsons):
        try:
            row = json.loads(r_json.read_text(encoding="utf-8"))
            rows.append(row)
        except Exception:
            print(f"[tp_bench] rank={rank} failed to parse result json")
            print(_tail(logs[rank]))
            return 1

    tputs = [float(r.get("actual_tokens_per_sec", 0.0)) for r in rows]
    run_secs = [float(r.get("run_seconds", r.get("seconds", 0.0))) for r in rows]
    for rank, row in enumerate(rows):
        print(
            "[tp_bench] "
            f"rank={rank} throughput_actual={float(row.get('actual_tokens_per_sec', 0.0)):.4f} tok/s "
            f"run_seconds={float(row.get('run_seconds', row.get('seconds', 0.0))):.4f}"
        )
    print(
        "[tp_bench] summary "
        f"throughput_actual_avg={sum(tputs)/max(1, len(tputs)):.4f} "
        f"throughput_actual_min={min(tputs):.4f} "
        f"throughput_actual_max={max(tputs):.4f} "
        f"run_seconds_avg={sum(run_secs)/max(1, len(run_secs)):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
