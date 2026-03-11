from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NovaInfer TP=2 bench in two processes.")
    parser.add_argument("--model-path", required=True, type=Path)
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
        help="Physical GPU list exported to CUDA_VISIBLE_DEVICES, e.g. 4,5",
    )
    parser.add_argument(
        "--tensor-parallel-device-ids",
        default="",
        type=str,
        help="Optional logical ids under CUDA_VISIBLE_DEVICES, e.g. 0,1. Empty means auto-select.",
    )
    parser.add_argument(
        "--init-method",
        default="file:///tmp/llaisys_tp_nccl_bench.id",
        type=str,
        help="TP init method shared by both ranks.",
    )
    args = parser.parse_args()

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
        "2",
        "--tensor-parallel-device-ids",
        str(args.tensor_parallel_device_ids),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "cudnn")
    env.setdefault("LLAISYS_TP_SINGLE_PROCESS", "0")
    env["LLAISYS_TP_INIT_METHOD"] = str(args.init_method)

    root = Path(tempfile.mkdtemp(prefix="tp2_bench_"))
    r0_log = root / "rank0.log"
    r1_log = root / "rank1.log"
    r0_json = root / "rank0.json"
    r1_json = root / "rank1.json"

    cmd0 = common + ["--tp-rank", "0", "--tp-local-rank", "0", "--result-json", str(r0_json)]
    cmd1 = common + ["--tp-rank", "1", "--tp-local-rank", "1", "--result-json", str(r1_json)]

    with r0_log.open("w", encoding="utf-8") as f0, r1_log.open("w", encoding="utf-8") as f1:
        p0 = subprocess.Popen(cmd0, stdout=f0, stderr=subprocess.STDOUT, env=env)
        p1 = subprocess.Popen(cmd1, stdout=f1, stderr=subprocess.STDOUT, env=env)
        rc0 = p0.wait()
        rc1 = p1.wait()

    print(f"[tp2_bench] rank0_rc={rc0} rank1_rc={rc1}")
    print(f"[tp2_bench] log_dir={root}")
    print(f"[tp2_bench] rank0_log={r0_log}")
    print(f"[tp2_bench] rank1_log={r1_log}")

    if rc0 != 0 or rc1 != 0:
        print("[tp2_bench] rank0 tail:")
        print(_tail(r0_log))
        print("[tp2_bench] rank1 tail:")
        print(_tail(r1_log))
        return 1

    try:
        row0 = json.loads(r0_json.read_text(encoding="utf-8"))
        row1 = json.loads(r1_json.read_text(encoding="utf-8"))
    except Exception:
        print("[tp2_bench] failed to parse result json")
        print("[tp2_bench] rank0 tail:")
        print(_tail(r0_log))
        print("[tp2_bench] rank1 tail:")
        print(_tail(r1_log))
        return 1

    print(
        "[tp2_bench] "
        f"rank0 throughput_actual={row0.get('actual_tokens_per_sec', 0.0):.4f} tok/s "
        f"run_seconds={row0.get('run_seconds', row0.get('seconds', 0.0)):.4f}"
    )
    print(
        "[tp2_bench] "
        f"rank1 throughput_actual={row1.get('actual_tokens_per_sec', 0.0):.4f} tok/s "
        f"run_seconds={row1.get('run_seconds', row1.get('seconds', 0.0)):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
