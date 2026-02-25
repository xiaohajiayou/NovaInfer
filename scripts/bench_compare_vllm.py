from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _build_dataset(
    seed: int,
    num_seqs: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    max_model_len: int,
) -> tuple[list[list[int]], list[int]]:
    rng = random.Random(seed)
    prompts: list[list[int]] = []
    out_lens: list[int] = []
    for _ in range(num_seqs):
        plen = rng.randint(min_input_len, max_input_len)
        plen = min(plen, max_model_len - 1)
        prompt = [rng.randint(0, 10000) for _ in range(max(1, plen))]
        want = rng.randint(min_output_len, max_output_len)
        remain = max(1, max_model_len - len(prompt))
        out_len = min(want, remain)
        prompts.append(prompt)
        out_lens.append(out_len)
    return prompts, out_lens


def _run_novainfer(
    model_path: Path,
    prompts: list[list[int]],
    out_lens: list[int],
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    kv_cache_capacity_mode: str,
    kv_cache_capacity_tokens: int,
    kv_cache_memory_utilization: float,
) -> dict:
    from llaisys.entrypoints.llm import LLM
    from llaisys.engine.types import SamplingParams
    from llaisys.libllaisys import DeviceType
    from llaisys.libllaisys.model import KvCacheLayout

    print(f"[bench] init backend=novainfer model={model_path}")
    t_total0 = time.perf_counter()
    t_init0 = t_total0

    sps = [
        SamplingParams(
            temperature=0.6,
            top_k=1,
            top_p=1.0,
            max_new_tokens=out_len,
            ignore_eos=True,
        )
        for out_len in out_lens
    ]
    expected_total_tokens = int(sum(out_lens))

    llm = LLM(
        model=model_path,
        model_type="qwen2",
        device=DeviceType.NVIDIA,
        kv_cache_layout=KvCacheLayout.BLOCK,
        kv_cache_block_size=16,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        kv_cache_capacity_tokens=(
            None if kv_cache_capacity_mode == "auto" else int(kv_cache_capacity_tokens)
        ),
        kv_cache_auto_capacity=(kv_cache_capacity_mode == "auto"),
        kv_cache_memory_utilization=float(kv_cache_memory_utilization),
    )
    t_init1 = time.perf_counter()
    t_warmup0 = t_init1
    t_warmup1 = t_warmup0
    t_run0 = t_warmup0
    t_run1 = t_run0
    kv_stats = {}
    try:
        print("[bench] warmup start backend=novainfer")
        llm.generate(["Benchmark: "], SamplingParams(max_new_tokens=1))
        kv_stats = llm.kv_cache_stats()
        if kv_stats:
            print(f"[bench] kv_stats_after_warmup backend=novainfer stats={kv_stats}")
        print("[bench] warmup done backend=novainfer")
        t_warmup1 = time.perf_counter()

        print("[bench] run start backend=novainfer")
        t_run0 = time.perf_counter()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        t_run1 = time.perf_counter()
        kv_stats = llm.kv_cache_stats()
        print(f"[bench] run done backend=novainfer kv_stats={kv_stats}")
    finally:
        print("[bench] close backend=novainfer")
        llm.close()
    t_total1 = time.perf_counter()
    init_dt = max(1e-9, t_init1 - t_init0)
    warmup_dt = max(1e-9, t_warmup1 - t_warmup0)
    run_dt = max(1e-9, t_run1 - t_run0)
    total_dt = max(1e-9, t_total1 - t_total0)
    print(
        "[bench] finish backend=novainfer "
        f"completion_tokens_expected={expected_total_tokens} "
        f"init_seconds={init_dt:.4f} warmup_seconds={warmup_dt:.4f} "
        f"run_seconds={run_dt:.4f} total_seconds={total_dt:.4f}"
    )
    actual_total_tokens = 0
    for out in outs if isinstance(outs, list) else []:
        if isinstance(out, dict):
            tids = out.get("token_ids")
            if isinstance(tids, list):
                actual_total_tokens += len(tids)

    return {
        "backend": "novainfer",
        "total_tokens": expected_total_tokens,
        "actual_total_tokens": int(actual_total_tokens),
        "init_seconds": init_dt,
        "warmup_seconds": warmup_dt,
        "run_seconds": run_dt,
        "seconds": total_dt,
        "tokens_per_sec": expected_total_tokens / run_dt,
        "actual_tokens_per_sec": max(0.0, float(actual_total_tokens)) / run_dt,
        "kv_cache_stats": kv_stats,
    }


def _run_vllm(
    model_path: Path,
    prompts: list[list[int]],
    out_lens: list[int],
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    vllm_fair_mode: bool,
) -> dict:
    from vllm import LLM, SamplingParams
    if vllm_fair_mode:
        print(
            "[bench] note vllm_fair_mode: this vLLM build may still use FLASH_ATTN "
            "for decoder attention on CUDA."
        )

    sps = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=out_len,
        )
        for out_len in out_lens
    ]
    expected_total_tokens = int(sum(out_lens))

    llm_kwargs = {
        "model": str(model_path),
        "max_model_len": int(max_model_len),
        "trust_remote_code": True,
        # default mode keeps vLLM normal optimized path.
        "enforce_eager": False,
        "max_num_seqs": int(max_num_seqs),
        "max_num_batched_tokens": int(max_num_batched_tokens),
    }

    if vllm_fair_mode:
        # Fair mode: disable vLLM-only runtime optimizations not present in NovaInfer yet.
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["enable_chunked_prefill"] = False
        # Keep prefix caching ON since NovaInfer BLOCK path supports prefix caching.
        llm_kwargs["enable_prefix_caching"] = True

    print(f"[bench] vllm_kwargs={llm_kwargs}")
    llm = LLM(**llm_kwargs)
    llm.generate(["Benchmark: "], SamplingParams(max_tokens=1))
    token_prompts = [{"prompt_token_ids": [int(t) for t in p]} for p in prompts]
    t0 = time.time()
    outs = llm.generate(token_prompts, sps, use_tqdm=False)
    dt = max(1e-9, time.time() - t0)
    actual_total_tokens = 0
    for out in outs if isinstance(outs, list) else []:
        try:
            # vLLM RequestOutput -> outputs: list[CompletionOutput], each has token_ids
            seq_outs = getattr(out, "outputs", None)
            if seq_outs is None and isinstance(out, dict):
                seq_outs = out.get("outputs")
            if not isinstance(seq_outs, list):
                continue
            for s in seq_outs:
                tids = getattr(s, "token_ids", None)
                if tids is None and isinstance(s, dict):
                    tids = s.get("token_ids")
                if isinstance(tids, list):
                    actual_total_tokens += len(tids)
        except Exception:
            continue
    return {
        "backend": "vllm",
        "total_tokens": expected_total_tokens,
        "actual_total_tokens": int(actual_total_tokens),
        "seconds": dt,
        "tokens_per_sec": expected_total_tokens / dt,
        "actual_tokens_per_sec": max(0.0, float(actual_total_tokens)) / dt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Throughput benchmark: NovaInfer vs vLLM (nano-vllm-style timing).")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--backend", default="both", choices=["novainfer", "vllm", "both"])
    parser.add_argument("--num-seqs", default=256, type=int)
    parser.add_argument("--min-input-len", default=100, type=int)
    parser.add_argument("--max-input-len", default=1024, type=int)
    parser.add_argument("--min-output-len", default=100, type=int)
    parser.add_argument("--max-output-len", default=1024, type=int)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-num-seqs", default=256, type=int)
    parser.add_argument("--max-num-batched-tokens", default=16384, type=int)
    parser.add_argument("--kv-cache-capacity-mode", default="auto", choices=["explicit", "auto"])
    parser.add_argument("--kv-cache-capacity-tokens", default=4096, type=int)
    parser.add_argument("--kv-cache-memory-utilization", default=0.9, type=float)
    parser.add_argument("--vllm-fair-mode", action="store_true")
    parser.add_argument("--result-json", default="", type=str)
    args = parser.parse_args()

    prompts, out_lens = _build_dataset(
        seed=int(args.seed),
        num_seqs=max(1, int(args.num_seqs)),
        min_input_len=max(1, int(args.min_input_len)),
        max_input_len=max(1, int(args.max_input_len)),
        min_output_len=max(1, int(args.min_output_len)),
        max_output_len=max(1, int(args.max_output_len)),
        max_model_len=max(2, int(args.max_model_len)),
    )
    print(
        "[bench] config "
        f"backend={args.backend} model={args.model_path} num_seqs={len(prompts)} "
        f"seed={args.seed} in=[{args.min_input_len},{args.max_input_len}] "
        f"out=[{args.min_output_len},{args.max_output_len}] max_model_len={args.max_model_len} "
        f"vllm_fair_mode={bool(args.vllm_fair_mode)} ignore_eos=True"
    )
    print(f"[bench] expected_total_tokens={sum(out_lens)}")

    def _print_row(row: dict) -> None:
        print(
            f"[{row['backend']}] expected_total_tokens={row['total_tokens']} "
            f"actual_total_tokens={row.get('actual_total_tokens', 0)} "
            f"time={row['seconds']:.4f}s "
            f"throughput_expected={row['tokens_per_sec']:.4f} tok/s "
            f"throughput_actual={row.get('actual_tokens_per_sec', 0.0):.4f} tok/s"
        )

    # To avoid CUDA context/process init conflicts, "both" runs each backend
    # in an isolated subprocess with the same benchmark arguments.
    if args.backend == "both":
        results: list[dict] = []
        failed: list[str] = []
        for backend in ("novainfer", "vllm"):
            with tempfile.NamedTemporaryFile(prefix=f"bench_{backend}_", suffix=".json", delete=False) as tf:
                result_json = tf.name
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model-path",
                str(args.model_path),
                "--backend",
                backend,
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
                "--kv-cache-capacity-mode",
                str(args.kv_cache_capacity_mode),
                "--kv-cache-capacity-tokens",
                str(args.kv_cache_capacity_tokens),
                "--kv-cache-memory-utilization",
                str(args.kv_cache_memory_utilization),
                *(["--vllm-fair-mode"] if bool(args.vllm_fair_mode) else []),
                "--result-json",
                result_json,
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                failed.append(backend)
                continue
            try:
                with open(result_json, "r", encoding="utf-8") as fh:
                    row = json.load(fh)
                if isinstance(row, dict):
                    results.append(row)
            except Exception:
                failed.append(backend)

        if len(results) == 2:
            n = next(r for r in results if r["backend"] == "novainfer")
            v = next(r for r in results if r["backend"] == "vllm")
            print(
                f"[speedup] novainfer/vllm expected={n['tokens_per_sec'] / max(1e-9, v['tokens_per_sec']):.4f}x "
                f"actual={n.get('actual_tokens_per_sec', 0.0) / max(1e-9, v.get('actual_tokens_per_sec', 0.0)):.4f}x"
            )
        if failed:
            print(f"[bench] failed_backends={failed}")
            return 1
        return 0

    results: list[dict] = []

    if args.backend == "novainfer":
        row = _run_novainfer(
            model_path=args.model_path,
            prompts=prompts,
            out_lens=out_lens,
            max_model_len=int(args.max_model_len),
            max_num_seqs=int(args.max_num_seqs),
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            kv_cache_capacity_mode=str(args.kv_cache_capacity_mode),
            kv_cache_capacity_tokens=int(args.kv_cache_capacity_tokens),
            kv_cache_memory_utilization=float(args.kv_cache_memory_utilization),
        )
        results.append(row)
        _print_row(row)
    if args.backend == "vllm":
        row = _run_vllm(
            model_path=args.model_path,
            prompts=prompts,
            out_lens=out_lens,
            max_model_len=int(args.max_model_len),
            max_num_seqs=int(args.max_num_seqs),
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            vllm_fair_mode=bool(args.vllm_fair_mode),
        )
        results.append(row)
        _print_row(row)
    if args.result_json and results:
        with open(args.result_json, "w", encoding="utf-8") as fh:
            json.dump(results[0], fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
