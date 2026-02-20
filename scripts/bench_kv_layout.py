from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

from llaisys.entrypoints.llm import LLM
from llaisys.engine.types import SamplingParams
from llaisys.libllaisys import DeviceType
from llaisys.libllaisys.model import KvCacheLayout


def _parse_device(name: str) -> DeviceType:
    n = name.strip().lower()
    if n == "cpu":
        return DeviceType.CPU
    if n in ("nvidia", "cuda", "gpu"):
        return DeviceType.NVIDIA
    raise ValueError(f"unsupported device: {name}")


def _mem_snapshot() -> tuple[float, float]:
    # Linux /proc/meminfo: values are in kB.
    mem_total_kb = 0
    mem_avail_kb = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
    except Exception:
        return 0.0, 0.0
    return mem_total_kb / 1024.0 / 1024.0, mem_avail_kb / 1024.0 / 1024.0


def _run_once(
    model_path: Path,
    device: DeviceType,
    layout: KvCacheLayout,
    block_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    kv_cache_capacity_tokens: int | None,
    kv_cache_auto_capacity: bool,
    kv_cache_memory_utilization: float,
    prompts: list[list[int]],
    sampling_params: list[SamplingParams],
    expected_total_tokens_per_round: int,
    rounds: int,
) -> dict:
    layout_name = "block" if layout == KvCacheLayout.BLOCK else "slot"
    print(f"[bench] init layout={layout_name} block_size={block_size} model={model_path}")
    llm = LLM(
        model=model_path,
        model_type="qwen2",
        device=device,
        kv_cache_layout=layout,
        kv_cache_block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        kv_cache_capacity_tokens=kv_cache_capacity_tokens,
        kv_cache_auto_capacity=kv_cache_auto_capacity,
        kv_cache_memory_utilization=kv_cache_memory_utilization,
    )
    kv_stats_last: dict = {}
    try:
        # warmup
        print(f"[bench] warmup start layout={layout_name}")
        _ = llm.generate([[1, 2, 3, 4]], sampling_params=SamplingParams(max_new_tokens=4, top_k=1, top_p=1.0, temperature=1.0))
        kv_stats_last = llm.kv_cache_stats()
        if kv_stats_last:
            print(f"[bench] kv_stats_after_warmup layout={layout_name} stats={kv_stats_last}")
        print(f"[bench] warmup done layout={layout_name}")

        t0 = time.perf_counter()
        total_reqs = 0
        for i in range(rounds):
            print(f"[bench] round {i + 1}/{rounds} start layout={layout_name}")
            outs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            total_reqs += len(outs)
            kv_stats_last = llm.kv_cache_stats()
            print(
                f"[bench] round {i + 1}/{rounds} done layout={layout_name} "
                f"cum_reqs={total_reqs} kv_stats={kv_stats_last}"
            )
        t1 = time.perf_counter()
    finally:
        print(f"[bench] close model layout={layout_name}")
        llm.close()

    dt = max(1e-9, t1 - t0)
    total_completion = int(expected_total_tokens_per_round) * int(rounds)
    print(
        f"[bench] finish layout={layout_name} total_reqs={total_reqs} "
        f"completion_tokens={total_completion} seconds={dt:.4f}"
    )
    return {
        "layout": layout_name,
        "rounds": rounds,
        "requests": total_reqs,
        "completion_tokens": total_completion,
        "seconds": dt,
        "tokens_per_sec": total_completion / dt,
        "avg_req_latency_ms": (dt / max(1, total_reqs)) * 1000.0,
        "kv_cache_stats": kv_stats_last,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SLOT vs BLOCK kv cache layouts.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--block-size", default=16, type=int)
    parser.add_argument("--layout", default="both", choices=["both", "slot", "block"])
    parser.add_argument("--omp-num-threads", default=8, type=int)
    parser.add_argument("--omp-proc-bind", default="spread")
    parser.add_argument("--omp-places", default="cores")
    parser.add_argument("--rounds", default=1, type=int)
    parser.add_argument("--num-prompts", default=256, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-input-len", default=1024, type=int)
    parser.add_argument("--max-output-len", default=1024, type=int)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--kv-cache-capacity-tokens", default=4096, type=int)
    parser.add_argument("--kv-cache-capacity-mode", default="explicit", choices=["explicit", "auto"])
    parser.add_argument("--kv-cache-memory-utilization", default=0.9, type=float)
    parser.add_argument("--max-num-seqs", default=8, type=int)
    parser.add_argument("--max-num-batched-tokens", default=0, type=int)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.omp_num_threads)))
    os.environ["OMP_PROC_BIND"] = str(args.omp_proc_bind)
    os.environ["OMP_PLACES"] = str(args.omp_places)
    print(
        "[bench] config "
        f"model_path={args.model_path} device={args.device} block_size={args.block_size} "
        f"num_prompts={args.num_prompts} rounds={args.rounds} seed={args.seed} "
        f"max_input_len={args.max_input_len} max_output_len={args.max_output_len} "
        f"max_model_len={args.max_model_len} kv_cache_capacity_tokens={args.kv_cache_capacity_tokens} "
        f"kv_cache_capacity_mode={args.kv_cache_capacity_mode} "
        f"kv_cache_memory_utilization={args.kv_cache_memory_utilization} "
        f"max_num_seqs={args.max_num_seqs} max_num_batched_tokens={args.max_num_batched_tokens} "
        f"omp_num_threads={os.environ.get('OMP_NUM_THREADS')} "
        f"omp_proc_bind={os.environ.get('OMP_PROC_BIND')} omp_places={os.environ.get('OMP_PLACES')}"
    )
    mem_total_gib, mem_avail_gib = _mem_snapshot()
    if mem_total_gib > 0.0:
        print(f"[bench] mem_total_gib={mem_total_gib:.2f} mem_available_gib={mem_avail_gib:.2f}")

    rng = random.Random(int(args.seed))
    n = max(1, int(args.num_prompts))
    max_input_len = max(100, int(args.max_input_len))
    max_output_len = max(100, int(args.max_output_len))
    prompts = [
        [rng.randint(0, 10000) for _ in range(rng.randint(100, max_input_len))]
        for _ in range(n)
    ]
    max_model_len = max(1, int(args.max_model_len))
    sampling_params = [
        SamplingParams(max_new_tokens=rng.randint(100, max_output_len), top_k=1, top_p=1.0, temperature=1.0)
        for _ in range(n)
    ]
    expected_total_tokens_per_round = sum(int(sp.max_new_tokens or 0) for sp in sampling_params)
    print(f"[bench] expected_total_tokens_per_round={expected_total_tokens_per_round}")
    device = _parse_device(args.device)
    explicit_capacity_tokens = max(1, int(args.kv_cache_capacity_tokens))
    use_auto_capacity = str(args.kv_cache_capacity_mode) == "auto"
    max_num_batched_tokens = (
        max(1, int(args.max_num_batched_tokens))
        if int(args.max_num_batched_tokens) > 0
        else explicit_capacity_tokens
    )

    slot = None
    block = None
    if args.layout in ("both", "slot"):
        slot = _run_once(
            model_path=args.model_path,
            device=device,
            layout=KvCacheLayout.SLOT,
            block_size=int(args.block_size),
            max_num_seqs=max(1, int(args.max_num_seqs)),
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            kv_cache_capacity_tokens=(None if use_auto_capacity else explicit_capacity_tokens),
            kv_cache_auto_capacity=use_auto_capacity,
            kv_cache_memory_utilization=float(args.kv_cache_memory_utilization),
            prompts=prompts,
            sampling_params=sampling_params,
            expected_total_tokens_per_round=expected_total_tokens_per_round,
            rounds=int(args.rounds),
        )
        print("=== SLOT ===")
        print(slot)
    if args.layout in ("both", "block"):
        block = _run_once(
            model_path=args.model_path,
            device=device,
            layout=KvCacheLayout.BLOCK,
            block_size=int(args.block_size),
            max_num_seqs=max(1, int(args.max_num_seqs)),
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            kv_cache_capacity_tokens=(None if use_auto_capacity else explicit_capacity_tokens),
            kv_cache_auto_capacity=use_auto_capacity,
            kv_cache_memory_utilization=float(args.kv_cache_memory_utilization),
            prompts=prompts,
            sampling_params=sampling_params,
            expected_total_tokens_per_round=expected_total_tokens_per_round,
            rounds=int(args.rounds),
        )
        print("=== BLOCK ===")
        print(block)
    if slot is not None and block is not None:
        speedup = block["tokens_per_sec"] / max(1e-9, slot["tokens_per_sec"])
        print(f"speedup(block/slot)={speedup:.4f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



