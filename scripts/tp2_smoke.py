from __future__ import annotations

import argparse
import os
import traceback
from multiprocessing import get_context
from pathlib import Path


def _default_init_method(tag: str) -> str:
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    return f"file://{(tmp_dir / f'llaisys_{tag}_{os.getpid()}.id').resolve()}"


def _parse_device_ids(raw: str) -> tuple[int, ...] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    ids = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    return ids if ids else None


def _worker(
    rank: int,
    model_path: str,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    device_ids: tuple[int, ...] | None,
) -> int:
    try:
        from llaisys.entrypoints.llm import LLM
        from llaisys.engine.types import SamplingParams
        from llaisys.libllaisys import DeviceType

        llm = LLM(
            model=model_path,
            model_type="qwen2",
            device=DeviceType.NVIDIA,
            kv_cache_block_size=16,
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            max_num_batched_tokens=int(max_num_batched_tokens),
            tensor_parallel_size=2,
            tp_rank=int(rank),
            tp_local_rank=int(rank),
            distributed_backend="nccl",
            tensor_parallel_device_ids=device_ids,
        )
        # Pretokenized input path avoids tokenizer/torch import side effects.
        prompt = [151646] + [100] * 127
        prompts = [prompt for _ in range(8)]
        params = [SamplingParams(max_new_tokens=16, top_k=1) for _ in range(8)]
        outs = llm.generate(prompts, params, use_tqdm=False)
        llm.close()
        print(f"[tp2_smoke] rank={rank} ok outputs={len(outs)} first={outs[0]['token_ids'][:4] if outs else []}")
        return 0
    except Exception as exc:
        print(f"[tp2_smoke] rank={rank} failed: {exc}")
        traceback.print_exc()
        return 1


def _worker_entry(*args) -> None:
    rc = int(_worker(*args))
    raise SystemExit(rc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TP=2 two-process smoke for NovaInfer.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--max-num-seqs", default=32, type=int)
    parser.add_argument("--max-num-batched-tokens", default=4096, type=int)
    parser.add_argument(
        "--tensor-parallel-device-ids",
        default="",
        type=str,
        help="Optional comma-separated logical GPU ids (under CUDA_VISIBLE_DEVICES). Empty means auto-select.",
    )
    args = parser.parse_args()

    if str(os.getenv("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "")).strip().lower() != "cudnn":
        print("[tp2_smoke] requires LLAISYS_CUDA_PAGED_ATTN_BACKEND=cudnn")
        return 1

    if not os.getenv("LLAISYS_TP_INIT_METHOD"):
        os.environ["LLAISYS_TP_INIT_METHOD"] = _default_init_method("tp2_nccl_smoke")
    if not os.getenv("LLAISYS_TP_SINGLE_PROCESS"):
        os.environ["LLAISYS_TP_SINGLE_PROCESS"] = "0"

    device_ids = _parse_device_ids(args.tensor_parallel_device_ids)
    ctx = get_context("spawn")
    p0 = ctx.Process(
        target=_worker_entry,
        args=(
            0,
            str(args.model_path),
            int(args.max_model_len),
            int(args.max_num_seqs),
            int(args.max_num_batched_tokens),
            device_ids,
        ),
    )
    p1 = ctx.Process(
        target=_worker_entry,
        args=(
            1,
            str(args.model_path),
            int(args.max_model_len),
            int(args.max_num_seqs),
            int(args.max_num_batched_tokens),
            device_ids,
        ),
    )
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    print(f"[tp2_smoke] exitcodes rank0={p0.exitcode} rank1={p1.exitcode}")
    return 0 if p0.exitcode == 0 and p1.exitcode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
