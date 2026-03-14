from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any


def _default_init_method(tag: str) -> str:
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return f"file://{(tmp_dir / f'llaisys_{tag}_{os.getpid()}.id').resolve()}"


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple()
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    return tuple(vals)


def _encode_prompts_for_llm(model_path: str, prompts: list[str]) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rows: list[list[int]] = []
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            rows.append([int(v) for v in tokenizer.encode(text)])
        else:
            rows.append([int(v) for v in tokenizer.encode(prompt)])
    return rows


def _render_token_piece(tokenizer: Any, token_id: int) -> str:
    text = str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if text == "":
        text = "<empty>"
    return text


def _print_aligned_answers(tokenizer: Any, prompts: list[str], hf_rows: list[list[int]], tp_rows: list[list[int]]) -> None:
    for i, prompt in enumerate(prompts):
        hf_row = hf_rows[i] if i < len(hf_rows) else []
        tp_row = tp_rows[i] if i < len(tp_rows) else []
        ncols = max(len(hf_row), len(tp_row))
        hf_cells = [_render_token_piece(tokenizer, tid) for tid in hf_row]
        tp_cells = [_render_token_piece(tokenizer, tid) for tid in tp_row]
        widths: list[int] = []
        for idx in range(ncols):
            hf_cell = hf_cells[idx] if idx < len(hf_cells) else "-"
            tp_cell = tp_cells[idx] if idx < len(tp_cells) else "-"
            widths.append(max(len(hf_cell), len(tp_cell), 1))

        def format_row(label: str, cells: list[str]) -> str:
            parts: list[str] = []
            for idx, width in enumerate(widths):
                cell = cells[idx] if idx < len(cells) else "-"
                parts.append(cell.ljust(width))
            return f"[tp_hf_parity] {label:<2} | " + " | ".join(parts)

        print(f"[tp_hf_parity] prompt[{i}]={prompt}")
        print(format_row("HF", hf_cells))
        print(format_row("TP", tp_cells))


def _hf_generate_tokens(model_path: str, prompt_token_ids: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to("cpu")
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    per_prompt: list[list[int]] = []
    for row in prompt_token_ids:
        input_ids = torch.tensor([[int(v) for v in row]], dtype=torch.long, device="cpu")
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cpu")
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=int(model.generation_config.pad_token_id),
                eos_token_id=int(model.generation_config.eos_token_id),
            )
        in_len = int(input_ids.shape[1])
        seq = out[0][in_len:]
        per_prompt.append([int(v) for v in seq.tolist()])
    return per_prompt


def _worker_generate(
    rank: int,
    tp_size: int,
    model_path: str,
    prompt_token_ids: list[list[int]],
    max_new_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    device_ids: tuple[int, ...] | None,
    distributed_executor_backend: str,
    out_json: str,
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
            tensor_parallel_size=int(tp_size),
            tp_rank=int(rank),
            tp_local_rank=int(rank),
            distributed_backend="nccl",
            tensor_parallel_device_ids=device_ids,
            distributed_executor_backend=str(distributed_executor_backend),
        )
        params = [
            SamplingParams(
                max_new_tokens=int(max_new_tokens),
                top_k=1,
                top_p=1.0,
                temperature=1.0,
            )
            for _ in prompt_token_ids
        ]
        outs = llm.generate(prompt_token_ids, params, use_tqdm=False)
        llm.close()
        tokens: list[list[int]] = []
        for out in outs:
            if not isinstance(out, dict):
                tokens.append([])
                continue
            tids = out.get("token_ids")
            if isinstance(tids, list):
                tokens.append([int(v) for v in tids])
            else:
                tokens.append([])
        Path(out_json).write_text(json.dumps({"rank": int(rank), "tokens": tokens}), encoding="utf-8")
        print(f"[tp_hf_parity] rank={rank} ok")
        return 0
    except Exception as exc:
        print(f"[tp_hf_parity] rank={rank} failed: {exc}")
        traceback.print_exc()
        return 1


def _first_diff(a: list[list[int]], b: list[list[int]]) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return f"prompt={i} a={a[i][:16]} b={b[i][:16]}"
    if len(a) != len(b):
        return f"different prompt count: {len(a)} vs {len(b)}"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="HF parity check for TP multi-process NovaInfer.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--tp-size", required=True, type=int)
    parser.add_argument("--max-new-tokens", default=8, type=int)
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--max-num-seqs", default=16, type=int)
    parser.add_argument("--max-num-batched-tokens", default=4096, type=int)
    parser.add_argument("--prompts", default="Who are you?,Explain KV cache in one sentence.", type=str)
    parser.add_argument(
        "--tensor-parallel-device-ids",
        default="",
        type=str,
        help="Logical ids under CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3. Empty means auto-select.",
    )
    parser.add_argument(
        "--init-method",
        default="",
        type=str,
    )
    parser.add_argument(
        "--distributed-executor-backend",
        default="mp",
        type=str,
        help="TP parity now requires mp. This flag is kept only for CLI compatibility.",
    )
    args = parser.parse_args()
    if str(args.distributed_executor_backend).strip().lower() != "mp":
        raise ValueError("tp_hf_parity.py only supports distributed_executor_backend=mp")

    tp_size = max(1, int(args.tp_size))
    prompts = [p.strip() for p in str(args.prompts).split(",") if p.strip()]
    if not prompts:
        raise RuntimeError("prompts must not be empty")
    device_ids = _parse_csv_ints(args.tensor_parallel_device_ids)
    device_ids_opt = device_ids if len(device_ids) > 0 else None

    init_method = str(args.init_method).strip() or _default_init_method("tp_nccl_parity")
    os.environ.setdefault("LLAISYS_CUDA_PAGED_ATTN_BACKEND", "cudnn")
    os.environ.setdefault("LLAISYS_TP_SINGLE_PROCESS", "0")
    os.environ["LLAISYS_TP_INIT_METHOD"] = init_method

    print(f"[tp_hf_parity] hf_generate start prompts={len(prompts)}")
    prompt_token_ids = _encode_prompts_for_llm(str(args.model_path), prompts)
    hf_tokens = _hf_generate_tokens(str(args.model_path), prompt_token_ids, int(args.max_new_tokens))
    print(f"[tp_hf_parity] hf_generate done")

    rank_tokens: list[list[list[int]]] = []
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"tp_hf_parity_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_json = tmp_dir / "rank0.json"
    rc = _worker_generate(
        0,
        int(tp_size),
        str(args.model_path),
        prompt_token_ids,
        int(args.max_new_tokens),
        int(args.max_model_len),
        int(args.max_num_seqs),
        int(args.max_num_batched_tokens),
        device_ids_opt,
        "mp",
        str(out_json),
    )
    print(f"[tp_hf_parity] exitcodes={[int(rc)]}")
    if rc != 0:
        return 1
    obj = json.loads(out_json.read_text(encoding="utf-8"))
    rank_tokens.append([[int(v) for v in row] for row in obj["tokens"]])

    base = rank_tokens[0]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    _print_aligned_answers(tokenizer, prompts, hf_tokens, base)

    for rank in range(1, len(rank_tokens)):
        if rank_tokens[rank] != base:
            print(f"[tp_hf_parity] mismatch between rank0 and rank{rank}: {_first_diff(base, rank_tokens[rank])}")
            return 2

    if base != hf_tokens:
        print(f"[tp_hf_parity] mismatch with HF: {_first_diff(base, hf_tokens)}")
        print(f"[tp_hf_parity] rank0={base}")
        print(f"[tp_hf_parity] hf={hf_tokens}")
        return 3

    print("[tp_hf_parity] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
