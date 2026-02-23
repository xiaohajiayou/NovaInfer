from __future__ import annotations

import os
import gc

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from test.parity.backend_matrix import parity_device_backend_layout_cases
from test.test_utils import llaisys_device, torch_device
from llaisys.libllaisys.model import KvCacheLayout
from test.utils.batch_builders import BlockBatchState, build_decode_batch


def _has_nvidia_runtime() -> bool:
    try:
        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        return api.get_device_count() > 0 and torch.cuda.is_available()
    except Exception:
        return False


def _set_attn_backend(backend: str | None):
    key = "LLAISYS_CUDA_PAGED_ATTN_BACKEND"
    old = os.environ.get(key)
    if backend is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = backend
    return old


def _restore_attn_backend(old: str | None):
    key = "LLAISYS_CUDA_PAGED_ATTN_BACKEND"
    if old is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old


def load_hf_model(model_path: str, device_name: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )
    return tokenizer, model


def hf_infer(
    prompt,
    tokenizer,
    model,
    max_new_tokens=128,
    top_p=0.8,
    top_k=50,
    temperature=0.8,
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    return outputs[0].tolist()


def llaisys_model_runner_infer(
    prompt,
    tokenizer,
    model_runner,
    max_new_tokens=128,
    top_p=0.8,
    top_k=50,
    temperature=0.8,
):
    _ = (top_p, top_k, temperature)
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_tokens = tokenizer.encode(input_content)
    output_tokens = [int(t) for t in input_tokens]
    is_block_layout = int(model_runner._kv_cache_layout) == int(KvCacheLayout.BLOCK)
    block_size = int(getattr(model_runner, "_kv_cache_block_size", 16))
    block_state = BlockBatchState()
    seq_id = 0

    def _decode_block(token_batch: list[int], pos_start: int):
        n = len(token_batch)
        pos_ids = [int(pos_start + i) for i in range(n)]
        built = build_decode_batch(
            token_batch,
            logits_mask=None,
            seq_ids=[seq_id] * n,
            pos_ids=pos_ids,
            layout=KvCacheLayout.BLOCK,
            block_size=block_size,
            block_state=block_state,
        )
        n_batch_seq = int(built.batch.n_batch_seq)
        width = int(built.batch.block_table_width)
        slot_mapping = [int(built.batch.slot_mapping[i]) for i in range(n)]
        context_lens = [int(built.batch.context_lens[i]) for i in range(n_batch_seq)]
        batch_seq_ids = [int(built.batch.batch_seq_ids[i]) for i in range(n_batch_seq)]
        block_tables = [int(built.batch.block_tables[i]) for i in range(n_batch_seq * width)]
        return model_runner.decode_batch(
            token_ids=token_batch,
            pos_ids=pos_ids,
            seq_ids=[seq_id] * n,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            batch_seq_ids=batch_seq_ids,
            block_tables=block_tables,
            block_table_width=width,
        )

    if is_block_layout:
        _, logits_rows = _decode_block(input_tokens, pos_start=0)
    else:
        _, logits_rows = model_runner.decode_batch(token_ids=input_tokens)
    if not logits_rows:
        raise RuntimeError("ModelRunner prefill returned no logits")

    next_token = int(np.argmax(logits_rows[-1]))
    output_tokens.append(next_token)
    decode_pos = len(input_tokens)

    for _ in range(max(0, int(max_new_tokens) - 1)):
        if next_token == int(model_runner.end_token_id):
            break
        if is_block_layout:
            _, logits_rows = _decode_block([next_token], pos_start=decode_pos)
        else:
            _, logits_rows = model_runner.decode_batch(token_ids=[next_token])
        if not logits_rows:
            raise RuntimeError("ModelRunner decode returned no logits")
        next_token = int(np.argmax(logits_rows[-1]))
        output_tokens.append(next_token)
        decode_pos += 1
        if next_token == int(model_runner.end_token_id):
            break
    return output_tokens


@pytest.mark.requires_model
@pytest.mark.requires_hf
@pytest.mark.parity
@pytest.mark.parametrize(("ll_device", "backend", "kv_layout"), parity_device_backend_layout_cases())
def test_infer_parity(require_model_path, ll_device, backend, kv_layout):
    if ll_device == "nvidia" and not _has_nvidia_runtime():
        pytest.skip("NVIDIA runtime unavailable")
    model_path = require_model_path
    prompt = "Who are you?"
    max_steps = 10
    top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, hf_model = load_hf_model(model_path, "cpu")
    hf_tokens = hf_infer(
        prompt,
        tokenizer,
        hf_model,
        max_new_tokens=max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    del hf_model
    gc.collect()

    old_backend = _set_attn_backend(backend if ll_device == "nvidia" else None)
    model_runner = None
    try:
        try:
            model_runner = llaisys.models.Qwen2(
                model_path=model_path,
                device=llaisys_device(ll_device),
                kv_cache_layout=KvCacheLayout.BLOCK if kv_layout == "block" else KvCacheLayout.SLOT,
                kv_cache_auto_capacity=True,
            )
        except Exception as exc:
            if ll_device == "nvidia" and backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise
        mr_tokens = llaisys_model_runner_infer(
            prompt,
            tokenizer,
            model_runner,
            max_new_tokens=max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        assert mr_tokens == hf_tokens
    finally:
        if model_runner is not None:
            model_runner.close()
        _restore_attn_backend(old_backend)


@pytest.mark.requires_model
@pytest.mark.test_device("cpu")
@pytest.mark.test_layout("block")
@pytest.mark.test_backend("native")
def test_infer_smoke(require_model_path):
    model_path = require_model_path
    prompt = "hello"
    max_steps = 2
    top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_runner = llaisys.models.Qwen2(
        model_path=model_path,
        device=llaisys_device("cpu"),
        kv_cache_auto_capacity=True,
    )
    try:
        out_tokens = llaisys_model_runner_infer(
            prompt,
            tokenizer,
            model_runner,
            max_new_tokens=max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        assert len(out_tokens) > 0
    finally:
        model_runner.close()


@pytest.mark.requires_model
@pytest.mark.test_device("nvidia")
@pytest.mark.test_layout("block")
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
@pytest.mark.parametrize("backend", ["native", "cudnn"])
def test_infer_smoke_nvidia(require_model_path, backend):
    model_path = require_model_path
    prompt = "hello"
    max_steps = 2
    top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    old_backend = _set_attn_backend(backend)
    model_runner = None
    try:
        try:
            model_runner = llaisys.models.Qwen2(
                model_path=model_path,
                device=llaisys_device("nvidia"),
                kv_cache_auto_capacity=True,
            )
        except Exception as exc:
            if backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise
        out_tokens = llaisys_model_runner_infer(
            prompt,
            tokenizer,
            model_runner,
            max_new_tokens=max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        assert len(out_tokens) > 0
    finally:
        if model_runner is not None:
            model_runner.close()
        _restore_attn_backend(old_backend)
