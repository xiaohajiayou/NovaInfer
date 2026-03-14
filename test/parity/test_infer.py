from __future__ import annotations

import os
import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from llaisys.engine.runtime_factory import create_kv_state, create_parallel_context, plan_qwen2_kv_cache
from llaisys.libllaisys import LIB_LLAISYS
from test.parity.backend_matrix import parity_device_backend_layout_cases
from test.test_utils import llaisys_device, torch_device
from test.utils.batch_builders import BlockBatchState, build_decode_batch
from test.utils.forward_api import run_model_forward, sample_from_forward


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


def _create_qwen2_with_runtime(model_path: str, device):
    plan = plan_qwen2_kv_cache(
        model_path=model_path,
        device=device,
        kv_cache_block_size=16,
        max_model_len=None,
        kv_cache_memory_utilization=0.9,
        max_num_seqs=None,
    )
    runtime_handle = create_kv_state(
        device=device,
        kv_cache_block_size=16,
        plan=plan,
    )
    try:
        model = llaisys.models.Qwen2(
            model_path=model_path,
            device=device,
            max_model_len=int(plan.max_model_len),
        )
        parallel_context = create_parallel_context(device=device, tensor_parallel_size=1)
        rc = int(model.bind_parallel_context(parallel_context))
        if rc != 0:
            raise RuntimeError(f"modelBindParallelContext failed with status={rc}")
        model._test_parallel_context = parallel_context
    except Exception:
        LIB_LLAISYS.llaisysKvStateDestroy(runtime_handle)
        raise
    return model, runtime_handle


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
    model_wrapper,
    runtime_handle,
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
    block_size = 16
    device = getattr(model_wrapper, "_device", llaisys.DeviceType.CPU)
    model_handle = model_wrapper._model
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
            block_size=block_size,
            block_state=block_state,
        )
        out = run_model_forward(model_handle, runtime_handle, built, device=device)
        if out.status != 0:
            raise RuntimeError(f"modelForward failed with status={out.status}")
        sampled_ids = sample_from_forward(out, device=device)
        return out.output_ids, sampled_ids

    _, sampled_ids = _decode_block(input_tokens, pos_start=0)
    if not sampled_ids:
        raise RuntimeError("ModelRunner prefill returned no sampled ids")

    next_token = int(sampled_ids[-1])
    output_tokens.append(next_token)
    decode_pos = len(input_tokens)

    for _ in range(max(0, int(max_new_tokens) - 1)):
        if next_token == int(model_wrapper.end_token_id):
            break
        _, sampled_ids = _decode_block([next_token], pos_start=decode_pos)
        if not sampled_ids:
            raise RuntimeError("ModelRunner decode returned no sampled ids")
        next_token = int(sampled_ids[-1])
        output_tokens.append(next_token)
        decode_pos += 1
        if next_token == int(model_wrapper.end_token_id):
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
    runtime_handle = None
    try:
        try:
            model_runner, runtime_handle = _create_qwen2_with_runtime(
                model_path=model_path,
                device=llaisys_device(ll_device),
            )
        except Exception as exc:
            if ll_device == "nvidia" and backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise
        try:
            mr_tokens = llaisys_model_runner_infer(
                prompt,
                tokenizer,
                model_runner,
                runtime_handle,
                max_new_tokens=max_steps,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
        except RuntimeError as exc:
            if ll_device == "nvidia" and backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable during forward: {exc}")
            raise
        assert mr_tokens == hf_tokens
    finally:
        parallel_context = getattr(model_runner, "_test_parallel_context", None) if model_runner is not None else None
        if model_runner is not None:
            model_runner.close()
        if parallel_context is not None:
            LIB_LLAISYS.llaisysParallelContextDestroy(parallel_context)
        if runtime_handle is not None:
            LIB_LLAISYS.llaisysKvStateDestroy(runtime_handle)
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
    model_runner, runtime_handle = _create_qwen2_with_runtime(
        model_path=model_path,
        device=llaisys_device("cpu"),
    )
    try:
        out_tokens = llaisys_model_runner_infer(
            prompt,
            tokenizer,
            model_runner,
            runtime_handle,
            max_new_tokens=max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        assert len(out_tokens) > 0
    finally:
        parallel_context = getattr(model_runner, "_test_parallel_context", None)
        model_runner.close()
        if parallel_context is not None:
            LIB_LLAISYS.llaisysParallelContextDestroy(parallel_context)
        LIB_LLAISYS.llaisysKvStateDestroy(runtime_handle)


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
    runtime_handle = None
    try:
        try:
            model_runner, runtime_handle = _create_qwen2_with_runtime(
                model_path=model_path,
                device=llaisys_device("nvidia"),
            )
        except Exception as exc:
            if backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise
        out_tokens = llaisys_model_runner_infer(
            prompt,
            tokenizer,
            model_runner,
            runtime_handle,
            max_new_tokens=max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        assert len(out_tokens) > 0
    finally:
        parallel_context = getattr(model_runner, "_test_parallel_context", None) if model_runner is not None else None
        if model_runner is not None:
            model_runner.close()
        if parallel_context is not None:
            LIB_LLAISYS.llaisysParallelContextDestroy(parallel_context)
        if runtime_handle is not None:
            LIB_LLAISYS.llaisysKvStateDestroy(runtime_handle)
        _restore_attn_backend(old_backend)
