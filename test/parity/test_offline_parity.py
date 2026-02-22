from __future__ import annotations

import os
import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from test.parity.backend_matrix import parity_device_backend_cases
from test.test_utils import llaisys_device, torch_device


MULTI_PROMPTS = [
    "Who are you?",
    "Give one short sentence about distributed inference.",
]

def _has_nvidia_runtime() -> bool:
    try:
        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        return api.get_device_count() > 0 and torch.cuda.is_available()
    except Exception:
        return False


def load_hf_model(model_path: str, device_name="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )
    return tokenizer, model


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
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


def hf_completion_tokens(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = tokenizer.encode(input_content)
    full_tokens = hf_infer(
        prompt,
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    return full_tokens[len(prompt_tokens) :]


def load_llaisys_llm(model_path, device_name):
    return llaisys.LLM(
        model=model_path,
        model_type="qwen2",
        device=llaisys_device(device_name),
        kv_cache_auto_capacity=True,
    )

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


def llaisys_offline_infer(
    prompt, tokenizer, llm, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = llm.generate(
        [inputs],
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    return outputs[0]["token_ids"]


@pytest.mark.requires_model
@pytest.mark.requires_hf
@pytest.mark.parity
@pytest.mark.parametrize(("ll_device", "backend"), parity_device_backend_cases())
def test_offline_parity_single(require_model_path, ll_device, backend):
    if ll_device == "nvidia" and not _has_nvidia_runtime():
        pytest.skip("NVIDIA runtime unavailable")
    tokenizer, model = load_hf_model(require_model_path, "cpu")
    prompt = "Who are you?"
    top_p, top_k, temperature = 1.0, 1, 1.0
    max_steps = 10

    hf_tokens = hf_completion_tokens(
        prompt,
        tokenizer,
        model,
        max_new_tokens=max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    old_backend = _set_attn_backend(backend if ll_device == "nvidia" else None)
    llm = None
    try:
        try:
            llm = load_llaisys_llm(require_model_path, ll_device)
        except Exception as exc:
            if ll_device == "nvidia" and backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise
        ll_tokens = llaisys_offline_infer(
            prompt,
            tokenizer,
            llm,
            max_new_tokens=max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        assert ll_tokens == hf_tokens
    finally:
        if llm is not None:
            llm.close()
        _restore_attn_backend(old_backend)

    del model
    gc.collect()


@pytest.mark.requires_model
@pytest.mark.requires_hf
@pytest.mark.parity
@pytest.mark.parametrize(("ll_device", "backend"), parity_device_backend_cases())
def test_offline_parity_multi(require_model_path, ll_device, backend):
    if ll_device == "nvidia" and not _has_nvidia_runtime():
        pytest.skip("NVIDIA runtime unavailable")
    tokenizer, model = load_hf_model(require_model_path, "cpu")
    top_p, top_k, temperature = 1.0, 1, 1.0
    max_steps = 10

    hf_expected_completion_tokens = []
    for prompt in MULTI_PROMPTS:
        completion = hf_completion_tokens(
            prompt,
            tokenizer,
            model,
            max_new_tokens=max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        hf_expected_completion_tokens.append(completion)

    old_backend = _set_attn_backend(backend if ll_device == "nvidia" else None)
    llm = None
    try:
        try:
            llm = load_llaisys_llm(require_model_path, ll_device)
        except Exception as exc:
            if ll_device == "nvidia" and backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise
        outputs = llm.generate(
            MULTI_PROMPTS,
            max_new_tokens=max_steps,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        for i in range(len(MULTI_PROMPTS)):
            assert outputs[i]["token_ids"] == hf_expected_completion_tokens[i]
    finally:
        if llm is not None:
            llm.close()
        _restore_attn_backend(old_backend)

    del model
    gc.collect()
