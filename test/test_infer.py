from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from test.test_utils import llaisys_device, torch_device


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

    _, logits_rows = model_runner.decode_batch(token_ids=input_tokens)
    if not logits_rows:
        raise RuntimeError("ModelRunner prefill returned no logits")

    next_token = int(np.argmax(logits_rows[-1]))
    output_tokens.append(next_token)

    for _ in range(max(0, int(max_new_tokens) - 1)):
        if next_token == int(model_runner.end_token_id):
            break
        _, logits_rows = model_runner.decode_batch(token_ids=[next_token])
        if not logits_rows:
            raise RuntimeError("ModelRunner decode returned no logits")
        next_token = int(np.argmax(logits_rows[-1]))
        output_tokens.append(next_token)
        if next_token == int(model_runner.end_token_id):
            break
    return output_tokens


@pytest.mark.requires_model
@pytest.mark.requires_hf
@pytest.mark.parity
def test_infer_parity(require_model_path):
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

    model_runner = llaisys.models.Qwen2(
        model_path=model_path,
        device=llaisys_device("cpu"),
    )
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


@pytest.mark.requires_model
def test_infer_smoke(require_model_path):
    model_path = require_model_path
    prompt = "hello"
    max_steps = 2
    top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_runner = llaisys.models.Qwen2(
        model_path=model_path,
        device=llaisys_device("cpu"),
    )
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
