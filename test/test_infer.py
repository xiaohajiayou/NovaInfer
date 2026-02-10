import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


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
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_llaisys_model_runner(model_path, device_name):
    model_runner = llaisys.models.Qwen2(
        model_path=model_path,
        device=llaisys_device(device_name),
    )
    return model_runner


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

    output = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return output_tokens, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    start_time = time.time()
    hf_tokens, hf_output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    del model
    gc.collect()

    print("\n=== HF Answer ===\n")
    print("Tokens:")
    print(hf_tokens)
    print("\nContents:")
    print(hf_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    model_runner = load_llaisys_model_runner(model_path, args.device)
    start_time = time.time()
    mr_tokens, mr_output = llaisys_model_runner_infer(
        args.prompt,
        tokenizer,
        model_runner,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    print("\n=== ModelRunner Result ===\n")
    print("Tokens:")
    print(mr_tokens)
    print("\nContents:")
    print(mr_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    if args.test:
        assert mr_tokens == hf_tokens
        print("\033[92mTest passed!\033[0m\n")
