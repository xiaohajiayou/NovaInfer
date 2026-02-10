import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


MULTI_PROMPTS = [
    "Who are you?",
    "Give one short sentence about distributed inference.",
]


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


def hf_completion_tokens(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = tokenizer.encode(input_content)
    full_tokens, _ = hf_infer(
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
    )


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
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


def run_single_case(args, tokenizer, model, model_path, top_p, top_k, temperature):
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

    print("\n=== HF Answer (single) ===\n")
    print("Tokens:")
    print(hf_tokens)
    print("\nContents:")
    print(hf_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    llm = load_llaisys_llm(model_path, args.device)
    start_time = time.time()
    ll_tokens, ll_output = llaisys_offline_infer(
        args.prompt,
        tokenizer,
        llm,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    print("\n=== Engine Offline Result (single) ===\n")
    print("Tokens:")
    print(ll_tokens)
    print("\nContents:")
    print(ll_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    if args.test:
        assert ll_tokens == hf_tokens, "single-seq parity mismatch"


def run_multi_case(args, tokenizer, model, model_path, top_p, top_k, temperature):
    prompts = MULTI_PROMPTS
    hf_expected_completion_tokens = []
    for prompt in prompts:
        completion = hf_completion_tokens(
            prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        hf_expected_completion_tokens.append(completion)

    llm = load_llaisys_llm(model_path, args.device)
    outputs = llm.generate(
        prompts,
        max_new_tokens=args.max_steps,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    print("\n=== Engine Offline Result (multi) ===\n")
    for i, prompt in enumerate(prompts):
        got = outputs[i]["token_ids"]
        exp = hf_expected_completion_tokens[i]
        print(f"[multi][{i}] prompt={prompt}")
        print(f"[multi][{i}] expected={exp}")
        print(f"[multi][{i}] got={got}")
        if args.test:
            assert got == exp, f"multi-seq parity mismatch at idx={i}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--case", default="all", choices=["single", "multi", "all"], type=str)
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
    if args.case in ("single", "all"):
        run_single_case(args, tokenizer, model, model_path, top_p, top_k, temperature)

    if args.case in ("multi", "all"):
        run_multi_case(args, tokenizer, model, model_path, top_p, top_k, temperature)

    del model
    gc.collect()

    print("\033[92mtest_offline_parity passed!\033[0m\n")
