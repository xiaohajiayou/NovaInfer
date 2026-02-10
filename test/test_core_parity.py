import argparse
import io
import sys
import time
from ctypes import POINTER, c_int32, c_int64, c_int8

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.model import LlaisysBatch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _torch_device(device_name: str):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "nvidia":
        return torch.device("cuda:0")
    raise ValueError(f"Unsupported device: {device_name}")


def _argmax_and_top5_from_numpy(row: np.ndarray):
    token = int(np.argmax(row))
    top5 = np.argsort(row)[-5:][::-1].astype(np.int64).tolist()
    return token, top5


def _argmax_and_top5_from_torch(row: torch.Tensor):
    token = int(torch.argmax(row).item())
    top5 = torch.topk(row, k=5, dim=-1).indices.detach().cpu().to(torch.int64).tolist()
    return token, top5




def _decode_batch(model_handle, tokens, seq_ids, poss, logits_mask):
    n = len(tokens)
    token_buf = (c_int64 * n)(*tokens)
    pos_buf = (c_int64 * n)(*poss)
    logits_buf = (c_int8 * n)(*logits_mask)

    n_seq_buf = (c_int32 * n)()
    seq_ptr_buf = (POINTER(c_int64) * n)()
    seq_rows = []
    for i, sid in enumerate(seq_ids):
        row = (c_int64 * 1)(sid)
        seq_rows.append(row)
        n_seq_buf[i] = 1
        seq_ptr_buf[i] = row

    batch = LlaisysBatch(
        n_tokens=c_int32(n),
        token=token_buf,
        embd=None,
        pos=pos_buf,
        n_seq_id=n_seq_buf,
        seq_id=seq_ptr_buf,
        logits=logits_buf,
    )
    status = int(LIB_LLAISYS.llaisysModelDecode(model_handle, batch))
    if status != 0:
        raise RuntimeError(f"llaisysModelDecode failed with status={status}")

    n_outputs = int(LIB_LLAISYS.llaisysModelNOutputs(model_handle))
    if n_outputs == 0:
        return [], []
    output_ids = np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(model_handle), shape=(n_outputs,)).astype(np.int64).tolist()
    logits_rows = []
    for i in range(n_outputs):
        ptr = LIB_LLAISYS.llaisysModelGetLogitsIth(model_handle, c_int32(i))
        logits_rows.append(ptr)
    return output_ids, logits_rows


def _hf_generate_batch(model, prompt_ids, max_new_tokens):
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = model.generation_config.eos_token_id

    batch_size = len(prompt_ids)
    max_len = max(len(x) for x in prompt_ids)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=model.device)
    for i, ids in enumerate(prompt_ids):
        start = max_len - len(ids)
        input_ids[i, start:] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attention_mask[i, start:] = 1

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
        )

    generated = []
    for i in range(batch_size):
        gen_tokens = out[i, max_len : max_len + max_new_tokens].detach().cpu().to(torch.int64).tolist()
        generated.append(gen_tokens)
    return generated


def _llaisys_argmax_batch(llaisys_model, prompt_ids, max_new_tokens):
    seq_ids = [1000 + i for i in range(len(prompt_ids))]
    generated = [[] for _ in range(len(prompt_ids))]
    traces = [[] for _ in range(len(prompt_ids))]
    vocab = int(llaisys_model._meta_info.voc)
    handle = llaisys_model._model

    max_prompt = max(len(x) for x in prompt_ids)
    for t in range(max_prompt):
        btok = []
        bseq = []
        bpos = []
        blogits = []
        bidx_to_seq = []
        for i, ids in enumerate(prompt_ids):
            if t < len(ids):
                btok.append(int(ids[t]))
                bseq.append(seq_ids[i])
                bpos.append(t)
                blogits.append(1 if t == len(ids) - 1 else 0)
                bidx_to_seq.append(i)
        out_ids, out_rows = _decode_batch(handle, btok, bseq, bpos, blogits)
        for ridx, bidx in enumerate(out_ids):
            seq_i = bidx_to_seq[bidx]
            row = np.ctypeslib.as_array(out_rows[ridx], shape=(vocab,))
            tok, top5 = _argmax_and_top5_from_numpy(row)
            generated[seq_i].append(tok)
            traces[seq_i].append(top5)

    for i in range(len(prompt_ids)):
        if len(generated[i]) != 1:
            raise RuntimeError(f"prefill sampling rows mismatch for seq[{i}], got {len(generated[i])}")

    for step in range(1, max_new_tokens):
        btok = []
        bseq = []
        bpos = []
        blogits = []
        for i, ids in enumerate(prompt_ids):
            btok.append(int(generated[i][-1]))
            bseq.append(seq_ids[i])
            bpos.append(len(ids) + step - 1)
            blogits.append(1)

        out_ids, out_rows = _decode_batch(handle, btok, bseq, bpos, blogits)
        for ridx, bidx in enumerate(out_ids):
            row = np.ctypeslib.as_array(out_rows[ridx], shape=(vocab,))
            tok, top5 = _argmax_and_top5_from_numpy(row)
            generated[bidx].append(tok)
            traces[bidx].append(top5)

    return generated, traces



def _build_prompt_to_token_ids(tokenizer, prompts):
    prompt_ids = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids.append(tokenizer.encode(text))
    return prompt_ids

def _build_token_to_ids_text(tokenizer, token_ids):
    texts = []
    for ids in token_ids:
        text = tokenizer.decode(ids)
        texts.append(text)
    return texts

def _run_parity_case(case_name, hf_model, tokenizer, model_path, max_new_tokens, prompts):
    prompt_ids = _build_prompt_to_token_ids(tokenizer, prompts)

    t0 = time.time()
    hf_tokens = _hf_generate_batch(hf_model, prompt_ids, max_new_tokens)
    t1 = time.time()

    llaisys_model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
    t2 = time.time()
    ll_tokens, ll_trace = _llaisys_argmax_batch(llaisys_model, prompt_ids, max_new_tokens)
    t3 = time.time()

    ok = True
    for i in range(len(prompts)):
        for step in range(max_new_tokens):
            if hf_tokens[i][step] != ll_tokens[i][step]:
                ok = False
                print(
                    f"[mismatch][{case_name}] seq={i} step={step} expected={hf_tokens[i][step]} got={ll_tokens[i][step]} "
                    f"ll_top5={ll_trace[i][step]}"
                )

    if not ok:
        raise AssertionError(f"core parity check failed: {case_name}")
    
    hf_tokens = _build_token_to_ids_text(tokenizer, hf_tokens)
    ll_tokens = _build_token_to_ids_text(tokenizer, ll_tokens)
    for i in range(len(prompts)):
        print(f"[parity][{case_name}] hf_ref_tokens for seq_{i}: {hf_tokens[i]}")
        print(f"[parity][{case_name}] our_token_ans for seq_{i}: {ll_tokens[i]}")

    print(
        f"[parity][{case_name}] passed "
        f"(hf={t1 - t0:.2f}s, llaisys_init={t2 - t1:.2f}s, llaisys_decode={t3 - t2:.2f}s)"
    )
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu"], type=str)
    parser.add_argument("--max_new_tokens", default=5, type=int)
    parser.add_argument("--case", default="all", choices=["single", "multi", "all"], type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.to(_torch_device(args.device))
    if hf_model.generation_config.pad_token_id is None:
        hf_model.generation_config.pad_token_id = hf_model.generation_config.eos_token_id
    hf_model.eval()

    if args.case in ("single", "all"):
        _run_parity_case(
            case_name="single-seq",
            hf_model=hf_model,
            tokenizer=tokenizer,
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            prompts=["Who are you?"],
        )

    if args.case in ("multi", "all"):
        _run_parity_case(
            case_name="multi-seq-interleaved",
            hf_model=hf_model,
            tokenizer=tokenizer,
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            prompts=["Who are you?", "Explain KV cache in one sentence."],
        )

    print("test_core_parity passed!")


if __name__ == "__main__":
    main()
