from __future__ import annotations

import os
from ctypes import c_int32

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from test.parity.backend_matrix import parity_device_backend_layout_cases
from llaisys.libllaisys import LIB_LLAISYS
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


def _decode_batch(model_handle, tokens, seq_ids, poss, logits_mask, *, is_block_layout=False, block_state=None, block_size=16):
    built = build_decode_batch(
        tokens,
        logits_mask=logits_mask,
        seq_ids=seq_ids,
        pos_ids=poss,
        layout=KvCacheLayout.BLOCK if is_block_layout else KvCacheLayout.SLOT,
        block_size=block_size,
        block_state=block_state,
    )
    status = int(LIB_LLAISYS.llaisysModelDecode(model_handle, built.batch))
    if status != 0:
        raise RuntimeError(f"llaisysModelDecode failed with status={status}")

    n_outputs = int(LIB_LLAISYS.llaisysModelNOutputs(model_handle))
    if n_outputs == 0:
        return [], []
    output_ids = (
        np.ctypeslib.as_array(LIB_LLAISYS.llaisysModelOutputIds(model_handle), shape=(n_outputs,))
        .astype(np.int64)
        .tolist()
    )
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
    is_block_layout = int(llaisys_model._kv_cache_layout) == int(KvCacheLayout.BLOCK)
    block_size = int(getattr(llaisys_model, "_kv_cache_block_size", 16))
    block_state = BlockBatchState()

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
        out_ids, out_rows = _decode_batch(
            handle,
            btok,
            bseq,
            bpos,
            blogits,
            is_block_layout=is_block_layout,
            block_state=block_state,
            block_size=block_size,
        )
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

        out_ids, out_rows = _decode_batch(
            handle,
            btok,
            bseq,
            bpos,
            blogits,
            is_block_layout=is_block_layout,
            block_state=block_state,
            block_size=block_size,
        )
        for ridx, bidx in enumerate(out_ids):
            row = np.ctypeslib.as_array(out_rows[ridx], shape=(vocab,))
            tok, _ = _argmax_and_top5_from_numpy(row)
            generated[bidx].append(tok)
    return generated


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


@pytest.mark.requires_model
@pytest.mark.requires_hf
@pytest.mark.parity
@pytest.mark.parametrize(
    "prompts, ll_device, backend, kv_layout",
    [
        (prompts, d, b, layout)
        for prompts in (
            ["Who are you?"],
            ["Who are you?", "Explain KV cache in one sentence."],
        )
        for d, b, layout in parity_device_backend_layout_cases()
    ],
)
def test_core_parity(require_model_path, prompts, ll_device, backend, kv_layout):
    if ll_device == "nvidia" and not _has_nvidia_runtime():
        pytest.skip("NVIDIA runtime unavailable")

    tokenizer = AutoTokenizer.from_pretrained(require_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    hf_model = AutoModelForCausalLM.from_pretrained(
        require_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    hf_model.to(_torch_device("cpu"))
    if hf_model.generation_config.pad_token_id is None:
        hf_model.generation_config.pad_token_id = hf_model.generation_config.eos_token_id
    hf_model.eval()

    prompt_ids = _build_prompt_to_token_ids(tokenizer, prompts)
    hf_tokens = _hf_generate_batch(hf_model, prompt_ids, max_new_tokens=5)

    old_backend = _set_attn_backend(backend if ll_device == "nvidia" else None)
    llaisys_model = None
    try:
        try:
            llaisys_model = llaisys.models.Qwen2(
                require_model_path,
                llaisys.DeviceType.NVIDIA if ll_device == "nvidia" else llaisys.DeviceType.CPU,
                kv_cache_layout=(
                    KvCacheLayout.BLOCK if kv_layout == "block" else KvCacheLayout.SLOT
                ),
                kv_cache_auto_capacity=True,
            )
        except Exception as exc:
            if ll_device == "nvidia" and backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise
        ll_tokens = _llaisys_argmax_batch(llaisys_model, prompt_ids, max_new_tokens=5)
        assert ll_tokens == hf_tokens
    finally:
        if llaisys_model is not None:
            llaisys_model.close()
        _restore_attn_backend(old_backend)
