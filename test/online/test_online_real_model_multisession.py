from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from llaisys.engine.llm_engine import LLMEngine
from llaisys.server.async_engine import AsyncLLMEngine
from llaisys.server.openai_server import OpenAIServer
from llaisys.server.schemas import ChatCompletionRequest, ChatMessage


def _collect_stream(server: OpenAIServer, prompt: str, max_tokens: int = 12) -> list[dict]:
    req = ChatCompletionRequest(
        model="qwen2",
        messages=[ChatMessage(role="user", content=prompt)],
        stream=True,
        max_tokens=max_tokens,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    return list(server.handle_chat_stream(req))


def _assert_single_request_stream(chunks: list[dict]) -> str:
    assert chunks, "stream must not be empty"
    req_ids = [c.get("request_id") for c in chunks if c.get("request_id")]
    assert req_ids, "stream must contain request_id"
    unique = set(req_ids)
    assert len(unique) == 1, f"mixed request ids in one stream: {unique}"

    final = chunks[-1]
    assert final.get("is_finished") is True
    assert final.get("choices", [{}])[0].get("finish_reason") is not None
    return next(iter(unique))


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


def _hf_completion_tokens(model_path: str, prompt: str, max_new_tokens: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(torch.device("cpu"))
    model.eval()

    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
    full = out[0].detach().cpu().to(torch.int64).tolist()
    return full[len(prompt_ids):]


@pytest.mark.requires_model
@pytest.mark.online
def test_online_real_model_multisession_stream_isolation(require_model_path: str):
    engine = LLMEngine(
        model_type="qwen2",
        model_path=require_model_path,
        kv_cache_auto_capacity=True,
    )
    async_engine = AsyncLLMEngine(engine=engine)
    server = OpenAIServer(async_engine)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_collect_stream, server, "请用一句中文介绍你自己。", 12)
            fut_b = pool.submit(_collect_stream, server, "Reply in English: what are you doing now?", 12)
            chunks_a = fut_a.result(timeout=180)
            chunks_b = fut_b.result(timeout=180)

        req_id_a = _assert_single_request_stream(chunks_a)
        req_id_b = _assert_single_request_stream(chunks_b)
        assert req_id_a != req_id_b, "two sessions should map to different request ids"

        # Non-final chunks should carry token ids. This catches common stream corruption.
        non_final_a = [c for c in chunks_a if not c.get("is_finished")]
        non_final_b = [c for c in chunks_b if not c.get("is_finished")]
        assert non_final_a and non_final_b
        assert any(c.get("token_id") is not None for c in non_final_a)
        assert any(c.get("token_id") is not None for c in non_final_b)
    finally:
        async_engine.close()


@pytest.mark.requires_model
@pytest.mark.requires_hf
@pytest.mark.online
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
@pytest.mark.parametrize("backend", ["native", "cudnn"])
def test_online_real_model_hf_parity_nvidia_backend(require_model_path: str, backend: str):
    prompt = "请用一句中文介绍你自己。"
    max_tokens = 10
    hf_tokens = _hf_completion_tokens(require_model_path, prompt, max_new_tokens=max_tokens)

    old_backend = _set_attn_backend(backend)
    async_engine = None
    try:
        try:
            engine = LLMEngine(
                model_type="qwen2",
                model_path=require_model_path,
                device=llaisys.DeviceType.NVIDIA,
                kv_cache_auto_capacity=True,
                max_model_len=4096,
                max_batch_size=8,
            )
            async_engine = AsyncLLMEngine(engine=engine)
            server = OpenAIServer(async_engine)
        except Exception as exc:
            if backend == "cudnn":
                pytest.skip(f"cudnn backend unavailable or failed to initialize: {exc}")
            raise

        req = ChatCompletionRequest(
            model="qwen2",
            messages=[ChatMessage(role="user", content=prompt)],
            stream=False,
            max_tokens=max_tokens,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        resp = server.handle_chat(req)
        assert [int(t) for t in resp["token_ids"]] == hf_tokens

        stream_chunks = list(server.handle_chat_stream(req))
        stream_ids = [int(c["token_id"]) for c in stream_chunks if not c.get("is_finished") and c.get("token_id") is not None]
        assert stream_ids == hf_tokens
    finally:
        if async_engine is not None:
            async_engine.close()
        _restore_attn_backend(old_backend)
