import pytest
import torch
import llaisys

from test.ops import add as add_ops
from test.ops import argmax as argmax_ops
from test.ops import embedding as embedding_ops
from test.ops import linear as linear_ops
from test.ops import rms_norm as rms_norm_ops
from test.ops import rope as rope_ops
from test.ops import self_attention as self_attention_ops
from test.ops import swiglu as swiglu_ops


def _has_nvidia_runtime() -> bool:
    try:
        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        return api.get_device_count() > 0 and torch.cuda.is_available()
    except Exception:
        return False


def _has_nvidia_bf16() -> bool:
    if not _has_nvidia_runtime():
        return False
    try:
        x = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda:0")
        _ = x + x
        return True
    except Exception:
        return False


@pytest.mark.ops
def test_ops_add_cpu():
    add_ops.test_op_add(shape=(2, 3), dtype_name="f32", device_name="cpu", profile=False)


@pytest.mark.ops
def test_ops_argmax_cpu():
    argmax_ops.test_op_argmax(shape=(16,), dtype_name="f32", device_name="cpu", profile=False)


@pytest.mark.ops
def test_ops_embedding_cpu():
    embedding_ops.test_op_embedding(idx_shape=(8,), embd_shape=(64, 32), dtype_name="f32", device_name="cpu", profile=False)


@pytest.mark.ops
def test_ops_linear_cpu():
    linear_ops.test_op_linear(
        out_shape=(8, 16),
        x_shape=(8, 32),
        w_shape=(16, 32),
        use_bias=True,
        dtype_name="f32",
        device_name="cpu",
        profile=False,
    )


@pytest.mark.ops
def test_ops_rms_norm_cpu():
    rms_norm_ops.test_op_rms_norm(shape=(8, 32), dtype_name="f32", device_name="cpu", profile=False)


@pytest.mark.ops
def test_ops_rope_cpu():
    rope_ops.test_op_rope(shape=(8, 2, 16), start_end=(0, 8), dtype_name="f32", device_name="cpu", profile=False)


@pytest.mark.ops
def test_ops_self_attention_cpu():
    self_attention_ops.test_op_self_attention(
        qlen=4, kvlen=8, nh=4, nkvh=2, hd=8, dtype_name="f32", device_name="cpu", profile=False
    )


@pytest.mark.ops
def test_ops_swiglu_cpu():
    swiglu_ops.test_op_swiglu(shape=(8, 32), dtype_name="f32", device_name="cpu", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_embedding_nvidia_f32():
    embedding_ops.test_op_embedding(idx_shape=(8,), embd_shape=(64, 32), dtype_name="f32", device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_rms_norm_nvidia_f32():
    rms_norm_ops.test_op_rms_norm(shape=(8, 32), dtype_name="f32", device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_linear_nvidia_f32():
    linear_ops.test_op_linear(
        out_shape=(8, 16),
        x_shape=(8, 32),
        w_shape=(16, 32),
        use_bias=True,
        dtype_name="f32",
        device_name="nvidia",
        profile=False,
    )


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_linear_nvidia_f16():
    linear_ops.test_op_linear(
        out_shape=(8, 16),
        x_shape=(8, 32),
        w_shape=(16, 32),
        use_bias=True,
        dtype_name="f16",
        atol=1e-3,
        rtol=1e-3,
        device_name="nvidia",
        profile=False,
    )


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_bf16(), reason="NVIDIA bf16 unavailable")
def test_ops_linear_nvidia_bf16():
    linear_ops.test_op_linear(
        out_shape=(8, 16),
        x_shape=(8, 32),
        w_shape=(16, 32),
        use_bias=True,
        dtype_name="bf16",
        atol=1e-2,
        rtol=1e-2,
        device_name="nvidia",
        profile=False,
    )


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_swiglu_nvidia_f32():
    swiglu_ops.test_op_swiglu(shape=(8, 32), dtype_name="f32", device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_swiglu_nvidia_f16():
    swiglu_ops.test_op_swiglu(shape=(8, 32), dtype_name="f16", atol=1e-3, rtol=1e-3, device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_bf16(), reason="NVIDIA bf16 unavailable")
def test_ops_swiglu_nvidia_bf16():
    swiglu_ops.test_op_swiglu(shape=(8, 32), dtype_name="bf16", atol=1e-2, rtol=1e-2, device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_argmax_nvidia_f32():
    argmax_ops.test_op_argmax(shape=(16,), dtype_name="f32", device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_argmax_nvidia_f16():
    argmax_ops.test_op_argmax(shape=(16,), dtype_name="f16", device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_bf16(), reason="NVIDIA bf16 unavailable")
def test_ops_argmax_nvidia_bf16():
    argmax_ops.test_op_argmax(shape=(16,), dtype_name="bf16", device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_rope_nvidia_f32():
    rope_ops.test_op_rope(shape=(8, 2, 16), start_end=(0, 8), dtype_name="f32", device_name="nvidia", profile=False)


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_rope_nvidia_f16():
    rope_ops.test_op_rope(
        shape=(8, 2, 16), start_end=(0, 8), dtype_name="f16", atol=1e-3, rtol=1e-3, device_name="nvidia", profile=False
    )


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_bf16(), reason="NVIDIA bf16 unavailable")
def test_ops_rope_nvidia_bf16():
    rope_ops.test_op_rope(
        shape=(8, 2, 16), start_end=(0, 8), dtype_name="bf16", atol=1e-2, rtol=1e-2, device_name="nvidia", profile=False
    )


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_self_attention_nvidia_f32():
    self_attention_ops.test_op_self_attention(
        qlen=4, kvlen=8, nh=4, nkvh=2, hd=8, dtype_name="f32", device_name="nvidia", profile=False
    )


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_runtime(), reason="NVIDIA runtime unavailable")
def test_ops_self_attention_nvidia_f16():
    self_attention_ops.test_op_self_attention(
        qlen=4, kvlen=8, nh=4, nkvh=2, hd=8, dtype_name="f16", atol=1e-3, rtol=1e-3, device_name="nvidia", profile=False
    )


@pytest.mark.ops
@pytest.mark.skipif(not _has_nvidia_bf16(), reason="NVIDIA bf16 unavailable")
def test_ops_self_attention_nvidia_bf16():
    self_attention_ops.test_op_self_attention(
        qlen=4, kvlen=8, nh=4, nkvh=2, hd=8, dtype_name="bf16", atol=1e-2, rtol=1e-2, device_name="nvidia", profile=False
    )
