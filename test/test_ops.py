import pytest

from test.ops import add as add_ops
from test.ops import argmax as argmax_ops
from test.ops import embedding as embedding_ops
from test.ops import linear as linear_ops
from test.ops import rms_norm as rms_norm_ops
from test.ops import rope as rope_ops
from test.ops import self_attention as self_attention_ops
from test.ops import swiglu as swiglu_ops


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
