import llaisys
import pytest

import torch
from test.test_utils import *
import argparse


def test_tensor():
    torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    llaisys_tensor = llaisys.Tensor(
        (3, 4, 5), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )

    # Test load
    print("===Test load===")
    llaisys_tensor.load(torch_tensor.data_ptr())
    llaisys_tensor.debug()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor, torch_tensor)

    # Test view
    print("===Test view===")
    torch_tensor_view = torch_tensor.view(6, 10)
    llaisys_tensor_view = llaisys_tensor.view(6, 10)
    llaisys_tensor_view.debug()
    assert llaisys_tensor_view.shape() == torch_tensor_view.shape
    assert llaisys_tensor_view.strides() == torch_tensor_view.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_view, torch_tensor_view)

    # Test permute
    print("===Test permute===")
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    llaisys_tensor_perm = llaisys_tensor.permute(2, 0, 1)
    llaisys_tensor_perm.debug()
    assert llaisys_tensor_perm.shape() == torch_tensor_perm.shape
    assert llaisys_tensor_perm.strides() == torch_tensor_perm.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_perm, torch_tensor_perm)

    # Test slice
    print("===Test slice===")
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    llaisys_tensor_slice = llaisys_tensor.slice(2, 1, 4)
    llaisys_tensor_slice.debug()
    assert llaisys_tensor_slice.shape() == torch_tensor_slice.shape
    assert llaisys_tensor_slice.strides() == torch_tensor_slice.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_slice, torch_tensor_slice)


def test_tensor_contiguous_reshape_to_cpu():
    torch_tensor = torch.arange(24, dtype=torch_dtype("i64")).reshape(2, 3, 4)
    llaisys_tensor = llaisys.Tensor(
        (2, 3, 4), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )
    llaisys_tensor.load(torch_tensor.data_ptr())

    # non-contiguous -> contiguous
    torch_perm = torch_tensor.permute(2, 0, 1)
    llaisys_perm = llaisys_tensor.permute(2, 0, 1)
    assert not llaisys_perm.is_contiguous()
    llaisys_ctg = llaisys_perm.contiguous()
    assert llaisys_ctg.is_contiguous()
    assert check_equal(llaisys_ctg, torch_perm.contiguous())

    # reshape on non-contiguous tensor should still be valid (contiguous fallback)
    llaisys_r = llaisys_perm.reshape(4, 6)
    torch_r = torch_perm.reshape(4, 6)
    assert llaisys_r.shape() == torch_r.shape
    assert check_equal(llaisys_r, torch_r)

    # to(cpu) roundtrip
    llaisys_cpu2 = llaisys_r.to(llaisys.DeviceType.CPU, 0)
    assert llaisys_cpu2.device_type() == llaisys.DeviceType.CPU
    assert check_equal(llaisys_cpu2, torch_r)


def test_tensor_contiguous_reshape_to_nvidia():
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    if api.get_device_count() <= 0 or (not torch.cuda.is_available()):
        pytest.skip("NVIDIA device is not available")

    torch_tensor = torch.arange(24, dtype=torch_dtype("i64"), device=torch_device("nvidia", 0)).reshape(2, 3, 4)
    llaisys_tensor = llaisys.Tensor(
        (2, 3, 4),
        dtype=llaisys_dtype("i64"),
        device=llaisys_device("nvidia"),
        device_id=0,
    )
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        torch_tensor.numel() * torch_tensor.element_size(),
        llaisys.MemcpyKind.D2D,
    )

    # Force non-contiguous layout and run device-side contiguous path.
    torch_perm = torch_tensor.permute(2, 0, 1)
    llaisys_perm = llaisys_tensor.permute(2, 0, 1)
    assert not llaisys_perm.is_contiguous()
    llaisys_ctg = llaisys_perm.contiguous()
    assert llaisys_ctg.is_contiguous()
    assert check_equal(llaisys_ctg, torch_perm.contiguous())

    # reshape + to(cpu) for cross-device copy validation.
    llaisys_r = llaisys_perm.reshape(4, 6)
    torch_r = torch_perm.reshape(4, 6)
    assert check_equal(llaisys_r, torch_r)

    llaisys_cpu = llaisys_r.to(llaisys.DeviceType.CPU, 0)
    assert llaisys_cpu.device_type() == llaisys.DeviceType.CPU
    assert check_equal(llaisys_cpu, torch_r.cpu())


if __name__ == "__main__":
    test_tensor()
    test_tensor_contiguous_reshape_to_cpu()
    test_tensor_contiguous_reshape_to_nvidia()

    print("\n\033[92mTest passed!\033[0m\n")
