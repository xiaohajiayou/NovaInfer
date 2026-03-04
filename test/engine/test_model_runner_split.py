from __future__ import annotations

from types import SimpleNamespace

import pytest

import llaisys
from llaisys.engine.buffers import CpuGpuBuffer
from llaisys.engine.cpu_model_runner import CPUModelRunner
from llaisys.engine.gpu_model_runner import GPUModelRunner
from llaisys.engine.types import BatchPlan


class _FakeSampler:
    def __init__(self, values: list[int]):
        self._values = values

    def sample_tokens(self, **kwargs):
        out_ids_dev = kwargs["out_ids_dev"]
        out_ids_dev.copy_from_sequence(self._values)
        return out_ids_dev


class _FakeRuntimeAPI:
    def set_device(self, _device_id: int):
        return None

    def event_record(self, _event, _stream):
        return None

    def stream_wait_event(self, _stream, _event):
        return None


def _make_state(n: int, req_ids: list[str]):
    logits = llaisys.Tensor((n, 4), llaisys.DataType.F32, llaisys.DeviceType.CPU, 0)
    plan = BatchPlan(
        token_ids=[1] * n,
        logits_mask=[1] * n,
        temperatures=[1.0] * n,
        top_ps=[1.0] * n,
        top_ks=[0] * n,
        seeds=[0] * n,
        has_seeds=[0] * n,
    )
    return SimpleNamespace(logits=logits, plan=plan, sampled_req_ids=req_ids, keepalive=[])


def test_sample_tokens_cpu_gpu_runner_semantics_match():
    req_ids = ["req-1", "req-2"]
    state_cpu = _make_state(2, req_ids)
    state_gpu = _make_state(2, req_ids)

    cpu_runner = CPUModelRunner.__new__(CPUModelRunner)
    cpu_runner._execute_model_state = state_cpu
    cpu_runner._sampled_ids_buf = llaisys.Tensor((4,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    cpu_runner._max_num_reqs = 4
    cpu_runner.sampler = _FakeSampler([11, 22])

    sampled_cpu, sampled_req_ids_cpu = cpu_runner.sample_tokens(None)
    assert sampled_cpu.tolist() == [11, 22]
    assert sampled_req_ids_cpu == req_ids

    gpu_runner = GPUModelRunner.__new__(GPUModelRunner)
    gpu_runner._execute_model_state = state_gpu
    sampled_buf = CpuGpuBuffer.__new__(CpuGpuBuffer)
    sampled_buf.cpu = llaisys.Tensor(
        (4,),
        llaisys.DataType.I64,
        llaisys.DeviceType.CPU,
        0,
    )
    sampled_buf.gpu = llaisys.Tensor((4,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    gpu_runner._sampled_ids_buf = sampled_buf
    gpu_runner._max_num_reqs = 4
    gpu_runner.sampler = _FakeSampler([11, 22])
    gpu_runner._device = llaisys.DeviceType.NVIDIA
    gpu_runner._runtime_api = _FakeRuntimeAPI()
    gpu_runner._get_runtime_compute_stream = lambda _device_id: object()
    gpu_runner._get_sampler_done_event = lambda _device_id: object()
    gpu_runner._get_d2h_stream = lambda _device_id: object()

    sampled_gpu, sampled_req_ids_gpu = gpu_runner.sample_tokens(None)
    assert sampled_gpu.tolist() == [11, 22]
    assert sampled_req_ids_gpu == req_ids


def test_cpu_runner_block_metadata_capacity_guard():
    runner = CPUModelRunner.__new__(CPUModelRunner)
    runner._max_block_meta_i32 = 4
    runner._block_meta_buf = llaisys.Tensor((4,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._batch_seq_ids_buf = llaisys.Tensor((2,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)

    seq_ids = llaisys.Tensor((2,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    pos_ids_host = llaisys.Tensor((2,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    plan = BatchPlan(
        token_ids=[101, 102],
        logits_mask=[1, 1],
        q_seq_rows=[0, 1],
        q_pos=[0, 1],
        slot_mapping=[0, 1],
        context_lens=[2, 2],
        batch_seq_ids=[10, 11],
        block_tables=[7, 8],
        block_table_width=1,
    )

    with pytest.raises(RuntimeError, match="max_block_meta_i32"):
        runner._build_attention_metadata(
            plan=plan,
            ntoken=2,
            seq_ids=seq_ids,
            pos_ids_host=pos_ids_host,
            is_block_layout=True,
        )
