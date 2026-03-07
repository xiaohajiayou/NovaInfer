from __future__ import annotations

import pytest

import llaisys
from llaisys.engine.buffers import CpuGpuBuffer
from llaisys.engine.cpu_model_runner import CPUModelRunner
from llaisys.engine.gpu_input_batch import InputBatch
from llaisys.engine.gpu_model_runner import GPUModelRunner, _ExecuteModelState
from llaisys.engine.sequence import Sequence
from llaisys.engine.types import SamplingParams


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


def _make_state(n: int, seq_ids: list[int]):
    logits = llaisys.Tensor((n, 4), llaisys.DataType.F32, llaisys.DeviceType.CPU, 0)
    return _ExecuteModelState(
        logits=logits,
        sampled_seq_ids=seq_ids,
        keepalive=[],
        temperatures=[1.0] * n,
        top_ps=[1.0] * n,
        top_ks=[0] * n,
        seeds=[0] * n,
        has_seeds=[0] * n,
    )


def test_sample_tokens_cpu_gpu_runner_semantics_match():
    seq_ids = [1, 2]
    state_cpu = _make_state(2, seq_ids)
    state_gpu = _make_state(2, seq_ids)

    cpu_runner = CPUModelRunner.__new__(CPUModelRunner)
    cpu_runner._execute_model_state = state_cpu
    cpu_runner._sampled_ids_buf = llaisys.Tensor((4,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    cpu_runner._max_num_reqs = 4
    cpu_runner.sampler = _FakeSampler([11, 22])

    sampled_cpu = cpu_runner.sample_tokens(None)
    assert sampled_cpu == [11, 22]

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

    sampled_gpu = gpu_runner.sample_tokens(None)
    assert sampled_gpu == [11, 22]


def test_cpu_runner_block_metadata_capacity_guard():
    runner = CPUModelRunner.__new__(CPUModelRunner)
    runner.input_batch = InputBatch(max_num_reqs=2, max_num_batched_tokens=4, max_block_table_width=1)
    sp = SamplingParams(max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0)
    seq1 = Sequence(seq_id=1, token_ids=[1, 2], sampling_params=sp, block_size=1)
    seq2 = Sequence(seq_id=2, token_ids=[3], sampling_params=sp, block_size=1)
    seq1.block_table = [7]
    seq2.block_table = [8]
    runner.input_batch.scheduled_seqs = [seq1, seq2]
    runner.input_batch.seq_id_to_index = {1: 0, 2: 1}
    runner.input_batch.req_num_scheduled_tokens_step = [2, 1]
    runner.input_batch.req_num_computed_tokens_step = [0, 0]
    runner.input_batch.block_table_width = 1
    runner.input_batch.block_table_cpu[0, 0] = 7
    runner.input_batch.block_table_cpu[1, 0] = 8

    seq_ids = llaisys.Tensor((2,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    pos_ids_host = llaisys.Tensor((2,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)

    with pytest.raises(ValueError, match="sum\\(req_num_scheduled_tokens\\)"):
        runner._build_attention_metadata(
            ntoken=2,
            seq_ids=seq_ids,
            pos_ids_host=pos_ids_host,
            is_block_layout=True,
        )
