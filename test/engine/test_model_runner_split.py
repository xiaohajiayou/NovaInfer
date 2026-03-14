from __future__ import annotations

import pytest
import types

import llaisys
from llaisys.engine.buffers import CpuGpuBuffer
from llaisys.engine.config import EngineConfig
from llaisys.engine.cpu_model_runner import CPUModelRunner
from llaisys.engine.gpu_model_runner import GPUModelRunner, PreparedTensors
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
    return (
        logits,
        len(seq_ids),
        [],
        [1.0] * n,
        [1.0] * n,
        [0] * n,
        [0] * n,
        [0] * n,
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
    gpu_runner._compute_streams = {}
    gpu_runner._get_compute_stream = lambda _device_id: object()
    gpu_runner._get_sampler_done_event = lambda _device_id: object()
    gpu_runner._get_d2h_stream = lambda _device_id: object()

    sampled_gpu = gpu_runner.sample_tokens(None)
    assert sampled_gpu == [11, 22]


def test_cpu_runner_block_metadata_capacity_guard():
    runner = CPUModelRunner.__new__(CPUModelRunner)
    runner._config = EngineConfig(
        kv_cache_block_size=1,
    )
    runner._max_num_reqs = 2
    runner._max_num_tokens = 4
    runner._max_block_table_width = 1
    runner._input_ids_buf = llaisys.Tensor((4,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    runner._pos_ids_buf = llaisys.Tensor((4,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    runner._seq_ids_buf = llaisys.Tensor((4,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    runner._output_ids_buf = llaisys.Tensor((4,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    runner._cu_seqlens_q_buf = llaisys.Tensor((3,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._cu_seqlens_k_buf = llaisys.Tensor((3,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._slot_mapping_buf = llaisys.Tensor((4,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._block_tables_buf = llaisys.Tensor((2,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    sp = SamplingParams(max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0)
    seq1 = Sequence(seq_id=1, token_ids=[1, 2], sampling_params=sp, block_size=1)
    seq2 = Sequence(seq_id=2, token_ids=[3], sampling_params=sp, block_size=1)
    seq1.block_table = [7, 8]
    seq2.block_table = [8]

    with pytest.raises(RuntimeError, match="n_block_elems exceeds configured BLOCK metadata capacity"):
        runner.prepare_prefill([seq1, seq2])


def test_gpu_runner_prefill_recompute_replays_full_prefix_not_prompt_only():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._config = EngineConfig(
        kv_cache_block_size=4,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        max_model_len=16,
        device=llaisys.DeviceType.NVIDIA,
    )
    runner._device = llaisys.DeviceType.CPU
    runner._paged_attn_backend = "native"
    runner._max_num_reqs = 2
    runner._max_num_tokens = 16
    runner._max_num_cudnn_rows = 2
    runner._max_block_table_width = 4
    runner._input_ids_buf = llaisys.Tensor((16,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    runner._pos_ids_buf = llaisys.Tensor((16,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    runner._output_ids_buf = llaisys.Tensor((16,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    runner._cu_seqlens_q_buf = llaisys.Tensor((3,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._cu_seqlens_k_buf = llaisys.Tensor((3,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._slot_mapping_buf = llaisys.Tensor((16,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._block_tables_buf = llaisys.Tensor((8,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._cudnn_seq_lens_q_buf = llaisys.Tensor((2,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._cudnn_seq_lens_kv_buf = llaisys.Tensor((2,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._cudnn_page_table_buf = llaisys.Tensor((8,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    runner._cudnn_qo_ragged_offset_buf = llaisys.Tensor((3,), llaisys.DataType.I32, llaisys.DeviceType.CPU, 0)
    captured = {}

    def _fake_build_block_tensors(self, **kwargs):
        captured["input_ids"] = kwargs["input_ids"].tolist()
        captured["positions"] = kwargs["positions"].tolist()
        return PreparedTensors(
            input_ids=self._input_ids_buf,
            pos_ids=self._pos_ids_buf,
            logits_indices=self._output_ids_buf,
            n_outputs=len(kwargs["seqs"]),
            keepalive=[],
        )

    runner._build_block_tensors = types.MethodType(_fake_build_block_tensors, runner)

    sp = SamplingParams(max_new_tokens=4, top_k=1, top_p=1.0, temperature=1.0)
    seq = Sequence(seq_id=1, token_ids=[10, 11, 12, 13, 21, 22], sampling_params=sp, block_size=4)
    seq.num_prompt_tokens = 4
    seq.num_cached_tokens = 4
    seq.block_table = [3, 4]

    prepared = runner.prepare_prefill([seq])
    assert prepared is not None
    assert prepared.n_outputs == 1
    assert captured["input_ids"] == [21, 22]
    assert captured["positions"] == [4, 5]
