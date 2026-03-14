from __future__ import annotations

import pytest

import llaisys
from llaisys.engine.buffers import CpuGpuBuffer
from llaisys.engine.config import EngineConfig
from llaisys.engine.cpu_model_runner import CPUModelRunner
from llaisys.engine.gpu_model_runner import GPUModelRunner
from llaisys.engine.sampler import Sampler
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


def test_prepare_sample_auto_seed_is_stable_across_batch_order():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = Sampler(llaisys.DeviceType.CPU, max_num_seqs=8)

    params = SamplingParams(max_new_tokens=4, top_k=40, top_p=0.95, temperature=0.6)
    seq_a = Sequence(seq_id=101, token_ids=[1, 2, 3], sampling_params=params, block_size=1)
    seq_b = Sequence(seq_id=202, token_ids=[4, 5, 6], sampling_params=params, block_size=1)

    temps_ab, top_ps_ab, top_ks_ab, seeds_ab, has_seeds_ab = runner.prepare_sample([seq_a, seq_b])
    temps_ba, top_ps_ba, top_ks_ba, seeds_ba, has_seeds_ba = runner.prepare_sample([seq_b, seq_a])

    assert temps_ab == temps_ba[::-1]
    assert top_ps_ab == top_ps_ba[::-1]
    assert top_ks_ab == top_ks_ba[::-1]
    assert has_seeds_ab == [1, 1]
    assert has_seeds_ba == [1, 1]

    seeds_by_seq_ab = {seq.seq_id: seed for seq, seed in zip([seq_a, seq_b], seeds_ab)}
    seeds_by_seq_ba = {seq.seq_id: seed for seq, seed in zip([seq_b, seq_a], seeds_ba)}
    assert seeds_by_seq_ab == seeds_by_seq_ba

    seq_a.append_token(7)
    _, _, _, advanced_seeds, advanced_has_seeds = runner.prepare_sample([seq_a])
    assert advanced_has_seeds == [1]
    assert advanced_seeds[0] != seeds_by_seq_ab[seq_a.seq_id]


def test_prepare_sample_keeps_greedy_rows_seedless():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = Sampler(llaisys.DeviceType.CPU, max_num_seqs=4)

    greedy = SamplingParams(max_new_tokens=2, top_k=0, top_p=1.0, temperature=1.0)
    seq = Sequence(seq_id=7, token_ids=[1, 2], sampling_params=greedy, block_size=1)

    _, _, _, seeds, has_seeds = runner.prepare_sample([seq])
    assert seeds == [0]
    assert has_seeds == [0]
