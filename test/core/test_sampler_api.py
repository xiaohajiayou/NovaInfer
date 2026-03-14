from __future__ import annotations

import llaisys
from llaisys.engine.sampler import Sampler


def _make_logits(rows: list[list[float]]) -> llaisys.Tensor:
    nrow = len(rows)
    ncol = len(rows[0])
    flat = [v for row in rows for v in row]
    logits = llaisys.Tensor((nrow, ncol), llaisys.DataType.F32, llaisys.DeviceType.CPU, 0)
    logits.copy_from_sequence(flat)
    return logits


def test_sampler_top_k_one_matches_argmax():
    sampler = Sampler(llaisys.DeviceType.CPU, max_num_seqs=4)
    logits = _make_logits([[0.1, 0.2, 0.9, 0.3], [2.0, 1.0, -1.0, 0.0]])
    out = llaisys.Tensor((2,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    sampled = sampler.sample_tokens(
        logits_tensor=logits,
        out_ids_dev=out,
        temperatures=[1.0, 1.0],
        top_ps=[1.0, 1.0],
        top_ks=[1, 1],
        seeds=[123, 456],
        has_seeds=[1, 1],
    )
    assert sampled is not None
    assert sampled.tolist() == [2, 0]


def test_sampler_default_controls_stay_greedy():
    sampler = Sampler(llaisys.DeviceType.CPU, max_num_seqs=2)
    logits = _make_logits([[0.1, 0.8, 0.7, 0.2]])
    out = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    sampled = sampler.sample_tokens(
        logits_tensor=logits,
        out_ids_dev=out,
        temperatures=[1.0],
        top_ps=[1.0],
        top_ks=[0],
        seeds=[0],
        has_seeds=[0],
    )
    assert sampled is not None
    assert sampled.tolist() == [1]


def test_sampler_seed_is_reproducible_and_top_p_respects_candidate_prefix():
    sampler = Sampler(llaisys.DeviceType.CPU, max_num_seqs=2)
    logits = _make_logits([[8.0, 7.0, 1.0, -1.0]])
    out_a = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    out_b = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
    sampled_a = sampler.sample_tokens(
        logits_tensor=logits,
        out_ids_dev=out_a,
        temperatures=[0.8],
        top_ps=[0.6],
        top_ks=[4],
        seeds=[1234],
        has_seeds=[1],
    )
    sampled_b = sampler.sample_tokens(
        logits_tensor=logits,
        out_ids_dev=out_b,
        temperatures=[0.8],
        top_ps=[0.6],
        top_ks=[4],
        seeds=[1234],
        has_seeds=[1],
    )
    assert sampled_a is not None and sampled_b is not None
    assert sampled_a.tolist() == sampled_b.tolist()
    assert sampled_a.tolist()[0] == 0
