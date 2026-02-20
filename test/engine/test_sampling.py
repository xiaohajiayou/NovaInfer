from __future__ import annotations

import numpy as np

from llaisys.engine.sampling import Sampler
from llaisys.engine.types import SamplingParams


def test_sampling_argmax_when_temperature_zero():
    sampler = Sampler(seed=0)
    logits = [np.array([0.1, 0.2, 0.9, 0.4], dtype=np.float32)]
    out = sampler.sample(logits, SamplingParams(temperature=0.0))
    assert out == [2]


def test_sampling_topk1_is_greedy():
    sampler = Sampler(seed=42)
    logits = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)]
    out = sampler.sample(logits, SamplingParams(top_k=1, top_p=1.0, temperature=1.0))
    assert out == [3]


def test_sampling_topp_filters_tail_tokens():
    sampler = Sampler(seed=42)
    logits = [np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)]
    # With top_p=0.9, token 0 should be excluded for this distribution.
    got = set()
    for i in range(50):
        out = sampler.sample(logits, SamplingParams(top_k=0, top_p=0.9, temperature=1.0, seed=i))
        got.add(int(out[0]))
    assert 0 not in got
    assert got.issubset({1, 2, 3})


def test_sampling_topk_limits_candidate_set():
    sampler = Sampler(seed=1)
    logits = [np.array([0.5, 0.8, 1.0, 1.2], dtype=np.float32)]
    got = set()
    for i in range(30):
        out = sampler.sample(logits, SamplingParams(top_k=2, top_p=1.0, temperature=1.0, seed=i))
        got.add(int(out[0]))
    assert got.issubset({2, 3})
