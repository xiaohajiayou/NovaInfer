
from unittest.mock import patch

from llaisys.engine.config import EngineConfig
from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.types import RequestStatus, SamplingParams
from llaisys.libllaisys.model import KvCacheLayout
from test.utils.dummy_model_runner import DummyModelRunner



class DummyRunner(DummyModelRunner):
    pass


class PrefixProbeRunner(DummyRunner):
    def __init__(self, max_seq_len=32, end_token_id=99):
        super().__init__(
            max_seq_len=max_seq_len,
            end_token_id=end_token_id,
            kv_cache_layout=KvCacheLayout.BLOCK,
        )
        self.decode_calls = []

    def on_plan(self, step: dict) -> None:
        token_ids = []
        pos_ids = []
        seq_ids = []
        logits_mask = []
        for seq in step["scheduled_seqs"]:
            if step["is_prefill"]:
                start = max(0, int(seq.num_cached_tokens))
                toks = [int(t) for t in seq.prompt_token_ids[start:]]
                poss = list(range(start, start + len(toks)))
                mask = [0] * len(toks)
                if mask:
                    mask[-1] = 1
            else:
                toks = [int(seq.last_token)]
                poss = [len(seq) - 1]
                mask = [1]
            token_ids.extend(toks)
            pos_ids.extend(poss)
            seq_ids.extend([int(seq.seq_id)] * len(toks))
            logits_mask.extend(mask)

        self.decode_calls.append(
            {
                "token_ids": token_ids,
                "pos_ids": pos_ids,
                "seq_ids": seq_ids,
                "logits_mask": logits_mask,
                "req_num_scheduled_tokens": (
                    list(step["req_num_scheduled_tokens"]) if step["req_num_scheduled_tokens"] is not None else None
                ),
                "req_num_computed_tokens": (
                    list(step["req_num_computed_tokens"]) if step["req_num_computed_tokens"] is not None else None
                ),
                "block_table_width": int(step["block_table_width"]),
            }
        )


class KvStatsProbeRunner(DummyRunner):
    def __init__(self, max_seq_len=64, end_token_id=99):
        super().__init__(
            max_seq_len=max_seq_len,
            end_token_id=end_token_id,
            kv_cache_layout=KvCacheLayout.BLOCK,
        )
        self.used_tokens = 0
        self.capacity_tokens = 128

    def on_plan(self, step: dict) -> None:
        self.used_tokens = max(self.used_tokens, int(step["n_tokens"]))

    def request_free(self, seq_id: int) -> int:
        super().request_free(seq_id)
        self.used_tokens = 0
        return 0

    def kv_stats(self) -> dict:
        free_tokens = max(0, int(self.capacity_tokens - self.used_tokens))
        return {
            "capacity_tokens": int(self.capacity_tokens),
            "used_tokens": int(self.used_tokens),
            "free_tokens": int(free_tokens),
            # Force engine-level observed watermark path to take effect.
            "peak_used_tokens": 0,
        }


class _InjectedWorker:
    def __init__(self, runner: DummyModelRunner) -> None:
        self._runner = runner

    @property
    def model_runner(self):
        return self._runner

    def close(self) -> None:
        return None

    def execute_model(self, scheduler_outputs):
        return self._runner.execute_model(scheduler_outputs)

    def sample_tokens(self):
        # Unit path: avoid allocating native tensors from dummy runner.
        state = getattr(self._runner, "_execute_state", None)
        if state is not None:
            output_indices, token_ids = state
            self._runner._execute_state = None
            vocab_size = int(getattr(self._runner, "vocab_size", 32000))
            return [int((int(token_ids[i]) + 1) % vocab_size) for i in output_indices]
        sampled = self._runner.sample_tokens()
        if sampled is None:
            return None
        tolist = getattr(sampled, "tolist", None)
        if callable(tolist):
            return [int(t) for t in tolist()]
        return [int(t) for t in sampled]

    def free_request(self, seq_id: int) -> None:
        self._runner.request_free(int(seq_id))

    def decode_tokens(self, token_ids: list[int]) -> str | None:
        _ = token_ids
        return None


def _make_engine(runner: DummyModelRunner, **kwargs) -> LLMEngine:
    cfg_kwargs = dict(kwargs)
    if "max_num_seqs" not in cfg_kwargs and "max_batch_size" in cfg_kwargs:
        cfg_kwargs["max_num_seqs"] = int(cfg_kwargs.pop("max_batch_size"))
    else:
        cfg_kwargs.pop("max_batch_size", None)
    cfg_kwargs.setdefault("max_model_len", int(runner.max_seq_len))
    cfg_kwargs.setdefault("end_token_id", int(runner.end_token_id))
    cfg_kwargs.setdefault("kv_cache_layout", KvCacheLayout(int(runner.kv_cache_layout)))
    if (
        KvCacheLayout(int(cfg_kwargs["kv_cache_layout"])) == KvCacheLayout.BLOCK
        and int(cfg_kwargs.get("num_kvcache_blocks", 0) or 0) <= 0
    ):
        block_size = max(1, int(cfg_kwargs.get("kv_cache_block_size", 256)))
        max_model_len = max(1, int(cfg_kwargs["max_model_len"]))
        cfg_kwargs["num_kvcache_blocks"] = (max_model_len + block_size - 1) // block_size
    cfg = EngineConfig(**cfg_kwargs)
    fake_worker = _InjectedWorker(runner)
    with patch("llaisys.engine.llm_engine.Worker", side_effect=lambda *args, **_kw: fake_worker):
        return LLMEngine(config=cfg)


def test_state_machine_stopped_path():
    engine = _make_engine(DummyRunner(max_seq_len=32, end_token_id=4))
    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
    )

    assert out.status == RequestStatus.FINISHED_STOPPED
    assert out.finish_reason == "eos_token"
    assert out.token_ids == [1, 2, 3, 4]

    history = engine.get_request_history(out.request_id)
    assert history == [
        RequestStatus.WAITING,
        RequestStatus.RUNNING,
        RequestStatus.FINISHED_STOPPED,
    ]


def test_state_machine_length_capped_path():
    engine = _make_engine(DummyRunner(max_seq_len=32, end_token_id=99))
    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=2, top_k=1, top_p=1.0, temperature=1.0),
    )

    assert out.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert out.finish_reason == "length"
    assert out.token_ids == [1, 2, 3, 4]

    history = engine.get_request_history(out.request_id)
    assert history == [
        RequestStatus.WAITING,
        RequestStatus.RUNNING,
        RequestStatus.FINISHED_LENGTH_CAPPED,
    ]


def test_state_machine_aborted_on_invalid_prompt():
    engine = _make_engine(DummyRunner(max_seq_len=1, end_token_id=4))
    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=2, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.status == RequestStatus.FINISHED_ABORTED
    assert out.finish_reason == "aborted"

    req_id = engine.last_request_id
    assert req_id is not None
    assert engine.get_request_status(req_id) == RequestStatus.FINISHED_ABORTED
    history = engine.get_request_history(req_id)
    assert history == [
        RequestStatus.WAITING,
        RequestStatus.FINISHED_ABORTED,
    ]


def test_submit_step_collect_contract():
    engine = _make_engine(DummyRunner(max_seq_len=32, end_token_id=4))
    req_id = engine.submit(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
    )

    # drive engine loop explicitly
    for _ in range(16):
        out = engine.collect(req_id)
        if out is not None:
            break
        _ = engine.step()

    out = engine.collect(req_id)
    assert out is not None
    assert out.request_id == req_id
    assert out.status == RequestStatus.FINISHED_STOPPED
    assert out.finish_reason == "eos_token"
    assert out.token_ids == [1, 2, 3, 4]
    assert out.usage is not None
    assert out.usage["prompt_tokens"] == 2
    assert out.usage["completion_tokens"] == 2
    assert out.usage["total_tokens"] == 4
    assert out.text is None


def test_cancel_contract():
    engine = _make_engine(DummyRunner(max_seq_len=32, end_token_id=99))
    req_id = engine.submit(
        inputs=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert engine.cancel(req_id) is True

    out = engine.collect(req_id)
    assert out is not None
    assert out.status == RequestStatus.FINISHED_ABORTED
    assert out.finish_reason == "aborted"


def test_stop_string_contract():
    engine = _make_engine(DummyRunner(max_seq_len=32, end_token_id=99))
    out = engine.generate(
        inputs=[1, 2],
        sampling_params=SamplingParams(
            max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0, stop=("de",)
        ),
    )
    # Without tokenizer/model path, dummy runner does not produce text, so
    # stop-string matching is not applicable in this unit path.
    assert out.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert out.finish_reason == "length"
    assert out.text is None


def test_aborted_when_prompt_exceeds_scheduler_budget():
    engine = _make_engine(
        DummyRunner(max_seq_len=64, end_token_id=99),
        max_num_batched_tokens=4,
    )
    out = engine.generate(
        inputs=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_new_tokens=2, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.status == RequestStatus.FINISHED_ABORTED
    assert out.finish_reason == "aborted"


def test_engine_exposes_kv_cache_stats():
    engine = _make_engine(
        DummyRunner(max_seq_len=32, end_token_id=99),
        kv_cache_block_size=16,
    )
    stats = engine.kv_cache_stats()
    assert "allocator" in stats
    alloc = stats["allocator"]
    assert alloc["block_size"] == 16
    assert alloc["num_blocks"] == 2
    assert alloc["used_blocks"] == 0
    assert alloc["peak_used_blocks"] == 0
    assert alloc["free_blocks"] == 2
    assert alloc["prefix_hits"] == 0
    assert alloc["prefix_misses"] == 0
    assert alloc["prefix_saved_tokens"] == 0


def test_engine_reset_prefix_cache_contract():
    engine = _make_engine(DummyRunner(max_seq_len=32, end_token_id=99))
    assert engine.reset_prefix_cache() == 0


def test_prefix_attach_and_uncached_prefill_suffix():
    runner = PrefixProbeRunner(max_seq_len=64, end_token_id=99)
    engine = _make_engine(
        runner,
        kv_cache_block_size=2,
        kv_cache_layout=KvCacheLayout.BLOCK,
        max_batch_size=1,
    )

    req1 = engine.submit(
        inputs=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert req1
    _ = engine.step()  # prefill req1

    req2 = engine.submit(
        inputs=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(max_new_tokens=8, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert req2
    _ = engine.step()  # prefill req2 (should feed uncached suffix with explicit block metadata)

    # The second prefill should only feed the uncached suffix token 14 at position 4.
    last = runner.decode_calls[-1]
    assert last["token_ids"] == [14]
    assert last["pos_ids"] == [4]
    assert last["req_num_scheduled_tokens"] == [1]
    assert last["req_num_computed_tokens"] == [4]


def test_prefix_reuses_after_finished_request_freed():
    runner = PrefixProbeRunner(max_seq_len=64, end_token_id=99)
    engine = _make_engine(
        runner,
        kv_cache_block_size=2,
        kv_cache_layout=KvCacheLayout.BLOCK,
        max_batch_size=1,
    )

    out1 = engine.generate(
        inputs=[20, 21, 22, 23],
        sampling_params=SamplingParams(max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out1.status == RequestStatus.FINISHED_LENGTH_CAPPED
    # nano-vllm style: finished request can be freed and still leave hash index reusable.
    assert len(runner.request_free_calls) >= 1

    req2 = engine.submit(
        inputs=[20, 21, 22, 23, 24],
        sampling_params=SamplingParams(max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert req2
    _ = engine.step()  # prefill req2
    last = runner.decode_calls[-1]
    assert last["token_ids"] == [24]
    assert last["pos_ids"] == [4]


def test_finished_request_releases_blocks_in_block_mode():
    runner = PrefixProbeRunner(max_seq_len=64, end_token_id=99)
    engine = _make_engine(
        runner,
        kv_cache_block_size=2,
        kv_cache_layout=KvCacheLayout.BLOCK,
        max_batch_size=1,
    )
    out = engine.generate(
        inputs=[30, 31, 32, 33],
        sampling_params=SamplingParams(max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert len(runner.request_free_calls) >= 1


def test_engine_runtime_peak_watermark_is_observed_in_block_mode():
    runner = KvStatsProbeRunner(max_seq_len=64, end_token_id=99)
    engine = _make_engine(
        runner,
        kv_cache_block_size=2,
        kv_cache_layout=KvCacheLayout.BLOCK,
        max_batch_size=1,
    )
    out = engine.generate(
        inputs=[40, 41, 42, 43],
        sampling_params=SamplingParams(max_new_tokens=2, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.status == RequestStatus.FINISHED_LENGTH_CAPPED

    stats = engine.kv_cache_stats()
    assert "runtime" in stats
    runtime = stats["runtime"]
    assert isinstance(runtime, dict)
    assert int(runtime["used_tokens"]) == 0
    assert int(runtime["peak_used_tokens"]) > 0


if __name__ == "__main__":
    test_state_machine_stopped_path()
    test_state_machine_length_capped_path()
    test_state_machine_aborted_on_invalid_prompt()
    test_submit_step_collect_contract()
    test_cancel_contract()
    test_stop_string_contract()
    print("\033[92mtest_engine_state_machine passed!\033[0m")
