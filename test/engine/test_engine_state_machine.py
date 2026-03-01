
from llaisys.engine.llm_engine import LLMEngine
from llaisys.engine.types import RequestStatus, SamplingParams
from llaisys.engine.types import BatchPlan
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

    def on_plan(self, plan: BatchPlan) -> None:
        self.decode_calls.append(
            {
                "token_ids": list(plan.token_ids),
                "pos_ids": list(plan.pos_ids) if plan.pos_ids is not None else None,
                "seq_ids": list(plan.seq_ids) if plan.seq_ids is not None else None,
                "logits_mask": list(plan.logits_mask) if plan.logits_mask is not None else None,
                "slot_mapping": list(plan.slot_mapping) if plan.slot_mapping is not None else None,
                "context_lens": list(plan.context_lens) if plan.context_lens is not None else None,
                "batch_seq_ids": list(plan.batch_seq_ids) if plan.batch_seq_ids is not None else None,
                "block_tables": list(plan.block_tables) if plan.block_tables is not None else None,
                "block_table_width": int(plan.block_table_width),
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

    def on_plan(self, plan: BatchPlan) -> None:
        self.used_tokens = max(self.used_tokens, len(plan.token_ids))

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


def test_state_machine_stopped_path():
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=4))
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
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=99))
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
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=1, end_token_id=4))
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
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=4))
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
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=99))
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
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=99))
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
    engine = LLMEngine(
        model_runner=DummyRunner(max_seq_len=64, end_token_id=99),
        kv_cache_capacity_tokens=4,
    )
    out = engine.generate(
        inputs=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_new_tokens=2, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.status == RequestStatus.FINISHED_ABORTED
    assert out.finish_reason == "aborted"


def test_engine_exposes_kv_cache_stats():
    engine = LLMEngine(
        model_runner=DummyRunner(max_seq_len=32, end_token_id=99),
        kv_cache_capacity_tokens=32,
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
    engine = LLMEngine(model_runner=DummyRunner(max_seq_len=32, end_token_id=99))
    assert engine.reset_prefix_cache() == 0


def test_prefix_attach_and_uncached_prefill_suffix():
    runner = PrefixProbeRunner(max_seq_len=64, end_token_id=99)
    engine = LLMEngine(
        model_runner=runner,
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
    assert last["slot_mapping"] == [4]
    assert last["context_lens"] == [5]
    assert last["batch_seq_ids"] == [2]


def test_prefix_reuses_after_finished_request_freed():
    runner = PrefixProbeRunner(max_seq_len=64, end_token_id=99)
    engine = LLMEngine(
        model_runner=runner,
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
    assert 1 in runner.request_free_calls

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
    engine = LLMEngine(
        model_runner=runner,
        kv_cache_block_size=2,
        kv_cache_layout=KvCacheLayout.BLOCK,
        max_batch_size=1,
    )
    out = engine.generate(
        inputs=[30, 31, 32, 33],
        sampling_params=SamplingParams(max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0),
    )
    assert out.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert 1 in runner.request_free_calls


def test_engine_runtime_peak_watermark_is_observed_in_block_mode():
    runner = KvStatsProbeRunner(max_seq_len=64, end_token_id=99)
    engine = LLMEngine(
        model_runner=runner,
        kv_cache_block_size=2,
        kv_cache_layout=KvCacheLayout.BLOCK,
        kv_cache_capacity_tokens=64,
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
