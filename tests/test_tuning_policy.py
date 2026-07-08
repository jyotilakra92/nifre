import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autotune.classifier import WorkloadLabels
from autotune.policy import PolicyConfig, TuningGoal, TuningPolicy
from inference.data_model import EngineConfig


def _config(**kwargs) -> EngineConfig:
    defaults = dict(
        max_concurrent_requests=2,
        prefill_chunk_size=128,
        max_tokens_per_step=2048,
        use_paged_kv_cache=True,
        use_prefix_cache=True,
    )
    defaults.update(kwargs)
    return EngineConfig(**defaults)


def test_policy_holds_on_error_elevated():
    policy = TuningPolicy()
    plan = policy.propose(
        _config(),
        WorkloadLabels(error_elevated=True, latency_sensitive=True),
        TuningGoal.LATENCY,
    )
    assert plan is None


def test_policy_latency_sensitive_decreases_chunk_and_budget():
    policy = TuningPolicy()
    current = _config(prefill_chunk_size=256, max_tokens_per_step=1024)
    plan = policy.propose(
        current,
        WorkloadLabels(latency_sensitive=True),
        TuningGoal.LATENCY,
    )

    assert plan is not None
    assert plan.config.prefill_chunk_size == 192
    assert plan.config.max_tokens_per_step == 768
    assert "prefill_chunk_size" in plan.changes
    assert "max_tokens_per_step" in plan.changes


def test_policy_throughput_queue_high_increases_concurrency_before_cache_init():
    policy = TuningPolicy()
    current = _config(max_concurrent_requests=2)
    plan = policy.propose(
        current,
        WorkloadLabels(queue_high=True),
        TuningGoal.THROUGHPUT,
        cache_initialized=False,
    )

    assert plan is not None
    assert plan.config.max_concurrent_requests == 3
    assert "raise concurrency" in plan.reason


def test_policy_throughput_queue_high_skips_concurrency_after_cache_init():
    policy = TuningPolicy()
    plan = policy.propose(
        _config(max_concurrent_requests=2),
        WorkloadLabels(queue_high=True),
        TuningGoal.THROUGHPUT,
        cache_initialized=True,
    )
    assert plan is None


def test_policy_throughput_low_increases_token_budget():
    policy = TuningPolicy()
    current = _config(max_tokens_per_step=1024)
    plan = policy.propose(
        current,
        WorkloadLabels(throughput_low=True),
        TuningGoal.THROUGHPUT,
    )

    assert plan is not None
    assert plan.config.max_tokens_per_step == 1280
    assert "token budget" in plan.reason


def test_policy_prefix_friendly_enables_prefix_cache():
    policy = TuningPolicy()
    current = _config(use_prefix_cache=False)
    plan = policy.propose(
        current,
        WorkloadLabels(prefix_friendly=True),
        TuningGoal.BALANCED,
        cache_initialized=False,
    )

    assert plan is not None
    assert plan.config.use_prefix_cache is True


def test_policy_balanced_queue_high_reduces_chunks_after_cache_init():
    policy = TuningPolicy(PolicyConfig(prefill_chunk_step=64))
    current = _config(prefill_chunk_size=128)
    plan = policy.propose(
        current,
        WorkloadLabels(queue_high=True),
        TuningGoal.BALANCED,
        cache_initialized=True,
    )

    assert plan is not None
    assert plan.config.prefill_chunk_size == 96


def test_policy_returns_none_when_no_action():
    policy = TuningPolicy()
    plan = policy.propose(_config(), WorkloadLabels(), TuningGoal.BALANCED)
    assert plan is None


def test_policy_respects_lower_bound():
    policy = TuningPolicy(PolicyConfig(min_prefill_chunk_size=32, prefill_chunk_step=64))
    current = _config(prefill_chunk_size=48)
    plan = policy.propose(
        current,
        WorkloadLabels(latency_sensitive=True, queue_high=True),
        TuningGoal.LATENCY,
    )

    assert plan is not None
    assert plan.config.prefill_chunk_size == 32


def test_policy_accepts_string_goal():
    policy = TuningPolicy()
    plan = policy.propose(
        _config(max_tokens_per_step=1024),
        WorkloadLabels(throughput_low=True),
        "throughput",
    )
    assert plan is not None
    assert plan.config.max_tokens_per_step == 1280
