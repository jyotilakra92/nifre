import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotune.classifier import ClassifierConfig, WorkloadClassifier
from autotune.workload import WorkloadSnapshot, workload_snapshot_from_metrics


def _metrics_snapshot(
    *,
    queued=0,
    active=0,
    ttft_p95_ms=0.0,
    ttft_p50_ms=0.0,
    tokens_per_sec=0.0,
    prefix_cache_tokens_saved=0,
    total_prefill_tokens=0,
    error_rate=0.0,
    avg_prefill_tokens_per_step=0.0,
    timestamp=1000.0,
):
    return {
        "timestamp": timestamp,
        "request_health": {
            "queued_requests": queued,
            "active_requests": active,
            "error_rate": error_rate,
        },
        "latency": {
            "ttft": {"p95_ms": ttft_p95_ms, "p50_ms": ttft_p50_ms},
        },
        "throughput": {
            "tokens_per_sec": tokens_per_sec,
            "prefix_cache_tokens_saved": prefix_cache_tokens_saved,
            "total_prefill_tokens": total_prefill_tokens,
            "avg_prefill_tokens_per_step": avg_prefill_tokens_per_step,
        },
    }


def _snapshot(**kwargs) -> WorkloadSnapshot:
    return workload_snapshot_from_metrics(_metrics_snapshot(**kwargs))


def test_workload_snapshot_from_metrics():
    snap = _snapshot(
        queued=3,
        active=2,
        ttft_p95_ms=120.0,
        tokens_per_sec=45.0,
        prefix_cache_tokens_saved=30,
        total_prefill_tokens=70,
        error_rate=0.01,
    )

    assert snap.queued_requests == 3
    assert snap.active_requests == 2
    assert snap.ttft_p95_ms == 120.0
    assert snap.tokens_per_sec == 45.0
    assert snap.prefix_cache_hit_rate == 0.3
    assert snap.error_rate == 0.01


def test_workload_snapshot_prefix_hit_rate_zero_when_no_tokens():
    snap = _snapshot()
    assert snap.prefix_cache_hit_rate == 0.0


def test_classifier_latency_sensitive():
    classifier = WorkloadClassifier()
    labels = classifier.classify(_snapshot(ttft_p95_ms=600.0))

    assert labels.latency_sensitive is True
    assert "latency_sensitive" in labels.active()


def test_classifier_prefix_friendly():
    classifier = WorkloadClassifier()
    labels = classifier.classify(
        _snapshot(prefix_cache_tokens_saved=40, total_prefill_tokens=60)
    )

    assert labels.prefix_friendly is True
    assert "prefix_friendly" in labels.active()


def test_classifier_throughput_low_only_under_load():
    classifier = WorkloadClassifier(ClassifierConfig(throughput_low_tps=20.0))

    idle = classifier.classify(_snapshot(tokens_per_sec=5.0, active=0, queued=0))
    assert idle.throughput_low is False

    loaded = classifier.classify(_snapshot(tokens_per_sec=5.0, active=2, queued=1))
    assert loaded.throughput_low is True


def test_classifier_error_elevated():
    classifier = WorkloadClassifier()
    labels = classifier.classify(_snapshot(error_rate=0.10))

    assert labels.error_elevated is True


def test_classifier_queue_high_requires_sustained_depth():
    config = ClassifierConfig(
        queue_high_threshold=2,
        queue_high_window_sec=60.0,
        queue_high_min_samples=3,
    )
    classifier = WorkloadClassifier(config)

    labels = classifier.classify(_snapshot(queued=5))
    assert labels.queue_high is False

    classifier.observe(_snapshot(queued=3, timestamp=1000.0))
    classifier.observe(_snapshot(queued=4, timestamp=1030.0))
    labels = classifier.observe(_snapshot(queued=3, timestamp=1060.0))
    assert labels.queue_high is True
    assert "queue_high" in labels.active()


def test_classifier_queue_high_not_triggered_by_brief_spike():
    config = ClassifierConfig(
        queue_high_threshold=2,
        queue_high_window_sec=60.0,
        queue_high_min_samples=3,
    )
    classifier = WorkloadClassifier(config)

    classifier.observe(_snapshot(queued=5, timestamp=1000.0))
    classifier.observe(_snapshot(queued=0, timestamp=1030.0))
    labels = classifier.observe(_snapshot(queued=5, timestamp=1060.0))

    assert labels.queue_high is False


def test_classifier_multiple_labels():
    classifier = WorkloadClassifier()
    labels = classifier.classify(
        _snapshot(
            ttft_p95_ms=800.0,
            prefix_cache_tokens_saved=50,
            total_prefill_tokens=50,
            error_rate=0.2,
            active=1,
            tokens_per_sec=1.0,
        )
    )

    assert labels.active() == frozenset(
        {"latency_sensitive", "prefix_friendly", "throughput_low", "error_elevated"}
    )
