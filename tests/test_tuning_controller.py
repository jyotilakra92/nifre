import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotune.classifier import WorkloadLabels
from autotune.controller import ControllerConfig, EvaluationSnapshot, TuningController
from autotune.policy import TuningGoal, TuningPlan
from inference.backends.gpt import GptInferenceModel
from inference.data_model import EngineConfig
from inference.engine import Engine
from inference.models.gpt import GPT_CONFIG_124M, GptModel
from observability import Observability
from observability.optimization import OptimizationTracker


def _tiny_engine():
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    device = torch.device("cpu")
    model = GptInferenceModel(GptModel(cfg).to(device).eval())
    obs = Observability(model_name="gpt-test", runtime="custom")
    engine = Engine(model, max_concurrent_requests=2, device=device, metrics_collector=obs.collector)
    obs.attach(engine)
    return engine, obs


def _metrics(
    *,
    completed=5,
    ttft_p95_ms=100.0,
    tokens_per_sec=20.0,
    error_rate=0.0,
    queued=0,
    active=1,
    timestamp=1000.0,
):
    return {
        "timestamp": timestamp,
        "request_health": {
            "completed_requests": completed,
            "queued_requests": queued,
            "active_requests": active,
            "error_rate": error_rate,
        },
        "latency": {"ttft": {"p95_ms": ttft_p95_ms, "p50_ms": ttft_p95_ms / 2}},
        "throughput": {
            "tokens_per_sec": tokens_per_sec,
            "prefix_cache_tokens_saved": 0,
            "total_prefill_tokens": 100,
            "avg_prefill_tokens_per_step": 10.0,
        },
    }


class FakeObservability:
    def __init__(self, snapshots):
        self.optimization = OptimizationTracker()
        self._snapshots = list(snapshots)
        self._index = 0

    def snapshot(self):
        if self._index < len(self._snapshots):
            snap = self._snapshots[self._index]
            self._index += 1
            return snap
        return self._snapshots[-1]


class FakePolicy:
    def __init__(self, plans):
        self._plans = list(plans)
        self._index = 0

    def propose(self, current, labels, goal, *, cache_initialized=False):
        if self._index >= len(self._plans):
            return None
        plan = self._plans[self._index]
        self._index += 1
        return plan


class FakeClassifier:
    def __init__(self, labels):
        self._labels = labels

    def observe(self, snapshot):
        return self._labels


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


def test_controller_starts_attempt_and_applies_changes():
    engine, _ = _tiny_engine()
    plan = TuningPlan(
        config=_config(prefill_chunk_size=64),
        reason="test reduce chunks",
        changes={"prefill_chunk_size": 64},
    )
    obs = FakeObservability([_metrics(completed=3), _metrics(completed=4)])
    controller = TuningController(
        engine=engine,
        observability=obs,
        goal=TuningGoal.LATENCY,
        config=ControllerConfig(cooldown_sec=0, evaluation_sec=60),
        classifier=FakeClassifier(WorkloadLabels(latency_sensitive=True)),
        policy=FakePolicy([plan]),
        _time=lambda: 1000.0,
    )

    action = controller.tick(now=1000.0)

    assert action == "attempted"
    assert engine.prefill_chunk_size == 64
    assert controller.status().pending_attempt == "autotune:prefill_chunk_size"
    assert obs.optimization.attempted == 1


def test_controller_promotes_on_latency_improvement():
    engine, _ = _tiny_engine()
    plan = TuningPlan(
        config=_config(prefill_chunk_size=64),
        reason="test",
        changes={"prefill_chunk_size": 64},
    )
    obs = FakeObservability(
        [
            _metrics(completed=3, ttft_p95_ms=500.0),
            _metrics(completed=4, ttft_p95_ms=300.0),
        ]
    )
    controller = TuningController(
        engine=engine,
        observability=obs,
        goal=TuningGoal.LATENCY,
        config=ControllerConfig(cooldown_sec=0, evaluation_sec=10),
        classifier=FakeClassifier(WorkloadLabels(latency_sensitive=True)),
        policy=FakePolicy([plan]),
        _time=lambda: 1000.0,
    )

    controller.tick(now=1000.0)
    action = controller.tick(now=1011.0)

    assert action == "promoted"
    assert engine.prefill_chunk_size == 64
    assert obs.optimization.promoted == 1
    assert controller.status().pending_attempt is None


def test_controller_rolls_back_on_latency_regression():
    engine, _ = _tiny_engine()
    plan = TuningPlan(
        config=_config(prefill_chunk_size=64),
        reason="test",
        changes={"prefill_chunk_size": 64},
    )
    obs = FakeObservability(
        [
            _metrics(completed=3, ttft_p95_ms=400.0),
            _metrics(completed=4, ttft_p95_ms=700.0),
        ]
    )
    controller = TuningController(
        engine=engine,
        observability=obs,
        goal=TuningGoal.LATENCY,
        config=ControllerConfig(cooldown_sec=0, evaluation_sec=10),
        classifier=FakeClassifier(WorkloadLabels(latency_sensitive=True)),
        policy=FakePolicy([plan]),
        _time=lambda: 1000.0,
    )

    controller.tick(now=1000.0)
    action = controller.tick(now=1011.0)

    assert action == "rolled_back"
    assert engine.prefill_chunk_size == 128
    assert obs.optimization.rolled_back == 1


def test_controller_rolls_back_on_neutral_result():
    engine, _ = _tiny_engine()
    plan = TuningPlan(
        config=_config(max_tokens_per_step=2304),
        reason="test",
        changes={"max_tokens_per_step": 2304},
    )
    obs = FakeObservability(
        [
            _metrics(completed=3, tokens_per_sec=50.0),
            _metrics(completed=4, tokens_per_sec=50.0),
        ]
    )
    controller = TuningController(
        engine=engine,
        observability=obs,
        goal=TuningGoal.THROUGHPUT,
        config=ControllerConfig(cooldown_sec=0, evaluation_sec=10),
        classifier=FakeClassifier(WorkloadLabels(throughput_low=True)),
        policy=FakePolicy([plan]),
        _time=lambda: 1000.0,
    )

    controller.tick(now=1000.0)
    action = controller.tick(now=1011.0)

    assert action == "rolled_back"
    assert engine.max_tokens_per_step == 2048


def test_controller_skips_without_min_completed_requests():
    engine, _ = _tiny_engine()
    obs = FakeObservability([_metrics(completed=0)])
    controller = TuningController(
        engine=engine,
        observability=obs,
        goal=TuningGoal.BALANCED,
        config=ControllerConfig(min_completed_requests=1, cooldown_sec=0),
        classifier=FakeClassifier(WorkloadLabels(latency_sensitive=True)),
        policy=FakePolicy(
            [
                TuningPlan(
                    config=_config(prefill_chunk_size=64),
                    reason="should not run",
                    changes={"prefill_chunk_size": 64},
                )
            ]
        ),
        _time=lambda: 1000.0,
    )

    assert controller.tick(now=1000.0) is None
    assert engine.prefill_chunk_size == 128


def test_controller_respects_cooldown():
    engine, _ = _tiny_engine()
    obs = FakeObservability([_metrics(completed=0), _metrics(completed=0)])
    controller = TuningController(
        engine=engine,
        observability=obs,
        goal=TuningGoal.BALANCED,
        config=ControllerConfig(min_completed_requests=0, cooldown_sec=100),
        classifier=FakeClassifier(WorkloadLabels()),
        policy=FakePolicy([None, None]),
        _time=lambda: 1000.0,
    )
    controller._last_attempt_at = 1000.0

    assert controller.tick(now=1050.0) is None


def test_evaluation_helpers():
    engine, _ = _tiny_engine()
    controller = TuningController(engine=engine, observability=FakeObservability([_metrics()]))
    baseline = EvaluationSnapshot(ttft_p95_ms=500.0, tokens_per_sec=20.0, error_rate=0.0)

    controller.goal = TuningGoal.LATENCY
    assert controller._is_improvement(baseline, EvaluationSnapshot(400.0, 20.0, 0.0))
    assert controller._is_regression(baseline, EvaluationSnapshot(600.0, 20.0, 0.0))

    controller.goal = TuningGoal.THROUGHPUT
    assert controller._is_improvement(baseline, EvaluationSnapshot(500.0, 25.0, 0.0))
    assert controller._is_regression(baseline, EvaluationSnapshot(500.0, 10.0, 0.0))
