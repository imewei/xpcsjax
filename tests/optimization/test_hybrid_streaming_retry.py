"""Regression tests for the T029/T030 hybrid-streaming progressive-recovery retry.

Assessment finding: analysis/xpcsjax/ASSESSMENT.md technical debt #3 —
``fit_with_hybrid_streaming_optimizer`` used to raise immediately on the
first recoverable failure instead of retrying with ``HybridRecoveryConfig``'s
progressively conservative settings, as its own docstring/config promised.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xpcsjax.optimization.exceptions import NLSQOptimizationError
from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs


def _logger() -> Any:
    class _NullLogger:
        def info(self, *a: Any, **k: Any) -> None:
            pass

        def warning(self, *a: Any, **k: Any) -> None:
            pass

        def error(self, *a: Any, **k: Any) -> None:
            pass

    return _NullLogger()


def _residual_fn(x: np.ndarray, *params: float) -> np.ndarray:
    return x - params[0]


class _FlakyOptimizer:
    """Fails ``fail_count`` times, then succeeds; mirrors real optimizer.fit contract."""

    calls: list[Any] = []

    def __init__(self, config: Any, fail_count: int) -> None:
        self.config = config
        self.fail_count = fail_count

    def fit(self, **kwargs: Any) -> dict:
        _FlakyOptimizer.calls.append(kwargs)
        if len(_FlakyOptimizer.calls) <= self.fail_count:
            raise RuntimeError(f"transient failure {len(_FlakyOptimizer.calls)}")
        return {"x": np.array([1.0, 2.0]), "pcov": np.eye(2), "success": True}


class _AlwaysFailsOptimizer:
    def __init__(self, config: Any) -> None:
        self.config = config

    def fit(self, **kwargs: Any) -> dict:
        raise RuntimeError("permanent failure")


def _fake_hybrid_streaming_config(**kw: Any) -> Any:
    # Mirrors the real NLSQ ``HybridStreamingConfig`` dataclass: fields the
    # ``nlsq_config is None`` branch doesn't pass (e.g. regularization_factor,
    # trust_region_initial) still resolve via the dataclass's own defaults.
    defaults = {
        "warmup_lr_refinement": 1e-6,
        "warmup_lr_careful": 1e-5,
        "regularization_factor": 1e-10,
        "trust_region_initial": 1.0,
    }
    defaults.update(kw)
    return type("Cfg", (), defaults)()


def _enable(monkeypatch: pytest.MonkeyPatch, opt_factory: Any) -> None:
    monkeypatch.setattr(hs, "HYBRID_STREAMING_AVAILABLE", True)
    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", opt_factory)
    monkeypatch.setattr(hs, "HybridStreamingConfig", _fake_hybrid_streaming_config)


def test_retries_and_recovers_after_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails twice, succeeds on the 3rd attempt — within max_retries=3."""
    _FlakyOptimizer.calls = []
    _enable(monkeypatch, lambda config: _FlakyOptimizer(config, fail_count=2))

    popt, pcov, info = hs.fit_with_hybrid_streaming_optimizer(
        residual_fn=_residual_fn,
        xdata=np.zeros(5),
        ydata=np.zeros(5),
        initial_params=np.array([0.5, 0.5]),
        bounds=None,
        logger=_logger(),
        nlsq_config=None,
    )

    assert len(_FlakyOptimizer.calls) == 3
    np.testing.assert_array_equal(popt, np.array([1.0, 2.0]))
    assert info["recovery_attempt"] == 2


def test_succeeds_on_first_attempt_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _FlakyOptimizer.calls = []
    _enable(monkeypatch, lambda config: _FlakyOptimizer(config, fail_count=0))

    popt, pcov, info = hs.fit_with_hybrid_streaming_optimizer(
        residual_fn=_residual_fn,
        xdata=np.zeros(5),
        ydata=np.zeros(5),
        initial_params=np.array([0.5, 0.5]),
        bounds=None,
        logger=_logger(),
        nlsq_config=None,
    )

    assert len(_FlakyOptimizer.calls) == 1
    assert "recovery_attempt" not in info


def test_raises_after_exhausting_all_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, _AlwaysFailsOptimizer)

    with pytest.raises(NLSQOptimizationError, match="permanent failure"):
        hs.fit_with_hybrid_streaming_optimizer(
            residual_fn=_residual_fn,
            xdata=np.zeros(5),
            ydata=np.zeros(5),
            initial_params=np.array([0.5, 0.5]),
            bounds=None,
            logger=_logger(),
            nlsq_config=None,
        )
