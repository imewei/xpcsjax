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
    calls: list[Any] = []

    def __init__(self, config: Any) -> None:
        self.config = config

    def fit(self, **kwargs: Any) -> dict:
        _AlwaysFailsOptimizer.calls.append(kwargs)
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
    _AlwaysFailsOptimizer.calls = []
    _enable(monkeypatch, _AlwaysFailsOptimizer)

    with pytest.raises(NLSQOptimizationError, match="permanent failure") as excinfo:
        hs.fit_with_hybrid_streaming_optimizer(
            residual_fn=_residual_fn,
            xdata=np.zeros(5),
            ydata=np.zeros(5),
            initial_params=np.array([0.5, 0.5]),
            bounds=None,
            logger=_logger(),
            nlsq_config=None,
        )

    # 1 initial attempt + 3 retries = 4 calls; an off-by-one loop bound would
    # silently under/over-retry without this assertion catching it.
    assert len(_AlwaysFailsOptimizer.calls) == 4
    assert len(excinfo.value.error_context["attempt_errors"]) == 4


def test_config_multipliers_apply_from_original_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: retry settings must apply fresh from the ORIGINAL config each
    attempt, not compound onto the already-adjusted config from the previous
    attempt (that bug made later retries collapse the trust region/learning
    rate far faster than HybridRecoveryConfig's own documented schedule)."""
    _FlakyOptimizer.calls = []
    # ``config`` is a single mutable object reused/mutated across attempts, so
    # capture a POINT-IN-TIME snapshot of the fields at construction — storing
    # the live object reference would just record 4 pointers to the same
    # final (fully-mutated) state.
    snapshots: list[tuple[float, float, float]] = []

    def factory(config: Any) -> _FlakyOptimizer:
        snapshots.append(
            (config.trust_region_initial, config.warmup_lr_refinement, config.regularization_factor)
        )
        return _FlakyOptimizer(config, fail_count=3)

    _enable(monkeypatch, factory)

    popt, pcov, info = hs.fit_with_hybrid_streaming_optimizer(
        residual_fn=_residual_fn,
        xdata=np.zeros(5),
        ydata=np.zeros(5),
        initial_params=np.array([0.5, 0.5]),
        bounds=None,
        logger=_logger(),
        nlsq_config=None,
    )

    assert info["recovery_attempt"] == 3  # success on the last allowed attempt
    assert len(snapshots) == 4

    trust = [s[0] for s in snapshots]
    lr = [s[1] for s in snapshots]
    reg = [s[2] for s in snapshots]

    # trust_decay=0.5, lr_decay=0.5, lambda_growth=2.0 (HybridRecoveryConfig
    # defaults) applied as base * factor**attempt, NOT compounded.
    np.testing.assert_allclose(trust, [1.0, 0.5, 0.25, 0.125])
    np.testing.assert_allclose(lr, [1e-6, 5e-7, 2.5e-7, 1.25e-7])
    np.testing.assert_allclose(reg, [1e-10, 2e-10, 4e-10, 8e-10])


def test_non_recoverable_exception_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bug unrelated to the optimizer (e.g. KeyError from malformed result
    handling) must fail fast on the first attempt, not get treated as a
    recoverable optimizer failure and retried."""

    class _RaisesKeyError:
        calls: list[Any] = []

        def __init__(self, config: Any) -> None:
            pass

        def fit(self, **kwargs: Any) -> dict:
            _RaisesKeyError.calls.append(kwargs)
            raise KeyError("not a recoverable optimizer failure")

    _RaisesKeyError.calls = []
    _enable(monkeypatch, _RaisesKeyError)

    with pytest.raises(KeyError):
        hs.fit_with_hybrid_streaming_optimizer(
            residual_fn=_residual_fn,
            xdata=np.zeros(5),
            ydata=np.zeros(5),
            initial_params=np.array([0.5, 0.5]),
            bounds=None,
            logger=_logger(),
            nlsq_config=None,
        )

    assert len(_RaisesKeyError.calls) == 1
