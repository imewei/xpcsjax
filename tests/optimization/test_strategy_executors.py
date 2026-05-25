"""Tests for xpcsjax.optimization.nlsq.strategies.executors.

The executors wrap NLSQ's ``curve_fit`` / ``curve_fit_large`` /
``AdaptiveHybridStreamingOptimizer``. Those are module-level imports, so tests
monkeypatch them with canned returns — this exercises the x_scale-selection
branches, the several curve_fit_large return formats, the streaming pcov
fallback, and the error/re-raise paths without running real optimization.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.strategies import executors as ex


def _logger() -> MagicMock:
    return MagicMock(spec=logging.Logger)


def _resid(x: np.ndarray, *p: float) -> np.ndarray:  # pragma: no cover - dummy
    return np.zeros_like(x)


# ---------------------------------------------------------------------------
# ExecutionResult + executor metadata
# ---------------------------------------------------------------------------


def test_execution_result_dataclass() -> None:
    r = ex.ExecutionResult(
        popt=np.array([1.0]), pcov=np.eye(1), info={"a": 1},
        recovery_actions=["x"], convergence_status="converged",
    )
    assert r.convergence_status == "converged"
    assert r.info == {"a": 1}


def test_executor_names_and_progress() -> None:
    assert ex.StandardExecutor().name == "standard"
    assert ex.StandardExecutor().supports_progress is False
    assert ex.LargeDatasetExecutor().name == "large"
    assert ex.LargeDatasetExecutor().supports_progress is True
    assert ex.StreamingExecutor().name == "streaming"
    assert ex.StreamingExecutor().supports_progress is True


# ---------------------------------------------------------------------------
# StandardExecutor
# ---------------------------------------------------------------------------


def test_standard_executor_success_and_magnitude_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_curve_fit(fn: Any, x: Any, y: Any, **kw: Any):
        captured.update(kw)
        return np.array([1.0, 2.0]), np.eye(2)

    monkeypatch.setattr(ex, "curve_fit", fake_curve_fit)
    initial = np.array([0.5, -2.0])
    result = ex.StandardExecutor().execute(
        _resid, np.zeros(3), np.zeros(3), initial, None, "soft_l1", "jac", _logger()
    )
    assert result.convergence_status == "converged"
    assert result.info["strategy"] == "standard"
    # "jac" -> magnitude-based scaling |p| + 1e-3.
    np.testing.assert_allclose(captured["x_scale"], np.abs(initial) + 1e-3)


def test_standard_executor_ndarray_x_scale_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_curve_fit(fn: Any, x: Any, y: Any, **kw: Any):
        captured.update(kw)
        return np.array([1.0]), np.eye(1)

    monkeypatch.setattr(ex, "curve_fit", fake_curve_fit)
    x_scale = np.array([5.0])
    ex.StandardExecutor().execute(
        _resid, np.zeros(1), np.zeros(1), np.array([1.0]), None, "linear", x_scale, _logger()
    )
    np.testing.assert_array_equal(captured["x_scale"], x_scale)


def test_standard_executor_reraises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any):
        raise ValueError("solver exploded")

    monkeypatch.setattr(ex, "curve_fit", boom)
    log = _logger()
    with pytest.raises(ValueError, match="solver exploded"):
        ex.StandardExecutor().execute(
            _resid, np.zeros(1), np.zeros(1), np.array([1.0]), None, "linear", "jac", log
        )
    log.error.assert_called()


# ---------------------------------------------------------------------------
# LargeDatasetExecutor — return-format handling
# ---------------------------------------------------------------------------


def test_large_executor_two_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ex, "curve_fit_large", lambda *a, **k: (np.array([1.0, 2.0]), np.eye(2))
    )
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    res = ex.LargeDatasetExecutor().execute(
        _resid, np.zeros(3), np.zeros(3), np.array([1.0, 2.0]), bounds, "soft_l1", 1.0, _logger()
    )
    assert res.info["strategy"] == "large"
    assert res.convergence_status == "converged"


def test_large_executor_three_tuple_with_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ex, "curve_fit_large",
        lambda *a, **k: (np.array([1.0]), np.eye(1), {"success": False, "nfev": 9}),
    )
    res = ex.LargeDatasetExecutor().execute(
        _resid, np.zeros(1), np.zeros(1), np.array([1.0]), None, "soft_l1", 1.0, _logger()
    )
    assert res.info["nfev"] == 9
    assert res.convergence_status == "partial"  # success=False


def test_large_executor_optimize_result_with_pcov(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = SimpleNamespace(x=np.array([1.0, 2.0]), pcov=np.eye(2), nfev=7)
    monkeypatch.setattr(ex, "curve_fit_large", lambda *a, **k: obj)
    res = ex.LargeDatasetExecutor().execute(
        _resid, np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), None, "soft_l1", 1.0, _logger()
    )
    np.testing.assert_array_equal(res.pcov, np.eye(2))
    assert res.info["nfev"] == 7


def test_large_executor_optimize_result_without_pcov(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = SimpleNamespace(x=np.array([1.0, 2.0]), nfev=3)  # no pcov attribute
    monkeypatch.setattr(ex, "curve_fit_large", lambda *a, **k: obj)
    res = ex.LargeDatasetExecutor().execute(
        _resid, np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), None, "soft_l1", 1.0, _logger()
    )
    np.testing.assert_array_equal(res.pcov, np.eye(2))  # identity fallback


def test_large_executor_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any):
        raise RuntimeError("oom")

    monkeypatch.setattr(ex, "curve_fit_large", boom)
    with pytest.raises(RuntimeError, match="oom"):
        ex.LargeDatasetExecutor().execute(
            _resid, np.zeros(1), np.zeros(1), np.array([1.0]), None, "soft_l1", 1.0, _logger()
        )


# ---------------------------------------------------------------------------
# StreamingExecutor
# ---------------------------------------------------------------------------


class _FakeOpt:
    def __init__(self, config: Any) -> None:
        self.config = config

    def fit(self, fn: Any, x: Any, y: Any, p0: Any, bounds: Any) -> dict:
        return {"x": np.array([1.0, 2.0]), "pcov": np.eye(2), "success": True, "nit": 4}


def _enable_streaming(monkeypatch: pytest.MonkeyPatch, opt_cls: type) -> None:
    monkeypatch.setattr(ex, "STREAMING_AVAILABLE", True)
    monkeypatch.setattr(ex, "AdaptiveHybridStreamingOptimizer", opt_cls)
    monkeypatch.setattr(ex, "HybridStreamingConfig", lambda **kw: kw)


def test_streaming_executor_not_available_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "STREAMING_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="not available"):
        ex.StreamingExecutor().execute(
            _resid, np.zeros(1), np.zeros(1), np.array([1.0]), None, "soft_l1", 1.0, _logger()
        )


def test_streaming_executor_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_streaming(monkeypatch, _FakeOpt)
    res = ex.StreamingExecutor({"chunk_size": 5000}).execute(
        _resid, np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), None, "soft_l1", 1.0, _logger()
    )
    assert res.convergence_status == "converged"
    assert res.info["strategy"] == "hybrid_streaming"
    assert res.info["iterations"] == 4


def test_streaming_executor_missing_pcov_uses_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoPcov:
        def __init__(self, config: Any) -> None:
            pass

        def fit(self, *a: Any, **k: Any) -> dict:
            return {"x": np.array([1.0, 2.0]), "success": False}

    _enable_streaming(monkeypatch, _NoPcov)
    res = ex.StreamingExecutor().execute(
        _resid, np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), None, "soft_l1", 1.0, _logger()
    )
    np.testing.assert_array_equal(res.pcov, np.eye(2))  # identity fallback
    assert res.convergence_status == "partial"  # success=False


def test_streaming_executor_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomOpt:
        def __init__(self, config: Any) -> None:
            pass

        def fit(self, *a: Any, **k: Any) -> dict:
            raise ValueError("stream broke")

    _enable_streaming(monkeypatch, _BoomOpt)
    with pytest.raises(ValueError, match="stream broke"):
        ex.StreamingExecutor().execute(
            _resid, np.zeros(1), np.zeros(1), np.array([1.0]), None, "soft_l1", 1.0, _logger()
        )


# ---------------------------------------------------------------------------
# get_executor factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("standard", ex.StandardExecutor),
        ("large", ex.LargeDatasetExecutor),
        ("chunked", ex.LargeDatasetExecutor),
        ("streaming", ex.StreamingExecutor),
    ],
)
def test_get_executor_dispatch(name: str, cls: type) -> None:
    assert isinstance(ex.get_executor(name), cls)


def test_get_executor_streaming_passes_checkpoint_config() -> None:
    executor = ex.get_executor("streaming", checkpoint_config={"chunk_size": 999})
    assert isinstance(executor, ex.StreamingExecutor)
    assert executor.checkpoint_config == {"chunk_size": 999}


def test_get_executor_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        ex.get_executor("bogus")
