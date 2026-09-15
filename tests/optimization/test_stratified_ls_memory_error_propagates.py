"""A7: an OOM on the >=1M stratified-LS route must propagate, not fall
through to the dense in-memory ``curve_fit_large`` path.

The dense fallback re-materializes the full, unchunked residual/Jacobian --
strictly MORE memory than the stratified route that just raised
``MemoryError``. Falling through there only guarantees a second, worse OOM
after burning the time of the first attempt. ``ValueError``/``RuntimeError``/
``OSError`` are still genuinely recoverable and keep falling through.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq import wrapper as wrapper_module
from xpcsjax.optimization.nlsq.memory import NLSQStrategy, StrategyDecision
from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper


class _FakeStratifiedData:
    """Minimal stand-in for angle-stratified data: only what the wrapper
    touches before dispatching to ``_fit_with_stratified_least_squares`` --
    ``phi_flat``/``g2_flat`` sized to trigger the >=1M stratified-LS gate."""

    def __init__(self, n_phi: int, n_per_angle: int) -> None:
        phi = np.linspace(0.0, 180.0, n_phi, endpoint=False)
        self.phi_flat = np.repeat(phi, n_per_angle)
        self.g2_flat = np.zeros(n_phi * n_per_angle)


class _FakeConfig:
    def __init__(self) -> None:
        self.config: dict = {}


def _force_standard(monkeypatch) -> None:
    def _fixed(n_points, n_params, *args, **kwargs):
        return StrategyDecision(
            strategy=NLSQStrategy.STANDARD,
            threshold_gb=1e9,
            index_memory_gb=0.0,
            peak_memory_gb=0.0,
            reason="forced standard",
        )

    monkeypatch.setattr(wrapper_module, "select_nlsq_strategy", _fixed)


def _make_wrapper_and_fit_kwargs(monkeypatch, n_phi=5, n_physical=3):
    _force_standard(monkeypatch)
    stratified = _FakeStratifiedData(n_phi=n_phi, n_per_angle=250_000)  # >=1M points
    monkeypatch.setattr(
        NLSQWrapper, "_apply_stratification_if_needed", lambda self, *a, **k: stratified
    )
    initial_params = np.zeros(n_physical + 2)  # compact [physical | contrast, offset]
    lower = np.full(n_physical + 2, -1.0)
    upper = np.full(n_physical + 2, 1.0)
    return dict(
        data=object(),
        config=_FakeConfig(),
        initial_params=initial_params,
        bounds=(lower, upper),
        analysis_mode=AnalysisMode.STATIC_ANISOTROPIC,
        per_angle_scaling=True,
    )


def test_memory_error_propagates_instead_of_falling_through(monkeypatch):
    fit_kwargs = _make_wrapper_and_fit_kwargs(monkeypatch)
    monkeypatch.setattr(
        NLSQWrapper,
        "_fit_with_stratified_least_squares",
        lambda self, *a, **k: (_ for _ in ()).throw(MemoryError("simulated OOM")),
    )
    wrapper = NLSQWrapper()
    with pytest.raises(MemoryError):
        wrapper.fit(**fit_kwargs)


def test_value_error_still_falls_through_to_dense_path(monkeypatch):
    """Non-memory failures are still recoverable -- this must NOT raise; it
    should fall through and continue past the stratified-LS try/except (the
    dense path below will itself likely fail fast on the fake object() data,
    which is an acceptable/expected downstream error for this unit test --
    the point is that it is NOT the injected ValueError)."""
    fit_kwargs = _make_wrapper_and_fit_kwargs(monkeypatch)
    monkeypatch.setattr(
        NLSQWrapper,
        "_fit_with_stratified_least_squares",
        lambda self, *a, **k: (_ for _ in ()).throw(ValueError("recoverable")),
    )
    wrapper = NLSQWrapper()
    with pytest.raises(Exception) as excinfo:
        wrapper.fit(**fit_kwargs)
    assert "recoverable" not in str(excinfo.value)
