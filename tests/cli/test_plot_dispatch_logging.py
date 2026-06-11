"""Logging-behavior tests for ``xpcsjax.cli.plot_dispatch``.

These tests pin the *observational* logging contract introduced in the
logging overhaul:

* The per-phi hot loop in ``_save_fit_comparison_only`` rate-limits its
  render-failure WARNING via ``log_once`` so that a renderer that fails on
  every angle logs AT MOST ONCE, not once per angle — while still skipping to
  the next angle exactly as before (control flow unchanged).

The tests are control-flow assertions: the function must still return its
normal fallback (``plots_dir``) regardless of how many angles fail.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import xpcsjax.cli.plot_dispatch as pd
from xpcsjax.utils.logging import reset_log_once_cache


@pytest.fixture(autouse=True)
def _reset_log_once() -> None:
    """``log_once`` uses a process-global dedup cache; reset for determinism."""
    reset_log_once_cache()
    yield
    reset_log_once_cache()


@pytest.fixture(autouse=True)
def _stub_c2_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the fitted-surface evaluation so the render-logging path is reached.

    ``_save_fit_comparison_only`` resolves the fitted c2 surface through the
    shared ``xpcsjax.viz.nlsq_plots._evaluate_c2_per_angle`` extractor, which is a
    strict isinstance gate (homodyne/heterodyne families only) so it correctly
    handles the per-angle scaling layout. These tests exercise the *render*-
    failure rate-limiting contract with a minimal fake model, so stub the
    evaluation step to return a valid surface and let the patched renderers drive
    the failure path. (The function's local ``from ... import`` re-binds the name
    each call, so patching the module attribute takes effect.)
    """
    monkeypatch.setattr(
        "xpcsjax.viz.nlsq_plots._evaluate_c2_per_angle",
        lambda model, result, data, config, phi_deg: np.ones((4, 4), dtype=np.float64),
        raising=False,
    )


class _FakeModel:
    """Minimal stand-in model for the fit-comparison render-logging tests."""

    def compute_g2(self, params, t1, t2, phi, q, L, contrast, offset, dt):  # noqa: N803
        n1 = np.asarray(t1).shape[0]
        n2 = np.asarray(t2).shape[0]
        return np.ones((1, n1, n2), dtype=np.float64)


class _FakeConfigManager:
    def __init__(self) -> None:
        self.config = {}

    def get_model(self):
        return _FakeModel()


class _FakeResult:
    parameters = np.zeros(3, dtype=np.float64)
    reduced_chi_squared = 1.0
    contrast = 0.3
    offset = 1.0


def _make_data(n_phi: int) -> dict:
    n = 4
    c2 = np.ones((n_phi, n, n), dtype=np.float64)
    return {
        "c2_exp": c2,
        "phi_angles_list": np.arange(n_phi, dtype=np.float64) * 30.0,
        "t1": np.arange(n, dtype=np.float64),
        "t2": np.arange(n, dtype=np.float64),
    }


def test_per_phi_render_failure_logs_once_not_per_angle(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    """A renderer that fails on EVERY angle must emit one WARNING per site.

    The per-phi loop has two render call sites (``plot_nlsq_fit`` and
    ``plot_residual_map``). With ``log_once`` rate-limiting keyed by a stable
    per-site suffix, each site logs AT MOST ONCE across all N angles — so the
    total is the number of failing sites (2), NOT ``2 * n_phi``. The loop must
    still skip every failing angle and return its normal fallback.
    """
    n_phi = 3

    def _fit_raises(*args, **kwargs):
        raise RuntimeError("nlsq_fit boom")

    def _resid_raises(*args, **kwargs):
        raise RuntimeError("residual boom")

    # Patch the names as bound inside the function's local import.
    monkeypatch.setattr("xpcsjax.viz.plot_nlsq_fit", _fit_raises, raising=False)
    monkeypatch.setattr("xpcsjax.viz.plot_residual_map", _resid_raises, raising=False)

    data = _make_data(n_phi)

    with caplog.at_level(logging.WARNING, logger="xpcsjax"):
        out = pd._save_fit_comparison_only(_FakeConfigManager(), data, _FakeResult(), tmp_path)

    # Control flow unchanged: the loop skipped every failing angle and the
    # function still returned its normal fallback (the plots dir).
    assert out == tmp_path

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    fit_warnings = [r for r in warnings if "nlsq_fit boom" in r.getMessage()]
    resid_warnings = [r for r in warnings if "residual boom" in r.getMessage()]

    # Each of the two render sites fired exactly once despite N>=3 failures.
    assert len(fit_warnings) == 1, (
        f"plot_nlsq_fit warning not rate-limited: got {len(fit_warnings)} for {n_phi} angles"
    )
    assert len(resid_warnings) == 1, (
        f"plot_residual_map warning not rate-limited: got {len(resid_warnings)} for {n_phi} angles"
    )


def test_second_dispatch_call_is_not_cross_call_suppressed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    """A SECOND independent call must emit its own WARNING (no cross-call collapse).

    With no log context set (``run_id is None``), the per-phi ``log_once`` key
    is scoped by a monotonic per-call token, so a second dispatch-function call
    with a still-failing renderer must NOT be silenced by the first call's
    process-global dedup entry. Under the old static ``"None:..."`` key this
    second call would emit zero warnings.
    """
    n_phi = 3

    def _fit_raises(*args, **kwargs):
        raise RuntimeError("nlsq_fit boom")

    def _resid_raises(*args, **kwargs):
        raise RuntimeError("residual boom")

    monkeypatch.setattr("xpcsjax.viz.plot_nlsq_fit", _fit_raises, raising=False)
    monkeypatch.setattr("xpcsjax.viz.plot_residual_map", _resid_raises, raising=False)

    data = _make_data(n_phi)

    # First call — primes the process-global dedup cache.
    out1 = pd._save_fit_comparison_only(_FakeConfigManager(), data, _FakeResult(), tmp_path)
    assert out1 == tmp_path

    # Drop the first call's records so we count ONLY the second call's output.
    caplog.clear()

    # Second, independent call. Capture only its warnings.
    with caplog.at_level(logging.WARNING, logger="xpcsjax"):
        out2 = pd._save_fit_comparison_only(_FakeConfigManager(), data, _FakeResult(), tmp_path)
    assert out2 == tmp_path

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    fit_warnings = [r for r in warnings if "nlsq_fit boom" in r.getMessage()]
    resid_warnings = [r for r in warnings if "residual boom" in r.getMessage()]

    # The second call emits its OWN one-per-site warning — not suppressed by the
    # first call's dedup entry.
    assert len(fit_warnings) == 1, (
        f"second call's plot_nlsq_fit warning was cross-call suppressed (got {len(fit_warnings)})"
    )
    assert len(resid_warnings) == 1, (
        "second call's plot_residual_map warning was cross-call suppressed "
        f"(got {len(resid_warnings)})"
    )
