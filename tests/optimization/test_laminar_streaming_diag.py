"""Diagnostics-parity tests for laminar non-in-memory result builders.

The in-memory laminar path already emits the symmetric anti-degeneracy
activation keys (``hierarchical_active`` / ``regularization_active`` /
``shear_weighting`` / optional ``gradient_monitor``). These tests pin the shared,
presence-based ``_laminar_anti_degeneracy_block`` helper that brings the
HYBRID_STREAMING, stratified-LS, sequential, and out-of-core return paths up to
the same contract.

Diagnostics-only: the helper reads ``info['anti_degeneracy']`` sub-key presence
and never touches popt/pcov/chi2.
"""
from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.wrapper import _laminar_anti_degeneracy_block


def test_block_markers_when_no_info():
    b = _laminar_anti_degeneracy_block(None)
    assert b["hierarchical_active"] is False
    assert b["regularization_active"] is False
    assert b["shear_weighting"] == "laminar_flow_inactive"
    assert "gradient_monitor" not in b


def test_block_honest_active_from_streaming_info():
    info_ad = {
        "hierarchical": {"x": 1},
        "regularization": {"y": 2},
        "shear_weighting": {"active": True},
        "gradient_monitor": {"mechanism": "post_solve_fallback"},
    }
    b = _laminar_anti_degeneracy_block(info_ad)
    assert b["hierarchical_active"] is True
    assert b["regularization_active"] is True
    assert b["shear_weighting"] == {"active": True}
    assert b["gradient_monitor"] == {"mechanism": "post_solve_fallback"}


def test_block_inactive_for_stratified_controller_only_info():
    # stratified-LS info carries mode/controller_diagnostics but NOT the layer sub-keys
    info_ad = {"mode": "auto_averaged", "controller_diagnostics": {"version": 1}}
    b = _laminar_anti_degeneracy_block(info_ad)
    assert b["hierarchical_active"] is False
    assert b["regularization_active"] is False
    assert b["shear_weighting"] == "laminar_flow_inactive"


def _build_sequential_laminar_fit():
    """Reuse the small synthetic laminar fixture but force the SEQUENTIAL
    per-angle return path (Site 4).

    ``force_sequential_fallback=true`` makes ``_apply_stratification_if_needed``
    return a ``UseSequentialOptimization`` marker, routing the fit through
    ``_run_sequential_optimization`` whose inline ``OptimizationResult`` bypasses
    ``_create_fit_result``. Stratification must be ``"auto"`` (not ``False``) so
    the disable-early-return does not fire before the force-sequential check.
    """
    from tests.optimization.test_l4_callback_observational import _build_laminar_fit

    fit_nlsq, data, cfg = _build_laminar_fit()
    # Flip stratification on (auto) and force the sequential fallback.
    cfg.config["optimization"]["stratification"] = {
        "enabled": "auto",
        "force_sequential_fallback": True,
    }
    return fit_nlsq, data, cfg


def test_sequential_laminar_emits_symmetric_activation_keys(monkeypatch):
    """Site 4 (sequential per-angle) result carries the symmetric anti-degeneracy
    activation keys. The sequential path runs no L2/L3/L5, so they are honest
    inactive markers. Diagnostics-only: this asserts only on nlsq_diagnostics.

    The real ``optimize_per_angle_sequential`` solver hits an unrelated, pre-
    existing JAX TracerArrayConversionError on this tiny synthetic fixture (the
    homodyne sequential solver is otherwise exercised only via heterodyne tests),
    which an outer guard swallows into a stub result — so the Site 4 result-build
    is never reached. We stub the solver with a minimal successful
    ``SequentialResult`` so the *real* Site 4 payload-build + anti-degeneracy
    merge runs end-to-end. This isolates exactly the wiring under test.
    """
    import xpcsjax.optimization.nlsq.wrapper as wrapper_mod
    from xpcsjax.optimization.nlsq.strategies.sequential import SequentialResult

    fit_nlsq, data, cfg = _build_sequential_laminar_fit()

    # Expanded per-angle layout for 2 angles: [c0, c1, o0, o1, <7 physical>] = 11.
    # Use the true physical params so the post-solve residual eval stays finite.
    true_physical = [1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0]
    combined = np.array([0.3, 0.3, 1.0, 1.0, *true_physical], dtype=np.float64)
    n_p = combined.shape[0]

    def _fake_sequential(*args, **kwargs):
        return SequentialResult(
            combined_parameters=combined.copy(),
            combined_covariance=np.eye(n_p, dtype=np.float64) * 1e-6,
            per_angle_results=[
                {"phi_angle": 0.0, "n_iterations": 3, "success": True},
                {"phi_angle": 90.0, "n_iterations": 3, "success": True},
            ],
            n_angles_optimized=2,
            n_angles_failed=0,
            total_cost=1.0,
            success_rate=1.0,
        )

    monkeypatch.setattr(wrapper_mod, "optimize_per_angle_sequential", _fake_sequential)

    result = fit_nlsq(data, cfg)

    diag = result.nlsq_diagnostics
    assert isinstance(diag, dict), "sequential result must carry nlsq_diagnostics"
    assert {
        "hierarchical_active",
        "regularization_active",
        "shear_weighting",
    } <= set(diag)
    # Honest inactive markers (no anti-degeneracy runs on the sequential path).
    assert diag["hierarchical_active"] is False
    assert diag["regularization_active"] is False
    assert diag["shear_weighting"] == "laminar_flow_inactive"
    # Sequential payload keys still present (block merged, not replaced).
    assert "parameter_status" in diag
    # Confirm this is genuinely the sequential return path (Site 4).
    assert any("Sequential" in a for a in result.recovery_actions)
    # Fit values are real numbers (sanity; diagnostics-only change).
    assert np.isfinite(result.chi_squared)
