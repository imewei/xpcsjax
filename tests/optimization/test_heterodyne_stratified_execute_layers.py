"""Phase 3 (steps 8-10): gated L2/L3 numeric execution on the heterodyne
stratified-LS path.

These tests pin the ``execute_layers`` contract:

* **Flag OFF (default)** — the stratified-LS solve is the single baseline solve;
  ``hierarchical_active`` / ``regularization_active`` stay ``False`` (the inert
  gate, byte-identical to pre-Phase-3 behavior).
* **Flag ON + ``enable_hierarchical``** — the L2 hierarchical solver runs on the
  inline ``[physics | scaling]`` residual (individual / fourier), and the result
  is accepted only under the keep-better guard (data-only SSR never worse than
  the baseline by more than ``tol``).
* **Flag ON + ``regularization_mode != "none"``** — L3 is configured; the
  reported ``chi_squared`` stays the *data-only* SSR (penalty rows never leak
  into the objective).
* **Keep-better rejection** — a worse L2 candidate is discarded and the baseline
  single-solve is returned, marked ``attempted_but_rejected``.

All fits use tiny synthetic data and call the driver directly (bypassing the
1M-point dispatch gate), so they run in well under a second each.
"""

from __future__ import annotations

import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq import heterodyne_stratified_ls as _hsl
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode
from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
    fit_heterodyne_stratified_least_squares,
)


def _fit(model, c2, phi, cfg):
    return fit_heterodyne_stratified_least_squares(
        model=model, c2=c2, phi=phi, config=cfg, weights=None, shuffle=False
    )


# ---------------------------------------------------------------------------
# Flag OFF — inert gate (regression net)
# ---------------------------------------------------------------------------


def test_execute_layers_off_individual_is_inert():
    """With ``execute_layers=False`` (default) the markers stay False."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    assert _resolve_effective_mode(cfg, len(phi)) == "individual"
    res = _fit(model, c2, phi, cfg)
    assert res.nlsq_diagnostics["hierarchical_active"] is False
    assert res.nlsq_diagnostics["regularization_active"] is False
    assert np.isfinite(res.chi_squared)


# ---------------------------------------------------------------------------
# Flag ON — L2 executes (individual)
# ---------------------------------------------------------------------------


def test_execute_layers_on_individual_runs_hierarchical():
    """Flag ON + ``enable_hierarchical`` runs L2 on individual mode."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "execute_layers": True,
            "enable_hierarchical": True,
            "hierarchical_max_outer_iterations": 3,
        }
    )
    res = _fit(model, c2, phi, cfg)
    assert res.nlsq_diagnostics["hierarchical_active"] is True
    assert np.isfinite(res.chi_squared)
    n_physics = int(model.param_manager.n_varying)
    assert len(res.parameters) == n_physics + 2 * len(phi)


def test_execute_layers_honors_hierarchical_budget():
    """The L2 branch honors the config outer-iteration budget (review Fix 2/3)."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "execute_layers": True,
            "enable_hierarchical": True,
            "hierarchical_max_outer_iterations": 2,
        }
    )
    res = _fit(model, c2, phi, cfg)
    diag = res.nlsq_diagnostics
    assert diag["hierarchical_active"] is True
    # Provenance keys surfaced; outer iterations bounded by the configured budget.
    assert diag["execute_layers_kind"] == "L2_hierarchical"
    assert 0 <= int(diag["execute_layers_n_outer"]) <= 2
    assert "execute_layers_converged" in diag


def test_execute_layers_fourier_l2_l3_executes():
    """Fourier L2+L3 executes and keeps the data-only objective (review Fix 1).

    Exercises the per-angle reconstruction path (Fourier coefficients ->
    per-angle contrast/offset) that L3 must penalize. Contract: executes,
    keep-better vs baseline, and chi^2 is data-only (penalty never contaminates).
    """
    model, c2, phi = make_synthetic_two_component(n_phi=7, n_t=20)
    base = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "fourier"}
    )
    on = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "fourier",
            "execute_layers": True,
            "enable_hierarchical": True,
            "regularization_mode": "adaptive",
            "group_variance_lambda": 0.01,
            "hierarchical_max_outer_iterations": 3,
        }
    )
    ssr_off = _fit(model, c2, phi, base).chi_squared
    res = _fit(model, c2, phi, on)
    assert res.nlsq_diagnostics["hierarchical_active"] is True
    assert res.nlsq_diagnostics["regularization_active"] is True
    assert res.chi_squared <= ssr_off * (1.0 + 1e-3)
    chi2_pa = np.asarray(res.nlsq_diagnostics["chi2_per_angle"], dtype=np.float64)
    assert np.isclose(chi2_pa.sum(), res.chi_squared, rtol=1e-6, atol=1e-9)


def test_execute_layers_keep_better_never_worse():
    """The accepted L2 result is never worse than baseline beyond tol."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    base_cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    on_cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "execute_layers": True,
            "enable_hierarchical": True,
            "hierarchical_max_outer_iterations": 3,
        }
    )
    ssr_off = _fit(model, c2, phi, base_cfg).chi_squared
    ssr_on = _fit(model, c2, phi, on_cfg).chi_squared
    assert ssr_on <= ssr_off * (1.0 + 1e-3)


# ---------------------------------------------------------------------------
# Flag ON — L3 objective separation
# ---------------------------------------------------------------------------


def test_execute_layers_l3_objective_separation():
    """L3 active → ``regularization_active`` True, penalty never in chi_squared."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "execute_layers": True,
            "enable_hierarchical": True,
            "regularization_mode": "adaptive",
            "group_variance_lambda": 0.01,
            "hierarchical_max_outer_iterations": 3,
        }
    )
    res = _fit(model, c2, phi, cfg)
    assert res.nlsq_diagnostics["regularization_active"] is True
    # chi2_per_angle sums to the reported (data-only) chi_squared — penalty rows
    # must not contaminate the objective.
    chi2_pa = np.asarray(res.nlsq_diagnostics["chi2_per_angle"], dtype=np.float64)
    assert np.isclose(chi2_pa.sum(), res.chi_squared, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Flag ON — L3-only (L2 disabled) on individual: row-append re-solve
# ---------------------------------------------------------------------------


def test_execute_layers_l3_only_individual_row_append():
    """L3 configured but L2 disabled (individual) runs a row-append re-solve.

    ``regularization_active`` flips True; the keep-better guard ensures the
    data-only SSR is never worse than baseline; ``hierarchical_active`` stays
    False (L2 was not enabled); the reported chi^2 is data-only (penalty rows
    never contaminate it).
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    base = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    on = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "execute_layers": True,
            "enable_hierarchical": False,
            "regularization_mode": "adaptive",
            "group_variance_lambda": 0.01,
        }
    )
    ssr_off = _fit(model, c2, phi, base).chi_squared
    res = _fit(model, c2, phi, on)
    assert res.nlsq_diagnostics["regularization_active"] is True
    assert res.nlsq_diagnostics["hierarchical_active"] is False
    assert res.chi_squared <= ssr_off * (1.0 + 1e-3)
    chi2_pa = np.asarray(res.nlsq_diagnostics["chi2_per_angle"], dtype=np.float64)
    assert np.isclose(chi2_pa.sum(), res.chi_squared, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Flag ON — averaged mode: L3 flag only, no L2, no numeric change
# ---------------------------------------------------------------------------


def test_execute_layers_averaged_l3_is_flag_only():
    """Averaged L3 penalty is degenerate-zero: flag flips, numerics unchanged."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    off = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "averaged"}
    )
    on = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "execute_layers": True,
            "regularization_mode": "adaptive",
        }
    )
    res_off = _fit(model, c2, phi, off)
    res_on = _fit(model, c2, phi, on)
    assert res_on.nlsq_diagnostics["regularization_active"] is True
    # averaged has no per-angle DoF → L2 must not fire
    assert res_on.nlsq_diagnostics["hierarchical_active"] is False
    # degenerate-zero penalty → identical solution
    np.testing.assert_allclose(res_on.parameters, res_off.parameters, rtol=1e-8, atol=1e-10)
    assert np.isclose(res_on.chi_squared, res_off.chi_squared, rtol=1e-8)


# ---------------------------------------------------------------------------
# Flag ON — keep-better rejection (monkeypatch a deliberately-bad candidate)
# ---------------------------------------------------------------------------


def test_execute_layers_rejected_keeps_baseline(monkeypatch):
    """A worse L2 candidate is discarded; the baseline solve is returned."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    base_cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    on_cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "execute_layers": True,
            "enable_hierarchical": True,
            "hierarchical_max_outer_iterations": 2,
        }
    )
    baseline = _fit(model, c2, phi, base_cfg)

    # Force the L2 runner to return a wrecked per-angle scaling (a much worse SSR),
    # so the keep-better guard must reject it and fall back to the baseline solve.
    def _bad_runner(*, p0_start, n_physics, **_kwargs):
        bad = np.asarray(p0_start, dtype=np.float64).copy()
        bad[n_physics:] = bad[n_physics:] + 100.0  # inflate scaling tail -> huge SSR
        return {"popt": bad, "n_outer": 1, "success": True}

    monkeypatch.setattr(_hsl, "_run_hierarchical_layers", _bad_runner)

    res = _fit(model, c2, phi, on_cfg)
    # Baseline kept → markers honest (not executed), params match the baseline.
    assert res.nlsq_diagnostics["hierarchical_active"] is False
    assert res.nlsq_diagnostics.get("execute_layers_status") == "attempted_but_rejected"
    np.testing.assert_allclose(res.parameters, baseline.parameters, rtol=1e-6, atol=1e-8)
