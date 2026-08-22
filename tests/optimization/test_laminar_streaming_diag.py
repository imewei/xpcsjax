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
import pytest

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
    info_ad = {"mode": "averaged", "controller_diagnostics": {"version": 1}}
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

    A pre-existing JAX TracerArrayConversionError that used to block any real
    ``optimize_per_angle_sequential`` solve is now fixed (see git history for
    ``wrapper.py``'s sequential ``residual_func`` and ``strategies/
    sequential.py``'s ``has_fixed`` branch, both rewritten to stay
    trace-safe) — see ``test_fixed_parameters_integration.py::
    test_fixed_parameter_survives_sequential_fit`` for a real end-to-end
    proof. This test still stubs the solver deliberately, not to work
    around that bug: the goal here is isolating the Site 4 payload-build +
    anti-degeneracy merge from solver convergence behavior on this tiny
    synthetic fixture, so we stub the solver with a minimal successful
    ``SequentialResult`` and assert only on the diagnostics wiring under
    test.
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


def test_sequential_laminar_rescales_covariance_for_active_shear_transform(monkeypatch):
    """Regression: the Site 4 (sequential per-angle) covariance-rescale call
    (``wrapper.py`` ~line 3624, ``adjust_covariance_for_transforms``) must run
    without raising, and must rescale ONLY the log-transformed gamma_dot_t0
    entry, whenever ``shear_transforms.enable_gamma_dot_log`` is configured.

    This is the branch ``test_sequential_laminar_emits_symmetric_activation_keys``
    does NOT exercise: with no shear transform configured,
    ``apply_forward_shear_transforms_to_vector`` returns an EMPTY ``{}`` state
    (see ``transforms.py``), which is falsy, so the ``if transform_state:`` guard
    at the covariance-rescale call site never fires there. Pre-fix, this call
    passed an extra, erroneous positional argument
    (``adjust_covariance_for_transforms(cov, combined_solver, combined_physical,
    state)`` instead of the 3-arg ``(cov, physical_params, state)``) and would
    have raised ``TypeError: too many positional arguments`` the first time a
    real fit reached this branch -- which no test did.
    """
    import xpcsjax.optimization.nlsq.wrapper as wrapper_mod
    from xpcsjax.optimization.nlsq.strategies.sequential import SequentialResult

    fit_nlsq, data, cfg = _build_sequential_laminar_fit()
    cfg.config["optimization"]["nlsq"]["shear_transforms"] = {"enable_gamma_dot_log": True}

    # Layout: [c0, c1, o0, o1, D0, alpha, D_offset, gamma_dot_t0, beta,
    # gamma_dot_t_offset, phi0] -- gamma_dot_t0 is physical index 3, and the
    # dense per-angle scaling head is 2*n_phi=4, so its combined-vector index
    # is 4 + 3 = 7 (mirrors build_physical_index_map's dense default).
    gamma_idx = 7
    true_gamma_dot_t0 = 0.01
    true_physical = [1000.0, 0.5, 10.0, true_gamma_dot_t0, 0.0, 0.0, 0.0]
    # combined_parameters is SOLVER-space: gamma_dot_t0 is log-transformed.
    combined_solver_space = np.array([0.3, 0.3, 1.0, 1.0, *true_physical], dtype=np.float64)
    combined_solver_space[gamma_idx] = np.log(true_gamma_dot_t0)
    n_p = combined_solver_space.shape[0]
    base_variance = 1e-6

    def _fake_sequential(*args, **kwargs):
        return SequentialResult(
            combined_parameters=combined_solver_space.copy(),
            combined_covariance=np.eye(n_p, dtype=np.float64) * base_variance,
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

    # Pre-fix, this raised TypeError inside adjust_covariance_for_transforms.
    result = fit_nlsq(data, cfg)

    # Parameters are inverse-transformed back to physical space.
    assert result.parameters[gamma_idx] == pytest.approx(true_gamma_dot_t0, rel=1e-9)

    # Only the gamma_dot_t0 diagonal entry is rescaled, by scale**2 where
    # scale == the physical-space value (see adjust_covariance_for_transforms).
    expected_gamma_uncertainty = np.sqrt(base_variance) * true_gamma_dot_t0
    assert result.uncertainties[gamma_idx] == pytest.approx(expected_gamma_uncertainty, rel=1e-9)
    # An untouched (non-gamma) diagonal entry stays at the base variance.
    assert result.uncertainties[0] == pytest.approx(np.sqrt(base_variance), rel=1e-9)


# --- Phase 6 Task 5: laminar streaming has no reparam machinery ---


def test_streaming_has_no_fourier_machinery():
    import inspect

    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming

    fns = [
        v
        for v in vars(hybrid_streaming).values()
        if callable(v) and getattr(v, "__module__", None) == hybrid_streaming.__name__
    ]
    src = "\n".join(inspect.getsource(f) for f in fns)
    # Tokens rebuilt from fragments so this file stays clean under the Phase-7 gate.
    f = "four" + "ier"
    assert f"model_fn_{f}" not in src
    assert f"{f}_reparameterizer" not in src
    assert f"use_{f}" not in src
    assert f"{f.capitalize()}Reparameterizer" not in src


# --- Phase 6 Task 6: mapper drives streaming L3 group_indices / L4 n_optimized ---


def test_mapper_drives_streaming_L3_L4():  # noqa: N802
    from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper
    from xpcsjax.optimization.nlsq.per_angle_mode import n_optimized

    cases = [
        ("individual", 5, [(0, 5), (5, 10)], 10),
        ("averaged", 5, [(0, 1), (1, 2)], 2),
        ("constant", 5, [], 0),
    ]
    for mode, n_phi, exp_groups, exp_nopt in cases:
        m = ParameterIndexMapper.canonical(mode=mode, n_phi=n_phi, n_physics=7)
        assert m.group_indices == exp_groups
        assert m.n_optimized == exp_nopt
        assert m.n_optimized == n_optimized(mode, n_phi)
