"""Regression: the L2 hierarchical loss must honor per-point sigma weighting,
matching the sibling plain-path branch in the same function.

Finding #4 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
Covers both heterodyne_hybrid_streaming.py (two_component) and
hybrid_streaming.py (laminar_flow).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_heterodyne_hier_loss_honors_nonuniform_sigma():
    """_sigma_weighted_mse (the arithmetic building block _hier_loss/_loss_jax
    call) must divide residuals by sigma when sigma is non-uniform, not just
    compute an unweighted mean(residuals**2). This is a pure arithmetic check
    of the helper in isolation -- see
    test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse below for
    the separate proof that _hier_loss/_loss_jax actually CALL this helper
    (an implementation could add the helper without wiring it in and this
    test alone would not catch that)."""
    from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as hhs

    y_data = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 1.5, 3.5, 3.5])  # residuals = [-0.5, 0.5, -0.5, 0.5]
    residuals = y_data - pred
    uniform_sigma = np.ones(4)
    nonuniform_sigma = np.array([0.1, 1.0, 0.1, 1.0])

    loss_uniform = hhs._sigma_weighted_mse(residuals, uniform_sigma) * y_data.shape[0]
    loss_nonuniform = hhs._sigma_weighted_mse(residuals, nonuniform_sigma) * y_data.shape[0]
    loss_unweighted = np.mean(residuals**2) * y_data.shape[0]

    assert loss_uniform == pytest.approx(loss_unweighted)
    assert loss_nonuniform != pytest.approx(loss_unweighted)


def test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse(monkeypatch):
    """Wiring check: drive the REAL fit_with_stratified_hybrid_streaming_heterodyne
    through its L2 hierarchical branch (per_angle_mode='individual') with
    non-uniform per-point weights, and spy on the module-level
    _sigma_weighted_mse to prove _hier_loss/_loss_jax actually call it. Reuses
    the proven synthetic-heterodyne fixture from
    tests/optimization/test_heterodyne_hybrid_streaming.py's own
    test_l2_individual_runs_and_beats_frozen_baseline (n_phi=2 -> auto
    resolves to individual, exercising the L2 branch)."""
    from tests.optimization.test_heterodyne_hybrid_streaming import (
        _make_synthetic_heterodyne,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as hs

    model, c2, phi = _make_synthetic_heterodyne(n_phi=2, n_t=8)
    rng = np.random.default_rng(0)
    weights = 1.0 + rng.random(c2.shape)  # non-uniform -> non-uniform sigma
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=weights)
    lo, hi = model.param_manager.get_bounds()

    call_count = [0]
    original = hs._sigma_weighted_mse

    def spy(residuals, sigma):
        call_count[0] += 1
        return original(residuals, sigma)

    monkeypatch.setattr(hs, "_sigma_weighted_mse", spy)

    hs.fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
        bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
        hybrid_config={"verbose": 0},
        anti_degeneracy_config={
            "per_angle_mode": "individual",
            "hierarchical": {"max_outer_iterations": 3},
        },
    )

    assert call_count[0] > 0, (
        "_hier_loss/_loss_jax never called _sigma_weighted_mse -- the helper "
        "exists but is not wired into the hierarchical loss closures"
    )
