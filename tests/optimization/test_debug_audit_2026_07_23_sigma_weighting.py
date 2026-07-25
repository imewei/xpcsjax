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


class _FakeStratifiedDataWithSigma:
    def __init__(self, phi_flat, t1_flat, t2_flat, g2_flat, sigma):
        self.phi_flat = phi_flat
        self.t1_flat = t1_flat
        self.t2_flat = t2_flat
        self.g2_flat = g2_flat
        self.sigma = sigma
        self.q = 0.0237
        self.L = 2_000_000.0
        self.dt = 0.1


class _HierarchicalOptimizerSpy:
    """Stand-in for HierarchicalOptimizer: captures loss_fn/grad_fn and
    returns a stub result without running real optimization."""

    captured: dict = {}

    def __init__(self, config, n_phi, n_physical):
        self.config = config
        self.n_phi = n_phi
        self.n_physical = n_physical

    def get_diagnostics(self) -> dict:
        # ponytail: minimal stub -- real caller only needs the dict shape,
        # not real diagnostic values, since .fit() never actually runs.
        return {
            "enabled": True,
            "n_phi": self.n_phi,
            "n_physical": self.n_physical,
            "n_per_angle": 2,
            "max_outer_iterations": getattr(self.config, "max_outer_iterations", 1),
            "outer_tolerance": getattr(self.config, "outer_tolerance", 1e-6),
        }

    def fit(self, loss_fn, grad_fn, p0, bounds, outer_iteration_callback=None):
        type(self).captured["loss_fn"] = loss_fn
        type(self).captured["grad_fn"] = grad_fn

        class _Result:
            x = p0
            fun = 0.0
            success = True
            n_outer_iterations = 0
            message = "stub"
            history = []

        return _Result()


def test_laminar_loss_fn_honors_nonuniform_sigma(monkeypatch):
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

    n_phi = 2
    n_t = 4
    phi = np.repeat([0.0, 45.0], n_t * n_t)
    t1 = np.tile(np.repeat(np.arange(n_t, dtype=float), n_t), n_phi)
    t2 = np.tile(np.tile(np.arange(n_t, dtype=float), n_t), n_phi)
    g2 = np.ones_like(phi) + 0.01 * np.arange(phi.size)
    # sigma must be the raw (n_phi, n_t, n_t) GRID, matching production
    # StratifiedData.sigma (wrapper.py copies it verbatim from
    # original_data.sigma, never flattened) -- the plumbing this task adds
    # in Step 3 indexes it as sigma_3d[phi_idx_arr, t1_idx_arr, t2_idx_arr],
    # which requires 3 real grid axes, not a flat per-point array.
    sigma_3d = np.ones((n_phi, n_t, n_t))
    sigma_3d[0] = 0.1  # non-uniform: first phi angle tightly weighted

    stratified_data = _FakeStratifiedDataWithSigma(phi, t1, t2, g2, sigma_3d)

    monkeypatch.setattr(hs, "HierarchicalOptimizer", _HierarchicalOptimizerSpy)
    _HierarchicalOptimizerSpy.captured = {}

    n_physical = 7
    initial_params = np.concatenate([np.ones(2 * n_phi), np.ones(n_physical)])
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params) * 10)

    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=[
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ],
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma"),
    )

    loss_fn = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    assert loss_fn is not None, "hierarchical path was not entered -- check gating config"

    p0 = initial_params
    loss_with_sigma = float(loss_fn(p0))

    # Now re-run with uniform sigma and confirm the loss differs -- proving
    # sigma is actually consulted, not just present but unused.
    stratified_data_uniform = _FakeStratifiedDataWithSigma(
        phi, t1, t2, g2, np.ones((n_phi, n_t, n_t))
    )
    _HierarchicalOptimizerSpy.captured = {}
    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data_uniform,
        per_angle_scaling=True,
        physical_param_names=[
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ],
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma"),
    )
    loss_fn_uniform = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    loss_uniform = float(loss_fn_uniform(p0))

    assert loss_with_sigma != loss_uniform, (
        "loss must change when sigma is non-uniform -- sigma is not being consulted"
    )


def test_laminar_loss_fn_combines_sigma_with_shear_weighting(monkeypatch):
    """Audit follow-up (2026-07-23): the plan's initial fix only edited the
    `else` branch of loss_fn (no shear weighter), leaving sigma silently
    ignored whenever L5 shear weighting is ALSO active -- the common
    laminar_flow case: shear_weighter is constructed whenever
    `is_laminar_flow and shear_weighting_enabled(default True) and n_phi > 3`
    (hybrid_streaming.py). n_phi=4 here (>3) activates shear; explicit
    per_angle_mode='individual' keeps L2 hierarchical active too (n_phi>=3
    would otherwise auto-resolve to 'averaged', which disables hierarchical
    -- see use_constant/per_angle_mode_actual gating). Both layers active
    simultaneously is the scenario Step 4's combined fix must cover."""
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

    n_phi = 4
    n_t = 4
    phi = np.repeat([0.0, 30.0, 60.0, 90.0], n_t * n_t)
    t1 = np.tile(np.repeat(np.arange(n_t, dtype=float), n_t), n_phi)
    t2 = np.tile(np.tile(np.arange(n_t, dtype=float), n_t), n_phi)
    g2 = np.ones_like(phi) + 0.01 * np.arange(phi.size)
    sigma_3d = np.ones((n_phi, n_t, n_t))
    sigma_3d[0] = 0.1  # non-uniform: first phi angle tightly weighted

    stratified_data = _FakeStratifiedDataWithSigma(phi, t1, t2, g2, sigma_3d)

    monkeypatch.setattr(hs, "HierarchicalOptimizer", _HierarchicalOptimizerSpy)
    _HierarchicalOptimizerSpy.captured = {}

    n_physical = 7
    initial_params = np.concatenate([np.ones(2 * n_phi), np.ones(n_physical)])
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params) * 10)
    physical_param_names = [
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]

    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=physical_param_names,
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma_shear"),
        anti_degeneracy_config={"per_angle_mode": "individual"},
    )
    loss_fn = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    assert loss_fn is not None, "hierarchical path was not entered -- check gating config"
    loss_with_sigma = float(loss_fn(initial_params))

    stratified_data_uniform = _FakeStratifiedDataWithSigma(
        phi, t1, t2, g2, np.ones((n_phi, n_t, n_t))
    )
    _HierarchicalOptimizerSpy.captured = {}
    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data_uniform,
        per_angle_scaling=True,
        physical_param_names=physical_param_names,
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma_shear"),
        anti_degeneracy_config={"per_angle_mode": "individual"},
    )
    loss_fn_uniform = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    loss_uniform = float(loss_fn_uniform(initial_params))

    assert loss_with_sigma != loss_uniform, (
        "loss must change when sigma is non-uniform even with shear weighting "
        "active -- the shear_weighter_local branch must also honor sigma"
    )


class _CapturingAdaptiveOptimizerSpy:
    """Stand-in for NLSQ's AdaptiveHybridStreamingOptimizer: captures the
    fit() kwargs it was given (in particular sigma) and returns a stub
    result without running a real solve."""

    captured: dict = {}

    def __init__(self, config):
        self.config = config

    def fit(
        self,
        *,
        data_source,
        func,
        p0,
        bounds=None,
        sigma=None,
        absolute_sigma=False,
        callback=None,
        verbose=1,
    ):
        type(self).captured["sigma"] = sigma
        n = len(p0)
        return {
            "x": np.asarray(p0, dtype=float),
            "pcov": np.eye(n),
            "success": True,
            "streaming_diagnostics": {},
        }


def test_laminar_plain_path_threads_sigma_into_optimizer_fit(monkeypatch):
    """Audit follow-up (2026-07-23, PR #15 review): the plain (non-hierarchical)
    branch of fit_with_stratified_hybrid_streaming never passed sigma= to
    optimizer.fit(), unlike heterodyne_hybrid_streaming.py's equivalent plain
    path (which does, at line ~918) -- contradicting both the design spec's own
    premise ("the sibling plain-path branch...correctly threads sigma") and a
    comment added by this same PR that (incorrectly, until this fix) claimed
    that parity already held for laminar. per_angle_scaling=False forces the
    plain path (hierarchical requires per_angle_scaling=True)."""
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

    n_t = 4
    t1 = np.tile(np.arange(n_t, dtype=float), n_t)
    t2 = np.repeat(np.arange(n_t, dtype=float), n_t)
    keep = t1 != t2
    phi = np.zeros(keep.sum())
    t1 = t1[keep]
    t2 = t2[keep]
    g2 = np.ones_like(phi) + 0.01 * np.arange(phi.size)
    sigma_3d = np.ones((1, n_t, n_t))
    sigma_3d[0, 0, 1] = 0.1  # non-uniform, arbitrary off-diagonal entry

    stratified_data = _FakeStratifiedDataWithSigma(phi, t1, t2, g2, sigma_3d)

    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", _CapturingAdaptiveOptimizerSpy)
    _CapturingAdaptiveOptimizerSpy.captured = {}

    n_physical = 7
    initial_params = np.ones(n_physical)
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params) * 10)

    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=False,
        physical_param_names=[
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ],
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_plain_sigma"),
    )

    captured_sigma = _CapturingAdaptiveOptimizerSpy.captured.get("sigma")
    assert captured_sigma is not None, (
        "plain-path optimizer.fit() must receive sigma when stratified_data.sigma "
        "is set -- it was silently dropped before this fix"
    )
    assert np.asarray(captured_sigma).size == len(g2), (
        "sigma passed to optimizer.fit() must be aligned to y_data's flattened, "
        "non-diagonal-filtered length"
    )
