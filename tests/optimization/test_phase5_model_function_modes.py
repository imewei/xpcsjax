"""Phase 5 — the JIT model_function slices per resolved mode (no crash, correct length)."""
from __future__ import annotations

import numpy as np
import pytest


def _data_obj(n_phi=4, n_t=6):
    """Raw (non-stratified) grid data object the standard path builds."""
    class _D:
        pass

    d = _D()
    d.phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    d.t1 = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    d.t2 = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    d.g2 = np.ones((n_phi, n_t, n_t), dtype=np.float64)
    d.sigma = np.ones((n_phi, n_t, n_t), dtype=np.float64)
    d.q = 0.0237
    d.L = 2_000_000.0
    d.dt = 0.1
    return d


def _make_fn(mode, n_phi=4):
    # NOTE: plan named ``xpcsjax.core.analysis.AnalysisMode`` / ``NLSQFitter``; the
    # live names are ``xpcsjax.config.parameter_registry.AnalysisMode`` /
    # ``NLSQWrapper`` (adapted to real code).
    from xpcsjax.config.parameter_registry import AnalysisMode
    from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

    fitter = NLSQWrapper()
    fixed = None
    if mode == "constant":
        fixed = (np.full(n_phi, 0.3), np.full(n_phi, 1.0))
    return fitter._create_residual_function(
        _data_obj(n_phi=n_phi),
        AnalysisMode.LAMINAR_FLOW,
        per_angle_scaling=True,
        resolved_per_angle_mode=mode,
        fixed_scaling=fixed,
    )


def _phys():
    return [1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0]  # 7 physics


def test_individual_vector_runs():
    n_phi = 4
    fn = _make_fn("individual", n_phi)
    params = [0.3] * n_phi + [1.0] * n_phi + _phys()  # 2*n_phi + 7
    idx = np.arange(n_phi * 6 * 6, dtype=np.int64)
    out = np.asarray(fn(idx, *params))
    assert out.shape == idx.shape
    assert np.all(np.isfinite(out))


def test_averaged_vector_runs_and_broadcasts():
    n_phi = 4
    fn = _make_fn("averaged", n_phi)
    params = [0.3, 1.0] + _phys()  # 2 + 7 == 9
    idx = np.arange(n_phi * 6 * 6, dtype=np.int64)
    out = np.asarray(fn(idx, *params))
    assert out.shape == idx.shape
    assert np.all(np.isfinite(out))


def test_constant_vector_runs_physics_only():
    n_phi = 4
    fn = _make_fn("constant", n_phi)
    params = _phys()  # 7 physics ONLY
    idx = np.arange(n_phi * 6 * 6, dtype=np.int64)
    out = np.asarray(fn(idx, *params))
    assert out.shape == idx.shape
    assert np.all(np.isfinite(out))


def test_averaged_equals_individual_when_uniform():
    """Averaged with (c,o) must equal individual with all angles = (c,o)."""
    n_phi = 4
    idx = np.arange(n_phi * 6 * 6, dtype=np.int64)
    fa = _make_fn("averaged", n_phi)
    fi = _make_fn("individual", n_phi)
    oa = np.asarray(fa(idx, 0.3, 1.0, *_phys()))
    oi = np.asarray(fi(idx, *([0.3] * n_phi + [1.0] * n_phi + _phys())))
    np.testing.assert_allclose(oa, oi, rtol=1e-12, atol=0.0)
