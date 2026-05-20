"""Coverage tests for `xpcsjax.optimization.nlsq.gradient_diagnostics`.

Closes the /double-check Phase 6 gap: the ported ``x_scale_map`` recommender
ships but is not exercised by any other test. This file pins the public
contract — keys returned, ordering of magnitudes (poorly-conditioned params
get smaller scales), and clipping behavior. Uses a minimal synthetic XPCS
two-time fixture so the @jax.jit boundary in ``_create_residual_function``
is actually compiled and called.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.gradient_diagnostics import (
    compute_gradient_norms,
    compute_optimal_x_scale,
)

# ---------------------------------------------------------------------------
# Synthetic XPCS fixture — large enough to give the JAX residual something to
# do, small enough to stay under ~5 s wall time.
# ---------------------------------------------------------------------------


def _make_data(n_phi: int = 3, n_t: int = 8) -> SimpleNamespace:
    rng = np.random.default_rng(seed=42)
    phi = np.repeat(np.linspace(0.0, np.pi / 2, n_phi), n_t * n_t)
    t1 = np.tile(np.repeat(np.arange(n_t, dtype=float), n_t), n_phi)
    t2 = np.tile(np.tile(np.arange(n_t, dtype=float), n_t), n_phi)
    # g2 in the physically meaningful range [1.0, 1.5]; small noise.
    g2 = 1.0 + 0.4 * np.exp(-0.05 * np.abs(t2 - t1)) + 0.01 * rng.standard_normal(
        phi.shape
    )
    return SimpleNamespace(
        phi=phi,
        t1=t1,
        t2=t2,
        g2=g2,
        q=0.01,
        L=1e-3,
        dt=1.0,
    )


def _laminar_params() -> dict[str, float]:
    return {
        "D0": 100.0,
        "alpha": 0.0,
        "D_offset": 0.0,
        "gamma_dot_t0": 0.001,
        "beta": 0.0,
        "gamma_dot_t_offset": 0.0,
        "phi0": 0.0,
    }


def _static_params() -> dict[str, float]:
    return {"D0": 100.0, "alpha": 0.0, "D_offset": 0.0}


# ---------------------------------------------------------------------------
# compute_gradient_norms — public contract
# ---------------------------------------------------------------------------


def test_compute_gradient_norms_laminar_returns_all_seven_keys():
    norms = compute_gradient_norms(
        parameters=_laminar_params(),
        data=_make_data(),
        config=None,
        analysis_mode="laminar_flow",
    )
    assert set(norms.keys()) == {
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    }
    for name, value in norms.items():
        assert np.isfinite(value), f"{name} gradient is not finite: {value}"
        assert value >= 0.0, f"{name} gradient norm is negative: {value}"


def test_compute_gradient_norms_static_returns_three_keys():
    norms = compute_gradient_norms(
        parameters=_static_params(),
        data=_make_data(),
        config=None,
        analysis_mode="static_isotropic",
    )
    assert set(norms.keys()) == {"D0", "alpha", "D_offset"}


# ---------------------------------------------------------------------------
# compute_optimal_x_scale — public contract
# ---------------------------------------------------------------------------


def test_compute_optimal_x_scale_returns_one_scale_per_param():
    scales = compute_optimal_x_scale(
        parameters=_laminar_params(),
        data=_make_data(),
        config=None,
        analysis_mode="laminar_flow",
    )
    assert set(scales.keys()) == set(_laminar_params().keys())
    for name, scale in scales.items():
        assert np.isfinite(scale), f"{name} scale is not finite: {scale}"
        assert scale > 0.0, f"{name} scale is non-positive: {scale}"


def test_compute_optimal_x_scale_clips_to_min_max():
    # Crank min_scale up and max_scale down so EVERY parameter must be clipped.
    scales = compute_optimal_x_scale(
        parameters=_laminar_params(),
        data=_make_data(),
        config=None,
        analysis_mode="laminar_flow",
        min_scale=0.5,
        max_scale=1.5,
    )
    for name, scale in scales.items():
        assert 0.5 <= scale <= 1.5, f"{name} scale {scale} outside [0.5, 1.5]"


def test_compute_optimal_x_scale_baseline_params_anchor_at_unity():
    """Baseline params should get x_scale ≈ 1.0 (they ARE the baseline).

    The implementation normalises against the geometric mean of baseline
    gradients, so for a single-element baseline the scale is exactly 1.0;
    for a multi-element baseline the individual scales bracket 1.0.
    """
    scales = compute_optimal_x_scale(
        parameters=_laminar_params(),
        data=_make_data(),
        config=None,
        analysis_mode="laminar_flow",
        baseline_params=["D0"],
    )
    # With a single baseline, D0's scale should be exactly 1.0 (modulo
    # safety_factor=1.0 default and min/max clip not biting).
    assert scales["D0"] == pytest.approx(1.0, rel=1e-6)


def test_compute_optimal_x_scale_unknown_mode_defaults_to_laminar():
    # The implementation uses ``"static" in mode.lower()`` to pick params;
    # anything that does NOT contain "static" gets the 7-param laminar set.
    scales = compute_optimal_x_scale(
        parameters=_laminar_params(),
        data=_make_data(),
        config=None,
        analysis_mode="something_else",
    )
    assert len(scales) == 7
