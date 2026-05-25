"""Scientific tests for xpcsjax.optimization.nlsq.parameter_utils.

Pure helpers (labels, bound classification, subsampling) are covered directly.
``compute_jacobian_stats`` is exercised through JAX ``jacfwd`` on a known-linear
residual. The two per-angle estimators are validated by **analytical
recovery**: we synthesize ``C2 = offset + contrast * g1^2`` with known scaling
and assert the estimator recovers it, plus the fallback-to-defaults branches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.core.physics_utils import (
    calculate_diffusion_coefficient,
    trapezoid_cumsum,
)
from xpcsjax.optimization.nlsq import parameter_utils as pu

# ---------------------------------------------------------------------------
# build_parameter_labels
# ---------------------------------------------------------------------------


def test_build_parameter_labels_no_per_angle() -> None:
    assert pu.build_parameter_labels(False, 3, ["D0", "alpha"]) == ["D0", "alpha"]


def test_build_parameter_labels_per_angle() -> None:
    labels = pu.build_parameter_labels(True, 2, ["D0"])
    assert labels == ["contrast[0]", "contrast[1]", "offset[0]", "offset[1]", "D0"]


# ---------------------------------------------------------------------------
# classify_parameter_status
# ---------------------------------------------------------------------------


def test_classify_status_no_bounds_all_active() -> None:
    out = pu.classify_parameter_status(np.array([1.0, 2.0]), None, None)
    assert out == ["active", "active"]


def test_classify_status_detects_bounds() -> None:
    values = np.array([0.0, 5.0, 2.5])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 5.0, 5.0])
    assert pu.classify_parameter_status(values, lower, upper) == [
        "at_lower_bound",
        "at_upper_bound",
        "active",
    ]


# ---------------------------------------------------------------------------
# sample_xdata
# ---------------------------------------------------------------------------


def test_sample_xdata_passthrough_when_small() -> None:
    x = np.arange(5)
    assert pu.sample_xdata(x, 10) is x


def test_sample_xdata_subsamples() -> None:
    x = np.arange(100)
    out = pu.sample_xdata(x, 10)
    assert out.size == 10
    assert out[0] == 0 and out[-1] == 99  # endpoints preserved


def test_sample_xdata_nonpositive_max() -> None:
    x = np.arange(5)
    assert pu.sample_xdata(x, 0) is x


# ---------------------------------------------------------------------------
# compute_jacobian_stats (JAX jacfwd)
# ---------------------------------------------------------------------------


def test_jacobian_stats_plain_residual() -> None:
    def residual_fn(x: np.ndarray, a: float, b: float) -> jnp.ndarray:
        return jnp.asarray(a) * jnp.asarray(x) + jnp.asarray(b)

    x_subset = np.array([1.0, 2.0, 3.0])
    params = np.array([2.0, 1.0])
    jtj, col_norms = pu.compute_jacobian_stats(residual_fn, x_subset, params, 1.0)
    assert jtj is not None and col_norms is not None
    assert jtj.shape == (2, 2)
    assert col_norms.shape == (2,)
    # d/da = x -> column norm = ||x||; d/db = 1 -> column norm = sqrt(3)
    assert col_norms[0] == pytest.approx(np.linalg.norm(x_subset))
    assert col_norms[1] == pytest.approx(np.sqrt(3.0))


def test_jacobian_stats_uses_jax_residual_attr() -> None:
    class _Resid:
        def jax_residual(self, p: jnp.ndarray) -> jnp.ndarray:
            return p[0] * jnp.ones(3) + p[1]

    jtj, col_norms = pu.compute_jacobian_stats(
        cast(Any, _Resid()), np.array([0.0]), np.array([2.0, 1.0]), 2.0
    )
    assert jtj is not None and col_norms is not None
    assert jtj.shape == (2, 2)


def test_jacobian_stats_failure_returns_none() -> None:
    def bad_residual(x: np.ndarray, *p: float) -> jnp.ndarray:
        raise ValueError("synthetic residual failure")

    jtj, col_norms = pu.compute_jacobian_stats(
        bad_residual, np.array([1.0]), np.array([1.0]), 1.0
    )
    assert jtj is None
    assert col_norms is None


# ---------------------------------------------------------------------------
# compute_consistent_per_angle_init  (analytical recovery, static mode)
# ---------------------------------------------------------------------------


def _static_g1_sq(
    t1: np.ndarray, t2: np.ndarray, t_unique: np.ndarray, q: float, dt: float,
    D0: float, alpha: float, D_offset: float,
) -> np.ndarray:
    """Mirror the diffusion-only g1^2 the estimator computes, for self-consistency."""
    d_t = calculate_diffusion_coefficient(t_unique, D0, alpha, D_offset)
    d_cumsum = np.asarray(trapezoid_cumsum(d_t))
    wq = 0.5 * q**2 * dt
    i1 = np.clip(np.searchsorted(t_unique, t1), 0, len(t_unique) - 1)
    i2 = np.clip(np.searchsorted(t_unique, t2), 0, len(t_unique) - 1)
    log_g1 = -wq * np.abs(d_cumsum[i1] - d_cumsum[i2])
    g1 = np.clip(np.exp(np.clip(log_g1, -700.0, 0.0)), 1e-10, 1.0)
    return g1**2


def _static_stratified(g2: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray,
                       q: float, dt: float) -> SimpleNamespace:
    # No 'chunks' attribute -> the flat-array code path is used.
    return SimpleNamespace(
        phi_flat=phi, g2_flat=g2, t1_flat=t1, t2_flat=t2, q=q, L=1.0, dt=dt
    )


def test_consistent_init_recovers_known_scaling_static() -> None:
    q, dt = 0.01, 1.0
    D0, alpha, D_offset = 1.0e-3, 1.0, 0.0
    contrast_true, offset_true = 0.4, 1.0

    t_grid = np.arange(10.0)
    t1 = np.repeat(t_grid, 10)
    t2 = np.tile(t_grid, 10)
    t_unique = np.unique(np.concatenate([t1, t2]))
    g1_sq = _static_g1_sq(t1, t2, t_unique, q, dt, D0, alpha, D_offset)
    g2 = offset_true + contrast_true * g1_sq

    # Two angles sharing identical (phi-independent) diffusion structure.
    phi = np.concatenate([np.zeros(100), np.full(100, 45.0)])
    data = _static_stratified(
        np.tile(g2, 2), np.tile(t1, 2), np.tile(t2, 2), phi, q, dt
    )

    contrast, offset = pu.compute_consistent_per_angle_init(
        data, np.array([D0, alpha, D_offset]), ["D0", "alpha", "D_offset"]
    )
    assert contrast.shape == (2,)
    np.testing.assert_allclose(contrast, contrast_true, rtol=1e-6)
    np.testing.assert_allclose(offset, offset_true, rtol=1e-6)


def test_consistent_init_falls_back_to_defaults_on_flat_data() -> None:
    # Constant g2 -> degenerate fit (contrast ~ 0) fails the sanity gate, so
    # the supplied defaults are retained.
    t_grid = np.arange(10.0)
    t1 = np.repeat(t_grid, 10)
    t2 = np.tile(t_grid, 10)
    g2 = np.full(100, 1.2)
    data = _static_stratified(g2, t1, t2, np.zeros(100), 0.01, 1.0)
    contrast, offset = pu.compute_consistent_per_angle_init(
        data, np.array([1e-3, 1.0, 0.0]), ["D0", "alpha", "D_offset"],
        default_contrast=0.5, default_offset=1.0,
    )
    assert contrast[0] == pytest.approx(0.5)
    assert offset[0] == pytest.approx(1.0)


def test_consistent_init_chunks_format_smoke() -> None:
    t_grid = np.arange(8.0)
    t1 = np.repeat(t_grid, 8)
    t2 = np.tile(t_grid, 8)
    g2 = 1.0 + 0.3 * np.linspace(1.0, 0.0, t1.size)
    chunk = SimpleNamespace(
        phi=np.zeros(t1.size), g2=g2, t1=t1, t2=t2, q=0.01, L=1.0, dt=1.0
    )
    data = SimpleNamespace(chunks=[chunk])
    contrast, offset = pu.compute_consistent_per_angle_init(
        data, np.array([1e-3, 1.0, 0.0]), ["D0", "alpha", "D_offset"]
    )
    assert contrast.shape == (1,)
    assert np.all(np.isfinite(contrast))
    assert np.all(np.isfinite(offset))


# ---------------------------------------------------------------------------
# compute_quantile_per_angle_scaling  (analytical recovery, model-free)
# ---------------------------------------------------------------------------


def _quantile_flat(n: int, offset: float, contrast: float) -> SimpleNamespace:
    """Build flat data where small lags sit at the ceiling and large lags at the floor."""
    t1 = np.arange(float(n))
    t2 = np.zeros(n)  # delta_t == t1, spanning [0, n)
    delta_t = np.abs(t1 - t2)
    lo = np.quantile(delta_t, 0.20)
    hi = np.quantile(delta_t, 0.80)
    g2 = np.full(n, (offset + (offset + contrast)) / 2.0)  # mid-region filler
    g2[delta_t <= lo] = offset + contrast  # ceiling (small lag, g1^2 ~ 1)
    g2[delta_t >= hi] = offset  # floor (large lag, g1^2 ~ 0)
    return SimpleNamespace(
        phi_flat=np.zeros(n), g2_flat=g2, t1_flat=t1, t2_flat=t2, q=0.01, L=1.0, dt=1.0
    )


def test_quantile_recovers_floor_and_ceiling() -> None:
    offset_true, contrast_true = 1.0, 0.5
    data = _quantile_flat(400, offset_true, contrast_true)
    contrast, offset = pu.compute_quantile_per_angle_scaling(data)
    assert contrast.shape == (1,)
    np.testing.assert_allclose(offset, offset_true, atol=1e-6)
    np.testing.assert_allclose(contrast, contrast_true, atol=1e-6)


def test_quantile_insufficient_data_uses_midpoint_defaults() -> None:
    # Fewer than 100 points per angle -> midpoint of the bounds is returned.
    data = SimpleNamespace(
        phi_flat=np.zeros(50),
        g2_flat=np.full(50, 1.3),
        t1_flat=np.arange(50.0),
        t2_flat=np.zeros(50),
        q=0.01,
        L=1.0,
        dt=1.0,
    )
    contrast, offset = pu.compute_quantile_per_angle_scaling(
        data, contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5)
    )
    assert contrast[0] == pytest.approx(0.5)  # midpoint of (0, 1)
    assert offset[0] == pytest.approx(1.0)  # midpoint of (0.5, 1.5)


def test_quantile_chunks_format_smoke() -> None:
    n = 300
    t1 = np.arange(float(n))
    chunk = SimpleNamespace(
        phi=np.zeros(n),
        g2=1.0 + 0.4 * np.exp(-t1 / 50.0),
        t1=t1,
        t2=np.zeros(n),
        q=0.01,
        L=1.0,
        dt=1.0,
    )
    data = SimpleNamespace(chunks=[chunk])
    contrast, offset = pu.compute_quantile_per_angle_scaling(data)
    assert contrast.shape == (1,)
    assert np.all(np.isfinite(contrast)) and np.all(np.isfinite(offset))
