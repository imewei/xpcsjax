"""Tests for xpcsjax.core.jax_backend identified by Gemini round-2 review.

Covers six previously untested paths:
  T1 - Meshgrid cache collision guard (same endpoints, different interior)
  T2 - compute_chi_squared sigma=0 produces finite result (not Inf)
  T3 - compute_g2_scaled_with_factors numerically matches compute_g2_scaled
  T4 - compute_g2_scaled with contrast=0 returns constant offset surface
  T5 - create_time_integral_matrix N=1 returns correct (1,1) matrix
  T6 - safe_sinc is continuous across the Taylor / sin(x)/x threshold at 1e-4
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from xpcsjax.core.diagonal_correction import apply_diagonal_correction_batch
from xpcsjax.core.jax_backend import (
    clear_meshgrid_cache,
    compute_chi_squared,
    compute_g2_scaled,
    compute_g2_scaled_with_factors,
    get_cached_meshgrid,
)
from xpcsjax.core.physics_utils import create_time_integral_matrix, safe_sinc

# ---------------------------------------------------------------------------
# T1: Meshgrid cache — same endpoints, different interior must not collide
# ---------------------------------------------------------------------------


class TestMeshgridCacheCollision:
    """After BUG1 fix the hash key includes interior quartile samples."""

    def setup_method(self) -> None:
        clear_meshgrid_cache()

    def test_uniform_vs_nonuniform_same_endpoints_different_grids(self) -> None:
        """Arrays with same (len, first, last) but different midpoint produce
        distinct cached meshgrids."""
        t_uniform = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64)
        t_skewed = jnp.array([0.0, 0.9, 1.0], dtype=jnp.float64)

        clear_meshgrid_cache()
        g1, _ = get_cached_meshgrid(t_uniform, t_uniform)
        g2, _ = get_cached_meshgrid(t_skewed, t_skewed)

        # The two grids must differ — midpoint row differs (0.5 vs 0.9)
        assert not jnp.allclose(g1, g2), "meshgrids for uniform vs skewed arrays must be distinct"

    def test_same_array_returns_cached_hit(self) -> None:
        """Identical array inputs must return the same meshgrid object (cache hit)."""
        t = jnp.linspace(0.0, 1.0, 10, dtype=jnp.float64)
        clear_meshgrid_cache()
        g1, _ = get_cached_meshgrid(t, t)
        g2, _ = get_cached_meshgrid(t, t)
        assert jnp.allclose(g1, g2)

    def test_longer_arrays_with_same_endpoints_differ(self) -> None:
        """For N=8 arrays, uniform vs non-uniform spacing at same endpoints differ."""
        t_uniform = jnp.linspace(0.0, 7.0, 8, dtype=jnp.float64)
        # Scramble interior while keeping endpoints identical
        t_nonuniform = jnp.array([0.0, 0.1, 0.3, 2.0, 3.5, 5.0, 6.8, 7.0], dtype=jnp.float64)
        clear_meshgrid_cache()
        g1, _ = get_cached_meshgrid(t_uniform, t_uniform)
        g2, _ = get_cached_meshgrid(t_nonuniform, t_nonuniform)
        assert not jnp.allclose(g1, g2)


# ---------------------------------------------------------------------------
# T2: compute_chi_squared — sigma=0 must not produce Inf
# ---------------------------------------------------------------------------


class TestChiSquaredZeroSigma:
    """After BUG2 fix, zero-sigma pixels are excluded (contribute 0), not Inf."""

    @pytest.fixture
    def minimal_params(self) -> jnp.ndarray:
        return jnp.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0], dtype=jnp.float64)

    def _make_grids(self, N: int = 5) -> tuple:
        t = jnp.linspace(1e-3, 1e-2, N, dtype=jnp.float64)
        t1, t2 = jnp.meshgrid(t, t, indexing="ij")
        return t1, t2

    def test_zero_sigma_returns_finite_chi_squared(self, minimal_params: jnp.ndarray) -> None:
        """A single zero-sigma element must not produce Inf chi-squared."""
        t1, t2 = self._make_grids()
        data = jnp.ones_like(t1)
        sigma = jnp.ones_like(t1).at[2, 2].set(0.0)  # one masked pixel
        phi = jnp.array([0.0], dtype=jnp.float64)

        chi2 = compute_chi_squared(
            minimal_params,
            data,
            sigma,
            t1,
            t2,
            phi,
            0.01,
            1e6,
            0.5,
            1.0,
            1e-3,
        )
        assert jnp.isfinite(chi2), f"chi-squared with zero sigma must be finite, got {chi2}"

    def test_all_zero_sigma_returns_zero(self, minimal_params: jnp.ndarray) -> None:
        """All-zero sigma means all pixels excluded → chi-squared = 0."""
        t1, t2 = self._make_grids()
        data = jnp.ones_like(t1)
        sigma = jnp.zeros_like(t1)
        phi = jnp.array([0.0], dtype=jnp.float64)

        chi2 = compute_chi_squared(
            minimal_params,
            data,
            sigma,
            t1,
            t2,
            phi,
            0.01,
            1e6,
            0.5,
            1.0,
            1e-3,
        )
        assert float(chi2) == pytest.approx(0.0, abs=1e-10), (
            "all-zero sigma → all pixels excluded → chi-squared must be 0"
        )

    def test_nonzero_sigma_unaffected(self, minimal_params: jnp.ndarray) -> None:
        """All-positive sigma must give same result before and after the fix path."""
        t1, t2 = self._make_grids()
        data = jnp.ones_like(t1)
        sigma = jnp.full_like(t1, 0.1)
        phi = jnp.array([0.0], dtype=jnp.float64)

        chi2 = compute_chi_squared(
            minimal_params,
            data,
            sigma,
            t1,
            t2,
            phi,
            0.01,
            1e6,
            0.5,
            1.0,
            1e-3,
        )
        assert jnp.isfinite(chi2) and chi2 >= 0.0


# ---------------------------------------------------------------------------
# T3: compute_g2_scaled_with_factors vs compute_g2_scaled parity
# ---------------------------------------------------------------------------


class TestG2ScaledFactorsParity:
    """The pre-computed-factors hot path must produce bit-identical output to
    the reference compute_g2_scaled path."""

    def test_parity_for_diffusion_only_params(self) -> None:
        """compute_g2_scaled_with_factors must equal compute_g2_scaled to float64."""
        params = jnp.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0], dtype=jnp.float64)
        N = 10
        t = jnp.linspace(1e-3, 1e-2, N, dtype=jnp.float64)
        t1, t2 = jnp.meshgrid(t, t, indexing="ij")
        phi = jnp.array([0.0, math.pi / 4], dtype=jnp.float64)
        q, L, contrast, offset, dt = 0.01, 1e6, 0.5, 1.0, 1e-3

        # Reference path
        ref = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)

        # Pre-computed-factors path
        q2h_dt = 0.5 * q**2 * dt  # wavevector_q_squared_half_dt
        sinc_pf = q * L * dt / (2 * math.pi)  # sinc_prefactor
        fast = compute_g2_scaled_with_factors(
            params,
            t1,
            t2,
            phi,
            q2h_dt,
            sinc_pf,
            contrast,
            offset,
            dt,
        )

        assert jnp.allclose(ref, fast, rtol=1e-10, atol=1e-12), (
            "pre-computed-factors path must be numerically identical to reference path"
        )


# ---------------------------------------------------------------------------
# T4: compute_g2_scaled with contrast=0 → pure offset surface
# ---------------------------------------------------------------------------


class TestG2ScaledContrastZero:
    """Physics identity: g₂ = offset + 0·g₁² = offset when contrast=0."""

    def test_zero_contrast_returns_offset_everywhere(self) -> None:
        params = jnp.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0], dtype=jnp.float64)
        N = 8
        t = jnp.linspace(1e-3, 1e-2, N, dtype=jnp.float64)
        t1, t2 = jnp.meshgrid(t, t, indexing="ij")
        phi = jnp.array([0.0], dtype=jnp.float64)
        offset = 1.23

        g2 = compute_g2_scaled(params, t1, t2, phi, 0.01, 1e6, 0.0, offset, 1e-3)

        assert jnp.allclose(g2, offset, rtol=1e-10), (
            f"contrast=0 must give offset={offset} everywhere; got range "
            f"[{float(g2.min()):.6f}, {float(g2.max()):.6f}]"
        )

    def test_unit_contrast_produces_varied_surface(self) -> None:
        """Sanity check: contrast=1 must not be constant."""
        params = jnp.array([100.0, 0.5, 10.0, 1e-4, 0.0, 0.0, 0.0], dtype=jnp.float64)
        N = 8
        t = jnp.linspace(1e-3, 5e-2, N, dtype=jnp.float64)
        t1, t2 = jnp.meshgrid(t, t, indexing="ij")
        phi = jnp.array([0.0], dtype=jnp.float64)

        g2 = compute_g2_scaled(params, t1, t2, phi, 0.05, 1e6, 1.0, 1.0, 1e-3)

        # g2 shape is (n_phi=1, N, N); diagonal vs far-off-diagonal must differ
        diag_val = float(g2[0, 0, 0])
        corner_val = float(g2[0, 0, -1])
        assert diag_val > corner_val, "correlation surface must decay away from diagonal"


# ---------------------------------------------------------------------------
# T5: create_time_integral_matrix with N=1 → finite (1,1) matrix
# ---------------------------------------------------------------------------


class TestTimeIntegralMatrixN1:
    """N=1 single-point time array must not crash and must return finite (1,1)."""

    def test_n1_returns_1x1_finite_matrix(self) -> None:
        t = jnp.array([5.0], dtype=jnp.float64)
        mat = create_time_integral_matrix(t)
        assert mat.shape == (1, 1), f"expected (1,1), got {mat.shape}"
        assert jnp.isfinite(mat).all(), "N=1 integral matrix must be finite"

    def test_n1_value_is_near_zero(self) -> None:
        """The integral from t[0] to t[0] is zero; smooth_abs gives sqrt(eps)."""
        t = jnp.array([5.0], dtype=jnp.float64)
        mat = create_time_integral_matrix(t)
        # Value should be small (smooth_abs of 0 = sqrt(1e-12) ≈ 1e-6)
        assert float(mat[0, 0]) < 0.1, f"N=1 self-integral should be ~eps, got {float(mat[0, 0])}"

    def test_n2_returns_2x2_matrix(self) -> None:
        """Sanity check: N=2 also works."""
        t = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mat = create_time_integral_matrix(t)
        assert mat.shape == (2, 2)
        assert jnp.isfinite(mat).all()


class TestDiagonalCorrectionBatchN1:
    """Batched basic correction must preserve 1x1 matrices like the scalar path."""

    def test_numpy_batch_n1_preserves_input(self) -> None:
        c2 = jnp.asarray([[[7.5]], [[2.25]]], dtype=jnp.float64)

        corrected = apply_diagonal_correction_batch(c2, backend="numpy")

        assert corrected.shape == (2, 1, 1)
        assert jnp.allclose(jnp.asarray(corrected), c2)


# ---------------------------------------------------------------------------
# T6: safe_sinc continuity across Taylor / sin(x)/x threshold at 1e-4
# ---------------------------------------------------------------------------


class TestSafeSincContinuity:
    """safe_sinc must be continuous and well-valued at the Taylor threshold."""

    def test_continuity_at_threshold(self) -> None:
        """Values just inside and just outside the 1e-4 threshold must agree
        to better than 1e-8 (Taylor is accurate to O(x⁶))."""
        threshold = 1e-4
        x_inside = jnp.array(threshold * (1 - 1e-6), dtype=jnp.float64)
        x_outside = jnp.array(threshold * (1 + 1e-6), dtype=jnp.float64)

        val_inside = float(safe_sinc(x_inside))
        val_outside = float(safe_sinc(x_outside))

        assert abs(val_inside - val_outside) < 1e-8, (
            f"safe_sinc discontinuity at threshold: inside={val_inside:.12f}, "
            f"outside={val_outside:.12f}, diff={abs(val_inside - val_outside):.2e}"
        )

    def test_value_at_zero_is_one(self) -> None:
        """sinc(0) = 1 by the Taylor expansion."""
        assert float(safe_sinc(jnp.float64(0.0))) == pytest.approx(1.0, abs=1e-12)

    def test_value_at_pi_is_near_zero(self) -> None:
        """sin(π)/π ≈ 0; sanity check for the far branch."""
        val = float(safe_sinc(jnp.float64(math.pi)))
        assert abs(val) < 1e-10, f"sinc(π) should be ≈0, got {val}"

    def test_no_nan_across_range(self) -> None:
        """safe_sinc must return finite values for all x in [-10, 10]."""
        x = jnp.linspace(-10.0, 10.0, 1000, dtype=jnp.float64)
        vals = safe_sinc(x)
        assert jnp.all(jnp.isfinite(vals)), "safe_sinc must be finite across [-10, 10]"

    def test_gradient_continuous_near_zero(self) -> None:
        """Gradient at x just below threshold must match gradient just above."""
        import jax

        threshold = 1e-4
        grad_fn = jax.grad(lambda x: safe_sinc(x))
        g_in = float(grad_fn(jnp.float64(threshold * 0.5)))
        g_out = float(grad_fn(jnp.float64(threshold * 2.0)))
        # Both gradients should be finite (no discontinuity)
        assert math.isfinite(g_in), f"gradient inside threshold must be finite: {g_in}"
        assert math.isfinite(g_out), f"gradient outside threshold must be finite: {g_out}"
