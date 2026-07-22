"""Regression tests for the 2026-07-22 debug-audit fixes (Fix 1-3).

Fix 1: ``_compute_g1_shear_core``'s static (no-shear) short-circuit must
handle a 0-d scalar ``t1`` like the general branch does.
Fix 2: the element-wise g1 fallback must not silently truncate the
diffusion/shear integral for datasets with >10000 unique lag times when a
real ``time_grid`` is threaded through.
Fix 3: heterodyne ``compute_chi_squared`` must mask out the t=0 boundary and
the t1==t2 diagonal the same way ``compute_residuals`` does.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xpcsjax.core.heterodyne_jax_backend import (
    _offdiag_indices,
    compute_c2_heterodyne,
)
from xpcsjax.core.heterodyne_jax_backend import (
    compute_chi_squared as compute_chi_squared_het,
)
from xpcsjax.core.heterodyne_jax_backend import (
    compute_residuals as compute_residuals_het,
)
from xpcsjax.core.jax_backend import compute_g1_shear, compute_g1_total

# ---------------------------------------------------------------------------
# Fix 1: 0-d scalar t1 in the static (no-shear) branch must not IndexError
# ---------------------------------------------------------------------------


def test_compute_g1_shear_static_mode_accepts_0d_scalar_t1():
    """Static (<7-param) mode with a 0-d scalar t1 must not raise."""
    params = jnp.array([100.0, 0.0, 10.0], dtype=jnp.float64)  # len < 7 -> static
    t1 = jnp.array(0.5, dtype=jnp.float64)  # 0-d scalar
    t2 = jnp.array(0.5, dtype=jnp.float64)
    phi = jnp.array([0.0], dtype=jnp.float64)

    result = compute_g1_shear(params, t1, t2, phi, q=0.01, L=1e6, dt=1e-3)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, 1.0)  # g1_shear = 1 in static mode


# ---------------------------------------------------------------------------
# Fix 2: element-wise integral must not silently truncate beyond 10001 pts
# ---------------------------------------------------------------------------


def test_compute_g1_total_elementwise_time_grid_avoids_truncation():
    """With N > 10001 unique lag times, passing time_grid must change the
    (previously silently-truncated) result vs the fallback-grid default.

    Pairs each point against t1=0 so t2 sweeps past the fallback grid's
    extent (10000*dt). Without an explicit time_grid, every t2 beyond that
    extent is clamped (via searchsorted+clip) to the SAME last grid index,
    so the tail of the result plateaus to a constant value -- the silent
    truncation this fix addresses. With the explicit time_grid the tail
    keeps varying instead of flat-lining.
    """
    n_unique = 10500
    dt = 1e-4
    # Element-wise mode requires 1D t1/t2 that get_cached_meshgrid will NOT
    # convert to a 2D meshgrid (it only meshgrids arrays <= 2000 long).
    t_unique = (jnp.arange(n_unique, dtype=jnp.float64) + 1.0) * dt
    t1 = jnp.zeros(n_unique, dtype=jnp.float64)
    t2 = t_unique
    phi = jnp.zeros(n_unique, dtype=jnp.float64)
    # q chosen so the diffusion decay is visible (not fully floored) across
    # the ~1.0 time-unit range probed here; only the truncation behavior
    # under test depends on this, not the physical realism of q.
    q = 0.5
    params = jnp.array([100.0, 0.0, 10.0], dtype=jnp.float64)  # diffusion-only

    # Old behavior: no time_grid -> falls back to the hardcoded 10001-point
    # grid (extent = 10000*dt = 1.0), which is SHORTER than the requested
    # range (n_unique*dt = 1.05), silently clipping the integral.
    g1_truncated = compute_g1_total(params, t1, t2, phi, q=q, L=1e6, dt=dt)

    # New behavior: an explicit time_grid covering the full range fixes it.
    g1_full = compute_g1_total(params, t1, t2, phi, q=q, L=1e6, dt=dt, time_grid=t_unique)

    assert jnp.all(jnp.isfinite(g1_truncated))
    assert jnp.all(jnp.isfinite(g1_full))

    # Tail region: t2 values beyond the fallback grid's extent (1.0).
    tail_mask = np.asarray(t2) > 1.0 + dt
    assert tail_mask.sum() > 100, "test setup must exercise a real tail region"

    truncated_tail_std = float(jnp.std(g1_truncated[tail_mask]))
    full_tail_std = float(jnp.std(g1_full[tail_mask]))

    assert truncated_tail_std < 1e-10, (
        "without time_grid, the tail (t2 beyond the fallback grid's extent) "
        f"must plateau to a constant (clipped) value; got std={truncated_tail_std:.3e}"
    )
    assert full_tail_std > 1e-8, (
        "with an explicit time_grid, the tail must keep varying instead of "
        f"flat-lining; got std={full_tail_std:.3e}"
    )


# ---------------------------------------------------------------------------
# Fix 3: heterodyne compute_chi_squared must exclude t=0 / diagonal, exactly
# like compute_residuals does
# ---------------------------------------------------------------------------


def _het_params():
    # [D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample, D_offset_sample,
    #  v0, beta, v_offset, f0, f1, f2, f3, phi0]
    return jnp.asarray(
        [1000.0, 0.9, 0.0, 1500.0, 1.0, 0.0, 50.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.float64,
    )


def test_heterodyne_chi_squared_matches_residuals_masking():
    """compute_chi_squared's masked support must equal
    sum(compute_residuals**2) -- both must exclude t=0 and the diagonal."""
    n_t = 8
    t = jnp.arange(1, n_t + 1, dtype=jnp.float64) * 0.1
    q, dt, phi_angle = 0.0054, 0.1, 0.0
    contrast, offset = 0.18, 1.05
    params = _het_params()

    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    rng = np.random.default_rng(0)
    c2_data = jnp.asarray(np.asarray(c2_model) + 0.01 * rng.standard_normal((n_t, n_t)))
    weights = jnp.ones((n_t, n_t), dtype=jnp.float64)

    chi2 = compute_chi_squared_het(params, t, q, dt, phi_angle, c2_data, weights, contrast, offset)
    residuals = compute_residuals_het(
        params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )

    assert jnp.allclose(chi2, jnp.sum(residuals**2), rtol=1e-10)

    # Sanity: masked support size matches _offdiag_indices exactly.
    rows, cols = _offdiag_indices(n_t)
    assert residuals.shape[0] == len(rows)


def test_heterodyne_chi_squared_ignores_diagonal_and_frame0_perturbation():
    """Perturbing ONLY the t=0 row/col or the diagonal must not change chi2,
    since those points are excluded from the masked support (matches the
    residual path's exclusion)."""
    n_t = 6
    t = jnp.arange(1, n_t + 1, dtype=jnp.float64) * 0.1
    q, dt, phi_angle = 0.0054, 0.1, 0.0
    contrast, offset = 0.18, 1.05
    params = _het_params()

    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    weights = jnp.ones((n_t, n_t), dtype=jnp.float64)

    baseline_chi2 = compute_chi_squared_het(
        params, t, q, dt, phi_angle, c2_model, weights, contrast, offset
    )

    # Corrupt the excluded support only: t=0 row/col and the full diagonal.
    c2_data_np = np.asarray(c2_model).copy()
    c2_data_np[0, :] += 999.0
    c2_data_np[:, 0] += 999.0
    np.fill_diagonal(c2_data_np, c2_data_np.diagonal() + 999.0)
    c2_data_corrupted = jnp.asarray(c2_data_np)

    corrupted_chi2 = compute_chi_squared_het(
        params, t, q, dt, phi_angle, c2_data_corrupted, weights, contrast, offset
    )

    assert jnp.allclose(baseline_chi2, corrupted_chi2, atol=1e-10), (
        "corrupting only the excluded (t=0 / diagonal) support must not "
        "change chi-squared once masking matches compute_residuals"
    )
